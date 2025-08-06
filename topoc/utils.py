"""Utility Functions"""

from typing import Callable, Tuple, Any, Optional, TYPE_CHECKING
import jax
from jax import random
from jax import Array, lax, debug, profiler
import jax.numpy as jnp
import functools
from functools import partial

import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

import time

from topoc.types import *

if TYPE_CHECKING:
    from topoc.base import TOProblemDefinition, TOAlgorithm
    

# region: Mathematical Functions

def linearize(fun: Callable) -> Callable:
    """
    Returns a Jacobian function with respect to x and u inputs,
    or just x if only one input. Handles normal functions and functools.partial.

    Returns:
    - For one argument: Callable[[Array], Array]
        Returns the Jacobian ∂f/∂x as an array.
    - For two arguments: Callable[[Array, Array], Tuple[Array, Array]]
        Returns the tuple of Jacobians (∂f/∂x, ∂f/∂u).
    Why wrapped?:
        It ensures that the function you pass to jax.jacrev always takes exactly the
        arguments you want to differentiate with respect to, regardless of whether the
        original function is a plain function or a functools.partial.
    """

    # Count the number of arguments the function expects after partial application
    def getnumargs(f):
        if isinstance(f, functools.partial):
            total = f.func.__code__.co_argcount
            num_fixed = len(f.args) + len(f.keywords) if f.keywords else len(f.args)
            return total - num_fixed
        return f.__code__.co_argcount

    nargs = getnumargs(fun)

    if nargs == 1:
        def wrapped(x):
            return fun(x)
        return jax.jacrev(wrapped, argnums=0)
    elif nargs == 2:
        def wrapped(x, u):
            return fun(x, u)
        return jax.jacrev(wrapped, argnums=(0, 1))
    else:
        raise ValueError("Function must take one or two arguments (x) or (x, u)")

def quadratize(fun: Callable) -> Callable:
    """
    Returns a Hessian function with respect to x and u inputs,
    or just x if only one input. Handles normal functions and functools.partial.

    Returns:
    - For one argument: Callable[[Array], Array]
        Returns the Hessian ∂²f/∂x² as an array.
    - For two arguments: Callable[[Array, Array], Tuple[Tuple[Array, Array], Tuple[Array, Array]]]
        Returns the block Hessian:
            (
                ( ∂²f/∂x², ∂²f/∂x∂u ),
                ( ∂²f/∂u∂x, ∂²f/∂u² )
            )
    """
    import functools

    def getnumargs(f):
        if isinstance(f, functools.partial):
            total = f.func.__code__.co_argcount
            num_fixed = len(f.args) + len(f.keywords) if f.keywords else len(f.args)
            return total - num_fixed
        return f.__code__.co_argcount

    nargs = getnumargs(fun)

    if nargs == 1:
        def wrapped(x):
            return fun(x)
        return jax.hessian(wrapped, argnums=0)
    elif nargs == 2:
        def wrapped(x, u):
            return fun(x, u)
        return jax.jacfwd(jax.jacrev(wrapped, argnums=(0, 1)), argnums=(0, 1))
    else:
        raise ValueError("Function must take one or two arguments (x) or (x, u)")

# endregion: Mathematical Functions

# region: Cost Functions for TOProblemDefinition

def quadratic_running_cost(x: Array, u: Array , xg: Array, params: Optional[Any] = None) -> Array:
    """
    Quadratic running cost: 0.5 * (x^T Q x + u^T R u)
    params should be a dict with 'Q' and 'R' arrays.
    """
    Q = params["Q"] if params and "Q" in params else jnp.eye(x.shape[0])
    R = params["R"] if params and "R" in params else jnp.eye(u.shape[0])
    cost = 0.5 * ((x-xg).T @ Q @ (x-xg) + u.T @ R @ u)
    return cost

def quadratic_terminal_cost(x: Array, xg: Array, params: Optional[Any] = None) -> Array:
    """
    Quadratic terminal cost: 0.5 * (x^T P x)
    params should be a dict with 'P' array.
    """
    P = params["P"] if params and "P" in params else jnp.eye(x.shape[0])
    cost = 0.5 * ((x-xg).T @ P @ (x-xg))
    return cost

# endregion: Cost Functions for TOProblemDefinition

# region: Functions for Algorithms

@partial(jax.jit, static_argnums=4)
def forward_pass(
        Xs: Array,
        Us: Array,
        Ks: Array,
        ks: Array,
        toproblem: "TOProblemDefinition",
        eps: float = 1.0, # linear search parameter
    ) -> Tuple[Tuple[Array, Array], float]:
        """
        Perform a forward pass.

        Parameters
        ----------
        Xs : Array
            The target state trajectory.
        Us : Array
            The control trajectory.
        Ks : Gains
            The gains obtained from the Backward Pass.
        dynamics : Callable
            The dynamics function of the system.
        runningcost : Callable
            The running cost function.
        finalcost : Callable
            The final cost function.
        eps : float, optional
            The linesearch parameter, by default 1.0.

        Returns
        -------
        [[NewStates, NewControls], TotalCost] -> Tuple[Tuple[Array, Array], float]
            A tuple containing the updated state trajectory and control trajectory, and the total cost
            of the trajectory.
        """

        dynamics = toproblem.dynamics
        runningcost = toproblem.runningcost
        terminalcost = toproblem.terminalcost
        dt = toproblem.modelparams.dt

        def dynamics_step(scan_state, scan_input):
            x, traj_cost = scan_state
            x_bar, u_bar, K, k = scan_input

            delta_x = x - x_bar
            delta_u = K @ delta_x + eps * k
            u_hat = u_bar + delta_u
            nx_hat = dynamics(x, u_hat)
            traj_cost = traj_cost + dt * runningcost(x, u_hat)
            # (x_t+1, V_t), (x_t+1, u_t)
            return (nx_hat, traj_cost), (nx_hat, u_hat)

        (xf, traj_cost), (new_Xs, new_Us) = lax.scan(
            dynamics_step, init=(Xs[0], 0.0), xs=(Xs[:-1], Us, Ks, ks)
        )
        total_cost = traj_cost + terminalcost(xf)
        new_Xs = jnp.vstack([Xs[0], new_Xs])
        return (new_Xs, new_Us), total_cost

# @partial(jax.jit, static_argnums=2)
def backward_pass(
    trajderivatives: TrajDerivatives,
    reg: float = 0.0, # Regularization term for Quu
    use_second_order_info: bool = False
):
    """
    iLQR/DDP backward pass using JAX with regularization, Cholesky PD check,
    and computation of expected cost decrease dV.
    """
    scaninputs_dict = trajderivatives._asdict()
    lfx = scaninputs_dict.pop('lfx')
    lfxx = scaninputs_dict.pop('lfxx')
    scaninputs_dict = {TRAJ_TO_WAYPOINT_RENAME_MAP.get(k, k): v for k, v in scaninputs_dict.items()}
    scaninputs = WaypointDerivatives(**scaninputs_dict)
    init_scanstate = (0.0, lfx, lfxx, True)

    def backward_step(scanstate, scaninput):
        dV, Vx, Vxx, success = scanstate

        def skip_step(_):
            # Use the same shapes as in compute_step
            m = scaninput.fu.shape[1]
            n = scaninput.fx.shape[1]
            dummy_K = jnp.zeros((m, n))
            dummy_k = jnp.zeros((m,))
            # # Print shapes for debugging
            # print("skip_step shapes: dummy_K", dummy_K.shape, "dummy_k", dummy_k.shape, "Vx", Vx.shape, "Vxx", Vxx.shape)
            return (dV, Vx, Vxx, False), (dummy_K, dummy_k, Vx, Vxx)

        def compute_step(_):
            qinfo = QInfo(
                scaninput,
                NextTimeStepVFDerivatives(Vx=Vx, Vxx=Vxx),
                use_second_order_info=use_second_order_info
            )
            Qx, Qu, Qxx, Qux, Quu = qinfo.Qx, qinfo.Qu, qinfo.Qxx, qinfo.Qux, qinfo.Quu
            n = Qxx.shape[0]
            m = Quu.shape[0]
            Quu = Quu + reg * jnp.eye(m)
            Qxx = Qxx + reg * jnp.eye(n)

            def on_pd(L):
                k = -jax.scipy.linalg.cho_solve((L, True), Qu)
                K = -jax.scipy.linalg.cho_solve((L, True), Qux)
                V_x_new = Qx + Qux.T @ k
                V_xx_new = Qxx + Qux.T @ K
                dV_new = dV + Qu.T @ k
                # # Print shapes for debugging
                # print("compute_step shapes: K", K.shape, "k", k.shape, "V_x_new", V_x_new.shape, "V_xx_new", V_xx_new.shape)
                return (dV_new, V_x_new, V_xx_new, True), (K, k, V_x_new, V_xx_new)

            def on_fail():
                dummy_K = jnp.zeros((m, n))
                dummy_k = jnp.zeros((m,))
                # # Print shapes for debugging
                # print("on_fail shapes: dummy_K", dummy_K.shape, "dummy_k", dummy_k.shape, "Vx", Vx.shape, "Vxx", Vxx.shape)
                return (dV, Vx, Vxx, False), (dummy_K, dummy_k, Vx, Vxx)

            def try_chol_safe(Q_uu):
                eps = 1e-9
                Q_uu_test = Q_uu - eps * jnp.eye(m)
                eigvals = jnp.linalg.eigvalsh(Q_uu_test)
                is_pd = jnp.all(eigvals > 0.0)
                L = jax.scipy.linalg.cholesky(Q_uu + 1e-9 * jnp.eye(m), lower=True)
                return is_pd, L

            is_pd, L = try_chol_safe(Quu)
            return lax.cond(
                is_pd,
                lambda _: on_pd(L),
                lambda _: on_fail(),
                operand=None
            )

        return lax.cond(
            success,
            compute_step,
            skip_step,
            operand=None
        )

    (dV, Vx, Vxx, success), scan_outputs = lax.scan(
        backward_step,
        init=init_scanstate,
        xs=scaninputs,
        reverse=True
    )

    # # Print shapes of scan_outputs for debugging
    # print("scan_outputs lengths:", [len(x) for x in scan_outputs])
    # for i, x in enumerate(zip(*scan_outputs)):
    #     shapes = [a.shape for a in x]
    #     print(f"scan_output step {i} shapes:", shapes)

    # Unpack and reverse outputs to forward-time order
    K_seq_rev, k_seq_rev, Vx_seq_rev, Vxx_seq_rev = scan_outputs
    K_seq = K_seq_rev
    k_seq = k_seq_rev
    Vx_seq = jnp.concatenate([Vx_seq_rev, lfx[None, :]], axis=0)
    Vxx_seq = jnp.concatenate([Vxx_seq_rev, lfxx[None, :, :]], axis=0)

    return dV, success, K_seq, k_seq, Vx_seq, Vxx_seq

@partial(jax.jit, static_argnums=2)
def traj_batch_derivatives(
    Xs,  # (N, Nx)
    Us,  # (N-1, Nu)
    toproblem: "TOProblemDefinition",
):
    """
    Compute all derivatives for a trajectory using linearize/quadratize signatures.

    Returns:
        TrajDerivatives object with attributes:
            fx, fu: (N-1, Nx, Nx), (N-1, Nx, Nu)
            fxx, fxu, fux, fuu: (N-1, Nx, Nx, Nx), (N-1, Nx, Nx, Nu), (N-1, Nx, Nu, Nx), (N-1, Nx, Nu, Nu)
            lx, lu: (N-1, Nx), (N-1, Nu)
            lxx, lxu, lux, luu: (N-1, Nx, Nx), (N-1, Nx, Nu), (N-1, Nu, Nx), (N-1, Nu, Nu)
            lfx: (Nx,)
            lfxx: (Nx, Nx)
    """
    graddynamics = toproblem.graddynamics
    hessiandynamics = toproblem.hessiandynamics
    gradrunningcost = toproblem.gradrunningcost
    hessianrunningcost = toproblem.hessianrunningcost
    gradterminalcost = toproblem.gradterminalcost
    hessiantterminalcost = toproblem.hessiantterminalcost
    
    # Apply to Xs[:-1] and Us (first N elements)
    fxs, fus = jax.vmap(graddynamics)(Xs[:-1], Us)
    (fxxs, fxus), (fuxs, fuus) = jax.vmap(hessiandynamics)(Xs[:-1], Us)

    lxs, lus = jax.vmap(gradrunningcost)(Xs[:-1], Us)
    (lxxs, lxus), (luxs, luus) = jax.vmap(hessianrunningcost)(Xs[:-1], Us)

    # Terminal cost on last state (Xs[-1])
    lfx = gradterminalcost(Xs[-1])
    lfxx = hessiantterminalcost(Xs[-1])

    return TrajDerivatives(
        fxs=fxs, fus=fus,
        fxxs=fxxs, fxus=fxus, fuxs=fuxs, fuus=fuus,
        lxs=lxs, lus=lus,
        lxxs=lxxs, lxus=lxus, luxs=luxs, luus=luus,
        lfx=lfx, lfxx=lfxx
    )


# region: Smoothed Trajectory Derivatives
@partial(jax.jit, static_argnums=(2, 4))
def input_smoothed_traj_batch_derivatives(
    Xs,  # (N, Nx)
    Us,  # (N-1, Nu)
    toproblem: "TOProblemDefinition",
    sigma: float = 1e-3,
    N_samples: int = 50,
    seed: int = 0,
    key: jax.Array = None,
):
    """
    Compute all derivatives for a trajectory using input_smoothed_dynamics_derivatives for dynamics part.
    Returns:
        TrajDerivatives object with attributes:
            fx, fu: (N-1, Nx, Nx), (N-1, Nx, Nu)
            fxx, fxu, fux, fuu: (N-1, Nx, Nx, Nx), (N-1, Nx, Nx, Nu), (N-1, Nx, Nu, Nx), (N-1, Nx, Nu, Nu)
            lx, lu: (N-1, Nx), (N-1, Nu)
            lxx, lxu, lux, luu: (N-1, Nx, Nx), (N-1, Nx, Nu), (N-1, Nu, Nx), (N-1, Nu, Nu)
            lfx: (Nx,)
            lfxx: (Nx, Nx)
    """
    # Use provided key or create from seed
    if key is None:
        key = random.PRNGKey(seed)
    # Get cost derivatives as usual
    gradrunningcost = toproblem.gradrunningcost
    hessianrunningcost = toproblem.hessianrunningcost
    gradterminalcost = toproblem.gradterminalcost
    hessiantterminalcost = toproblem.hessiantterminalcost

    # Prepare dynamics functions
    f = toproblem.dynamics
    fx = jax.jacrev(f, argnums=0)
    fxx = jax.hessian(f, argnums=0)

    # For each (x, u), call input_smoothed_dynamics_derivatives with a split key
    def smoothed_dyn_for_t(args):
        x, u, key = args
        return input_smoothed_dynamics_derivatives(x, u, f, fx, fxx, key, sigma, N_samples)

    # Split key for each time step
    keys = random.split(key, Us.shape[0])
    # Map over time steps
    fx_s, fxx_s, fu_s, fux_s, fuu_s = jax.vmap(smoothed_dyn_for_t)((Xs[:-1], Us, keys))

    # Cost derivatives as usual
    lxs, lus = jax.vmap(gradrunningcost)(Xs[:-1], Us)
    (lxxs, lxus), (luxs, luus) = jax.vmap(hessianrunningcost)(Xs[:-1], Us)

    # Terminal cost on last state (Xs[-1])
    lfx = gradterminalcost(Xs[-1])
    lfxx = hessiantterminalcost(Xs[-1])

    return TrajDerivatives(
        fxs=fx_s, fus=fu_s,
        fxxs=fxx_s, fxus=None, fuxs=fux_s, fuus=fuu_s,
        lxs=lxs, lus=lus,
        lxxs=lxxs, lxus=lxus, luxs=luxs, luus=luus,
        lfx=lfx, lfxx=lfxx
    )
# endregion: Smoothed Trajectory Derivatives

def input_smoothed_dynamics_derivatives(
    x,
    u,
    f,
    fx,
    fxx,
    key,
    sigma: float = 1e-3,
    N: int = 1000,
):
    """
    JAX/jittable, vectorized version of smoothed_dynamicsInfo.
    Uses (N, Nx) and (N, Nu) conventions for batching.
    Returns: fx_s, fxx_s, fu_s, fux_s, fuu_s
    """
    C = jnp.sqrt(sigma)
    Nx = x.shape[0]
    Nu = u.shape[0]

    # Sample perturbations [N, Nu]
    eps = jax.random.normal(key, (N, Nu))
    u_plus = u + C * eps  # [N, Nu]
    u_minus = u - C * eps  # [N, Nu]

    # Vectorized evaluation of f, fx, fxx
    f_plus = jax.vmap(lambda up: f(x, up))(u_plus)  # [N, Nx]
    f_minus = jax.vmap(lambda um: f(x, um))(u_minus)  # [N, Nx]
    # fs = (f_plus + f_minus) / 2  # [N, Nx]

    fx_all = jax.vmap(lambda up: fx(x, up))(u_plus)  # [N, Nx, Nx]
    fx_s = jnp.mean(fx_all, axis=0)  # [Nx, Nx]

    fxx_all = jax.vmap(lambda up: fxx(x, up))(u_plus)  # [N, Nx, Nx, Nx]
    fxx_s = jnp.mean(fxx_all, axis=0)  # [Nx, Nx, Nx]

    # fu
    fu_diff = (f_plus - f_minus) / (2 * C)  # [N, Nx]
    # [N, Nx] @ [N, Nu] -> [Nx, Nu]
    fu_s = (fu_diff.T @ eps) / N  # [Nx, Nu]

    # fuu
    f0 = f(x, u)  # [Nx]
    delta_f = f_plus + f_minus - 2 * f0  # [N, Nx]
    # eps_outer: [N, Nu, Nu]
    eps_outer = eps[:, :, None] * eps[:, None, :]  # [N, Nu, Nu]
    eps_outer = eps_outer - jnp.eye(Nu)[None, :, :]  # [N, Nu, Nu]
    # fuu_s: [Nx, Nu, Nu]
    fuu_s = jnp.einsum('nd,nij->dij', delta_f, eps_outer) / (2 * C**2 * N)

    # fx(x, u ± sigma*eps)
    fx_plus = jax.vmap(lambda up: fx(x, up))(u_plus)  # [N, Nx, Nx]
    fx_minus = jax.vmap(lambda um: fx(x, um))(u_minus)  # [N, Nx, Nx]
    fx_diff = (fx_plus - fx_minus) / (2 * C)  # [N, Nx, Nx]
    # fux: [Nx, Nu, Nx]
    # [N, Nx, Nx] and [N, Nu] -> [Nx, Nu, Nx]
    fux_s = jnp.einsum('nij,nk->ikj', fx_diff, eps) / N

    return fx_s, fxx_s, fu_s, fux_s, fuu_s


# region: Smoothed Trajectory Derivatives
@partial(jax.jit, static_argnums=(2, 5))
def state_input_smoothed_traj_batch_derivatives(
    Xs,  # (N, Nx)
    Us,  # (N-1, Nu)
    toproblem: "TOProblemDefinition",
    sigma_x: float = 1e-4,
    sigma_u: float = 1e-3,
    N_samples: int = 50,
    seed: int = 0,
    key: jax.Array = None,
):
    """
    Compute all derivatives for a trajectory using state_input_smoothed_dynamics_derivatives for dynamics part.
    Returns:
        TrajDerivatives object with attributes:
            fx, fu: (N-1, Nx, Nx), (N-1, Nx, Nu)
            fxx, fxu, fux, fuu: (N-1, Nx, Nx, Nx), (N-1, Nx, Nx, Nu), (N-1, Nx, Nu, Nx), (N-1, Nx, Nu, Nu)
            lx, lu: (N-1, Nx), (N-1, Nu)
            lxx, lxu, lux, luu: (N-1, Nx, Nx), (N-1, Nx, Nu), (N-1, Nu, Nx), (N-1, Nu, Nu)
            lfx: (Nx,)
            lfxx: (Nx, Nx)
    """
    # Use provided key or create from seed
    if key is None:
        key = random.PRNGKey(seed)
    # Get cost derivatives as usual
    gradrunningcost = toproblem.gradrunningcost
    hessianrunningcost = toproblem.hessianrunningcost
    gradterminalcost = toproblem.gradterminalcost
    hessiantterminalcost = toproblem.hessiantterminalcost

    # Prepare dynamics functions
    f = toproblem.dynamics

    # For each (x, u), call state_input_smoothed_dynamics_derivatives with a split key
    def smoothed_dyn_for_t(args):
        x, u, key = args
        return state_input_smoothed_dynamics_derivatives(x, u, f, key, sigma_x, sigma_u, N_samples)

    # Split key for each time step
    keys = random.split(key, Us.shape[0])
    # Map over time steps
    fx_s, fxx_s, fu_s, fux_s, fuu_s = jax.vmap(smoothed_dyn_for_t)((Xs[:-1], Us, keys))

    # Cost derivatives as usual
    lxs, lus = jax.vmap(gradrunningcost)(Xs[:-1], Us)
    (lxxs, lxus), (luxs, luus) = jax.vmap(hessianrunningcost)(Xs[:-1], Us)

    # Terminal cost on last state (Xs[-1])
    lfx = gradterminalcost(Xs[-1])
    lfxx = hessiantterminalcost(Xs[-1])

    return TrajDerivatives(
        fxs=fx_s, fus=fu_s,
        fxxs=fxx_s, fxus=None, fuxs=fux_s, fuus=fuu_s,
        lxs=lxs, lus=lus,
        lxxs=lxxs, lxus=lxus, luxs=luxs, luus=luus,
        lfx=lfx, lfxx=lfxx
    )
# endregion: Smoothed Trajectory Derivatives

def state_input_smoothed_dynamics_derivatives(
    x,
    u,
    f,
    key,
    sigma_x: float = 1e-4,
    sigma_u: float = 1e-3,
    N: int = 1000,
):
    """
    JAX/jittable, vectorized version of smoothed_dynamicsInfo.
    Uses (N, Nx) and (N, Nu) conventions for batching.
    Returns: fx_s, fxx_s, fu_s, fux_s, fuu_s
    """
    C_x = jnp.sqrt(sigma_x)
    C_u = jnp.sqrt(sigma_u)
    Nx = x.shape[0]
    Nu = u.shape[0]
    C = jnp.concatenate([jnp.full((Nx,), C_x), jnp.full((Nu,), C_u)])
    C = jnp.diag(C)
    C_inv = jnp.linalg.inv(C)
    f0 = f(x, u)  # [Nx]

    # Sample perturbations
    eps_x = jax.random.normal(key, (N, Nx))
    x_plus = x + C_x * eps_x  # [N, Nx]
    x_minus = x - C_x * eps_x  # [N, Nx]
    eps_u = jax.random.normal(key, (N, Nu))
    u_plus = u + C_u * eps_u  # [N, Nu]
    u_minus = u - C_u * eps_u  # [N, Nu]
    eps = jnp.concatenate([eps_x, eps_u], axis=1)  # [N, Nx + Nu]

    # Vectorized evaluation of f, fx, fxx
    f_plus = jax.vmap(lambda xp, up: f(xp, up))(x_plus, u_plus)  # [N, Nx]
    f_minus = jax.vmap(lambda xm, um: f(xm, um))(x_minus, u_minus)  # [N, Nx]

    # # fx, fu
    # f_diff = (f_plus - f_minus) / (2)  # [N, Nx]
    # # f_diff = f_plus  # [N, Nx]
    # # [N, Nx] @ [N, Nu] -> [Nx, Nu]
    # fz_s = (f_diff.T @ (C_inv @ eps.T).T) / N  # [Nx, Nx+Nu]
    # fx_s = fz_s[:, :Nx]  # [Nx, Nx]
    # fu_s = fz_s[:, Nx:]  # [Nx, Nu]

    fx_s, fu_s = calc_AB_lstsq(x_plus, u_plus, f_plus, x, u)

    # fu_s = (f_diff.T @ eps_u) / N / C_u  # [Nx, Nu]
    # fx_s = (f_diff.T @ eps_x) / N / C_x  # [Nx, Nx]

    # fxx, fux, fuu
    delta_f = (f_plus + f_minus - 2 * f0)/2  # [N, Nx]
    # eps_outer: [N, Nx+Nu, Nx+Nu]
    eps_outer = eps[:, :, None] * eps[:, None, :]  # [N, Nx+Nu, Nx+Nu]
    eps_outer = eps_outer - jnp.eye(Nx + Nu)[None, :, :]  # [N, Nx+Nu, Nx+Nu]
    eps_outer_transformed = jax.vmap(lambda e: C_inv @ e @ C_inv)(eps_outer)  # [N, Nx+Nu, Nx+Nu]
    fzz_s = jnp.einsum('nd,nij->dij', delta_f, eps_outer_transformed) / N  # [Nx, Nx+Nu, Nx+Nu]
    fxx_s = fzz_s[:, :Nx, :Nx]  # [Nx, Nx, Nx]
    fux_s = fzz_s[:, Nx:, :Nx]  # [Nx, Nu, Nx]
    fuu_s = fzz_s[:, Nx:, Nx:]  # [Nu, Nu, Nu]

    return fx_s, fxx_s, fu_s, fux_s, fuu_s


def QInfo(
    ctsdinfo: WaypointDerivatives,
    ntsdinfo: NextTimeStepVFDerivatives,
    use_second_order_info: bool = False,
):
    """
    Concise Q-function derivatives computation with optional second-order dynamics.
    JIT-compatible and fully functional for DDP or iLQR.
    """
    fx, fu, fxx, fux, fuu = ctsdinfo.fx, ctsdinfo.fu, ctsdinfo.fxx, ctsdinfo.fux, ctsdinfo.fuu
    lx, lu, lxx, lux, luu = ctsdinfo.lx, ctsdinfo.lu, ctsdinfo.lxx, ctsdinfo.lux, ctsdinfo.luu
    Vx, Vxx = ntsdinfo.Vx, ntsdinfo.Vxx

    # Vxx = Vxx + 1e-3 * jnp.eye(Vxx.shape[0])
    
    # First-order
    Qx = lx + fx.T @ Vx
    Qu = lu + fu.T @ Vx

    # Remainder terms (always compute them — JAX is smart about pruning)
    Qxx_rem = jnp.einsum('i,ijk->jk', Vx, fxx)
    Qux_rem = jnp.einsum('i,imn->mn', Vx, fux)
    Quu_rem = jnp.einsum('i,imn->mn', Vx, fuu)

    # Second-order
    Qxx = lxx + fx.T @ Vxx @ fx + (Qxx_rem if use_second_order_info else 0.0)
    Qux = lux + fu.T @ Vxx @ fx + (Qux_rem if use_second_order_info else 0.0)
    Quu = luu + fu.T @ Vxx @ fu + (Quu_rem if use_second_order_info else 0.0)

    return QDerivatives(Qx=Qx, Qu=Qu, Qxx=Qxx, Qux=Qux, Quu=Quu)

@partial(jax.jit, static_argnums=(6, 7, 8))
def forward_iteration(
    Xs,
    Us,
    Ks,
    ks,
    Vprev,
    dV,
    toproblem: "TOProblemDefinition",
    algorithm: "TOAlgorithm",
    max_fi_iters,  # Maximum number of backtracking steps
):
    """
    JIT-compatible line search with backtracking for DDP/iLQR forward pass.

    Args:
        Xs: Nominal state trajectory [T+1, n]
        Us: Nominal control trajectory [T, m]
        Ks: Feedback gains [T, m, n]
        ks: Feedforward terms [T, m]
        dynamics: Callable
        runningcost: Callable
        terminalcost: Callable
        dt: float
        Vprev: Previous cost (scalar)
        dV: Expected cost decrease (scalar)
        params: object or dict with .gamma, .beta
        max_iters: int, maximum number of backtracking steps

    Returns:
        V: New cost after forward pass
        Xs_new: New state trajectory
        Us_new: New control trajectory
        eps: Step size used
    """
    gamma = algorithm.params.gamma
    beta = algorithm.params.beta

    # # JIT-compile forward_pass with static_argnums for toproblem
    # forward_pass_jit = jax.jit(forward_pass, static_argnums=4)

    def body_fn(scanstate, _):
        eps, V, Xs_new, Us_new, done = scanstate

        def do_update(_):
            (Xs_new1, Us_new1), V1 = forward_pass(
                Xs, Us, Ks, ks, toproblem, eps
            )
            accept = V1 < Vprev + gamma * eps * (1 - eps / 2) * dV
            new_eps = lax.select(accept, eps, beta * eps)
            new_done = done | accept
            Xs_out = lax.select(accept, Xs_new1, Xs_new)
            Us_out = lax.select(accept, Us_new1, Us_new)
            V_out = lax.select(accept, V1, V)

            return (new_eps, V_out, Xs_out, Us_out, new_done), None

        def do_nothing(_): # Latch the state
            return (eps, V, Xs_new, Us_new, done), None

        result = lax.cond(done, do_nothing, do_update, operand=None)

        return result

    # Initial values
    eps_ini = 1.0
    Xs_ini = Xs
    Us_ini = Us
    # V_ini = 0.0
    V_ini = Vprev
    done_ini = False

    scanstate = (eps_ini, V_ini, Xs_ini, Us_ini, done_ini)
    finalscanstate, _ = lax.scan(body_fn, scanstate, xs=None, length=max_fi_iters)
    

    eps, V, Xs_new, Us_new, done = finalscanstate

    return Xs_new, Us_new, V, eps, done


def calc_AB_lstsq(x_batch, u_batch, x_next_batch, x_nominal, u_nominal):
    """
    Estimate A, B from batch data.

    Args:
        x_batch: (N, n_x) batch of initial states
        u_batch: (N, n_u) batch of inputs
        x_next_batch: (N, n_x) batch of next states
        x_nominal: (n_x,) nominal state
        u_nominal: (n_u,) nominal input
    
    Returns:
        A: (n_x, n_x) state matrix
        B: (n_x, n_u) input matrix
        c: (n_x,) offset (mean of x_next_batch)
    """
    N, n_x = x_batch.shape
    n_u = u_batch.shape[1]
    
    # Center data
    dx = x_batch - x_nominal  # (N, n_x)
    du = u_batch - u_nominal  # (N, n_u)
    x_next_mean = jnp.mean(x_next_batch, axis=0)
    dy = x_next_batch - x_next_mean  # (N, n_x)

    # Stack inputs horizontally: [du | dx]
    input_mat = jnp.hstack([du, dx])  # (N, n_u + n_x)

    # Solve least squares: dy = input_mat @ [B.T; A.T]
    BA_T, residuals, rank, s = jnp.linalg.lstsq(input_mat, dy, rcond=None)
    BA = BA_T.T  # Transpose back to match shape
    
    B = BA[:, :n_u]  # first n_u columns
    A = BA[:, n_u:]  # remaining n_x columns

    return A, B


# endregion: Functions for Algorithms


# region: Plotting Functions

def plot_block_results(result, x0, xg, modelparams):
    """
    Plot block optimization results with three subplots:
    1. Position vs Velocity (phase plot, colored by time)
    2. Input vs Time (colored by time)
    3. Vstore vs Iterations (colored from red to green as values decrease)
    result: object with attributes xbar, ubar, Vstore
    x0: (2,) initial state
    xg: (2,) goal state
    modelparams: ModelParams object (for dt and horizon_len)
    """
    import numpy as np
    plt.style.use('seaborn-v0_8-darkgrid')
    mpl.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.figsize": (18, 6),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.3,
        "lines.linewidth": 2.5,
    })

    xbar = np.asarray(result.xbar)
    ubar = np.asarray(result.ubar)
    Vstore = np.asarray(result.Vstore)
    dt = modelparams.dt
    H = modelparams.horizon_len
    positions = xbar[:, 0]
    velocities = xbar[:, 1]
    timesteps = np.arange(len(positions))
    times = timesteps * dt

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Phase plot: position vs velocity
    ax = axs[0]
    points = np.stack([positions, velocities], axis=1).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(times[0], times[-1])
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(times)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    ax.scatter(x0[0], x0[1], color='#43a047', s=100, marker='o', label='Start', zorder=3)
    ax.scatter(xg[0], xg[1], color='#e53935', s=100, marker='*', label='Goal', zorder=3)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('State Trajectory')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    cbar = plt.colorbar(line, ax=ax, pad=0.02)
    cbar.set_label(f'Time (s) [Horizon: {H}]')


    # 2. Input vs time
    ax2 = axs[1]
    input_times = np.arange(len(ubar)) * dt
    uvals = ubar[:, 0] if ubar.ndim > 1 else ubar
    points_u = np.stack([input_times, uvals], axis=1).reshape(-1, 1, 2)
    segments_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
    norm_u = plt.Normalize(input_times[0], input_times[-1])
    lc_u = LineCollection(segments_u, cmap='viridis', norm=norm_u)
    lc_u.set_array(input_times[:-1])
    lc_u.set_linewidth(3)
    line_u = ax2.add_collection(lc_u)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Input (u)')
    ax2.set_title('Input Trajectory')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax2.set_axisbelow(True)
    ax2.set_xlim(input_times[0], input_times[-1])
    ax2.set_ylim(uvals.min() - 0.1 * abs(uvals.min()), uvals.max() + 0.1 * abs(uvals.max()))
    # Remove colorbar and y-axis label for time in input subplot


    # 3. Vstore vs iterations
    ax3 = axs[2]
    iterations = np.arange(len(Vstore))
    # Color from red (high) to green (low) -- reverse colormap so high=red, low=green
    from matplotlib.colors import LinearSegmentedColormap
    red_green = LinearSegmentedColormap.from_list('red_green', ['#43a047', '#e53935'])
    # For correct color mapping, use Vstore[:-1] for segments
    # Use log scale for color normalization
    vmin = np.clip(Vstore.min(), 1e-12, None)
    vmax = Vstore.max()
    from matplotlib.colors import LogNorm
    norm_v = LogNorm(vmin, vmax)
    points_v = np.stack([iterations, Vstore], axis=1).reshape(-1, 1, 2)
    segments_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
    # Color by Vstore value at start of each segment
    lc_v = LineCollection(segments_v, cmap=red_green, norm=norm_v)
    lc_v.set_array(Vstore[:-1])
    lc_v.set_linewidth(3)
    line_v = ax3.add_collection(lc_v)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Cost (Vstore)')
    ax3.set_title('Cost vs Iteration (Log Scale)')
    ax3.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax3.set_axisbelow(True)
    ax3.set_xlim(iterations[0], iterations[-1])
    ax3.set_yscale('log')
    # Set y-limits for log scale, avoid log(0)
    ax3.set_ylim(vmin, vmax * 1.05)
    cbar3 = plt.colorbar(line_v, ax=ax3, pad=0.02)
    cbar3.set_label('Cost (Vstore)')

    plt.tight_layout()
    plt.show()


def plot_pendulum_results(result, x0, xg, modelparams):
    """
    Plot pendulum optimization results with three subplots:
    1. Angular Position vs Angular Velocity (phase plot, colored by time)
    2. Input vs Time (colored by time)
    3. Vstore vs Iterations (colored from red to green as values decrease)
    result: object with attributes xbar, ubar, Vstore
    x0: (2,) initial state (angle, angular velocity)
    xg: (2,) goal state (angle, angular velocity)
    modelparams: ModelParams object (for dt and horizon_len)
    """
    import numpy as np
    plt.style.use('seaborn-v0_8-darkgrid')
    mpl.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.figsize": (18, 6),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.3,
        "lines.linewidth": 2.5,
    })

    xbar = np.asarray(result.xbar)
    ubar = np.asarray(result.ubar)
    Vstore = np.asarray(result.Vstore)
    dt = modelparams.dt
    H = modelparams.horizon_len
    angles = xbar[:, 0]
    ang_vels = xbar[:, 1]
    timesteps = np.arange(len(angles))
    times = timesteps * dt

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Phase plot: angular position vs angular velocity
    ax = axs[0]
    points = np.stack([angles, ang_vels], axis=1).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(times[0], times[-1])
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(times)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    ax.scatter(x0[0], x0[1], color='#43a047', s=100, marker='o', label='Start', zorder=3)
    ax.scatter(xg[0], xg[1], color='#e53935', s=100, marker='*', label='Goal', zorder=3)
    ax.set_xlabel('Angle (rad)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_title('Pendulum: State Trajectory')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    cbar = plt.colorbar(line, ax=ax, pad=0.02)
    cbar.set_label(f'Time (s) [Horizon: {H}]')

    # 2. Input vs time
    ax2 = axs[1]
    input_times = np.arange(len(ubar)) * dt
    uvals = ubar[:, 0] if ubar.ndim > 1 else ubar
    points_u = np.stack([input_times, uvals], axis=1).reshape(-1, 1, 2)
    segments_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
    norm_u = plt.Normalize(input_times[0], input_times[-1])
    lc_u = LineCollection(segments_u, cmap='viridis', norm=norm_u)
    lc_u.set_array(input_times[:-1])
    lc_u.set_linewidth(3)
    line_u = ax2.add_collection(lc_u)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Input (u)')
    ax2.set_title('Input Trajectory')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax2.set_axisbelow(True)
    ax2.set_xlim(input_times[0], input_times[-1])
    ax2.set_ylim(uvals.min() - 0.1 * abs(uvals.min()), uvals.max() + 0.1 * abs(uvals.max()))
    # Remove colorbar and y-axis label for time in input subplot

    # 3. Vstore vs iterations
    ax3 = axs[2]
    iterations = np.arange(len(Vstore))
    from matplotlib.colors import LinearSegmentedColormap
    red_green = LinearSegmentedColormap.from_list('red_green', ['#43a047', '#e53935'])
    vmin = np.clip(Vstore.min(), 1e-12, None)
    vmax = Vstore.max()
    from matplotlib.colors import LogNorm
    norm_v = LogNorm(vmin, vmax)
    points_v = np.stack([iterations, Vstore], axis=1).reshape(-1, 1, 2)
    segments_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
    lc_v = LineCollection(segments_v, cmap=red_green, norm=norm_v)
    lc_v.set_array(Vstore[:-1])
    lc_v.set_linewidth(3)
    line_v = ax3.add_collection(lc_v)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Cost (Vstore)')
    ax3.set_title('Cost vs Iteration (Log Scale)')
    ax3.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax3.set_axisbelow(True)
    ax3.set_xlim(iterations[0], iterations[-1])
    ax3.set_yscale('log')
    ax3.set_ylim(vmin, vmax * 1.05)
    cbar3 = plt.colorbar(line_v, ax=ax3, pad=0.02)
    cbar3.set_label('Cost (Vstore)')

    plt.tight_layout()
    plt.show()

def plot_cartpole_results(result, x0, xg, modelparams):
    """
    Plot cartpole optimization results with four subplots (2x2 grid):
    (1,1): Position vs Velocity (colored by time)
    (2,1): Angular Position vs Angular Velocity (colored by time)
    (1,2): Input vs Time
    (2,2): Vstore vs Iterations (log scale, red to green)
    result: object with attributes xbar, ubar, Vstore
    x0: (4,) initial state (pos, vel, angpos, angvel)
    xg: (4,) goal state (pos, vel, angpos, angvel)
    modelparams: ModelParams object (for dt and horizon_len)
    """
    import numpy as np
    plt.style.use('seaborn-v0_8-darkgrid')
    mpl.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.figsize": (14, 10),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.3,
        "lines.linewidth": 2.5,
    })

    xbar = np.asarray(result.xbar)
    ubar = np.asarray(result.ubar)
    Vstore = np.asarray(result.Vstore)
    dt = modelparams.dt
    H = modelparams.horizon_len
    pos = xbar[:, 0]
    vel = xbar[:, 1]
    angpos = xbar[:, 2]
    angvel = xbar[:, 3]
    timesteps = np.arange(len(pos))
    times = timesteps * dt

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # (1,1): Position vs Velocity
    ax = axs[0, 0]
    points = np.stack([pos, vel], axis=1).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(times[0], times[-1])
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(times)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    ax.scatter(x0[0], x0[1], color='#43a047', s=100, marker='o', label='Start', zorder=3)
    ax.scatter(xg[0], xg[1], color='#e53935', s=100, marker='*', label='Goal', zorder=3)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Cartpole: Position vs Velocity')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    cbar = plt.colorbar(line, ax=ax, pad=0.02)
    cbar.set_label(f'Time (s) [Horizon: {H}]')

    # (1,2): Angular Position vs Angular Velocity
    ax21 = axs[0, 1]
    points_a = np.stack([angpos, angvel], axis=1).reshape(-1, 1, 2)
    segments_a = np.concatenate([points_a[:-1], points_a[1:]], axis=1)
    norm_a = plt.Normalize(times[0], times[-1])
    lc_a = LineCollection(segments_a, cmap='viridis', norm=norm_a) # plasma
    lc_a.set_array(times)
    lc_a.set_linewidth(3)
    line_a = ax21.add_collection(lc_a)
    ax21.scatter(x0[2], x0[3], color='#43a047', s=100, marker='o', label='Start', zorder=3)
    ax21.scatter(xg[2], xg[3], color='#e53935', s=100, marker='*', label='Goal', zorder=3)
    ax21.set_xlabel('Angle (rad)')
    ax21.set_ylabel('Angular Velocity (rad/s)')
    ax21.set_title('Cartpole: Angle vs Angular Velocity')
    ax21.legend(loc='best', frameon=True)
    ax21.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax21.set_axisbelow(True)

    # (2,1): Input vs Time
    ax12 = axs[1, 0]
    input_times = np.arange(len(ubar)) * dt
    uvals = ubar[:, 0] if ubar.ndim > 1 else ubar
    points_u = np.stack([input_times, uvals], axis=1).reshape(-1, 1, 2)
    segments_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
    norm_u = plt.Normalize(input_times[0], input_times[-1])
    lc_u = LineCollection(segments_u, cmap='viridis', norm=norm_u)
    lc_u.set_array(input_times[:-1])
    lc_u.set_linewidth(3)
    line_u = ax12.add_collection(lc_u)
    ax12.set_xlabel('Time (s)')
    ax12.set_ylabel('Input (u)')
    ax12.set_title('Input Trajectory')
    ax12.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax12.set_axisbelow(True)
    ax12.set_xlim(input_times[0], input_times[-1])
    ax12.set_ylim(uvals.min() - 0.1 * abs(uvals.min()), uvals.max() + 0.1 * abs(uvals.max()))
    # No colorbar for input subplot

    # (2,2): Vstore vs Iterations
    ax22 = axs[1, 1]
    iterations = np.arange(len(Vstore))
    from matplotlib.colors import LinearSegmentedColormap
    red_green = LinearSegmentedColormap.from_list('red_green', ['#43a047', '#e53935'])
    vmin = np.clip(Vstore.min(), 1e-12, None)
    vmax = Vstore.max()
    from matplotlib.colors import LogNorm
    norm_v = LogNorm(vmin, vmax)
    points_v = np.stack([iterations, Vstore], axis=1).reshape(-1, 1, 2)
    segments_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
    lc_v = LineCollection(segments_v, cmap=red_green, norm=norm_v)
    lc_v.set_array(Vstore[:-1])
    lc_v.set_linewidth(3)
    line_v = ax22.add_collection(lc_v)
    ax22.set_xlabel('Iteration')
    ax22.set_ylabel('Cost (Vstore)')
    ax22.set_title('Cost vs Iteration (Log Scale)')
    ax22.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax22.set_axisbelow(True)
    ax22.set_xlim(iterations[0], iterations[-1])
    ax22.set_yscale('log')
    ax22.set_ylim(vmin, vmax * 1.05)
    cbar3 = plt.colorbar(line_v, ax=ax22, pad=0.02)
    cbar3.set_label('Cost (Vstore)')

    plt.tight_layout()
    plt.show()

# endregion: Plotting Functions
