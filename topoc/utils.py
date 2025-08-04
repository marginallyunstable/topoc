"""Utility Functions"""

from typing import Callable, Tuple, Any, Optional, TYPE_CHECKING
import jax
from jax import Array, lax, debug, profiler
import jax.numpy as jnp
import functools
from functools import partial
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

@jax.jit
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
    K_seq = K_seq_rev[::-1]
    k_seq = k_seq_rev[::-1]
    Vx_seq = jnp.concatenate([Vx_seq_rev[::-1], lfx[None, :]], axis=0)
    Vxx_seq = jnp.concatenate([Vxx_seq_rev[::-1], lfxx[None, :, :]], axis=0)

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
    max_iters: int = 50,  # Maximum number of backtracking steps
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
            print("Starting forward_pass...")  # Changed from forward_pass_jit
            start_fp = time.time()
            (Xs_new1, Us_new1), V1 = forward_pass(
                Xs, Us, Ks, ks, toproblem, eps
            )
            end_fp = time.time()
            print(f"forward_pass took {end_fp - start_fp:.8f} seconds")

            print("Evaluating acceptance condition...")
            start_cond = time.time()
            accept = V1 < Vprev + gamma * eps * (1 - eps / 2) * dV
            new_eps = lax.select(accept, eps, beta * eps)
            new_done = done | accept
            Xs_out = lax.select(accept, Xs_new1, Xs_new)
            Us_out = lax.select(accept, Us_new1, Us_new)
            V_out = lax.select(accept, V1, V)
            end_cond = time.time()
            print(f"Acceptance condition evaluation took {end_cond - start_cond:.8f} seconds")

            return (new_eps, V_out, Xs_out, Us_out, new_done), None

        def do_nothing(_): # Latch the state
            return (eps, V, Xs_new, Us_new, done), None

        print("Checking done condition...")
        start_done = time.time()
        result = lax.cond(done, do_nothing, do_update, operand=None)
        end_done = time.time()
        print(f"Done condition check took {end_done - start_done:.8f} seconds")

        return result

    # Initial values
    eps_ini = 1.0
    Xs_ini = jnp.zeros_like(Xs)
    Us_ini = jnp.zeros_like(Us)
    V_ini = 0.0
    done_ini = False

    scanstate = (eps_ini, V_ini, Xs_ini, Us_ini, done_ini)
    print("Starting lax.scan...")
    start_scan = time.time()
    finalscanstate, _ = lax.scan(body_fn, scanstate, xs=None, length=max_iters)
    

    eps, V, Xs_new, Us_new, done = finalscanstate

    end_scan = time.time()
    print(f"lax.scan took {end_scan - start_scan:.8f} seconds")

    return Xs_new, Us_new, V, eps, done


# endregion: Functions for Algorithms
