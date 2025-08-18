"""Utility Functions"""

from typing import Callable, Tuple, Any, Optional, TYPE_CHECKING
import jax
from jax import random
from jax import Array, lax, debug, profiler
import jax.numpy as jnp
import functools
from functools import partial

import itertools
import math
from jax.scipy.special import gammaln
from itertools import combinations, product

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

def quadratic_running_cost_qsim(x: Array, u: Array , xg: Array, params: Optional[Any] = None) -> Array:
    """
    Quadratic running cost: 0.5 * (x^T Q x + u^T R u)
    params should be a dict with 'Q' and 'R' arrays.
    """
    Q = params["Q"] if params and "Q" in params else jnp.eye(x.shape[0])
    R = params["R"] if params and "R" in params else jnp.eye(u.shape[0])
    nu = u.shape[0]
    cost = 0.5 * ((x-xg).T @ Q @ (x-xg) + (u - x[-nu:]).T @ R @ (u - x[-nu:]))
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

@partial(jax.jit, static_argnums=(4))
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
        xini = toproblem.starting_state

        def dynamics_step(scan_state, scan_input):
            x, traj_cost = scan_state
            x_bar, u_bar, K, k = scan_input

            delta_x = x - x_bar
            delta_u = K @ delta_x + eps * k
            u = u_bar + delta_u
            nx = dynamics(x, u)
            traj_cost = traj_cost + dt * runningcost(x, u)
            # (x_t+1, V_t), (x_t+1, u_t)
            return (nx, traj_cost), (nx, u)

        (xf, traj_cost), (new_next_Xs, new_Us) = lax.scan(
            dynamics_step, init=(xini, 0.0), xs=(Xs[:-1], Us, Ks, ks)
        )
        total_cost = traj_cost + terminalcost(xf)
        new_Xs = jnp.vstack([xini, new_next_Xs])
        return (new_Xs, new_Us), total_cost

@partial(jax.jit, static_argnames=("toproblem", "toalgorithm"))
def forward_pass_spm(
        Xs: Array,
        Us: Array,
        Ks: Array,
        ks: Array,
        cov_policy: Array,
        toproblem: "TOProblemDefinition",
        toalgorithm: "TOAlgorithm",
        eps: float = 1.0, # linear search parameter
    ):
        """
        Perform a forward pass using sigma point method.

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
        batched_dynamics = jax.vmap(dynamics)
        runningcost = toproblem.runningcost
        terminalcost = toproblem.terminalcost
        xdim = toproblem.modelparams.state_dim
        udim = toproblem.modelparams.input_dim
        H = toproblem.modelparams.horizon_len
        dt = toproblem.modelparams.dt
        xini = toproblem.starting_state
        cov_xini = toproblem.starting_state_cov
        spg_func = get_spg_func(toalgorithm.params.spg_method)
        # spg_func = globals()[toalgorithm.params.spg_method]
        
        wsp_r, usp_r = spg_func(xdim + udim) # sigma point weights (wsp_r) and unit sigma points (usp_r) for running part of trajectory
 
        def dynamics_step(scan_state, scan_input):
            x, cov_x, traj_cost = scan_state
            x_bar, u_bar, K, k, cov_policy = scan_input

            delta_x = x - x_bar
            delta_u = K @ delta_x + eps * k
            u = u_bar + delta_u

            cov_uu = cov_policy + K @  cov_x @ K.T
            cov_ux = K @ cov_x

            cov_z = jnp.block([
                [cov_x, cov_ux.T],
                [cov_ux, cov_uu]
            ]) # shape (nx + nu, nx + nu)

            chol_cov_z = jax.scipy.linalg.cholesky(cov_z, lower=True)

            # Generate sigma points
            # Stack x and u together
            z = jnp.concatenate([x, u])  # shape (nx + nu,)

            # Generate sigma points: z + cov_z_sqrt @ usp_r
            sigma_points = z[:, None] + chol_cov_z @ usp_r  # shape (nz, N_sigma)
            sigma_points = sigma_points.T  # shape (N_sigma, nz)

            x_sigma_points = sigma_points[:, :xdim]  # shape (N_sigma, nx)
            u_sigma_points = sigma_points[:, xdim:]  # shape (N_sigma, nu)

            # transported sigma points to state space through dynamics
            nx_sigma_points = batched_dynamics(x_sigma_points, u_sigma_points) # shape (N_sigma, nx) 

            nx = wsp_r @ nx_sigma_points  # shape (nx,)

            ncov_x = (nx_sigma_points - nx).T @ (wsp_r[:, None] * (nx_sigma_points - nx))  # shape (nx, nx)

            traj_cost = traj_cost + dt * runningcost(x, u)

            return (nx, ncov_x, traj_cost), (nx, u, sigma_points, nx_sigma_points, cov_z, chol_cov_z)

        (xf, cov_xf, traj_cost), (new_next_Xs, new_Us, new_SPs, new_nX_SPs, new_Covs_Zs, new_chol_Covs_Zs) = lax.scan(
            dynamics_step, init=(xini, cov_xini, 0.0), xs=(Xs[:-1], Us, Ks, ks, cov_policy)
        )
        
        total_cost = traj_cost + terminalcost(xf)
        new_Xs = jnp.vstack([xini, new_next_Xs])
        
        pad_width = ((0, udim), (0, udim))  # pad nu zeros to bottom and right
        cov_xf_padded = jnp.pad(cov_xf, pad_width, mode='constant')
        new_Covs_Zs = jnp.concatenate([new_Covs_Zs, cov_xf_padded[None, :, :]], axis=0)  # shape (T+1, nx+nu, nx+nu)

        chol_cov_xf = jax.scipy.linalg.cholesky(cov_xf, lower=True)
        chol_cov_xf_padded = jnp.pad(chol_cov_xf, pad_width, mode='constant')
        new_chol_Covs_Zs = jnp.concatenate([new_chol_Covs_Zs, chol_cov_xf_padded[None, :, :]], axis=0)  # shape (T+1, nx+nu, nx+nu)

        return (new_Xs, new_Us, new_SPs, new_nX_SPs, new_Covs_Zs, new_chol_Covs_Zs), total_cost


@partial(jax.jit, static_argnums=2)
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
            Quu_reg = Quu + reg * jnp.eye(m)
            # Qxx_reg = Qxx + reg * jnp.eye(n) # TODO: check if needed

            def on_pd(L):
                k = -jax.scipy.linalg.cho_solve((L, True), Qu)
                K = -jax.scipy.linalg.cho_solve((L, True), Qux)
                
                V_x_new = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
                V_xx_new = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
                # V_x_new = Qx + Qux.T @ k  # To use if Quu not regularized
                # V_xx_new = Qxx + Qux.T @ K   # To use if Quu not regularized
                dV_new = dV + Qu.T @ k # TODO: check this (eq 14 FHDDP paper)
                return (dV_new, V_x_new, V_xx_new, True), (K, k, V_x_new, V_xx_new)

            def on_fail():
                dummy_K = jnp.zeros((m, n))
                dummy_k = jnp.zeros((m,))
                return (dV, Vx, Vxx, False), (dummy_K, dummy_k, Vx, Vxx)

            def try_chol_safe(Q_uu):
                eps = 1e-9
                Q_uu_test = Q_uu - eps * jnp.eye(m)
                eigvals = jnp.linalg.eigvalsh(Q_uu_test)
                is_pd = jnp.all(eigvals > 0.0)
                L = jax.scipy.linalg.cholesky(Q_uu + 1e-9 * jnp.eye(m), lower=True)
                return is_pd, L

            is_pd, L = try_chol_safe(Quu_reg)
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

    # Unpack and reverse outputs to forward-time order
    K_seq_rev, k_seq_rev, Vx_seq_rev, Vxx_seq_rev = scan_outputs
    K_seq = K_seq_rev
    k_seq = k_seq_rev
    Vx_seq = jnp.concatenate([Vx_seq_rev, lfx[None, :]], axis=0)
    Vxx_seq = jnp.concatenate([Vxx_seq_rev, lfxx[None, :, :]], axis=0)

    return dV, success, K_seq, k_seq, Vx_seq, Vxx_seq


@partial(jax.jit, static_argnames=("toproblem", "toalgorithm"))
def backward_pass_spm(
    Xs,  # (N, Nx)
    Us,  # (N-1, Nu)
    ks, Ks, # Control gains
    SPs,  # (N-1, Nsigma, Nx+Nu)
    nX_SPs,  # (N-1, Nsigma, Nx)
    Covs_Zs,  # (N-1, Nsigma, Nx+Nu, Nx+Nu)
    chol_Covs_Zs,  # (N-1, Nsigma, Nx+Nu, Nx+Nu)
    zeta,
    Cov_policy,
    Cov_policy_inv,
    toproblem: "TOProblemDefinition",
    toalgorithm: "TOAlgorithm",
    reg: float = 0.0, # Regularization term for Quu
    use_second_order_info: bool = False
):
    """
    iLQR/DDP backward pass using JAX with regularization, Cholesky PD check,
    and computation of expected cost decrease dV.
    """

    runningcost = toproblem.runningcost
    terminalcost = toproblem.terminalcost
    xdim = toproblem.modelparams.state_dim
    udim = toproblem.modelparams.input_dim
    dt = toproblem.modelparams.dt
    lam_ = toalgorithm.params.lam
    eta = toalgorithm.params.eta
    spg_func = get_spg_func(toalgorithm.params.spg_method)
    # spg_func = globals()[toalgorithm.params.spg_method]
    wsp_r, usp_r = spg_func(xdim + udim)
    wsp_f, usp_f = spg_func(xdim)

    # Terminal Step
    lfxx, lfx, lf = VTerminalInfo_spm(Xs[-1], chol_Covs_Zs[-1][:xdim, :xdim], terminalcost, wsp_f, usp_f)

    nXs = Xs[1:]
    scaninputs = (nXs, SPs, nX_SPs, chol_Covs_Zs[:-1], Cov_policy, Cov_policy_inv, ks, Ks)
    init_scanstate = (0.0, lf, lfx, lfxx, True)

    def backward_step(scanstate, scaninput):
        dV, V, Vx, Vxx, success = scanstate
        nX, SPs, nX_SPs, chol_Cov, Cov_policy_, Cov_policy_inv_, k_, K_ = scaninput

        def skip_step(_):
            return (dV, V, Vx, Vxx, False), (K_, k_, V, Vx, Vxx, Cov_policy_, Cov_policy_inv_)

        def compute_step(_):
            qinfo = QInfo_spm(
                Vxx, Vx, V,
                nX, SPs, nX_SPs, chol_Cov,
                runningcost, dt, wsp_r, usp_r,
            )
            Q, Qx, Qu, Qxx, Qxu, Quu = qinfo
            n = Qxx.shape[0]
            m = Quu.shape[0]

            lam = lam_*zeta
            Cov_policy_inv_reg = eta * Cov_policy_inv_ + lam * Quu  + reg * jnp.eye(m)
            # Cov_policy_inv_reg = eta * Cov_policy_inv_ + lam * Quu  + zeta * jnp.eye(m)

            def on_pd():
                
                X = jnp.eye(m)  # if you need the explicit inverse
                Cov_policy_reg = jax.numpy.linalg.solve(Cov_policy_inv_reg, X)
                k = Cov_policy_reg @ (eta * Cov_policy_inv_ @ k_ - lam * Qu)
                K = Cov_policy_reg @ (eta * Cov_policy_inv_ @ K_ - lam * Qxu.T)

                Vxx_new   = Qxx + (eta * K_.T @ Cov_policy_inv_ @ K_ - K.T @ Cov_policy_inv_reg @ K) / (lam)
                Vx_new    = Qx  + (eta * K_.T @ Cov_policy_inv_ @ k_ - K.T @ Cov_policy_inv_reg @ k) / (lam)
                # TODO: correct this see eqn 46a
                V_new     = Q   + (eta * k_.T @ Cov_policy_inv_ @ k_ - k.T @ Cov_policy_inv_reg @ k) / (lam)
                dV_new    = dV  + (eta * k_.T @ Cov_policy_inv_ @ k_ - k.T @ Cov_policy_inv_reg @ k) / (lam) # TODO: correct this
                
                return (dV_new, V_new, Vx_new, Vxx_new, True), (K, k, V_new, Vx_new, Vxx_new, Cov_policy_reg, Cov_policy_inv_reg)

            def on_fail():
                return (dV, V, Vx, Vxx, False), (K_, k_, V, Vx, Vxx, Cov_policy_, Cov_policy_inv_)

            def try_chol_safe(Q_uu):
                eps = 1e-9
                Q_uu_test = Q_uu - eps * jnp.eye(m)
                eigvals = jnp.linalg.eigvalsh(Q_uu_test)
                is_pd = jnp.all(eigvals > 0.0)
                L = jax.scipy.linalg.cholesky(Q_uu + 1e-9 * jnp.eye(m), lower=True)

                # is_pd = True

                return is_pd, L

            is_pd, L = try_chol_safe(Cov_policy_inv_reg)
            return lax.cond(
                is_pd,
                lambda _: on_pd(),
                lambda _: on_fail(),
                operand=None
            )

        return lax.cond(
            success,
            compute_step,
            skip_step,
            operand=None
        )

    (dV, V, Vx, Vxx, success), scan_outputs = lax.scan(
        backward_step,
        init=init_scanstate,
        xs=scaninputs,
        reverse=True
    )

    # Unpack and reverse outputs to forward-time order
    K_seq_rev, k_seq_rev, V_seq_rev, Vx_seq_rev, Vxx_seq_rev, Cov_policy_seq_rev, Cov_policy_inv_seq_rev = scan_outputs
    K_seq = K_seq_rev
    k_seq = k_seq_rev
    Cov_policy_seq = Cov_policy_seq_rev
    Cov_policy_inv_seq = Cov_policy_inv_seq_rev
    V_seq = jnp.concatenate([V_seq_rev, jnp.array([lf])], axis=0)
    Vx_seq = jnp.concatenate([Vx_seq_rev, lfx[None, :]], axis=0)
    Vxx_seq = jnp.concatenate([Vxx_seq_rev, lfxx[None, :, :]], axis=0)

    return dV, success, K_seq, k_seq, V_seq, Vx_seq, Vxx_seq, Cov_policy_seq, Cov_policy_inv_seq    

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

@partial(jax.jit, static_argnums=(2, 4))
def input_smoothed_traj_batch_derivatives_qsim(
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
    graddynamics = toproblem.graddynamics
    # Get cost derivatives as usual
    gradrunningcost = toproblem.gradrunningcost
    hessianrunningcost = toproblem.hessianrunningcost
    gradterminalcost = toproblem.gradterminalcost
    hessiantterminalcost = toproblem.hessiantterminalcost


    fx_s, fu_s = graddynamics(Xs[:-1], Us, jnp.sqrt(sigma), N_samples)

    # Create zeros for higher-order derivatives
    N = Xs.shape[0]
    Nx = Xs.shape[1]
    Nu = Us.shape[1]
    fxx_s = jnp.zeros((N-1, Nx, Nx, Nx))
    fux_s = jnp.zeros((N-1, Nx, Nu, Nx))
    fuu_s = jnp.zeros((N-1, Nx, Nu, Nu))

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

# region: Smoothed Trajectory Derivatives (Chunked)
@partial(jax.jit, static_argnums=(2, 3, 5))
def input_smoothed_traj_chunked_batch_derivatives_qsim(
    Xs,  # (N, Nx)
    Us,  # (N-1, Nu)
    toproblem: "TOProblemDefinition",
    chunk_size: int,
    sigma: float = 1e-3,
    N_samples: int = 50,
    seed: int = 0,
    key: jax.Array = None,
):
    """
    Compute all derivatives for a trajectory using graddynamics in fixed-size chunks.
    Pads Xs[:-1] and Us to chunk_size, computes in chunks, then removes padding.
    Returns:
        TrajDerivatives object with attributes:
            fx, fu: (N-1, Nx, Nx), (N-1, Nx, Nu)
            fxx, fxu, fux, fuu: (N-1, Nx, Nx, Nx), (N-1, Nx, Nx, Nu), (N-1, Nx, Nu, Nx), (N-1, Nx, Nu, Nu)
            lx, lu: (N-1, Nx), (N-1, Nu)
            lxx, lxu, lux, luu: (N-1, Nx, Nx), (N-1, Nx, Nu), (N-1, Nu, Nx), (N-1, Nu, Nu)
            lfx: (Nx,)
            lfxx: (Nx, Nx)
    """
    if key is None:
        key = random.PRNGKey(seed)
    graddynamics = toproblem.graddynamics
    gradrunningcost = toproblem.gradrunningcost
    hessianrunningcost = toproblem.hessianrunningcost
    gradterminalcost = toproblem.gradterminalcost
    hessiantterminalcost = toproblem.hessiantterminalcost

    N = int(Xs.shape[0])
    Nx = int(Xs.shape[1])
    Nu = int(Us.shape[1])
    N_minus_1 = N - 1

    # Calculate padding needed (Python ints)
    n_chunks = (N_minus_1 + chunk_size - 1) // chunk_size
    pad_len = n_chunks * chunk_size - N_minus_1

    # Pad Xs[:-1] and Us with last element (Python ints for pad)
    if pad_len > 0:
        Xs_pad = jnp.pad(Xs[:-1], ((0, pad_len), (0, 0)), mode='edge')
        Us_pad = jnp.pad(Us, ((0, pad_len), (0, 0)), mode='edge')
    else:
        Xs_pad = Xs[:-1]
        Us_pad = Us

    # Reshape into chunks
    Xs_chunks = Xs_pad.reshape(n_chunks, chunk_size, Nx)
    Us_chunks = Us_pad.reshape(n_chunks, chunk_size, Nu)

    # graddynamics should accept (chunk_size, Nx), (chunk_size, Nu), ...
    def chunk_graddynamics(xs_chunk, us_chunk):
        return graddynamics(xs_chunk, us_chunk, jnp.sqrt(sigma), N_samples)

    # vmap over chunks
    fx_chunks, fu_chunks = jax.vmap(chunk_graddynamics)(Xs_chunks, Us_chunks)

    # Reshape back and remove padding
    fx_s = fx_chunks.reshape(n_chunks * chunk_size, Nx, Nx)[:N_minus_1]
    fu_s = fu_chunks.reshape(n_chunks * chunk_size, Nx, Nu)[:N_minus_1]

    # Create zeros for higher-order derivatives
    fxx_s = jnp.zeros((N_minus_1, Nx, Nx, Nx))
    fux_s = jnp.zeros((N_minus_1, Nx, Nu, Nx))
    fuu_s = jnp.zeros((N_minus_1, Nx, Nu, Nu))

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

@partial(jax.jit, static_argnames=("terminalcost", "wsp_f", "usp_f"))
def VTerminalInfo_spm(x_f, chol_cov, terminalcost, wsp_f, usp_f):

    xdim = x_f.shape[0]

    def sigma_propagatation(usp, wsp):

        V_f_i = terminalcost(chol_cov @ usp + x_f)

        Vx_f_i = V_f_i * usp
        Vxx_f_i = V_f_i * (jnp.outer(usp, usp) - jnp.eye(xdim))

        V_f_i = wsp * V_f_i
        Vx_f_i = wsp * Vx_f_i
        Vxx_f_i = wsp * Vxx_f_i

        return V_f_i, Vx_f_i, Vxx_f_i

    V_f_batch, Vx_f_batch, Vxx_f_batch = jax.vmap(sigma_propagatation, in_axes=(1, 0))(usp_f, wsp_f)

    V_f = jnp.sum(V_f_batch)
    Vx_f = jnp.sum(Vx_f_batch, axis=0)
    Vxx_f = jnp.sum(Vxx_f_batch, axis=0)

    V_f = V_f - 0.5 * jnp.trace(Vxx_f)
    Vx_f  = jnp.linalg.solve(chol_cov.T, Vx_f)

    Vxx_f  = jnp.linalg.solve(chol_cov.T, Vxx_f) @ jnp.linalg.inv(chol_cov)

    return Vxx_f, Vx_f, V_f


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

@partial(jax.jit, static_argnames=("runningcost", "dt", "wsp_r", "usp_r"))
def QInfo_spm(nVxx, nVx, nV, nX, SPs, nX_SPs, chol_cov, runningcost, dt, wsp_r, usp_r):

    xdim = nX.shape[0]
    zdim = chol_cov.shape[0]

    def sigma_propagatation(usp, wsp, SP, nX_SP):

        x, u = SP[:xdim], SP[xdim:]

        Q_i = dt * runningcost(x, u) + 0.5 * (nX_SP - nX).T @ nVxx @ (nX_SP - nX) + nVx.T @ (nX_SP - nX) + nV

        Qz_i = Q_i * usp
        Qzz_i = Q_i * (jnp.outer(usp, usp) - jnp.eye(zdim))

        Q_i = wsp * Q_i
        Qz_i = wsp * Qz_i
        Qzz_i = wsp * Qzz_i

        return Q_i, Qz_i, Qzz_i

    batched_compute_terms = jax.vmap(sigma_propagatation, in_axes=(1, 0, 0, 0))

    Q_batch, Qz_batch, Qzz_batch = batched_compute_terms(usp_r, wsp_r, SPs, nX_SPs)

    # Sum over all samples
    Q = jnp.sum(Q_batch)
    Qz = jnp.sum(Qz_batch, axis=0)
    Qzz = jnp.sum(Qzz_batch, axis=0)

    Q = Q - 0.5 * jnp.trace(Qzz)

    tmp = jnp.linalg.solve(chol_cov.T, Qz)
    Qx  = tmp[:xdim]
    Qu  = tmp[xdim:]
    
    tmp2 = jnp.linalg.solve(chol_cov.T, Qzz) @ jnp.linalg.inv(chol_cov)
    Qxx = tmp2[:xdim, :xdim]
    Qxu = tmp2[:xdim, xdim:]
    # Qux = tmp2[xdim:, :xdim]
    Quu = tmp2[xdim:, xdim:]

    return Q, Qx, Qu, Qxx, Qxu, Quu

@partial(jax.jit, static_argnums=(6, 7, 8))
def forward_iteration(
    Xs,
    Us,
    Ks,
    ks,
    Vprev,
    dV,
    toproblem: "TOProblemDefinition",
    toalgorithm: "TOAlgorithm",
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
    gamma = toalgorithm.params.gamma
    beta = toalgorithm.params.beta

    # # JIT-compile forward_pass with static_argnums for toproblem
    # forward_pass_jit = jax.jit(forward_pass, static_argnums=4)

    def body_fn(scanstate, _):
        eps, V_new, Xs_new, Us_new, done = scanstate

        def do_update(_):
            (Xs_new1, Us_new1), Vnew1 = forward_pass(
                Xs, Us, Ks, ks, toproblem, eps
            )
            accept = Vnew1 < Vprev # + gamma * eps * (1 - eps / 2) * dV
            new_eps = lax.select(accept, eps, beta * eps)
            new_done = done | accept
            Xs_out = lax.select(accept, Xs_new1, Xs_new)
            Us_out = lax.select(accept, Us_new1, Us_new)
            V_out = lax.select(accept, Vnew1, V_new)

            return (new_eps, V_out, Xs_out, Us_out, new_done), None

        def do_nothing(_): # Latch the state
            return (eps, V_new, Xs_new, Us_new, done), None

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
    

    eps, V_new, Xs_new, Us_new, done = finalscanstate

    return Xs_new, Us_new, V_new, eps, done

@partial(jax.jit, static_argnames=("toproblem", "toalgorithm"))
def forward_iteration_alpha(
    Xs,
    Us,
    Ks,
    ks,
    Vprev,
    dV,
    toproblem: "TOProblemDefinition",
    toalgorithm: "TOAlgorithm",
    eps_list,  # 1D array of candidate eps values (try in order)
):
    """
    Try each eps in eps_list in order. Accept the first eps that makes forward_pass
    produce a lower cost than Vprev. Returns (Xs_new, Us_new, V_new, eps_used, done).
    """
    # ensure array
    eps_list = jnp.asarray(eps_list)
    # initial carry: use a sentinel eps (first element) as current held eps
    eps0 = eps_list[0]
    carry0 = (eps0, Vprev, Xs, Us, False)  # (eps_current, V_current, Xs_current, Us_current, done)

    def body(carry, eps):
        eps_curr, V_curr, Xs_curr, Us_curr, done = carry

        def try_eps(_):
            (Xs_try, Us_try), V_try = forward_pass(Xs, Us, Ks, ks, toproblem, eps)
            accept = V_try < Vprev
            eps_out = lax.select(accept, eps, eps_curr)
            V_out = lax.select(accept, V_try, V_curr)
            Xs_out = lax.select(accept, Xs_try, Xs_curr)
            Us_out = lax.select(accept, Us_try, Us_curr)
            done_out = done | accept
            return (eps_out, V_out, Xs_out, Us_out, done_out), None

        def skip(_):
            return (eps_curr, V_curr, Xs_curr, Us_curr, done), None

        carry_out, _ = lax.cond(done, skip, try_eps, operand=None)
        return carry_out, None

    final_carry, _ = lax.scan(body, carry0, eps_list)
    eps_used, V_new, Xs_new, Us_new, done = final_carry
    return Xs_new, Us_new, V_new, eps_used, done

@partial(jax.jit, static_argnames=("toproblem", "toalgorithm", "max_fi_iters"))
def forward_iteration_spm(
    Xs, Us,
    Ks, ks,
    Vprev, dV,
    cov_policy,
    SPs, nX_SPs, Covs_Zs, chol_Covs_Zs,
    eps_ini,
    toproblem: "TOProblemDefinition",
    toalgorithm: "TOAlgorithm",
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
    gamma = toalgorithm.params.gamma
    beta = toalgorithm.params.beta

    # # JIT-compile forward_pass with static_argnums for toproblem
    # forward_pass_jit = jax.jit(forward_pass, static_argnums=4)

    def body_fn(scanstate, _):
        eps, V_new, Xs_new, Us_new, done, SPs_new, nX_SPs_new, Covs_Z_new, chol_Covs_Z_new = scanstate

        def do_update(_):
            (Xs_new1, Us_new1, SPs_new1, nX_SPs_new1, Covs_Z_new1, chol_Covs_Z_new1), Vnew1 = forward_pass_spm(
                Xs, Us, Ks, ks, cov_policy, toproblem, toalgorithm, eps
            )
            accept = Vnew1 < Vprev # + gamma * eps * (1 - eps / 2) * dV
            new_eps = lax.select(accept, eps, beta * eps)
            new_done = done | accept
            Xs_out = lax.select(accept, Xs_new1, Xs_new)
            Us_out = lax.select(accept, Us_new1, Us_new)
            V_out = lax.select(accept, Vnew1, V_new)
            SPs_out = lax.select(accept, SPs_new1, SPs_new)
            nX_SPs_out = lax.select(accept, nX_SPs_new1, nX_SPs_new)
            Covs_Zs_out = lax.select(accept, Covs_Z_new1, Covs_Z_new)
            chol_Covs_Zs_out = lax.select(accept, chol_Covs_Z_new1, chol_Covs_Z_new)

            return (new_eps, V_out, Xs_out, Us_out, new_done, SPs_out, nX_SPs_out, Covs_Zs_out, chol_Covs_Zs_out), None

        def do_nothing(_): # Latch the state
            return (eps, V_new, Xs_new, Us_new, done, SPs_new, nX_SPs_new, Covs_Z_new, chol_Covs_Z_new), None

        result = lax.cond(done, do_nothing, do_update, operand=None)

        return result

    # Initial values
    Xs_ini = Xs
    Us_ini = Us
    SPs_ini = SPs
    nX_SPs_ini = nX_SPs
    Covs_Zs_ini = Covs_Zs
    chol_Covs_Zs_ini = chol_Covs_Zs
    V_ini = Vprev
    done_ini = False

    scanstate = (eps_ini, V_ini, Xs_ini, Us_ini, done_ini, SPs_ini, nX_SPs_ini, Covs_Zs_ini, chol_Covs_Zs_ini)
    finalscanstate, _ = lax.scan(body_fn, scanstate, xs=None, length=max_fi_iters)
    

    eps, V_new, Xs_new, Us_new, done, SPs_new, nX_SPs_new, Covs_Zs_new, chol_Covs_Zs_new = finalscanstate

    return Xs_new, Us_new, V_new, eps, done, SPs_new, nX_SPs_new, Covs_Zs_new, chol_Covs_Zs_new

@partial(jax.jit, static_argnames=("toproblem", "toalgorithm"))
def forward_iteration_alpha_spm(
    Xs, Us,
    Ks, ks,
    Vprev, dV,
    cov_policy,
    SPs, nX_SPs, Covs_Zs, chol_Covs_Zs,
    toproblem: "TOProblemDefinition",
    toalgorithm: "TOAlgorithm",
    eps_list,
):
    """
    Like forward_iteration_alpha but for the sigma-point forward_pass_spm.
    Returns (Xs_new, Us_new, V_new, eps_used, done, SPs_new, nX_SPs_new, Covs_Zs_new, chol_Covs_Zs_new)
    """
    eps_list = jnp.asarray(eps_list)
    eps0 = eps_list[0]
    carry0 = (eps0, Vprev, Xs, Us, False, SPs, nX_SPs, Covs_Zs, chol_Covs_Zs)

    def body(carry, eps):
        eps_curr, V_curr, Xs_curr, Us_curr, done, SPs_curr, nX_SPs_curr, Covs_curr, chol_Covs_curr = carry

        def try_eps(_):
            (Xs_try, Us_try, SPs_try, nX_SPs_try, Covs_try, chol_try), V_try = forward_pass_spm(
                Xs, Us, Ks, ks, cov_policy, toproblem, toalgorithm, eps
            )
            accept = V_try < Vprev
            eps_out = lax.select(accept, eps, eps_curr)
            V_out = lax.select(accept, V_try, V_curr)
            Xs_out = lax.select(accept, Xs_try, Xs_curr)
            Us_out = lax.select(accept, Us_try, Us_curr)
            SPs_out = lax.select(accept, SPs_try, SPs_curr)
            nX_SPs_out = lax.select(accept, nX_SPs_try, nX_SPs_curr)
            Covs_out = lax.select(accept, Covs_try, Covs_curr)
            chol_out = lax.select(accept, chol_try, chol_Covs_curr)
            done_out = done | accept
            return (eps_out, V_out, Xs_out, Us_out, done_out, SPs_out, nX_SPs_out, Covs_out, chol_out), None

        def skip(_):
            return (eps_curr, V_curr, Xs_curr, Us_curr, done, SPs_curr, nX_SPs_curr, Covs_curr, chol_Covs_curr), None

        carry_out, _ = lax.cond(done, skip, try_eps, operand=None)
        return carry_out, None

    final_carry, _ = lax.scan(body, carry0, eps_list)
    (eps_used, V_new, Xs_new, Us_new, done,
     SPs_new, nX_SPs_new, Covs_Zs_new, chol_Covs_Zs_new) = final_carry

    return Xs_new, Us_new, V_new, eps_used, done, SPs_new, nX_SPs_new, Covs_Zs_new, chol_Covs_Zs_new


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


# region: Sigma Points and Weights

def get_spg_func(name: str):
    """Plain-Python mapping: name -> function object. Call this outside or at compile-time."""
    mapping = {
        "gh_ws": gh_ws,
        "sym_set": sym_set,
        "ut5_ws": ut5_ws,
        "ut7_ws": ut7_ws,
        "ut9_ws": ut9_ws,
        "ut3_ws": ut3_ws,
    }
    name = str(name)
    try:
        return mapping[name]
    except KeyError:
        raise ValueError(f"Unknown spg method '{name}'. Valid: {list(mapping.keys())}")

# ---- Gaussian-Hermite Quadrature Weights and Sigma Points ----
def gh_ws(n, p):

    Hpm = jnp.array([1.0])
    Hp = jnp.array([1.0, 0.0])
    for i in range(1, p):
        tmp = Hp
        Hp = jnp.pad(Hp, (0, 1)) - jnp.pad(Hpm * i, (2, 0))
        Hpm = tmp


    xi1 = jnp.roots(Hp, strip_zeros=False).real
    W1 = jnp.array([math.factorial(p) / (p ** 2 * (jnp.polyval(Hpm, x) ** 2)) for x in xi1])


    idx = jnp.array(list(itertools.product(range(p), repeat=n)))
    XI = xi1[idx]
    W = jnp.prod(W1[idx], axis=1)
    return W, XI.T  # XI shape (n, p^n)



# ---- Cubature Weights and Sigma Points ----

def sym_set(n, gen):
    """
    Generate the symmetric-set XI for dimension n given generator vector gen.

    Args:
      n   : ambient dimension
      gen : 1D array of length m (m <= n)

    Returns:
      XI  : (n, Npts) array of symmetrically placed points, where Npts = comb(n, m) * 2^m
    """
    gen = jnp.atleast_1d(gen)
    m = gen.shape[0]

    if m == 0:
        return jnp.zeros((n, 1))

    # all 2^m sign-combinations of gen
    # e.g. for gen=[u,v] → [( u, v), (-u, v), (u,-v), (-u,-v)]
    gen_combos = jnp.array(list(product(*[(g, -g) for g in gen])))  # (2^m, m)

    X_list = []
    # all index-tuples of length m from {0,...,n-1}
    for idx in combinations(range(n), m):
        # for each sign-combo, place values into a zero-vector of length n
        idx_arr = jnp.array(idx, dtype=int)
        for signs in gen_combos:
            xi = jnp.zeros((n,), dtype=gen.dtype)
            xi = xi.at[idx_arr].set(signs)
            X_list.append(xi)

    # stack → shape (Npts, n), then transpose → (n, Npts)
    if len(X_list) == 0:
        XI = jnp.zeros((n, 0))
    else:  
        XI = jnp.stack(X_list, axis=0).T  # (n, Npts)
    return XI



def ut5_ws(n):
    """
    Compute weights and sigma-points for 5th-order Unscented Transform in dimension n.
    Returns (W, XI, u) where
      - W  has shape (Npts,)
      - XI has shape (n, Npts)
      - u  is a scalar
    """
    # McNamee & Stenger moments
    I0, I2, I4, I22 = 1.0, 1.0, 3.0, 1.0

    u = jnp.sqrt(I4 / I2)

    # Weights
    A0 = I0 - n * (I2 / I4)**2 * (I4 - 0.5 * (n - 1) * I22)
    A1 = 0.5 * (I2 / I4)**2 * (I4 - (n - 1) * I22)
    A11 = 0.25 * (I2 / I4)**2 * I22

    # Sigma-point sets
    U0 = sym_set(n, jnp.array([]))        # central point
    U1 = sym_set(n, jnp.array([u]))       # first-order
    U2 = sym_set(n, jnp.array([u, u]))    # second-order

    # concatenate horizontally
    XI = jnp.concatenate([U0, U1, U2], axis=1)  # (n, N0+N1+N2)

    # build weight vector
    W0 = A0 * jnp.ones(U0.shape[1])
    W1 = A1 * jnp.ones(U1.shape[1])
    W2 = A11 * jnp.ones(U2.shape[1])
    W  = jnp.concatenate([W0, W1, W2], axis=0)  # (N0+N1+N2,)

    return W, XI





def ut7_ws(n):
    n = jnp.array(n, dtype=jnp.float64)
    
    # Constants
    I222 = jnp.array(1.0, dtype=jnp.float64)
    I22  = jnp.array(1.0, dtype=jnp.float64)
    I24  = jnp.array(3.0, dtype=jnp.float64)
    I2   = jnp.array(1.0, dtype=jnp.float64)
    I6   = jnp.array(15.0, dtype=jnp.float64)
    I4   = jnp.array(3.0, dtype=jnp.float64)
    I0   = jnp.array(1.0, dtype=jnp.float64)

    # Compute roots
    coeffs = jnp.array([
        I2**2 - I0*I4,
        0.0,
        -(I2*I4 - I0*I6),
        0.0,
        (I4**2 - I2*I6)
    ], dtype=jnp.float64)

    roots_all = jnp.roots(coeffs)
    # Keep positive roots only
    positive_roots = roots_all[roots_all > 0.0]
    
    # Ensure we have two roots
    v, u = jnp.sort(positive_roots)

    u = u.real
    v = v.real

    # Compute powers
    u2, u4, u6 = u**2, u**4, u**6
    v2, v4, v6 = v**2, v**4, v**6

    # A111
    A111 = I222 / 8.0 / u6

    # Solve for A11 and A22
    M1 = jnp.array([[u4, v4],
                    [u6, v6]], dtype=jnp.float64)
    rhs1 = jnp.array([I22, I24], dtype=jnp.float64) - 8.0*(n-2.0)*jnp.array([u4, u6], dtype=jnp.float64)*A111
    sol1 = 0.25 * jnp.linalg.solve(M1, rhs1)
    A11, A22 = sol1[0], sol1[1]

    # Solve for A1 and A2
    M2 = jnp.array([[u2, v2],
                    [u4, v4]], dtype=jnp.float64)
    rhs2 = jnp.array([I2, I4], dtype=jnp.float64) - 8.0*(n-1.0)*(n-2.0)/2.0 * jnp.array([u2, u4], dtype=jnp.float64) * A111
    sol2 = -2.0*(n-1.0)*jnp.array([A11, A22], dtype=jnp.float64) + 0.5 * jnp.linalg.solve(M2, rhs2)
    A1, A2 = sol2[0], sol2[1]

    # A0
    A0 = I0 - 2.0*n*(A1 + A2) - 4.0*n*(n-1.0)/2.0*(A11 + A22) - 8.0*n*(n-1.0)*(n-2.0)/6.0*A111

    # Generate U, V sets via sym_set
    n = int(n)
    U0 = sym_set(n, jnp.array([], dtype=jnp.float64))
    U1 = sym_set(n, jnp.array([u], dtype=jnp.float64))
    V1 = sym_set(n, jnp.array([v], dtype=jnp.float64))
    U2 = sym_set(n, jnp.array([u, u], dtype=jnp.float64))
    V2 = sym_set(n, jnp.array([v, v], dtype=jnp.float64))
    U3 = sym_set(n, jnp.array([u, u, u], dtype=jnp.float64))

    # Stack sigma points
    XI = jnp.concatenate([U0, U1, V1, U2, V2, U3], axis=1)

    # Stack weights
    W = jnp.concatenate([
        A0   * jnp.ones(U0.shape[1], dtype=jnp.float64),
        A1   * jnp.ones(U1.shape[1], dtype=jnp.float64),
        A2   * jnp.ones(V1.shape[1], dtype=jnp.float64),
        A11  * jnp.ones(U2.shape[1], dtype=jnp.float64),
        A22  * jnp.ones(V2.shape[1], dtype=jnp.float64),
        A111 * jnp.ones(U3.shape[1], dtype=jnp.float64)
    ], axis=0)


    return  W, XI


def ut9_ws(n):
    I2222, I224, I222 = 1., 3., 1.
    I44, I26, I24, I22 = 9., 15., 3., 1.
    I8, I6, I4, I2, I0     = 105., 15., 3., 1., 1.

    # solve quartic for u,v
    a4 = I4**2 - I2*I6
    a2 = -(I4*I6 - I2*I8)
    a0 =  (I6**2 - I4*I8)
    C = jnp.array([
        [0.,    -a2/a4,  0.,    -a0/a4],
        [1.,     0.,     0.,     0.    ],
        [0.,     1.,     0.,     0.    ],
        [0.,     0.,     1.,     0.    ],
    ])
    roots = jnp.linalg.eigvals(C)
    r_real = jnp.real(roots)
    r_imag = jnp.imag(roots)
    good = (jnp.abs(r_imag) < 1e-6) & (r_real > 0)
    LARGE = 1e6
    masked = jnp.where(good, r_real, LARGE)
    sorted_roots = jnp.sort(masked)
    u, v = sorted_roots[0], sorted_roots[1]

    # powers
    u2, u4, u6, u8 = u**2, u**4, u**6, u**8
    v2, v4, v6, v8 = v**2, v**4, v**6, v**8

    # coefficients
    A1111 = I2222 / (16 * u8)

    M_a = jnp.array([[u6, v6],
                     [u8, v8]])
    rhs_a = jnp.array([I222, I224]) - 16*(n-3)*A1111 * jnp.array([u6, u8])
    tmp_a = (1/8) * jnp.linalg.solve(M_a, rhs_a)
    A111, A222 = tmp_a[0], tmp_a[1]

    A12 = (I26 - I44) / (4 * u2 * v2 * (u2 - v2)**2)

    rhs_b = (jnp.array([I24, I26])
             - 4 * jnp.array([u4*v2 + u2*v4, u6*v2 + u2*v6]) * A12
             - 16 * ((n-2)*(n-3)//2) * A1111 * jnp.array([u6, u8]))
    tmp_b = -2*(n-2)*jnp.array([A111, A222]) + 0.25 * jnp.linalg.solve(M_a, rhs_b)
    A11, A22 = tmp_b[0], tmp_b[1]

    M_c = jnp.array([[u2, v2],
                     [u4, v4]])
    rhs_c = (jnp.array([I2, I4])
             - 16 * ((n-1)*(n-2)*(n-3)//6) * A1111 * jnp.array([u2, u4]))
    tmp_c = (-2*(n-1)*jnp.array([A11 + A12, A22 + A12])
             - 4 * ((n-1)*(n-2)//2) * jnp.array([A111, A222])
             + 0.5 * jnp.linalg.solve(M_c, rhs_c))
    A1, A2 = tmp_c[0], tmp_c[1]

    # binomial via lgamma
    comb = lambda N, K: jnp.exp(gammaln(N+1) - gammaln(K+1) - gammaln(N-K+1))
    A0 = (I0
          - 2*n*(A1 + A2)
          - 4*comb(n,2)*(A11 + 2*A12 + A22)
          - 8*comb(n,3)*(A111 + A222)
          - 16*comb(n,4)*A1111)

    # build sigma-point sets
    U0  = sym_set(n, jnp.array([]))
    U1  = sym_set(n, jnp.array([u]))
    V1  = sym_set(n, jnp.array([v]))
    U2  = sym_set(n, jnp.array([u, u]))
    UV  = sym_set(n, jnp.array([u, v]))
    VU  = sym_set(n, jnp.array([v, u]))
    V2  = sym_set(n, jnp.array([v, v]))
    U3  = sym_set(n, jnp.array([u, u, u]))
    V3  = sym_set(n, jnp.array([v, v, v]))
    U4  = sym_set(n, jnp.array([u, u, u, u]))

    XI = jnp.concatenate([U0, U1, V1, U2, UV, VU, V2, U3, V3, U4], axis=1)

    W = jnp.concatenate([
        A0    * jnp.ones(U0.shape[1]),
        A1    * jnp.ones(U1.shape[1]),
        A2    * jnp.ones(V1.shape[1]),
        A11   * jnp.ones(U2.shape[1]),
        A12   * jnp.ones(UV.shape[1]),
        A12   * jnp.ones(VU.shape[1]),
        A22   * jnp.ones(V2.shape[1]),
        A111  * jnp.ones(U3.shape[1]),
        A222  * jnp.ones(V3.shape[1]),
        A1111 * jnp.ones(U4.shape[1]),
    ], axis=0)
    

    return W, XI



def ut3_ws(n, kappa=None):
    """
    Compute weights and sigma points for 3rd order Unscented Transform (UT)
    for dimension n with parameter kappa (default 1-n).

    Returns:
        W: (2n+1,) array of weights
        XI: (n, 2n+1) array of sigma points
        u: scalar scaling factor
    """
    if kappa is None:
        kappa = 1 - n

    # Weights
    W = jnp.full((2 * n + 1,), 1 / (2 * (n + kappa)))
    W = W.at[0].set(kappa / (n + kappa))

    # Sigma points
    XI = jnp.concatenate(
        [jnp.zeros((n, 1)), jnp.eye(n), -jnp.eye(n)], axis=1
    )
    XI = jnp.sqrt(n + kappa) * XI

    # Scaling factor
    u = jnp.sqrt(n + kappa)

    return W, XI


# endregion: Sigma Points and Weights

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
    # plt.style.use('seaborn-v0_8-darkgrid')
    plt.style.use('seaborn-darkgrid')
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


    # 2. Input vs time (left y-axis) and velocity vs time (right y-axis)
    ax2 = axs[1]
    input_times = np.arange(len(ubar)) * dt
    uvals = ubar[:, 0] if ubar.ndim > 1 else ubar
    vel_times = np.arange(len(velocities)) * dt
    # Input trajectory
    points_u = np.stack([input_times, uvals], axis=1).reshape(-1, 1, 2)
    segments_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
    norm_u = plt.Normalize(input_times[0], input_times[-1])
    lc_u = LineCollection(segments_u, cmap='viridis', norm=norm_u)
    lc_u.set_array(input_times[:-1])
    lc_u.set_linewidth(3)
    line_u = ax2.add_collection(lc_u)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Input (u)', color='tab:blue')
    ax2.set_title('Input and Velocity Trajectory')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax2.set_axisbelow(True)
    ax2.set_xlim(input_times[0], input_times[-1])
    ax2.set_ylim(uvals.min() - 0.1 * abs(uvals.min()), uvals.max() + 0.1 * abs(uvals.max()))
    # Velocity trajectory (right y-axis, faint)
    ax2r = ax2.twinx()
    ax2r.plot(vel_times, velocities, color='tab:red', lw=1.5, alpha=0.3, label='Velocity')
    ax2r.set_ylabel('Velocity (m/s)', color='tab:red')
    ax2r.tick_params(axis='y', labelcolor='tab:red')
    ax2r.set_ylim(velocities.min() - 0.1 * abs(velocities.min()), velocities.max() + 0.1 * abs(velocities.max()))

    # Use a variable to control static friction threshold lines
    show_static_friction = True
    if show_static_friction:
        mu = 0.8
        m = 1.0
        g = 9.81
        threshold = mu * m * g
        ax2.axhline(threshold, color='gray', linestyle='dotted', linewidth=2, alpha=0.7, label='Static Friction +')
        ax2.axhline(-threshold, color='gray', linestyle='dotted', linewidth=2, alpha=0.7, label='Static Friction -')

    # Legends
    if show_static_friction:
        ax2.legend(['Input', 'Static Friction +', 'Static Friction -'], loc='upper left')
    else:
        ax2.legend(['Input'], loc='upper left')
    ax2r.legend(['Velocity'], loc='upper right')


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
    # plt.style.use('seaborn-v0_8-darkgrid')
    plt.style.use('seaborn-darkgrid')
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

    # 3. Vstore vs Iterations
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
    # plt.style.use('seaborn-v0_8-darkgrid')
    plt.style.use('seaborn-darkgrid')
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
