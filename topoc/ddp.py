"""DDP algorithm Skeleton"""

from typing import NamedTuple, Callable, Optional, Tuple, Any
from functools import partial
from jax import Array
import jax
from jax import lax
import jax.numpy as jnp
from jax.typing import ArrayLike

def DDP():
    """Finite horizon Discrete-time Differential Dynamic Programming(DDP)"""

    def __init__(
        self,
        Nx: int,
        Nu: int,
        dynamics: Callable,
        inst_cost: Callable,
        terminal_cost: Callable,
        tolerance: float = 1e-5,
        max_iters: int = 200,
        with_hessians: bool = False,
        constrain: bool = False,
        alphas: ArrayLike = [1.0],
    ):
        self.Nx = Nx
        self.Nu = Nu
        self.dynamics = dynamics
        self.inst_cost = inst_cost
        self.terminal_cost = terminal_cost
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.with_hessians = with_hessians
        self.constrain = constrain
        self.alphas = alphas

    def forward_pass(
        Xs: Array,
        Us: Array,
        Ks: Array,
        ks: Array,
        dynamics: Callable,
        runningcost: Callable,
        finalcost: Callable,
        dt: float,
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
        total_cost = traj_cost + finalcost(xf)
        new_Xs = jnp.vstack([Xs[0], new_Xs])
        return (new_Xs, new_Us), total_cost

    def backward_pass(fx, fu, lx, lu, lxx, luu, lux, l_final_x, l_final_xx, reg=1e-5):
        """
        iLQR/DDP backward pass using JAX with regularization, Cholesky PD check,
        and computation of expected cost decrease dV.

        Args:
            fx, fu, lx, lu, lxx, luu, lux: time-major cost/dynamics derivatives
            l_final_x, l_final_xx: final value function derivatives
            reg: regularization scalar

        Returns:
            K_seq: [T, m, n] feedback gains
            k_seq: [T, m] feedforward terms
            V_xs: [T+1, n] value function gradient trajectory
            V_xxs: [T+1, n, n] value function Hessian trajectory
            dV_seq: [T] expected scalar cost decrease at each step
            success: boolean, True if all Quu are PD
        """

        T = fx.shape[0]
        n = fx.shape[1]
        m = fu.shape[2]

        def backward_step(carry, inputs):
            V_x, V_xx, success = carry
            fx_t, fu_t, lx_t, lu_t, lxx_t, luu_t, lux_t = inputs

            Q_x = lx_t + fx_t.T @ V_x
            Q_u = lu_t + fu_t.T @ V_x
            Q_xx = lxx_t + fx_t.T @ V_xx @ fx_t
            Q_ux = lux_t + fu_t.T @ V_xx @ fx_t
            Q_uu = luu_t + fu_t.T @ V_xx @ fu_t

            # Add regularization to Qxx and Quu
            Q_xx = Q_xx + reg * jnp.eye(n)
            Q_uu = Q_uu + reg * jnp.eye(m)

            # Try Cholesky decomposition
            def on_pd(L):
                # Solve using Cholesky
                k = -jax.scipy.linalg.cho_solve((L, True), Q_u)
                K = -jax.scipy.linalg.cho_solve((L, True), Q_ux)

                # Value function update
                V_x_new = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
                V_xx_new = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K

                # Scalar cost decrease approximation
                dV = -Q_u.T @ k

                return (V_x_new, V_xx_new, True), (K, k, V_x_new, V_xx_new, dV)

            def on_fail():
                dummy_K = jnp.zeros((m, n))
                dummy_k = jnp.zeros((m,))
                dV = jnp.array(0.0)
                return (V_x, V_xx, False), (dummy_K, dummy_k, V_x, V_xx, dV)

            def try_chol(Q_uu):
                try:
                    L = jax.scipy.linalg.cholesky(Q_uu, lower=True)
                    return True, L
                except:
                    return False, jnp.zeros_like(Q_uu)

            def try_chol_safe(Q_uu):
                # JAX needs branchable logic
                eps = 1e-9
                Q_uu_test = Q_uu - eps * jnp.eye(m)
                eigvals = jnp.linalg.eigvalsh(Q_uu_test)
                is_pd = jnp.all(eigvals > 0.0)

                L = jax.scipy.linalg.cholesky(Q_uu + 1e-9 * jnp.eye(m), lower=True)
                return is_pd, L

            is_pd, L = try_chol_safe(Q_uu)

            # Branch based on PD status
            (carry_out, output) = lax.cond(
                is_pd & success,
                lambda _: on_pd(L),
                lambda _: on_fail(),
                operand=None
            )

            return carry_out, output

        # Prepare input sequence
        scaninputs = (fx, fu, lx, lu, lxx, luu, lux)

        # Initial state
        init_scanstate = (l_final_x, l_final_xx, True)

        # Run backward scan
        (V_x_T, V_xx_T, success), (K_seq_rev, k_seq_rev, V_xs_rev, V_xxs_rev, dV_seq_rev) = lax.scan(
            backward_step,
            init=init_scanstate,
            xs=scaninputs,
            reverse=True
        )

        # Time forward results
        K_seq = K_seq_rev[::-1]
        k_seq = k_seq_rev[::-1]
        V_xs = jnp.concatenate([V_xs_rev[::-1], l_final_x[None, :]], axis=0)
        V_xxs = jnp.concatenate([V_xxs_rev[::-1], l_final_xx[None, :, :]], axis=0)
        dV_seq = dV_seq_rev[::-1]

        return K_seq, k_seq, V_xs, V_xxs, dV_seq, success
