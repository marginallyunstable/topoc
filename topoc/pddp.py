from typing import NamedTuple, Callable, Optional, Tuple, Any, TYPE_CHECKING
from collections import namedtuple
from jax import config
config.update("jax_enable_x64", True)
from functools import partial
from jax import Array
import jax
from jax import lax
import jax.numpy as jnp
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from topoc.base import TOProblemDefinition, TOAlgorithm
from topoc.types import *
from topoc.utils import *
import topoc.utils as utils

class PDDP():
    """
    Probabilistic Differential Dynamic Programming (PDDP)
    """

    def __init__(
        self,
        toproblem: "TOProblemDefinition",
        toalgorithm: "TOAlgorithm",
    ):
        self.toproblem = toproblem
        self.toalgorithm = toalgorithm

    def solve(self):

        xini = self.toproblem.starting_state
        uini = self.toproblem.starting_input
        Nx = self.toproblem.modelparams.state_dim
        Nu = self.toproblem.modelparams.input_dim
        H = self.toproblem.modelparams.horizon_len
        zeta = self.toalgorithm.params.zeta
        zeta_factor = self.toalgorithm.params.zeta_factor
        zeta_min = self.toalgorithm.params.zeta_min
        sigma_u = self.toalgorithm.params.sigma_u
        use_second_order_info = self.toalgorithm.params.use_second_order_info

        # region: Initialize Simulation
    
        xbar = jnp.zeros((H, Nx))
        xbar = xbar.at[0].set(xini) # set first element to initial state
        ubar = jnp.tile(uini, (H-1, 1))
        k = jnp.zeros((H-1, Nu))
        K = jnp.zeros((H-1, Nu, Nx))
        cov_policy = jnp.zeros((H-1, Nu, Nu))  # covariance of the policy
        cov_policy = sigma_u * jnp.broadcast_to(jnp.eye(Nu), (H-1, Nu, Nu))
        cov_policy_inv = (1/sigma_u) * jnp.broadcast_to(jnp.eye(Nu), (H-1, Nu, Nu))

        (xbar, ubar, SPs, nX_SPs, Covs_Zs, chol_Covs_Zs), Vbar = forward_pass_wup(xbar, ubar, K, k, cov_policy, self.toproblem, self.toalgorithm)
        # (xbar, ubar, SPs, nX_SPs, Covs_Zs, chol_Covs_Zs), Vbar = forward_pass_qsim_wup(xbar, ubar, K, k, cov_policy, self.toproblem, self.toalgorithm)
        # (xbar, ubar, SPs, nX_SPs, Covs_Zs, chol_Covs_Zs), Vbar = forward_pass_wup_init(xbar, ubar, K, k, cov_policy, self.toproblem, self.toalgorithm)
        
        # endregion: Initialize Simulation
        # self.toalgorithm.params.lam = Vbar
        # zeta = (H+1)/Vbar
        # zeta = Vbar

        # region: Algorithm Iterations

        iter = 1
        Change = jnp.inf
        Vstore = [Vbar]
        Vred = []
        regularization = 0.0
        # call the initialization backward pass only once
        first_backward = True

        while True:

            while True:

                iter += 1
                print(f"Iteration: {iter}")

                # region: Backward Pass

                success = False

                while not success:

                    # choose init-backward for the very first backward pass, otherwise use regular backward
                    # back_fn = backward_pass_wup_init if first_backward else backward_pass_wup
                    back_fn = backward_pass_wup

                    dV, success, K_new, k_new, V_new, Vx_new, Vxx_new, \
                    cov_policy_new, cov_policy_inv_new = back_fn( xbar,ubar,
                                                                    k, K, SPs, nX_SPs,
                                                                    Covs_Zs, chol_Covs_Zs, zeta,
                                                                    cov_policy, cov_policy_inv,
                                                                    self.toproblem, self.toalgorithm,
                                                                    regularization, use_second_order_info)

                    if not success:
                        regularization = max(regularization * 4, 1e-3)
                        print(f"Backward Pass failed. Setting regularization to: {regularization}")

                # after a successful backward pass, mark that init has been used
                if first_backward:
                    first_backward = False

                # Note: this doesn't need to be done for other algorithms, because others calculate k, K directly and not UPDATE it.
                K = K_new
                k = k_new

                # # Regularize cov_policy_new and compute its inverse using solve with vmap
                # eps_reg = 1e-5
                # Nu_local = cov_policy_new.shape[-1]
                # I_Nu = jnp.eye(Nu_local)
                # cov_policy_reg = cov_policy_new + eps_reg * jnp.broadcast_to(I_Nu, cov_policy_new.shape)
                # # inverse each matrix via solve(A, I)
                # cov_policy_inv_new = jax.vmap(lambda A: jnp.linalg.solve(A, I_Nu))(cov_policy_reg)
                # cov_policy = cov_policy_reg
                # cov_policy_inv = cov_policy_inv_new

                cov_policy = cov_policy_new
                cov_policy_inv = cov_policy_inv_new

                regularization = regularization / 20
                if regularization < 1e-6:
                    regularization = 0.0

                dV = V_new[0] - Vbar # Expected change in value function
                Vprev = Vbar # Store previous value to check post forward iteration

                # To warm start forward_iteration_wup (forward_iteration is cheaper than forward_iteration_wup)
                _, _, _, eps_opt, done = forward_iteration(
                    xbar, ubar, K, k, Vprev, dV, self.toproblem, self.toalgorithm,
                )

                xbar_new, ubar_new, Vbar_new, eps, done, \
                      SPs_new, nX_SPs_new, Covs_Zs_new, chol_Covs_Zs_new = forward_iteration_wup(
                    xbar, ubar, K, k, Vprev, dV, cov_policy, SPs, nX_SPs, Covs_Zs, chol_Covs_Zs,
                    self.toproblem, self.toalgorithm
                )
                # xbar_new, ubar_new, Vbar_new, eps, done, \
                #       SPs_new, nX_SPs_new, Covs_Zs_new, chol_Covs_Zs_new = forward_iteration_qsim_wup(
                #     xbar, ubar, K, k, Vprev, dV, cov_policy, SPs, nX_SPs, Covs_Zs, chol_Covs_Zs,
                #     self.toproblem, self.toalgorithm
                # )

                print(f"eps_opt: {eps_opt}")

                if not done:
                    Vstore.append(Vprev)
                    Vred.append(jnp.nan)
                    zeta = max(zeta / zeta_factor, zeta_min)
                    print(f"Line Search exhausted. dV value expected was: {dV}. Zeta value is set to: {zeta}")
                    break

                
                # xbar_new, ubar_new, Vbar_new, eps, _, \
                #       SPs_new, nX_SPs_new, Covs_Zs_new, chol_Covs_Zs_new = forward_iteration_wup_once(
                #     xbar, ubar, K, k, Vprev, dV, cov_policy, SPs, nX_SPs, Covs_Zs, chol_Covs_Zs, eps_opt,
                #     self.toproblem, self.toalgorithm
                # )

                print(f"eps after forward_iteration_wup: {eps}")

                xbar = xbar_new
                ubar = ubar_new
                Vbar = Vbar_new
                SPs = SPs_new
                nX_SPs = nX_SPs_new
                Covs_Zs = Covs_Zs_new
                chol_Covs_Zs = chol_Covs_Zs_new

                Vstore.append(Vbar)
                Change = Vprev - Vbar
                Vred.append(Change)

                print(f"Forward Iteration: eps = {eps}, done = {done}")
                print(f"Vprev: {Vprev}, Vbar: {Vbar}, Change: {Change}")

                zeta = zeta * zeta_factor
                # # zeta = 1
                # zeta = (H+1)/Vbar
                print(f"Line Search succeeded. Zeta value is set to: {zeta}")
                
                # Stupid! but keeping this structure in case complete M-step of EM is performed
                if True:
                    break

            print(f"Vprev: {Vprev}, Vbar: {Vbar}, Change: {Change}")
            
            if (iter > self.toalgorithm.params.max_iters
                or abs(Change) <= self.toalgorithm.params.stopping_criteria
            ):
                print(f"Maximum iterations reached/Converged in {iter} iteration(s).")
                break


        Result = namedtuple('Result', ['xbar', 'ubar', 'Vstore', 'Vred'])
        return Result(xbar=xbar, ubar=ubar, Vstore=Vstore, Vred=Vred)











