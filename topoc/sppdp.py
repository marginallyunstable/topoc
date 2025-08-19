from typing import NamedTuple, Callable, Optional, Tuple, Any, TYPE_CHECKING
from collections import namedtuple
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

class SPPDP():
    """
    TO with Sigma Point-based Probabilistic Differential Dynamic Programming (SPPDP)
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
        # spg_func = getattr(utils, self.toalgorithm.params.spg_method)
        # wsp_r, usp_r = spg_func(Nx+Nu) # sigma point weights (wsp_r) and unit sigma points (usp_r) for running part of trajectory
        # wsp_f, usp_f = spg_func(Nx) # sigma point weights (wsp_f) and unit sigma points (usp_f) for final step of trajectory
        # sp_dict = {'wsp_r': wsp_r, 'usp_r': usp_r, 'wsp_f': wsp_f, 'usp_f': usp_f}
        zeta = self.toalgorithm.params.zeta
        zeta_factor = self.toalgorithm.params.zeta_factor
        zeta_min = self.toalgorithm.params.zeta_min
        sigma_u = self.toalgorithm.params.sigma_u

        # region: Initialize Simulation
    
        xbar = jnp.zeros((H, Nx))
        xbar = xbar.at[0].set(xini) # set first element to initial state
        ubar = jnp.tile(uini, (H-1, 1))
        k = jnp.zeros((H-1, Nu))
        K = jnp.zeros((H-1, Nu, Nx))
        cov_policy = jnp.zeros((H-1, Nu, Nu))  # covariance of the policy
        cov_policy = sigma_u * jnp.broadcast_to(jnp.eye(Nu), (H-1, Nu, Nu))
        cov_policy_inv = (1/sigma_u) * jnp.broadcast_to(jnp.eye(Nu), (H-1, Nu, Nu))

        (xbar, ubar, SPs, nX_SPs, Covs_Zs, chol_Covs_Zs), Vbar = forward_pass_spm(xbar, ubar, K, k, cov_policy, self.toproblem, self.toalgorithm)
        # endregion: Initialize Simulation

        # region: Algorithm Iterations

        iter = 0
        Change = jnp.inf
        Vstore = [Vbar]
        regularization = 0.0

        while True:

            while True:

                iter += 1
                print(f"Iteration: {iter}")

                # region: Backward Pass

                success = False

                while not success:

                    dV, success, K_new, k_new, V_new, Vx_new, Vxx_new, \
                    cov_policy_new, cov_policy_inv_new = backward_pass_spm( xbar,ubar,
                                                                            k, K, SPs, nX_SPs,
                                                                            Covs_Zs, chol_Covs_Zs, zeta,
                                                                            cov_policy, cov_policy_inv,
                                                                            self.toproblem, self.toalgorithm,
                                                                            regularization)

                    if not success:
                        regularization = max(regularization * 4, 1e-3)
                        print(f"Backward Pass failed. Setting regularization to: {regularization}")

                # Note: this doesn't need to be done for other algorithms, because others calculate k, K directly and not update it.
                K = K_new
                k = k_new
                cov_policy = cov_policy_new
                cov_policy_inv = cov_policy_inv_new

                regularization = regularization / 20
                if regularization < 1e-6:
                    regularization = 0.0

                dV = V_new[0] - Vbar
                Vprev = Vbar # Store previous value to check post forward iteration

                # TODO: change the criterion for forward iteration V < Vprev

                # To warm start forward_iteration_spm
                xbar_, ubar_, Vbar, eps, done = forward_iteration(
                    xbar, ubar, K, k, Vprev, dV, self.toproblem, self.toalgorithm, max_fi_iters=self.toalgorithm.params.max_fi_iters
                )

                # xbar, ubar, Vbar, eps, done, SPs, nX_SPs, Covs_Zs, chol_Covs_Zs = forward_iteration_spm(
                #     xbar, ubar, K, k, Vprev, dV, cov_policy, SPs, nX_SPs, Covs_Zs, chol_Covs_Zs, eps,
                #     self.toproblem, self.toalgorithm, max_fi_iters=self.toalgorithm.params.max_fi_iters,
                # )
                xbar, ubar, Vbar, eps, done, SPs, nX_SPs, Covs_Zs, chol_Covs_Zs = forward_iteration_list_spm(
                    xbar, ubar, K, k, Vprev, dV, cov_policy, SPs, nX_SPs, Covs_Zs, chol_Covs_Zs,
                    self.toproblem, self.toalgorithm
                )

                

                if not done:
                    Vstore.append(Vprev)
                    zeta = max(zeta / zeta_factor, zeta_min)
                    print(f"Line Search exhausted. dV value expected was: {dV}. Zeta value is set to: {zeta}")
                    break

                Vstore.append(Vbar)
                Change = Vprev - Vbar

                print(f"Forward Iteration: eps = {eps}, done = {done}")
                print(f"Vprev: {Vprev}, Vbar: {Vbar}, Change: {Change}")

                zeta = zeta * zeta_factor
                print(f"Line Search succeeded. Zeta value is set to: {zeta}")

                # if Change < self.toalgorithm.params.alpha or iter > self.toalgorithm.params.max_iters:
                #     print(f"Converged in {iter} iteration(s). [Inner Loop]")
                #     break
                
                # Stupid, but keeping this structure in case complete EM-step is performed
                if True:
                    break

            if (iter == self.toalgorithm.params.max_iters
                or Change <= self.toalgorithm.params.stopping_criteria
            ):
                print(f"Maximum iterations reached/Converged in {iter} iteration(s).")
                break


        Result = namedtuple('Result', ['xbar', 'ubar', 'Vstore'])
        return Result(xbar=xbar, ubar=ubar, Vstore=Vstore)


                
        




        


