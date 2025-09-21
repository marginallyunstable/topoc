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

class CSDDP():
    """
    Control Smoothed DDP
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
        sigma = self.toalgorithm.params.sigma
        alpha = self.toalgorithm.params.alpha

        # region: Initialize Simulation
    
        xbar = jnp.zeros((H, Nx))
        xbar = xbar.at[0].set(xini) # set first element to initial state
        ubar = jnp.tile(uini, (H-1, 1))
        k = jnp.zeros((H-1, Nu))
        K = jnp.zeros((H-1, Nu, Nx))

        (xbar, ubar), Vbar = forward_pass(xbar, ubar, K, k, self.toproblem)

        # endregion: Initialize Simulation

        # region: Algorithm Iterations

        iter = 1
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

                    # trajderivatives = input_smoothed_traj_batch_derivatives(
                    #     xbar, ubar, self.toproblem,
                    #     sigma=sigma,
                    #     N_samples=self.toalgorithm.params.mcsamples,
                    #     key=jax.random.PRNGKey(42)

                    # )
                    trajderivatives = input_smoothed_traj_batch_derivatives_spm(
                        xbar, ubar, sigma,
                        self.toproblem, self.toalgorithm,
                    )
                    # trajderivatives = input_smoothed_traj_batch_derivatives_qsim_spm(
                    #     xbar, ubar, sigma,
                    #     self.toproblem, self.toalgorithm,
                    # )

                    # trajderivatives = input_smoothed_traj_batch_derivatives_qsim(
                    #     xbar, ubar, self.toproblem,
                    #     sigma=sigma,
                    #     N_samples=self.toalgorithm.params.mcsamples,
                    #     key=jax.random.PRNGKey(42)
                    # )
                    # trajderivatives = input_smoothed_traj_chunked_batch_derivatives_qsim(
                    #     xbar, ubar, self.toproblem, chunk_size=5,
                    #     sigma=sigma,
                    #     N_samples=self.toalgorithm.params.mcsamples,
                    #     key=jax.random.PRNGKey(42)
                    # )
                    dV, success, K, k, Vx, Vxx = backward_pass(trajderivatives,
                                                        regularization,
                                                        use_second_order_info=self.toalgorithm.params.use_second_order_info)

                    # print(f"dV: {dV}")
                    
                    if not success:
                        regularization = max(regularization * 4, 1e-3)

                regularization = regularization / 20
                if regularization < 1e-6:
                    regularization = 0.0


                Vprev = Vbar # Store previous value to check post forward iteration

                xbar, ubar, Vbar, eps, done = forward_iteration(
                    xbar, ubar, K, k, Vprev, dV, self.toproblem, self.toalgorithm
                )

                print(f"Vbar: {Vbar}")
                
                if not done or iter > self.toalgorithm.params.max_iters:
                    Vstore.append(Vprev)
                    print(f"Line Search exhausted. dV value expected was: {dV}")
                    break

                Vstore.append(Vbar)
                Change = Vprev - Vbar

                if Change < self.toalgorithm.params.alpha or iter > self.toalgorithm.params.max_iters:
                    print(f"Converged in {iter} iteration(s). [Inner Loop]")
                    break

            # if (
            #     (alpha <= self.toalgorithm.params.targetalpha and
            #      sigma <= self.toalgorithm.params.targetsigma)
            #     or iter > self.toalgorithm.params.max_iters
            #     or Change <= self.toalgorithm.params.stopping_criteria
            # ):
            #     print(f"Converged in {iter} iteration(s). [Outer Loop]")
            #     break

            if (
                iter > self.toalgorithm.params.max_iters
                or abs(Change) <= self.toalgorithm.params.stopping_criteria
            ):
                print(f"Converged in {iter} iteration(s). [Outer Loop]")
                break

            
            print(Vprev)
            
            print(f"Alpha set from {alpha} to {alpha / self.toalgorithm.params.alpha_red}")
            alpha = alpha / self.toalgorithm.params.alpha_red

            print(f"Sigma set from {sigma} to {sigma / self.toalgorithm.params.sigma_red}")
            sigma = sigma / self.toalgorithm.params.sigma_red

        Result = namedtuple('Result', ['xbar', 'ubar', 'Vstore'])
        return Result(xbar=xbar, ubar=ubar, Vstore=Vstore)


                
        




        


