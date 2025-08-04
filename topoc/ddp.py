"""DDP algorithm Skeleton"""

from typing import NamedTuple, Callable, Optional, Tuple, Any, TYPE_CHECKING
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

class DDP():
    """Finite horizon Discrete-time Differential Dynamic Programming(DDP)"""

    def __init__(
        self,
        toproblem: "TOProblemDefinition",
        toalgorithm: "TOAlgorithm",
    ):
        self.toproblem = toproblem
        self.toalgorithm = toalgorithm

    def solve(self):

        xini = self.toproblem.starting_state
        Nx = self.toproblem.modelparams.state_dim
        Nu = self.toproblem.modelparams.input_dim
        H = self.toproblem.modelparams.horizon_len

        # region: Initialize Simulation
    
        xbar = jnp.zeros((H, Nx))
        xbar = xbar.at[0].set(xini) # set first element to initial state
        ubar = jnp.ones((H-1, Nu))
        k = jnp.zeros((H-1, Nu))
        K = jnp.zeros((H-1, Nu, Nx))

        (xbar, ubar), Vbar = forward_pass(xbar, ubar, K, k, self.toproblem)

        # endregion: Initialize Simulation

        # region: Algorithm Iterations

        iter = 1
        Vstore = [Vbar]
        regularization = 0.0

        while True:

            print(f"Iteration: {iter}")

            # region: Backward Pass

            success = False

            while not success:

                trajderivatives = traj_batch_derivatives(xbar, ubar, self.toproblem)
                dV, success, K, k, Vx, Vxx = backward_pass(trajderivatives,
                                                    regularization,
                                                    use_second_order_info=self.toalgorithm.params.use_second_order_info)

                if not success:
                    regularization = max(regularization * 4, 1e-3)

            regularization = regularization / 20
            if regularization < 1e-6:
                regularization = 0.0


            Vprev = Vbar # Store previous value to check post forward iteration

            xbar, ubar, Vbar, eps, done = forward_iteration(
                xbar, ubar, K, k, Vprev, dV, self.toproblem, self.toalgorithm, max_iters=20
            )

            if not done:
                print("Line Search exhaused. Try playing with gamma/beta/max_iters.")
                break

            Vstore.append(Vbar)
            Change = Vprev - Vbar

            iter += 1

            if Change < 1e-6 or iter > 100:
                print(f"Converged in {iter} iterations.")
                break

        return xbar, ubar, Vstore


                
        




        


