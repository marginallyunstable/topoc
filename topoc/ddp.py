"""DDP algorithm Skeleton"""

from typing import NamedTuple, Callable, Optional, Tuple, Any
from functools import partial
from jax import Array
import jax
from jax import lax
import jax.numpy as jnp
from jax.typing import ArrayLike

from topoc.base import TOProblemDefinition, TOAlgorithm
from topoc.types import *
from topoc.utils import *

class DDP():
    """Finite horizon Discrete-time Differential Dynamic Programming(DDP)"""

    def __init__(
        self,
        toproblem: TOProblemDefinition,
        toalgorithm: TOAlgorithm,
    ):
        
        self.toproblem = toproblem
        self.toalgorithm = toalgorithm

        Nx = toproblem.modelparams.state_dim
        Nu = toproblem.modelparams.input_dim
        H = toproblem.modelparams.horizon_len

        def solve():

            # region: Initialize Simulation

            # TODO: Add starting start state and goal state in TOProblemDefinition
        
            xbar = jnp.zeros((H, Nx)) # TODO: replace with initial state
            ubar = jnp.zeros((H-1, Nu))
            k = jnp.zeros((H-1, Nu))
            K = jnp.zeros((H-1, Nu, Nx))

            (xbar, ubar), Vbar = forward_pass(xbar, ubar, K, k, toproblem)

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

                    trajderivatives = traj_batch_derivatives(xbar, ubar, toproblem)
                     # TODO: Add use of second order information based on algorithm type
                    dV, success, K, k = backward_pass(trajderivatives, use_second_order_info=False)

                    if not success:
                        regularization = max(regularization * 4, 1e-3)

                regularization = regularization / 20
                if regularization < 1e-6:
                    regularization = 0.0


                Vprev = Vbar

                xbar, ubar, Vbar, eps, done = forward_iteration(
                    xbar, ubar, K, k, Vprev, dV, toproblem, toalgorithm, max_iters=20
                )

                if not done:
                    print("Line Search didn't succeed. Try playing with gamma/beta/max_iters.")
                    break

                Vstore.append(Vbar)
                Change = Vprev - Vbar

                iter += 1

                if Change < 1e-9 or iter > 100:
                    print(f"Converged in {iter} iterations.")
                    break


            return xbar, ubar, Vstore      

                    

                
        




        


