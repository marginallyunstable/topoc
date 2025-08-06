"""Define data structures and types"""

from typing import NamedTuple, Callable, Any, Union, Tuple, Optional
from functools import partial
import jax
from jax import lax, Array
from jax.typing import ArrayLike
import jax.numpy as jnp
from enum import Enum

from topoc.utils import linearize, quadratize
from topoc.types import *
from topoc.ddp import DDP
from topoc.rddp2 import RDDP2
from topoc.rddp1 import RDDP1

class TOSolve:
    """
    Class to solve the trajectory optimization problem using the specified algorithm.
    Stores results and provides utility methods.
    """
    def __init__(self, problem: "TOProblemDefinition", algorithm: "TOAlgorithm"):
        self.problem = problem
        self.algorithm = algorithm
        self.result = self._solve()

    def _solve(self):
        # Dispatch to the correct solver based on algorithm type
        # Replace with your actual solver logic
        if self.algorithm.algo_type == AlgorithmName.DDP:
            ddp = DDP(self.problem, self.algorithm)
            return ddp.solve()
        elif self.algorithm.algo_type == AlgorithmName.RDDP1:
            rddp1 = RDDP1(self.problem, self.algorithm)
            return rddp1.solve()
        elif self.algorithm.algo_type == AlgorithmName.RDDP2:
            rddp2 = RDDP2(self.problem, self.algorithm)
            return rddp2.solve()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm.algo_type}")

    def _rddp1_solver(self):
        # Implement RDDP1 solver logic here
        # Example: return {"status": "solved", "data": ...}
        return {"status": "solved", "algorithm": "RDDP1"}

    def _rddp2_solver(self):
        # Implement RDDP2 solver logic here
        return {"status": "solved", "algorithm": "RDDP2"}

    def _spddp_solver(self):
        # Implement SPDDP solver logic here
        return {"status": "solved", "algorithm": "SPDDP"}

    def _sppdp_solver(self):
        # Implement SPPDP solver logic here
        return {"status": "solved", "algorithm": "SPPDP"}

    def visualize(self):
        # Implement visualization logic here
        print(f"Visualizing result for {self.algorithm.algo_type}")

    def show_log(self):
        # Implement logging or summary logic here
        print(f"Log for {self.algorithm.algo_type}: {self.result}")

class TOProblemDefinition():
    """
    Trajectory Optimization Problem Definition.
    """
    def __init__(
        self,
        runningcost: RunningCostFn,
        terminalcost: TerminalCostFn,
        dynamics: DynamicsFn,
        starting_state: ArrayLike,
        goal_state: ArrayLike,
        modelparams: ModelParams,
        graddynamics: Optional[GradDynamicsFn] = None,
        hessiandynamics: Optional[HessianDynamicsFn] = None,
        gradrunningcost: Optional[GradRunningCostFn] = None,
        hessianrunningcost: Optional[HessianRunningCostFn] = None,
        gradterminalcost: Optional[GradTerminalCostFn] = None,
        hessiantterminalcost: Optional[HessianTerminalCostFn] = None,
    ):
        """
        Trajectory Optimization Problem initialization.

        Parameters
        ----------
        runningcost : RunningCostFn
            Running cost function (x, u).
        terminalcost : TerminalCostFn
            Terminal cost function (xf).
        dynamics : DynamicsFn
            Dynamics function (x, u).
        modelparams : ModelParams
            Model Parameters e.g. state_dim, input_dim, horizon_len, dt.
        """
        self.runningcost = runningcost
        self.terminalcost = terminalcost
        self.dynamics = dynamics
        self.starting_state = jnp.array(starting_state)
        self.goal_state = jnp.array(goal_state)
        self.modelparams = modelparams
        self.graddynamics = graddynamics if graddynamics is not None else linearize(dynamics)
        self.hessiandynamics = hessiandynamics if hessiandynamics is not None else quadratize(dynamics)
        self.gradrunningcost = gradrunningcost if gradrunningcost is not None else linearize(runningcost)
        self.hessianrunningcost = hessianrunningcost if hessianrunningcost is not None else quadratize(runningcost)
        self.gradterminalcost = gradterminalcost if gradterminalcost is not None else linearize(terminalcost)
        self.hessiantterminalcost = hessiantterminalcost if hessiantterminalcost is not None else quadratize(terminalcost)


class TOAlgorithm:
    def __init__(self, algo_type: AlgorithmName, **kwargs):
        param_class = algorithm_param_classes.get(algo_type)
        if param_class is None:
            raise ValueError(f"Unknown algorithm: {algo_type}")
        self.algo_type = algo_type
        self.params = param_class(**kwargs)


