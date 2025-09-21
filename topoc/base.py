"""Define data structures and types"""

from typing import NamedTuple, Callable, Any, Union, Tuple, Optional
from functools import partial
import jax
from jax import lax, Array
from jax.typing import ArrayLike
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from enum import Enum

from topoc.utils import linearize, quadratize
from topoc.types import *
from topoc.ddp import DDP
from topoc.csddp import CSDDP
from topoc.scsddp import SCSDDP
from topoc.pddp import PDDP

class TOSolve:
    """
    Class to solve the trajectory optimization problem using the specified algorithm.
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
        elif self.algorithm.algo_type == AlgorithmName.SCSDDP:
            scsddp = SCSDDP(self.problem, self.algorithm)
            return scsddp.solve()
        elif self.algorithm.algo_type == AlgorithmName.CSDDP:
            csddp = CSDDP(self.problem, self.algorithm)
            return csddp.solve()
        elif self.algorithm.algo_type == AlgorithmName.PDDP:
            pddp = PDDP(self.problem, self.algorithm)
            return pddp.solve()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm.algo_type}")

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
        starting_input: ArrayLike,
        goal_state: ArrayLike,
        modelparams: ModelParams,
        graddynamics: Optional[GradDynamicsFn] = None,
        hessiandynamics: Optional[HessianDynamicsFn] = None,
        gradrunningcost: Optional[GradRunningCostFn] = None,
        hessianrunningcost: Optional[HessianRunningCostFn] = None,
        gradterminalcost: Optional[GradTerminalCostFn] = None,
        hessiantterminalcost: Optional[HessianTerminalCostFn] = None,
        starting_state_cov: Optional[ArrayLike] = None,
    ):
        """
        Trajectory Optimization Problem initialization.
        """
        self.runningcost = runningcost
        self.terminalcost = terminalcost
        self.dynamics = dynamics
        self.starting_state = jnp.array(starting_state)
        self.starting_input = jnp.array(starting_input)
        self.goal_state = jnp.array(goal_state)
        self.modelparams = modelparams
        self.graddynamics = graddynamics if graddynamics is not None else linearize(dynamics)
        self.hessiandynamics = hessiandynamics if hessiandynamics is not None else quadratize(dynamics)
        self.gradrunningcost = gradrunningcost if gradrunningcost is not None else linearize(runningcost)
        self.hessianrunningcost = hessianrunningcost if hessianrunningcost is not None else quadratize(runningcost)
        self.gradterminalcost = gradterminalcost if gradterminalcost is not None else linearize(terminalcost)
        self.hessiantterminalcost = hessiantterminalcost if hessiantterminalcost is not None else quadratize(terminalcost)
        self.starting_state_cov = jnp.array(starting_state_cov) if starting_state_cov is not None else jnp.zeros((self.starting_state.shape[0], self.starting_state.shape[0]))


class TOAlgorithm:
    def __init__(self, algo_type: AlgorithmName, **kwargs):
        param_class = algorithm_param_classes.get(algo_type)
        if param_class is None:
            raise ValueError(f"Unknown algorithm: {algo_type}")
        self.algo_type = algo_type
        self.params = param_class(**kwargs)


