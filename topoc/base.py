"""Define data structures and types"""

from typing import NamedTuple, Callable, Any, Union, Tuple, Optional
from functools import partial
import jax
from jax import lax, Array
from jax.typing import ArrayLike
import jax.numpy as jnp
from enum import Enum

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
        if self.algorithm.algo_type == AlgorithmName.RDDP1:
            return self._rddp1_solver()
        elif self.algorithm.algo_type == AlgorithmName.RDDP2:
            return self._rddp2_solver()
        elif self.algorithm.algo_type == AlgorithmName.SPDDP:
            return self._spddp_solver()
        elif self.algorithm.algo_type == AlgorithmName.SPPDP:
            return self._sppdp_solver()
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

# region: TOProblemDefinition
class TOProblemDefinition():
    """
    Trajectory Optimization Problem Definition.
    """
    def __init__(
        self,
        runningcost: "RunningCostFn",
        finalcost: "TerminalCostFn",
        dynamics: "DynamicsFn",
        modelparams: "ModelParams",
    ):
        """
        Trajectory Optimization Problem initialization.

        Parameters
        ----------
        runningcost : RunningCostFn
            Running cost function (t, x, u, params).
        finalcost : FinalCostFn
            Final cost function (xf, params).
        dynamics : DynamicsFn
            Dynamics function (t, x, u, params).
        modelparams : ModelParams
            Model Parameters e.g. state_dim, input_dim, horizon_len, dt.
        """
        self.runningcost = runningcost
        self.finalcost = finalcost
        self.dynamics = dynamics
        self.modelparams = modelparams

class ModelParams(NamedTuple):
    """Model Parameters"""

    state_dim: int
    input_dim: int
    horizon_len: int
    dt: float

RunningCostFn = Callable[[Array, Array], Array]
GradRunningCostFn = Callable[[Array, Array], Tuple[Array, Array]]
HessianRunningCostFn = Callable[
    [Array, Array], Tuple[Tuple[Array, Array], Tuple[Array, Array]]
]

TerminalCostFn = Callable[[Array, Array], Array]
GradTerminalCostFn = Callable[[Array, Array], Tuple[Array, Array]]
HessianTerminalCostFn = Callable[
    [Array, Array], Tuple[Tuple[Array, Array], Tuple[Array, Array]]
]

DynamicsFn = Callable[[Array, Array], Array]
GradDynamicsFn = Callable[[Array, Array], Tuple[Array, Array]]
HessianDynamicsFn = Callable[
    [Array, Array], Tuple[Tuple[Array, Array], Tuple[Array, Array]]
]

# endregion: TOProblemDefinition

# region: TOAlgorithm

class TOAlgorithm:
    def __init__(self, algo_type: "AlgorithmName", **kwargs):
        param_class = algorithm_param_classes.get(algo_type)
        if param_class is None:
            raise ValueError(f"Unknown algorithm: {algo_type}")
        self.algo_type = algo_type
        self.params = param_class(**kwargs)

class AlgorithmName(Enum):
    RDDP1 = "RDDP1: Randomized DDP with Smoothing in state and control space"
    RDDP2 = "RDDP2: Randomized DDP with Smoothing only in control space"
    SPDDP = "SPDDP: Sigma Point Differential Dynamic Programming"
    SPPDP = "SPPDP: Sigma Point Probabilistic Dynamic Programming"

class AlgorithmParams:
    """Namespace for all algorithm parameter classes."""

    class RDDP1Params:
        def __init__(self, alpha: float, beta: int):
            self.alpha = alpha
            self.beta = beta
    class RDDP2Params:
        def __init__(self, gamma: float):
            self.gamma = gamma

    class SPDDPParams:
        def __init__(self, delta: float):
            self.delta = delta

    class SPPDPParams:
        def __init__(self, epsilon: float):
            self.epsilon = epsilon

# Mapping from AlgorithmName to parameter class
algorithm_param_classes = {
    AlgorithmName.RDDP1: AlgorithmParams.RDDP1Params,
    AlgorithmName.RDDP2: AlgorithmParams.RDDP2Params,
    AlgorithmName.SPDDP: AlgorithmParams.SPDDPParams,
    AlgorithmName.SPPDP: AlgorithmParams.SPPDPParams,
}

# endregion: TOAlgorithm
