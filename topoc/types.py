from typing import NamedTuple, Callable, Tuple
from jax import Array
from enum import Enum

# region: TOProblemDefinition

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

# region: TOAlgorithm Parameters

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

# endregion: TOAlgorithm Parameters

# region: DDP: Data Structures
class TrajDerivatives(NamedTuple): # Derivatives along complete trajectory
    fxs: Array # (N-1, Nx, Nx)
    fus: Array # (N-1, Nx, Nu)
    fxxs: Array # (N-1, Nx, Nx, Nx)
    fxus: Array # (N-1, Nx, Nx, Nu)
    fuxs: Array # (N-1, Nx, Nu, Nx)
    fuus: Array # (N-1, Nx, Nu, Nu)
    lxs: Array # (N-1, Nx)
    lus: Array # (N-1, Nu)
    lxxs: Array # (N-1, Nx, Nx)
    lxus: Array # (N-1, Nx, Nu)
    luxs: Array # (N-1, Nu, Nx)
    luus: Array # (N-1, Nu, Nu)
    lfx: Array # (Nx,)
    lfxx: Array # (Nx, Nx)

class WaypointDerivatives(NamedTuple):
    # Derivatives along waypoint NOTE: Terminal: lfx and lfxx not required 
    fx: Array # (. , Nx, Nx)
    fu: Array # (. , Nx, Nu)
    fxx: Array # (. , Nx, Nx, Nx)
    fxu: Array # (. , Nx, Nx, Nu)
    fux: Array # (. , Nx, Nu, Nx)
    fuu: Array # (. , Nx, Nu, Nu)
    lx: Array # (. , Nx)
    lu: Array # (. , Nu)
    lxx: Array # (. , Nx, Nx)
    lxu: Array # (. , Nx, Nu)
    lux: Array # (. , Nu, Nx)
    luu: Array # (. , Nu, Nu)

class NextTimeStepVFDerivatives(NamedTuple): # Next Time Step Value Function Derivatives
    Vx: Array # (Nx,)
    Vxx: Array # (Nx, Nx)

class QDerivatives(NamedTuple):
    Qx: Array # (Nx,)
    Qu: Array # (Nu,)
    Qxx: Array # (Nx, Nx)
    Qux: Array # (Nu, Nx)
    Quu: Array # (Nu, Nu)

class Gains(NamedTuple):
    K: Array # (Nu, Nx)
    k: Array # (Nu,)

# endregion: DDP: Data Structures