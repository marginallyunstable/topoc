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
    DDP = "DDP: Vanilla DDP"
    RDDP1 = "RDDP1: Randomized DDP with Smoothing in state and control space"
    RDDP2 = "RDDP2: Randomized DDP with Smoothing only in control space"
    SPDP = "SPDDP: Sigma Point Dynamic Programming"
    SPPDP = "SPPDP: Sigma Point Probabilistic Dynamic Programming"

class AlgorithmParams:
    """
    Namespace for all algorithm parameter classes.
        Common parameters for all algorithms:
            beta: line search reduction parameter
            gamma: scaling(reduction) factor for dV 
            use_second_order_info: whether to use second order information in the algorithm
            max_iters: maximum number of iterations for the algorithm
            max_fi_iters: maximum number of backtracking steps for forward iteration
            stopping criteria: if this change in value function is reached, the algorithm stops
    """

    

    class DDPParams:
        def __init__(self,
                     beta: float = 0.5,
                     gamma: float = 0.01,
                     use_second_order_info: bool = False,
                     max_iters: int = 200,
                     max_fi_iters: int = 50,
                     stopping_criteria: float = 1e-6):
            self.beta = beta
            self.gamma = gamma
            self.use_second_order_info = use_second_order_info
            self.max_iters = max_iters
            self.max_fi_iters = max_fi_iters
            self.stopping_criteria = stopping_criteria

    class RDDP1Params:
        """
        RDDP1 Parameters
            alpha: with this change in value function perform next DDP iteration with reduced noise
            alpha_red: reduction factor for alpha for next iteration
            sigma_x: initial noise for state space
            sigma_u: initial noise for control space
            sigma_red: reduction factor for sigma_x for next iteration
            targetalpha: target value for alpha to stop the algorithm
            targetsigma: target value for sigma to stop the algorithm
            mcsamples: number of samples for Monte Carlo estimation
        """
        def __init__(self,
                     beta: float = 0.5,
                     gamma: float = 0.01,
                     use_second_order_info: bool = False,
                     max_iters: int = 200,
                     max_fi_iters: int = 50,
                     stopping_criteria: float = 1e-6,
                     # RDDP2 specific parameters
                     alpha: float = 10, # change in value function
                     alpha_red: float = 2.0,
                     sigma_x: float = 2.0,
                     sigma_u: float = 2.0,
                     sigma_red: float = 2.0,
                     targetalpha: float = 1e-6,
                     targetsigma: float = 1e-6,
                     mcsamples: int = 50):
            self.beta = beta
            self.gamma = gamma
            self.use_second_order_info = use_second_order_info
            self.max_iters = max_iters
            self.max_fi_iters = max_fi_iters
            self.stopping_criteria = stopping_criteria
            self.alpha = alpha
            self.alpha_red = alpha_red
            self.sigma_x = sigma_x
            self.sigma_u = sigma_u
            self.sigma_red = sigma_red
            self.targetalpha = targetalpha  
            self.targetsigma = targetsigma
            self.mcsamples = mcsamples
    class RDDP2Params:
        """
        RDDP2 Parameters
            alpha: with this change in value function perform next DDP iteration with reduced noise
            alpha_red: reduction factor for alpha for next iteration
            sigma: initial noise for control space
            sigma_red: reduction factor for sigma for next iteration
            targetalpha: target value for alpha to stop the algorithm
            targetsigma: target value for sigma to stop the algorithm
            mcsamples: number of samples for Monte Carlo estimation
        """
        def __init__(self,
                     beta: float = 0.5,
                     gamma: float = 0.01,
                     use_second_order_info: bool = False,
                     max_iters: int = 200,
                     max_fi_iters: int = 50,
                     stopping_criteria: float = 1e-6,
                     # RDDP2 specific parameters
                     alpha: float = 10, # change in value function
                     alpha_red: float = 2.0,
                     sigma: float = 2.0,
                     sigma_red: float = 2.0,
                     targetalpha: float = 1e-6,
                     targetsigma: float = 1e-6,
                     mcsamples: int = 50):
            self.beta = beta
            self.gamma = gamma
            self.use_second_order_info = use_second_order_info
            self.max_iters = max_iters
            self.max_fi_iters = max_fi_iters
            self.stopping_criteria = stopping_criteria
            self.alpha = alpha
            self.alpha_red = alpha_red
            self.sigma = sigma
            self.sigma_red = sigma_red
            self.targetalpha = targetalpha  
            self.targetsigma = targetsigma
            self.mcsamples = mcsamples

    class SPPDPParams:
        """
        SPPDP Parameters
            spg_method: method for sigma point generation
            sigma_x: initial noise for state space
            sigma_u: initial noise for control space
            sigma_red: reduction factor for sigma_x for next iteration
            targetalpha: target value for alpha to stop the algorithm
            targetsigma: target value for sigma to stop the algorithm
        """
        def __init__(self,
                     beta: float = 0.5,
                     gamma: float = 0.01,
                     use_second_order_info: bool = False,
                     max_iters: int = 200,
                     max_fi_iters: int = 50,
                     stopping_criteria: float = 1e-6,
                     # SPPDP specific parameters
                     spg_method: str = 'ut5_ws',  # Allowed: 'gh_ws', 'sym_set', 'ut5_ws', 'ut7_ws', 'ut9_ws', 'ut3_ws'
                     eta: float = 1,  # forget factor for control policy
                     lam: float = 100.0,  # temperature parameter
                     zeta: float = 1.0,  # temperature change parameter
                     zeta_factor: float = 2.0,  # temperature change factor
                     zeta_min: float = 1e-2,  # minimum temperature parameter
                     sigma_u: float = 2.0,):
            self.beta = beta
            self.gamma = gamma
            self.use_second_order_info = use_second_order_info
            self.max_iters = max_iters
            self.max_fi_iters = max_fi_iters
            self.stopping_criteria = stopping_criteria
            self.spg_method = spg_method
            self.eta = eta
            self.lam = lam
            self.zeta = zeta
            self.zeta_factor = zeta_factor
            self.zeta_min = zeta_min
            self.sigma_u = sigma_u

    class SPDPParams:
        def __init__(self, epsilon: float):
            self.epsilon = epsilon

# Mapping from AlgorithmName to parameter class
algorithm_param_classes = {
    AlgorithmName.DDP: AlgorithmParams.DDPParams,
    AlgorithmName.RDDP1: AlgorithmParams.RDDP1Params,
    AlgorithmName.RDDP2: AlgorithmParams.RDDP2Params,
    AlgorithmName.SPDP: AlgorithmParams.SPDPParams,
    AlgorithmName.SPPDP: AlgorithmParams.SPPDPParams,
}

SPG_METHOD_NAMES = ('gh_ws', 'sym_set', 'ut5_ws', 'ut7_ws', 'ut9_ws', 'ut3_ws')

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

TRAJ_TO_WAYPOINT_RENAME_MAP = {
    'fxs': 'fx',
    'fus': 'fu',
    'fxxs': 'fxx',
    'fxus': 'fxu',
    'fuxs': 'fux',
    'fuus': 'fuu',
    'lxs': 'lx',
    'lus': 'lu',
    'lxxs': 'lxx',
    'lxus': 'lxu',
    'luxs': 'lux',
    'luus': 'luu',
}

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