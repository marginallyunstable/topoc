import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from topoc.utils import quadratic_running_cost, quadratic_terminal_cost, plot_compare_cartpole_results
from topoc.base import TOProblemDefinition, TOAlgorithm, TOSolve
from models.cartpole import cartpole_f, rk_f 
from topoc.types import ModelParams, AlgorithmName

# Define model parameters (example values)
state_dim = 4
input_dim = 1
# horizon_len = 121
# dt = 0.05
horizon_len = 400
dt = 0.01

modelparams = ModelParams(
    state_dim=state_dim,
    input_dim=input_dim,
    horizon_len=horizon_len,
    dt=dt
)

# Define initial and goal states
x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
covx0 = 1e-6 * jnp.eye(state_dim)
xg = jnp.array([0.0, 0.0, jnp.pi, 0.0])
# Define initial input (control)
u0 = jnp.array([0.0])

# Define cost matrices
# P = 100*jnp.eye(state_dim)
# Q = (0.01/dt)*jnp.eye(state_dim)
# R = (0.01/dt)*jnp.eye(input_dim)
P = 1000000*jnp.eye(state_dim)
Q = 1*jnp.eye(state_dim)
R = 5*jnp.eye(input_dim)

params_dynamics = {"mc": 1.0, "mp": 0.1, "g": 9.81, "pl": 0.5}
params_terminal = {"P": P}
params_running = {"Q": Q, "R": R}

# Define cost functions using partial

# dynamics_ = partial(cartpole_f, params=params_dynamics)
# dynamics = lambda x, u: rk_f(x, u, dt, dynamics_)
dynamics = partial(cartpole_f, params=params_dynamics)
terminalcost = partial(quadratic_terminal_cost, xg=xg, params=params_terminal)
runningcost = partial(quadratic_running_cost, xg=xg, params=params_running)


# region DDP

toprob_ddp = TOProblemDefinition(
    runningcost=runningcost,
    terminalcost=terminalcost,
    dynamics=dynamics,
    starting_state=x0,
    starting_input=u0,
    goal_state=xg,
    modelparams=modelparams
)

algorithm_ddp = TOAlgorithm(AlgorithmName.DDP, 
                            use_second_order_info=True, # NOTE
                            max_iters=50)

tosolve_ddp = TOSolve(toprob_ddp, algorithm_ddp)
xbar_ddp, ubar_ddp, Vstore_ddp = tosolve_ddp.result.xbar, tosolve_ddp.result.ubar, tosolve_ddp.result.Vstore

# endregion DDP


# region RDDP1

toprob_rddp1 = TOProblemDefinition(
    runningcost=runningcost,
    terminalcost=terminalcost,
    dynamics=dynamics,
    starting_state=x0,
    starting_input=u0,
    goal_state=xg,
    modelparams=modelparams
)

# Define RDDP1 algorithm parameters (example values)
algorithm_rddp1 = TOAlgorithm(
    AlgorithmName.RDDP1,
    use_second_order_info=True, # NOTE
    sigma_x=1e-6,
    sigma_u=1e-2,
    alpha=0.1,
    alpha_red=2.0,
    sigma_red=2.0,
    targetalpha=1e-6,
    targetsigma=1e-6,
    mcsamples=25, # NOTE
    max_iters=50,
    spg_method='gh_ws',
    spg_params={"order": 5},
)

# Example usage: create and solve the problem with RDDP1
tosolve_rddp1 = TOSolve(toprob_rddp1, algorithm_rddp1)
xbar_rddp1, ubar_rddp1, Vstore_rddp1 = tosolve_rddp1.result.xbar, tosolve_rddp1.result.ubar, tosolve_rddp1.result.Vstore


# endregion RDDP1


# region RDDP2

toprob_rddp2 = TOProblemDefinition(
    runningcost=runningcost,
    terminalcost=terminalcost,
    dynamics=dynamics,
    starting_state=x0,
    starting_input=u0,
    goal_state=xg,
    modelparams=modelparams
)

# Define RDDP2 algorithm parameters (example values)
algorithm_rddp2 = TOAlgorithm(
    AlgorithmName.RDDP2,
    use_second_order_info=True, # NOTE
    sigma=1e-2,
    alpha=0.1,
    alpha_red=2.0,
    sigma_red=2.0,
    targetalpha=1e-6,
    targetsigma=1e-6,
    mcsamples=25, # NOTE
    max_iters=50,
    spg_method='gh_ws',
    spg_params={"order": 5},
)

tosolve_rddp2 = TOSolve(toprob_rddp2, algorithm_rddp2)
xbar_rddp2, ubar_rddp2, Vstore_rddp2 = tosolve_rddp2.result.xbar, tosolve_rddp2.result.ubar, tosolve_rddp2.result.Vstore


# endregion RDDP2

# region SPPDP

toprob_sppdp = TOProblemDefinition(
    runningcost=runningcost,
    terminalcost=terminalcost,
    dynamics=dynamics,
    starting_state=x0,
    starting_state_cov=covx0,
    starting_input=u0,
    goal_state=xg,
    modelparams=modelparams
)

# Define SPPDP algorithm parameters (example values)
algorithm_sppdp = TOAlgorithm(
    AlgorithmName.SPPDP,
    spg_method='gh_ws',
    spg_params={"order": 5},
    eta=0.01,
    lam=100,
    zeta=1,
    zeta_factor=2,
    zeta_min=1e-2,
    sigma_u=1e-2,
    max_iters=50
)


# Example usage: create and solve the problem with SPPDP
tosolve_sppdp = TOSolve(toprob_sppdp, algorithm_sppdp)
xbar_sppdp, ubar_sppdp, Vstore_sppdp = tosolve_sppdp.result.xbar, tosolve_sppdp.result.ubar, tosolve_sppdp.result.Vstore

# endregion SPPDP


# recreate algorithms list and call the beautified plotting function
algorithms = [
    ("DDP", xbar_ddp, ubar_ddp, Vstore_ddp),
    ("RDDP1", xbar_rddp1, ubar_rddp1, Vstore_rddp1),
    ("RDDP2", xbar_rddp2, ubar_rddp2, Vstore_rddp2),
    ("SPPDP", xbar_sppdp, ubar_sppdp, Vstore_sppdp),
]

plot_compare_cartpole_results(algorithms, x0, xg, modelparams)