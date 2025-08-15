import jax
import jax.numpy as jnp
from functools import partial
from topoc.utils import quadratic_running_cost, quadratic_terminal_cost, plot_cartpole_results
from topoc.base import TOProblemDefinition, TOAlgorithm, TOSolve
from models.cartpole import cartpole  # Assumes this exists
from topoc.types import ModelParams, AlgorithmName

# Define model parameters (example values)
state_dim = 4
input_dim = 1
horizon_len = 500
dt = 0.01

modelparams = ModelParams(
    state_dim=state_dim,
    input_dim=input_dim,
    horizon_len=horizon_len,
    dt=dt
)

# Define initial and goal states
x0 = jnp.array([0.0, 0.0, jnp.pi, 0.0])
xg = jnp.array([0.0, 0.0, 0.0, 0.0])
# Define initial input (control)
u0 = jnp.array([0.0])

# Define cost matrices
P = 1000000*jnp.eye(state_dim)
Q = 1*jnp.eye(state_dim)
R = 1*jnp.eye(input_dim)

params_dynamics = {"mc": 1.0, "mp": 0.1, "g": 9.81, "l": 1.0, "dt": dt}
params_terminal = {"P": P}
params_running = {"Q": Q, "R": R}

# Define cost functions using partial

dynamics = partial(cartpole, params=params_dynamics)
terminalcost = partial(quadratic_terminal_cost, xg=xg, params=params_terminal)
runningcost = partial(quadratic_running_cost, xg=xg, params=params_running)

toprob = TOProblemDefinition(
    runningcost=runningcost,
    terminalcost=terminalcost,
    dynamics=dynamics,
    starting_state=x0,
    starting_input=u0,
    goal_state=xg,
    modelparams=modelparams
)

# Define algorithm (example: DDP)
algorithm = TOAlgorithm(AlgorithmName.DDP, gamma=0.01, beta=0.5, use_second_order_info=False, max_iters=200, max_fi_iters=50, stopping_criteria=1e-6)

print("Algorithm parameters:")
print("Name:", algorithm.algo_type)
print("Gamma:", algorithm.params.gamma)
print("Beta:", algorithm.params.beta)
print("Use second order info:", algorithm.params.use_second_order_info)

tosolve = TOSolve(toprob, algorithm)
xbar, ubar, Vstore = tosolve.result.xbar, tosolve.result.ubar, tosolve.result.Vstore

# ---- Call plotting function ----
plot_cartpole_results(tosolve.result, x0, xg, modelparams)
