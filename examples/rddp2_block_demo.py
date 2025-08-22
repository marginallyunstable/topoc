import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from topoc.utils import quadratic_running_cost, quadratic_terminal_cost, plot_block_results
from topoc.base import TOProblemDefinition, TOAlgorithm, TOSolve
from models.block import block_on_ground, block_on_ground_with_friction
from topoc.types import ModelParams, AlgorithmName



# Define model parameters (example values)
state_dim = 2
input_dim = 1
horizon_len = 200
dt = 0.01

modelparams = ModelParams(
    state_dim=state_dim,
    input_dim=input_dim,
    horizon_len=horizon_len,
    dt=dt
)

# Define initial and goal states
x0 = jnp.array([0.0, 0.0])
xg = jnp.array([4.0, 0.0])
# Define initial input (control)
u0 = jnp.array([0.0])

# Define cost matrices
P = 1000000*jnp.eye(state_dim)
Q = 1*jnp.eye(state_dim)
R = 5*jnp.eye(input_dim)

params_dynamics = {"m": 1.0, "dt": dt}
params_terminal = {"P": P}
params_running = {"Q": Q, "R": R}

# Define cost functions using partial
dynamics = partial(block_on_ground, params=params_dynamics)
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

# Define RDDP2 algorithm parameters (example values)
algorithm = TOAlgorithm(
    AlgorithmName.RDDP2,
    gamma=0.01,
    beta=0.5,
    use_second_order_info=True,
    sigma=1e-2,
    alpha=0.1,
    alpha_red=2.0,
    sigma_red=2.0,
    targetalpha=1e-6,
    targetsigma=1e-6,
    mcsamples=9,
    max_iters=50,
)

print("Algorithm parameters:")
print("Name:", algorithm.algo_type)
print("Gamma:", algorithm.params.gamma)
print("Beta:", algorithm.params.beta)
print("Sigma:", algorithm.params.sigma)
print("Alpha:", algorithm.params.alpha)
print("Use second order info:", algorithm.params.use_second_order_info)

# Example usage: create and solve the problem with RDDP2
tosolve = TOSolve(toprob, algorithm)
xbar, ubar, Vstore = tosolve.result.xbar, tosolve.result.ubar, tosolve.result.Vstore

print(Vstore[-1])  # Print the last value of the cost function

# ---- Call plotting function ----
plot_block_results(tosolve.result, x0, xg, modelparams)

