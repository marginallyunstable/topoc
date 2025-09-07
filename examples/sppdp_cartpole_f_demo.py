import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from topoc.utils import quadratic_running_cost, quadratic_terminal_cost, plot_cartpole_results
from topoc.base import TOProblemDefinition, TOAlgorithm, TOSolve
from models.cartpole import cartpole_f, rk_f  # Assumes this exists
from topoc.types import ModelParams, AlgorithmName

# Define model parameters (example values)
state_dim = 4
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
x0 = jnp.array([-1.0, 0.0, 0.0, 0.0])
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
R = 1*jnp.eye(input_dim)

params_dynamics = {"mc": 1.0, "mp": 0.1, "g": 9.81, "pl": 0.5, "dt": dt}
params_terminal = {"P": P}
params_running = {"Q": Q, "R": R}

# Define cost functions using partial

# dynamics_ = partial(cartpole__, params=params_dynamics)
# dynamics = lambda x, u: rk_f(x, u, dt, dynamics_)
dynamics = partial(cartpole_f, params=params_dynamics)
terminalcost = partial(quadratic_terminal_cost, xg=xg, params=params_terminal)
runningcost = partial(quadratic_running_cost, xg=xg, params=params_running)


toprob = TOProblemDefinition(
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
algorithm = TOAlgorithm(
    AlgorithmName.SPPDP,
    use_second_order_info=True,
    spg_method='gh_ws',
    spg_params={"order": 5},
    eta=0.01,
    lam=100,
    zeta=1,
    zeta_factor=2,
    zeta_min=1e-2,
    sigma_u=1e-2,
    max_iters=100,
)

print("Algorithm parameters:")
print("Name:", algorithm.algo_type)
print("Gamma:", algorithm.params.gamma)
print("Beta:", algorithm.params.beta)
print("SPG Method:", algorithm.params.spg_method)
print("Eta:", algorithm.params.eta)
print("Lam:", algorithm.params.lam)
print("Zeta:", algorithm.params.zeta)
print("Lam Factor:", algorithm.params.zeta_factor)
print("Lam Min:", algorithm.params.zeta_min)
print("Sigma_u:", algorithm.params.sigma_u)
print("Max iters:", algorithm.params.max_iters)
print("Max fi iters:", algorithm.params.max_fi_iters)

tosolve = TOSolve(toprob, algorithm)
xbar, ubar, Vstore = tosolve.result.xbar, tosolve.result.ubar, tosolve.result.Vstore

print(Vstore[0])  # Print the first value of the cost function
print(Vstore[-1])  # Print the last value of the cost function

# ---- Call plotting function ----
plot_cartpole_results(tosolve.result, x0, xg, modelparams)
