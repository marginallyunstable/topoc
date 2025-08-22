import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from topoc.utils import quadratic_running_cost, quadratic_terminal_cost, plot_block_results
from topoc.base import TOProblemDefinition, TOAlgorithm, TOSolve
from models.cartpole import cartpole, cartpole_with_friction
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
x0 = jnp.array([0.0, 0.0, jnp.pi, 0.0])
covx0 = 1e-6 * jnp.eye(state_dim)
xg = jnp.array([0.0, 0.0, 0.0, 0.0])
# Define initial input (control)
u0 = jnp.array([0.0])

# Define cost matrices
P = 1000000*jnp.eye(state_dim)
Q = 1*jnp.eye(state_dim)
R = 5*jnp.eye(input_dim)

params_dynamics = {"mc": 1.0, "mp": 0.1, "g": 9.81, "l": 1.0, "dt": dt}
params_terminal = {"P": P}
params_running = {"Q": Q, "R": R}

# Define cost functions using partial

dynamics = partial(cartpole_with_friction, params=params_dynamics)
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
                            use_second_order_info=False, # NOTE
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
    use_second_order_info=True,
    sigma_x=1e-6,
    sigma_u=10,
    alpha=0.1,
    alpha_red=2.0,
    sigma_red=2.0,
    targetalpha=1e-6,
    targetsigma=1e-6,
    mcsamples=500, # NOTE
    max_iters=50,
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
    use_second_order_info=True,
    sigma=1e-2,
    alpha=0.1,
    alpha_red=2.0,
    sigma_red=2.0,
    targetalpha=1e-6,
    targetsigma=1e-6,
    mcsamples=100,
    max_iters=50,
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


# Compare results
import matplotlib.pyplot as plt

algorithms = [
    ("DDP", xbar_ddp, ubar_ddp, Vstore_ddp),
    ("RDDP1", xbar_rddp1, ubar_rddp1, Vstore_rddp1),
    ("RDDP2", xbar_rddp2, ubar_rddp2, Vstore_rddp2),
    ("SPPDP", xbar_sppdp, ubar_sppdp, Vstore_sppdp),
]

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
linestyles = ["-", "--", "-.", ":"]

# Create a 2x2 grid of subplots: top-left = cart pos vs vel, top-right = pendulum pos vs vel,
# bottom-left = input vs time, bottom-right = Vstore over iterations
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1 (top-left): cart position vs cart velocity
for i, (name, xbar, _, _) in enumerate(algorithms):
    axs[0, 0].plot(xbar[:, 0], xbar[:, 1], label=name, color=colors[i], linestyle=linestyles[i])
axs[0, 0].set_xlabel("Cart Position")
axs[0, 0].set_ylabel("Cart Velocity")
axs[0, 0].set_title("Cart: Velocity vs Position")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Subplot 2 (top-right): pendulum angular position vs angular velocity
for i, (name, xbar, _, _) in enumerate(algorithms):
    axs[0, 1].plot(xbar[:, 2], xbar[:, 3], label=name, color=colors[i], linestyle=linestyles[i])
axs[0, 1].set_xlabel("Pendulum Angle")
axs[0, 1].set_ylabel("Pendulum Angular Velocity")
axs[0, 1].set_title("Pendulum: Angular Velocity vs Angle")
axs[0, 1].legend()
axs[0, 1].grid(True)

# Subplot 3 (bottom-left): input vs time
# Use per-algorithm control timesteps to match ubar length (controls often have one less timestep than states)
for i, (name, _, ubar, _) in enumerate(algorithms):
    nt = ubar.shape[0]
    timesteps_ctrl = jnp.arange(nt) * dt
    axs[1, 0].plot(timesteps_ctrl, ubar[:, 0], label=name, color=colors[i], linestyle=linestyles[i])
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].set_ylabel("Input")
axs[1, 0].set_title("Input vs Time")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Subplot 4 (bottom-right): Vstore over iterations
for i, (name, _, _, Vstore) in enumerate(algorithms):
    axs[1, 1].plot(jnp.arange(len(Vstore)), Vstore, label=name, color=colors[i], linestyle=linestyles[i])
axs[1, 1].set_xlabel("Iteration")
axs[1, 1].set_ylabel("Vstore")
axs[1, 1].set_title("Vstore over Iterations")
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()