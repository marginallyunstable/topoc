import jax
import jax.numpy as jnp
from functools import partial
from topoc.utils import quadratic_running_cost, quadratic_terminal_cost
from topoc.base import TOProblemDefinition, TOAlgorithm, TOSolve
from models.qsim_box_push import BoxPushSimulator
from topoc.types import ModelParams, AlgorithmName
from typing import Callable, Tuple

# Define model parameters (example values)
state_dim = 5
input_dim = 2
horizon_len = 100
dt = 0.01

modelparams = ModelParams(
    state_dim=state_dim,
    input_dim=input_dim,
    horizon_len=horizon_len,
    dt=dt
)

# Define initial and goal states
# State: [box_x, box_y, box_z, hand_x, hand_y] (5 elements total)
x0 = jnp.array([0.0, 0.7, 0.0, 0.0, 0.0])  # Initial: box and hand at same y position
xg = jnp.array([0.2, 0.9, jnp.pi/8, 0.0, 0.0])  # Goal: push box forward to y=0.9, hand_y=pi/8

# Define cost matrices
P = jnp.diag(jnp.array([1000000, 1000000, 1000000, 1, 1]))
Q = jnp.diag(jnp.array([1, 1, 1, 1, 1]))
R = 1*jnp.eye(input_dim)

params_terminal = {"P": P}
params_running = {"Q": Q, "R": R}

# Define cost functions using partial
box = BoxPushSimulator(dt)
dynamics = box.dynamics 
graddynamics = box.graddynamics
def dummy_func(x: jnp.ndarray, u: jnp.ndarray) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    zeros_x = jnp.zeros_like(x)
    zeros_u = jnp.zeros_like(u)
    return (zeros_x, zeros_u), (zeros_x, zeros_u)
hessiandynamics = dummy_func  # Placeholder for Hessian dynamics
terminalcost = partial(quadratic_terminal_cost, xg=xg, params=params_terminal)
runningcost = partial(quadratic_running_cost, xg=xg, params=params_running)

toprob = TOProblemDefinition(
    runningcost=runningcost,
    terminalcost=terminalcost,
    dynamics=dynamics,
    graddynamics=graddynamics,
    hessiandynamics=hessiandynamics,
    starting_state=x0,
    goal_state=xg,
    modelparams=modelparams
)

# Define RDDP2 algorithm parameters (example values)
algorithm = TOAlgorithm(
    AlgorithmName.RDDP2,
    gamma=0.01,
    beta=0.5,
    use_second_order_info=False,
    sigma=0.2,
    alpha=0.1,
    alpha_red=2.0,
    sigma_red=2.0,
    targetalpha=1e-6,
    targetsigma=1e-6,
    mcsamples=50,
    max_iters=5,
    max_fi_iters=5
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

# # ---- Call plotting function ----
print("Plotting results...")

import matplotlib.pyplot as plt

# Plot the trajectory of the square block in 2D space
block_size = 0.05  # Size of the block (square side length)

for i in range(xbar.shape[0]):
    x = xbar[i, 0]
    y = xbar[i, 1]
    theta = xbar[i, 3]
    # Calculate the corners of the square block
    half_size = block_size / 2
    corners = jnp.array([
        [-half_size, -half_size],
        [half_size, -half_size],
        [half_size, half_size],
        [-half_size, half_size],
        [-half_size, -half_size]
    ])
    # Rotation matrix
    rot = jnp.array([
        [jnp.cos(theta), -jnp.sin(theta)],
        [jnp.sin(theta),  jnp.cos(theta)]
    ])
    rotated = corners @ rot.T
    translated = rotated + jnp.array([x, y])
    plt.plot(translated[:, 0], translated[:, 1], 'b-', alpha=0.3)

# Plot the trajectory as a line
plt.plot(xbar[:, 0], xbar[:, 1], 'r-', label='Block trajectory')
plt.scatter(xbar[0, 0], xbar[0, 1], color='green', label='Start')
plt.scatter(xbar[-1, 0], xbar[-1, 1], color='orange', label='Goal')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('2D Block Push Trajectory')
plt.axis('equal')
plt.legend()
plt.show()

# Save results to a txt file in readable format
output_file = "/workspace/topoc/examples/rddp2_box_push_results.txt"
with open(output_file, "w") as f:
    f.write("xbar (state trajectory):\n")
    for i, x in enumerate(xbar):
        f.write(f"Step {i}: {x.tolist()}\n")
    f.write("\nubar (control trajectory):\n")
    for i, u in enumerate(ubar):
        f.write(f"Step {i}: {u.tolist()}\n")
    f.write("\nVstore (value function trajectory):\n")
    for i, V in enumerate(Vstore):
        f.write(f"Step {i}: {V}\n")
print(f"Results saved to {output_file}")