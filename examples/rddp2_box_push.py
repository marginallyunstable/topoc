import jax
import jax.numpy as jnp
from functools import partial
from topoc.utils import quadratic_running_cost, quadratic_terminal_cost
from topoc.base import TOProblemDefinition, TOAlgorithm, TOSolve
from models.qsim_box_push import BoxPushSimulator
from topoc.types import ModelParams, AlgorithmName
from typing import Callable, Tuple
from matplotlib import animation
import os

# Define model parameters (example values)
state_dim = 5
input_dim = 2
horizon_len = 200
dt = 0.01

modelparams = ModelParams(
    state_dim=state_dim,
    input_dim=input_dim,
    horizon_len=horizon_len,
    dt=dt
)

# Define initial and goal states
# State: [box_x, box_y, box_z, hand_x, hand_y] (5 elements total)
x0 = jnp.array([0.5, 0.5, 0.0, -0.4, 0.0])  # Initial: box and hand at same y position
xg = jnp.array([1.5, 1.5, 0.0, 0.0, 0.0])  # Goal: push box forward to y=0.9, hand_y=pi/8

# Define cost matrices
P = jnp.diag(jnp.array([1000000, 1000000, 1000000, 1, 1]))
Q = 1*jnp.eye(state_dim)
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

# # Plot the trajectory of the square block in 2D space
# block_size = 0.05  # Size of the block (square side length)

# for i in range(xbar.shape[0]):
#     x = xbar[i, 0]
#     y = xbar[i, 1]
#     theta = xbar[i, 3]
#     # Calculate the corners of the square block
#     half_size = block_size / 2
#     corners = jnp.array([
#         [-half_size, -half_size],
#         [half_size, -half_size],
#         [half_size, half_size],
#         [-half_size, half_size],
#         [-half_size, -half_size]
#     ])
#     # Rotation matrix
#     rot = jnp.array([
#         [jnp.cos(theta), -jnp.sin(theta)],
#         [jnp.sin(theta),  jnp.cos(theta)]
#     ])
#     rotated = corners @ rot.T
#     translated = rotated + jnp.array([x, y])
#     plt.plot(translated[:, 0], translated[:, 1], 'b-', alpha=0.3)

# # Plot the trajectory as a line
# plt.plot(xbar[:, 0], xbar[:, 1], 'r-', label='Block trajectory')
# plt.scatter(xbar[0, 0], xbar[0, 1], color='green', label='Start')
# plt.scatter(xbar[-1, 0], xbar[-1, 1], color='orange', label='Goal')
# plt.xlabel('X position')
# plt.ylabel('Y position')
# plt.title('2D Block Push Trajectory')
# plt.axis('equal')
# plt.legend()
# plt.show()

# # Save results to a txt file in readable format
# output_file = "/workspace/topoc/examples/rddp2_box_push_results.txt"
# with open(output_file, "w") as f:
#     f.write("xbar (state trajectory):\n")
#     for i, x in enumerate(xbar):
#         f.write(f"Step {i}: {x.tolist()}\n")
#     f.write("\nubar (control trajectory):\n")
#     for i, u in enumerate(ubar):
#         f.write(f"Step {i}: {u.tolist()}\n")
#     f.write("\nVstore (value function trajectory):\n")
#     for i, V in enumerate(Vstore):
#         f.write(f"Step {i}: {V}\n")
# print(f"Results saved to {output_file}")




def animate_box_push(xbar, ubar, block_size=1.0, finger_radius=0.1, robot_radius=0.1, dt=0.01, save_path=None):
    """
    Animates the box push scene.
    Args:
        xbar: (N, 5) array, state trajectory [block_x, block_y, block_theta, finger_x, finger_y]
        ubar: (N, 2) array, robot command trajectory [robot_x, robot_y]
        block_size: float, size of the block (square side length)
        finger_radius: float, radius of the finger
        robot_radius: float, radius of the robot command
        dt: float, time step used in simulation
        save_path: str or None, if provided, saves the animation to this path
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title("Box Push Animation")

    # Block (square)
    block_patch, = ax.plot([], [], 'b-', lw=2)
    # Finger (circle)
    finger_patch = plt.Circle((0, 0), finger_radius, color='g', alpha=0.5)
    ax.add_patch(finger_patch)
    # Robot command (circle)
    robot_patch = plt.Circle((0, 0), robot_radius, color='r', alpha=0.5)
    ax.add_patch(robot_patch)
    # Arrow for box orientation
    arrow_patch = ax.arrow(0, 0, 0, 0, head_width=0.07*block_size, head_length=0.12*block_size, fc='k', ec='k')

    # Plot start and goal positions of the box
    start_x, start_y = xbar[0, 0], xbar[0, 1]
    goal_x, goal_y = xbar[-1, 0], xbar[-1, 1]
    start_marker = ax.scatter(start_x, start_y, color='green', s=80, label='Start', zorder=5)
    goal_marker = ax.scatter(goal_x, goal_y, color='orange', s=80, label='Goal', zorder=5)
    ax.legend()

    def get_block_corners(x, y, theta, size):
        half = size / 2
        corners = jnp.array([
            [-half, -half],
            [half, -half],
            [half, half],
            [-half, half],
            [-half, -half]
        ])
        rot = jnp.array([
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta),  jnp.cos(theta)]
        ])
        rotated = corners @ rot.T
        translated = rotated + jnp.array([x, y])
        return translated

    def init():
        block_patch.set_data([], [])
        finger_patch.center = (0, 0)
        robot_patch.center = (0, 0)
        arrow_patch.set_visible(False)
        # Start and goal markers are static
        return block_patch, finger_patch, robot_patch, arrow_patch, start_marker, goal_marker

    def animate(i):
        nonlocal arrow_patch
        # If i >= xbar.shape[0], just hold the last frame
        idx = min(i, xbar.shape[0] - 1)
        bx, by, btheta = xbar[idx, 0], xbar[idx, 1], xbar[idx, 2]
        fx, fy = xbar[idx, 3], xbar[idx, 4]
        rx, ry = ubar[idx, 0], ubar[idx, 1]
        corners = get_block_corners(bx, by, btheta, block_size)
        block_patch.set_data(corners[:, 0], corners[:, 1])
        finger_patch.center = (fx, fy)
        robot_patch.center = (rx, ry)
        # Arrow for orientation
        arrow_length = block_size * 0.7
        dx = arrow_length * jnp.cos(btheta)
        dy = arrow_length * jnp.sin(btheta)
        arrow_patch.remove()
        arrow_patch_new = ax.arrow(
            bx, by, dx, dy,
            head_width=0.07*block_size, head_length=0.12*block_size, fc='k', ec='k'
        )
        arrow_patch = arrow_patch_new
        # Start and goal markers are static
        return block_patch, finger_patch, robot_patch, arrow_patch, start_marker, goal_marker

    # Add 2 second pause at the end
    interval_ms = int(dt * 1000)
    pause_frames = int(2.0 / dt)
    total_frames = xbar.shape[0] + pause_frames

    ani = animation.FuncAnimation(
        fig, animate, frames=total_frames, init_func=init,
        blit=True, interval=interval_ms, repeat=False
    )

    if save_path:
        try:
            # Determine format based on file extension
            if save_path.endswith('.gif'):
                print(f"Saving as GIF: {save_path}")
                ani.save(save_path, writer='pillow')
            elif save_path.endswith('.mp4'):
                print(f"Saving as MP4: {save_path}")
                ani.save(save_path, writer='ffmpeg')
            else:
                print(f"Unknown format, saving as GIF: {save_path}")
                ani.save(save_path, writer='pillow')
            print(f"✓ Animation successfully saved to {save_path}")
        except Exception as e:
            print(f"❌ Error saving animation: {e}")
    else:
        plt.show()

    return ani


# Save animations in both GIF and MP4 formats
script_folder = os.path.dirname(os.path.abspath(__file__))

# Save as GIF
gif_path = os.path.join(script_folder, "box_push_animation.gif")
animate_box_push(xbar, ubar, block_size=1, finger_radius=0.2, robot_radius=0.2, dt=dt, save_path=gif_path)

# Save as MP4 video
mp4_path = os.path.join(script_folder, "box_push_animation.mp4")
animate_box_push(xbar, ubar, block_size=1, finger_radius=0.2, robot_radius=0.2, dt=dt, save_path=mp4_path)