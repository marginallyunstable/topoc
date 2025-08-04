import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate smooth dummy data for cartpole trajectory
T = 200
timesteps = np.linspace(0, 4 * np.pi, T)
pos = 1.0 * np.sin(0.5 * timesteps)
theta = 0.2 * np.sin(timesteps)
xbar = np.stack([pos, theta], axis=1)  # shape (T, 2)

# Visualization parameters
cart_width = 0.4
cart_height = 0.2
pole_length = 1.0

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(-2, 2)
ax.set_ylim(-0.5, 1.5)
ax.set_aspect('equal')
ax.set_xlabel("Cart Position")
ax.set_ylabel("Height")

cart_patch = plt.Rectangle((0, 0), cart_width, cart_height, color='blue')
pole_line, = ax.plot([], [], lw=4, color='red')
ax.add_patch(cart_patch)

def init():
    cart_patch.set_xy((-cart_width/2, 0))
    pole_line.set_data([], [])
    return cart_patch, pole_line

def animate(i):
    cart_x = xbar[i, 0]
    theta = xbar[i, 1]
    # Cart
    cart_patch.set_xy((cart_x - cart_width/2, 0))
    # Pole
    pole_x0 = cart_x
    pole_y0 = cart_height
    pole_x1 = cart_x + pole_length * np.sin(theta)
    pole_y1 = cart_height + pole_length * np.cos(theta)
    pole_line.set_data([pole_x0, pole_x1], [pole_y0, pole_y1])
    return cart_patch, pole_line

ani = animation.FuncAnimation(
    fig, animate, frames=T, init_func=init, blit=True, interval=30
)

# Save as GIF
ani.save("cartpole_animation.gif", writer="pillow", fps=30)
# Save as MP4
ani.save("cartpole_animation.mp4", writer="ffmpeg", fps=30)

plt.show()