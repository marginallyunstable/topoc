"""Animate cartpole results saved in an .npz file.

Creates a row per algorithm with 4 columns:
 - col 1: cart + pole view (cart translates along x, pole pivots on cart)
 - col 2: cart phase plot (position vs velocity)
 - col 3: pole phase plot (angle vs angular velocity)
 - col 4: input vs time (force)

Usage:
    python animate_cartpole_results.py --npz results/cartpole_solution.npz --out-dir /workspace/topoc/scripts --fps 30

Expect keys in the .npz like '<sanitized>_xbar' (shape (T+1,4): [pos, vel, angle, angvel]) and '<sanitized>_ubar' (shape (T,1)).
Angle convention: state angle 0 = bottom (pole down), pi = vertically up. Mapping for plotting converts state angle -> plotting angle where 0 = +x and increases CCW.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

# Visual params
CART_TRAIL_LENGTH = 50
POLE_TRAIL_LENGTH = 30
PHASE_MARGIN = 0.15
INPUT_MARGIN = 0.10
ARROW_MIN = 1e-3

# Friction threshold
SHOW_FRICTION_LINE = False
FRICTION_THRESHOLD = -5.0

plt.rcParams.update({
    "mathtext.fontset": "stixsans",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "text.usetex": False,
    "mathtext.default": "regular",
})

DEFAULT_ALG_ORDER = ["ddp", "scs_ddp", "cs_ddp", "pddp"]
DISPLAY_NAMES = ["DDP", "SCS-DDP", "CS-DDP", "PDDP"]


def _load_npz(path):
    return np.load(path, allow_pickle=True)


def _get_algo_arrays(data, sanitized_name):
    xkey = f"{sanitized_name}_xbar"
    ukey = f"{sanitized_name}_ubar"
    vkey = f"{sanitized_name}_Vstore"
    if xkey not in data or ukey not in data:
        return None
    xbar = np.array(data[xkey])
    ubar = np.array(data[ukey])
    Vstore = np.array(data[vkey]) if vkey in data else None
    return xbar, ubar, Vstore


def _sanitize_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def build_animation(npz_path, out_dir, fps=30, dt=None):
    data = _load_npz(npz_path)

    # read start/goal if present
    x0_data = None
    xg_data = None
    if 'x0' in data:
        try:
            x0_data = np.array(data['x0']).astype(float)
        except Exception:
            x0_data = None
    if 'xg' in data:
        try:
            xg_data = np.array(data['xg']).astype(float)
        except Exception:
            xg_data = None

    # dt
    if dt is None and 'dt' in data:
        try:
            dt = float(np.array(data['dt']))
        except Exception:
            dt = None
    if dt is None:
        raise ValueError('dt not provided and not found in npz file. Provide --dt')

    # algos
    if 'sanitized_names' in data:
        san = [str(s) for s in np.array(data['sanitized_names'])]
    else:
        san = DEFAULT_ALG_ORDER

    algos = []
    for s in san:
        arrays = _get_algo_arrays(data, s)
        if arrays is None:
            continue
        xbar, ubar, Vstore = arrays
        algos.append((s, xbar, ubar))

    if not algos:
        raise ValueError('No algorithm data found in npz file')

    # horizon
    horizons = [min(len(ubar), max(0, len(xbar) - 1)) for (_, xbar, ubar) in algos]
    horizon = min(horizons)

    nrows = len(algos)
    ncols = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 2.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    artists = []

    # precompute
    all_pos = []
    all_vel = []
    all_angle = []
    all_angv = []
    all_inputs = []
    tvecs = []

    for (_, xbar, ubar) in algos:
        pos = np.asarray(xbar)[:horizon+1, 0]
        vel = np.asarray(xbar)[:horizon+1, 1]
        ang = np.asarray(xbar)[:horizon+1, 2]
        angv = np.asarray(xbar)[:horizon+1, 3]
        u = np.asarray(ubar)[:horizon, 0]
        all_pos.append(pos)
        all_vel.append(vel)
        all_angle.append(ang)
        all_angv.append(angv)
        all_inputs.append(u)
        tvecs.append(np.arange(len(u)) * dt)

    pos_min = min(p.min() for p in all_pos)
    pos_max = max(p.max() for p in all_pos)
    pos_range = max(1e-6, pos_max - pos_min)
    xpad = max(0.5, 0.05 * pos_range)

    # arrow/force scaling
    all_u_vals = np.concatenate(all_inputs)
    u_max = max(1.0, np.max(np.abs(all_u_vals)) + 1e-6)
    arrow_scale = 0.5 * (pos_max - pos_min + 1e-6) / u_max
    arrow_eps = ARROW_MIN * (pos_max - pos_min + 1e-6)

    # phase limits cart
    all_pos_concat = np.concatenate(all_pos)
    all_vel_concat = np.concatenate(all_vel)
    p_min, p_max = float(all_pos_concat.min()), float(all_pos_concat.max())
    v_min, v_max = float(all_vel_concat.min()), float(all_vel_concat.max())
    p_range = max(1e-6, p_max - p_min)
    v_range = max(1e-6, v_max - v_min)
    cart_xlim = (p_min - PHASE_MARGIN * p_range, p_max + PHASE_MARGIN * p_range)
    cart_ylim = (v_min - PHASE_MARGIN * v_range, v_max + PHASE_MARGIN * v_range)

    # phase limits pole
    all_ang_concat = np.concatenate(all_angle)
    all_angv_concat = np.concatenate(all_angv)
    a_min, a_max = float(all_ang_concat.min()), float(all_ang_concat.max())
    w_min, w_max = float(all_angv_concat.min()), float(all_angv_concat.max())
    a_range = max(1e-6, a_max - a_min)
    w_range = max(1e-6, w_max - w_min)
    pole_xlim = (a_min - PHASE_MARGIN * a_range, a_max + PHASE_MARGIN * a_range)
    pole_ylim = (w_min - PHASE_MARGIN * w_range, w_max + PHASE_MARGIN * w_range)

    # input plot limits
    all_u_concat = np.concatenate(all_inputs)
    u_min, u_max = float(all_u_concat.min()), float(all_u_concat.max())
    u_range = max(1e-6, u_max - u_min)
    input_ylim = (u_min - INPUT_MARGIN * u_range, u_max + INPUT_MARGIN * u_range)
    time_full_common = np.arange(horizon) * dt
    input_xlim = (0.0, float(time_full_common[-1]) if len(time_full_common) else 1.0)

    # visual params for cart/pole
    pend_length = 0.6
    cart_width = 0.3
    cart_height = 0.12
    wheel_r = 0.05
    bob_radius = 0.04

    def _pole_to_xy(r, theta_state):
        # state theta: 0 = bottom (down), pi = up. plotting angle: 0 = +x, CCW positive.
        # when state theta=0 -> plotting should be -pi/2 (down) => theta_plot = theta_state - pi/2
        theta_plot = theta_state - np.pi/2
        return (r * np.cos(theta_plot), r * np.sin(theta_plot))

    for i, (alg_name, xbar, ubar) in enumerate(algos):
        pos = all_pos[i]
        vel = all_vel[i]
        ang = all_angle[i]
        angv = all_angv[i]
        inputs = all_inputs[i]
        tvec = tvecs[i]

        # col 1: cart + pole view
        ax0 = axes[i, 0]
        ax0.set_aspect('equal')
        ax0.set_xlim(pos_min - xpad, pos_max + xpad)
        # vertical limits: allow pole up/down
        ylim_low = - (pend_length + 0.4)
        ylim_high = pend_length + 0.4
        ax0.set_ylim(ylim_low, ylim_high)
        ax0.get_yaxis().set_visible(False)
        ax0.set_xlabel('Position')
        ax0.set_title(DISPLAY_NAMES[i] if i < len(DISPLAY_NAMES) else alg_name)

        # ground line at y=0
        ground_y = 0.0
        ax0.plot([pos_min - xpad, pos_max + xpad], [ground_y, ground_y], color='k', lw=2, zorder=1)

        # place wheels so they rest on the ground line
        cart_x0 = pos[0]
        wheel_center_y = ground_y + wheel_r
        # cart center y so cart sits on top of wheels
        cart_y = wheel_center_y + wheel_r + cart_height/2
        cart = Rectangle((cart_x0 - cart_width/2, cart_y - cart_height/2), cart_width, cart_height, facecolor='#2b7bba', edgecolor='k', zorder=6)
        ax0.add_patch(cart)
        # wheels as small circles with centers at wheel_center_y (touching ground)
        wheel_l = Circle((cart_x0 - cart_width/3, wheel_center_y), wheel_r, color='k', zorder=6)
        wheel_rp = Circle((cart_x0 + cart_width/3, wheel_center_y), wheel_r, color='k', zorder=6)
        ax0.add_patch(wheel_l)
        ax0.add_patch(wheel_rp)

        # pole initial (pivot at center of cart rectangle)
        pole_end = _pole_to_xy(pend_length, ang[0])
        pivot = (cart_x0, cart_y)
        pole_line, = ax0.plot([pivot[0], pivot[0] + pole_end[0]], [pivot[1], pivot[1] + pole_end[1]], color='k', lw=2, zorder=7)
        pole_bob = Circle((pivot[0] + pole_end[0], pivot[1] + pole_end[1]), bob_radius, facecolor='#d94810', edgecolor='k', zorder=8)
        ax0.add_patch(pole_bob)

        # trail for cart (positions)
        trail_line, = ax0.plot([], [], color="#d84810", lw=2, alpha=0.9, zorder=4)
        # trail for pole tip
        pole_trail_line, = ax0.plot([], [], color="#1755b1", lw=2, alpha=0.9, zorder=3)

        # arrow placeholder
        arrow = None

        # col 2: cart phase
        ax1 = axes[i, 1]
        ax1.set_xlabel('position')
        ax1.set_ylabel('velocity')
        cart_phase_line, = ax1.plot(pos, vel, color='tab:blue', lw=2)
        cart_dot, = ax1.plot([pos[0]], [vel[0]], 'o', color='red')
        if x0_data is not None:
            ax1.scatter(x0_data[0], x0_data[1], color="#1755b1", s=80, marker='o', zorder=6)
        if xg_data is not None:
            ax1.scatter(xg_data[0], xg_data[1], color="#48e750", s=100, marker='*', zorder=6)
        ax1.set_xlim(cart_xlim)
        ax1.set_ylim(cart_ylim)
        if i == 0:
            ax1.set_title('Phase Plot (Cart)', pad=24)
            legend_handles = []
            if x0_data is not None:
                legend_handles.append(Line2D([0], [0], marker='o', color="#1755b1", markerfacecolor="#1755b1", linestyle='None', markersize=8, label='Start'))
            if xg_data is not None:
                legend_handles.append(Line2D([0], [0], marker='*', color="#48e750", markerfacecolor="#48e750", linestyle='None', markersize=10, label='Goal'))
            if legend_handles:
                try:
                    ax1.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), bbox_transform=ax1.transAxes, ncol=len(legend_handles), fontsize='small')
                except Exception:
                    pass

        # col 3: pole phase
        ax2 = axes[i, 2]
        ax2.set_xlabel('angle (rad)')
        ax2.set_ylabel('ang vel (rad/s)')
        pole_phase_line, = ax2.plot(ang, angv, color='tab:blue', lw=2)
        pole_dot, = ax2.plot([ang[0]], [angv[0]], 'o', color='red')
        if x0_data is not None:
            # start and goal angle components
            ax2.scatter(x0_data[2], x0_data[3], color="#1755b1", s=80, marker='o', zorder=6)
        if xg_data is not None:
            ax2.scatter(xg_data[2], xg_data[3], color="#48e750", s=100, marker='*', zorder=6)
        ax2.set_xlim(pole_xlim)
        ax2.set_ylim(pole_ylim)
        if i == 0:
            ax2.set_title('Phase Plot (Pole)', pad=24)
            # create centered legend below the title for start/goal markers
            legend_handles_pole = []
            if x0_data is not None:
                legend_handles_pole.append(Line2D([0], [0], marker='o', color="#1755b1", markerfacecolor="#1755b1", linestyle='None', markersize=8, label='Start'))
            if xg_data is not None:
                legend_handles_pole.append(Line2D([0], [0], marker='*', color="#48e750", markerfacecolor="#48e750", linestyle='None', markersize=10, label='Goal'))
            if legend_handles_pole:
                try:
                    ax2.legend(handles=legend_handles_pole, loc='lower center', bbox_to_anchor=(0.5, 1.0), bbox_transform=ax2.transAxes, ncol=len(legend_handles_pole), fontsize='small')
                except Exception:
                    pass

        # col 4: input
        ax3 = axes[i, 3]
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Force')
        time_full = np.arange(len(inputs)) * dt
        input_line, = ax3.plot(time_full, inputs, color='tab:blue', lw=1.5)
        input_dot, = ax3.plot([0.0], [inputs[0]], 'o', color='red')
        ax3.set_xlim(input_xlim)
        ax3.set_ylim(input_ylim)
        if i == 0:
            ax3.set_title('Input vs Time', pad=24)
        if SHOW_FRICTION_LINE:
            fr_line = ax3.axhline(FRICTION_THRESHOLD, color="#686565", linestyle='--', linewidth=3, alpha=0.5, zorder=3, label='Friction threshold')
            if i == 0:
                try:
                    ax3.legend(handles=[fr_line], loc='lower center', bbox_to_anchor=(0.5, 1.0), bbox_transform=ax3.transAxes, fontsize='small')
                except Exception:
                    pass

        artists.append({
            'cart': cart,
            'wheel_l': wheel_l,
            'wheel_r': wheel_rp,
            'pole_line': pole_line,
            'pole_bob': pole_bob,
            'pivot': pivot,
            'trail': trail_line,
            'pole_trail': pole_trail_line,
            'arrow': arrow,
            'pos': pos,
            'vel': vel,
            'ang': ang,
            'angv': angv,
            'inputs': inputs,
            'tvec': tvec,
            'ax0': ax0,
            'ax1': ax1,
            'ax2': ax2,
            'ax3': ax3,
            'cart_dot': cart_dot,
            'pole_dot': pole_dot,
            'input_dot': input_dot,
        })

    plt.tight_layout()

    def update(frame):
        for art in artists:
            p = float(art['pos'][frame])
            v = float(art['vel'][frame])
            th = float(art['ang'][frame])
            w = float(art['angv'][frame])
            u = float(art['inputs'][frame])

            # update cart position and wheels
            cart_x = p
            art['cart'].set_x(cart_x - cart_width/2)
            # keep wheels on ground (y coordinate stays wheel_center_y)
            art['wheel_l'].center = (cart_x - cart_width/3, wheel_center_y)
            art['wheel_r'].center = (cart_x + cart_width/3, wheel_center_y)

            # pivot based on cart center (joint at center of rectangle)
            pivot = (cart_x, wheel_center_y + wheel_r + cart_height/2)
            art['pivot'] = pivot

            # pole end
            end = _pole_to_xy(pend_length, th)
            art['pole_line'].set_data([pivot[0], pivot[0] + end[0]], [pivot[1], pivot[1] + end[1]])
            art['pole_bob'].center = (pivot[0] + end[0], pivot[1] + end[1])

            # trail (cart trajectory)
            start_idx = max(0, frame - CART_TRAIL_LENGTH + 1)
            xs = art['pos'][start_idx: frame + 1]
            ys = np.zeros_like(xs) + (wheel_center_y + wheel_r + cart_height/2)
            art['trail'].set_data(xs, ys)

            # pole tip trail
            angs = art['ang'][max(0, frame - POLE_TRAIL_LENGTH + 1): frame + 1]
            if len(angs):
                pts = np.array([_pole_to_xy(pend_length, a) for a in angs])
                # shift by current pivot location
                piv = art.get('pivot', pivot)
                pts[:, 0] += piv[0]
                pts[:, 1] += piv[1]
                art['pole_trail'].set_data(pts[:, 0], pts[:, 1])
            else:
                art['pole_trail'].set_data([], [])

            # update arrow: remove previous
            if art.get('arrow') is not None:
                try:
                    art['arrow'].remove()
                except Exception:
                    pass
                art['arrow'] = None

            # draw horizontal arrow from cart showing force
            dx = u * arrow_scale
            start = (cart_x, pivot[1])
            head_len = 0.02 * (pos_max - pos_min + 1e-6)
            end = (cart_x + dx + np.sign(dx) * head_len, pivot[1])
            if abs(dx) >= arrow_eps:
                art['arrow'] = art['ax0'].annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='red', lw=2), zorder=8)

            # update phase dots
            art['cart_dot'].set_data([p], [v])
            art['pole_dot'].set_data([th], [w])
            art['input_dot'].set_data([art['tvec'][frame]], [u])

        return []

    interval = 1000 * dt
    anim = animation.FuncAnimation(fig, update, frames=horizon, interval=interval, blit=False)

    os.makedirs(out_dir, exist_ok=True)
    mp4_path = os.path.join(out_dir, 'cartpole_compare_animation.mp4')
    gif_path = os.path.join(out_dir, 'cartpole_compare_animation.gif')

    # Try saving MP4 with ffmpeg
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='topoc'), bitrate=4000)
        anim.save(mp4_path, writer=writer)
        print(f"Saved MP4 to: {mp4_path}")
    except Exception as e:
        print("Failed to save MP4 using ffmpeg:", e)

    # Save GIF using Pillow writer
    try:
        from matplotlib.animation import PillowWriter
        pw = PillowWriter(fps=fps)
        anim.save(gif_path, writer=pw)
        print(f"Saved GIF to: {gif_path}")
    except Exception as e:
        print("Failed to save GIF:", e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=str, default='results/cartpole_with_friction_solution.npz', help='Path to .npz results file')
    parser.add_argument('--out-dir', type=str, default=os.path.dirname(__file__), help='Directory to save animations')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for saved animations')
    parser.add_argument('--dt', type=float, default=None, help='Timestep (s). If not provided, the script will try to read from the npz file')

    args = parser.parse_args()
    build_animation(args.npz, args.out_dir, fps=args.fps, dt=args.dt)
