"""Animate pendulum results saved in an .npz file.

Similar interface to animate_block_results.py but shows an anchored pendulum (pivot at joint)
with a circular arc indicator around the pivot that is scaled by the input torque magnitude.

Usage:
    python animate_pendulum_results.py --npz results/pendulum_solution.npz --out-dir /workspace/topoc/scripts --fps 30

Expect keys in the .npz like '<sanitized>_xbar' (angles, angvel) and '<sanitized>_ubar' (torques).
If `dt` is not provided on the command line the script will try to read it from the npz file.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Arc, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# Visualization params
PHASE_MARGIN = 0.15
INPUT_MARGIN = 0.10
TRAIL_LENGTH = 10
ARROW_MIN = 1e-3

# Friction threshold line settings (similar to animate_block_results.py)
SHOW_FRICTION_LINE = True
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
    data = np.load(path, allow_pickle=True)
    return data


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

    # read start and goal if present in file
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

    # try read dt
    if dt is None and "dt" in data:
        try:
            dt = float(np.array(data["dt"]))
        except Exception:
            dt = None
    if dt is None:
        raise ValueError("dt not provided and not found in npz file. Provide --dt")

    # determine algorithms present
    if "sanitized_names" in data:
        san = [str(s) for s in np.array(data["sanitized_names"])]
    else:
        san = DEFAULT_ALG_ORDER

    algos = []
    for sname in san:
        arrays = _get_algo_arrays(data, sname)
        if arrays is None:
            continue
        xbar, ubar, Vstore = arrays
        algos.append((sname, xbar, ubar))

    if not algos:
        raise ValueError("No algorithm data found in npz file")

    # horizon (min over algorithms)
    horizons = []
    for _, xbar, ubar in algos:
        horizons.append(min(len(ubar), max(0, len(xbar) - 1)))
    horizon = min(horizons)

    nrows = len(algos)
    ncols = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 2.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    artists = []

    # precompute series and extents
    all_angles = []
    all_angv = []
    all_inputs = []
    tvecs = []

    for (_, xbar, ubar) in algos:
        angles = np.asarray(xbar)[:horizon+1, 0]  # radians
        angv = np.asarray(xbar)[:horizon+1, 1]
        inputs = np.asarray(ubar)[:horizon, 0]
        all_angles.append(angles)
        all_angv.append(angv)
        all_inputs.append(inputs)
        tvecs.append(np.arange(len(inputs)) * dt)

    # angle limits for phase plot
    all_angles_concat = np.concatenate(all_angles)
    all_angv_concat = np.concatenate(all_angv)
    a_min, a_max = float(all_angles_concat.min()), float(all_angles_concat.max())
    w_min, w_max = float(all_angv_concat.min()), float(all_angv_concat.max())
    a_range = max(1e-6, a_max - a_min)
    w_range = max(1e-6, w_max - w_min)
    phase_xlim = (a_min - PHASE_MARGIN * a_range, a_max + PHASE_MARGIN * a_range)
    phase_ylim = (w_min - PHASE_MARGIN * w_range, w_max + PHASE_MARGIN * w_range)

    # input limits
    all_u_concat = np.concatenate(all_inputs)
    u_min, u_max = float(all_u_concat.min()), float(all_u_concat.max())
    u_range = max(1e-6, u_max - u_min)
    input_ylim = (u_min - INPUT_MARGIN * u_range, u_max + INPUT_MARGIN * u_range)
    time_full_common = np.arange(horizon) * dt
    input_xlim = (0.0, float(time_full_common[-1]) if len(time_full_common) else 1.0)

    # arrow/arc scaling
    u_abs_concat = np.abs(all_u_concat)
    u_max_abs = max(1.0, float(np.max(u_abs_concat)) + 1e-6)
    arc_scale = 0.5 / u_max_abs  # arc radial scaling factor (units relative to pendulum length)
    arrow_eps = ARROW_MIN

    # pendulum visual params
    pend_length = 1.0
    bob_radius = 0.06 * pend_length

    for i, (alg_name, xbar, ubar) in enumerate(algos):
        angles = all_angles[i]
        angv = all_angv[i]
        inputs = all_inputs[i]
        tvec = tvecs[i]

        # panel 1: pendulum view
        ax0 = axes[i, 0]
        ax0.set_aspect('equal')
        # make axes symmetric so pendulum from bottom (pi) to top (0) is fully visible
        lim = 1.4 * pend_length
        ax0.set_xlim(-lim, lim)
        ax0.set_ylim(-lim, lim)
        ax0.axis('off')
        ax0.set_title(DISPLAY_NAMES[i] if i < len(DISPLAY_NAMES) else alg_name)

        # pivot
        pivot = (0.0, 0.0)
        pivot_dot = Circle(pivot, 0.02, color='k', zorder=8)
        ax0.add_patch(pivot_dot)

        # initial bob position
        theta0 = float(angles[0])
        # convert state angle (0=up, pi=down) to plotting angle used by _polar_to_xy
        # NOTE: use pi/2 + theta so positive theta rotates CCW in plotting coords
        theta_plot0 = np.pi/2 + theta0
        bob_x = pend_length * np.cos(theta_plot0)
        bob_y = pend_length * np.sin(theta_plot0)
        bob = Circle((bob_x, bob_y), bob_radius, facecolor='#2b7bba', edgecolor='k', zorder=6)
        ax0.add_patch(bob)

        # rod line
        rod_line, = ax0.plot([pivot[0], bob_x], [pivot[1], bob_y], color='k', lw=2, zorder=5)

        # trail of bob
        trail_line, = ax0.plot([], [], color="#d84810", lw=2, alpha=0.9, zorder=4)

        # arc and arrow placeholders
        arc_artist = None
        arrow_artist = None
        # box border around the first-column axes to match cartpole style
        xmin, xmax = ax0.get_xlim()
        ymin, ymax = ax0.get_ylim()
        box = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='k', lw=1.2, zorder=2)
        ax0.add_patch(box)

        # panel 2: phase plot
        ax1 = axes[i, 1]
        ax1.set_xlabel('angle (rad)')
        ax1.set_ylabel('ang vel (rad/s)')
        phase_line, = ax1.plot(angles, angv, color='tab:blue', lw=2)
        phase_dot, = ax1.plot([angles[0]], [angv[0]], 'o', color='red')
        # plot start and goal markers if available
        if x0_data is not None:
            ax1.scatter(x0_data[0], x0_data[1], color="#1755b1", s=80, marker='o', zorder=6)
        if xg_data is not None:
            ax1.scatter(xg_data[0], xg_data[1], color="#48e750", s=100, marker='*', zorder=6)
        # set shared limits with margin
        ax1.set_xlim(phase_xlim)
        ax1.set_ylim(phase_ylim)
        if i == 0:
            ax1.set_title('Phase Plot', pad=24)
            # create centered legend below the title for start/goal markers
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

        # panel 3: input vs time
        ax2 = axes[i, 2]
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Torque')
        time_full = np.arange(len(inputs)) * dt
        input_line, = ax2.plot(time_full, inputs, color='tab:blue', lw=1.5)
        input_dot, = ax2.plot([0.0], [inputs[0]], 'o', color='red')
        ax2.set_xlim(input_xlim)
        ax2.set_ylim(input_ylim)
        if i == 0:
            ax2.set_title('Input vs Time', pad=24)

        # friction threshold line
        if SHOW_FRICTION_LINE:
            fr_line = ax2.axhline(FRICTION_THRESHOLD, color="#686565", linestyle='--', linewidth=3, alpha=0.5, zorder=3, label='Friction threshold')
            if i == 0:
                try:
                    ax2.legend(handles=[fr_line], loc='lower center', bbox_to_anchor=(0.5, 1.0), bbox_transform=ax2.transAxes, fontsize='small')
                except Exception:
                    pass

        artists.append({
            'ax0': ax0,
            'bob': bob,
            'rod': rod_line,
            'box': box,
            'pivot': pivot,
            'trail': trail_line,
            'arc': arc_artist,
            'arrow': arrow_artist,
            'angles': angles,
            'angv': angv,
            'inputs': inputs,
            'tvec': tvec,
            'phase_dot': phase_dot,
            'input_dot': input_dot,
        })

    plt.tight_layout()

    def _polar_to_xy(r, theta):
        # convert state angle (theta) where 0 = vertically up, pi = vertically down
        # to plotting polar angle where 0 = +x (right) and increases CCW.
        # Use theta_plot = pi/2 + theta so positive state theta moves CCW in the plot.
        theta_plot = np.pi/2 + theta
        return (r * np.cos(theta_plot), r * np.sin(theta_plot))

    def update(frame):
        for art in artists:
            # keep border box sized to axes limits in case of changes
            b = art.get('box')
            if b is not None:
                xmin, xmax = art['ax0'].get_xlim()
                ymin, ymax = art['ax0'].get_ylim()
                try:
                    b.set_xy((xmin, ymin))
                    b.set_width(xmax - xmin)
                    b.set_height(ymax - ymin)
                except Exception:
                    pass
            theta = float(art['angles'][frame])
            w = float(art['angv'][frame])
            u = float(art['inputs'][frame])

            # bob position
            bx, by = _polar_to_xy(pend_length, theta)
            art['bob'].center = (bx, by)
            art['rod'].set_data([art['pivot'][0], bx], [art['pivot'][1], by])

            # trail
            start_idx = max(0, frame - TRAIL_LENGTH + 1)
            xs = art['angles'][start_idx: frame + 1]
            # convert angles into xy for trail
            xy = np.array([_polar_to_xy(pend_length, a) for a in xs])
            if len(xy):
                art['trail'].set_data(xy[:, 0], xy[:, 1])
            else:
                art['trail'].set_data([], [])

            # remove previous arc/arrow if present
            if art.get('arc') is not None and art['arc'] in art['ax0'].patches:
                try:
                    art['ax0'].patches.remove(art['arc'])
                except Exception:
                    pass
                art['arc'] = None
            if art.get('arrow') is not None:
                try:
                    art['arrow'].remove()
                except Exception:
                    pass
                art['arrow'] = None

            # draw circular arc around pivot scaled by |u|
            r_base = 0.05 #0.28 * pend_length
            r = r_base + arc_scale * abs(u)
            # plotting degrees where 0deg is +x axis: start = 135deg, end = 45deg
            plot_start_deg = 135.0
            plot_end_deg = 45.0

            # draw CCW arc from 135 -> 45 by using 45+360 = 405 as end
            theta1 = plot_start_deg
            theta2 = plot_end_deg + 360.0

            arc = Arc(art['pivot'], width=2 * r, height=2 * r, angle=0.0, theta1=theta1, theta2=theta2,
                      color='red', lw=2, zorder=7, alpha=0.9)
            art['ax0'].add_patch(arc)
            art['arc'] = arc

            # place arrow head: for positive torque head should be on plot_end (45deg) side;
            # for negative torque head should be on plot_start (135deg) side.
            small_deg = 8.0
            if u >= 0:
                tip_plot_deg = plot_end_deg + 360.0  # 405deg
                interior_plot_deg = tip_plot_deg - small_deg
            else:
                tip_plot_deg = plot_start_deg  # 135deg
                interior_plot_deg = tip_plot_deg + small_deg

            tip_rad = np.deg2rad(tip_plot_deg)
            interior_rad = np.deg2rad(interior_plot_deg)
            # compute points in plotting coords (Arc uses plotting coordinates)
            end_pt = (r * np.cos(tip_rad), r * np.sin(tip_rad))
            interior_pt = (r * np.cos(interior_rad), r * np.sin(interior_rad))

            # draw a short FancyArrowPatch from interior_pt -> end_pt to create arrow head
            if abs(u) >= arrow_eps:
                arr = FancyArrowPatch(posA=interior_pt, posB=end_pt, arrowstyle='-|>', mutation_scale=12,
                                      color='red', lw=1.5, zorder=8)
                art['ax0'].add_patch(arr)
                art['arrow'] = arr

            # update phase and input dots
            art['phase_dot'].set_data([theta], [w])
            art['input_dot'].set_data([art['tvec'][frame]], [u])

        return []

    interval = 1000 * dt
    anim = animation.FuncAnimation(fig, update, frames=horizon, interval=interval, blit=False)

    os.makedirs(out_dir, exist_ok=True)
    mp4_path = os.path.join(out_dir, 'pendulum_compare_animation.mp4')
    gif_path = os.path.join(out_dir, 'pendulum_compare_animation.gif')

    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='topoc'), bitrate=4000)
        anim.save(mp4_path, writer=writer)
        print(f"Saved MP4 to: {mp4_path}")
    except Exception as e:
        print("Failed to save MP4 using ffmpeg:", e)

    try:
        from matplotlib.animation import PillowWriter
        pw = PillowWriter(fps=fps)
        anim.save(gif_path, writer=pw)
        print(f"Saved GIF to: {gif_path}")
    except Exception as e:
        print("Failed to save GIF:", e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=str, default='results/pendulum_with_friction_solution.npz', help='Path to .npz results file')
    parser.add_argument('--out-dir', type=str, default=os.path.dirname(__file__), help='Directory to save animations')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for saved animations')
    parser.add_argument('--dt', type=float, default=None, help='Timestep (s). If not provided, the script will try to read from the npz file')

    args = parser.parse_args()
    build_animation(args.npz, args.out_dir, fps=args.fps, dt=args.dt)
