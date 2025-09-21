"""Animate block results saved in an .npz file.

Creates a 4-row (DDP, SCS-DDP, CS-DDP, PDDP) x 3-column animation per row:
 - column 1: 2D block-on-ground view (block rectangle moves along x), trail, and arrow for input force
 - column 2: phase plot (position vs velocity) with red dot
 - column 3: input vs time with red dot

Saves outputs to MP4 and GIF in the same folder as this script.

Usage:
    python animate_block_results.py --npz results/all_algorithms_solution.npz --out-dir /workspace/topoc/scripts --fps 30

If dt is stored in the file it will be used, otherwise pass --dt.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Add visualization parameters
TRAIL_LENGTH = 50  # number of previous timesteps to show in the trail
PHASE_MARGIN = 0.15  # fraction of range to pad phase plot axes
INPUT_MARGIN = 0.10  # fraction of range to pad input plot axes

# Friction threshold line settings
SHOW_FRICTION_LINE = True
FRICTION_THRESHOLD = 7.848

# Minimum arrow length (fraction of position range) below which we show a small marker instead of an arrow
ARROW_MIN = 1e-3

# Apply requested matplotlib rc parameters to match style
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
    """Return xbar (T+1, state_dim), ubar (T, input_dim), Vstore if present.
    Keys are expected like '<sanitized>_xbar', '<sanitized>_ubar'."""
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


def build_animation(npz_path, out_dir, fps=30, dt=None, friction_thresh=None):
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
    # Try to infer dt
    if dt is None and "dt" in data:
        try:
            dt = float(np.array(data["dt"]))
        except Exception:
            dt = None
    if dt is None:
        raise ValueError("dt not provided and not found in npz file. Provide --dt")

    # Determine algorithm list
    if "sanitized_names" in data:
        san = [str(s) for s in np.array(data["sanitized_names"]) ]
    else:
        # fallback to default sanitized names
        san = DEFAULT_ALG_ORDER

    algos = []
    for sname in san:
        arrays = _get_algo_arrays(data, sname)
        if arrays is None:
            # skip missing
            continue
        xbar, ubar, Vstore = arrays
        algos.append((sname, xbar, ubar))

    if not algos:
        raise ValueError("No algorithm data found in npz file")

    # Use minimum length among algorithms for animation horizon
    horizons = []
    for _, xbar, ubar in algos:
        # xbar may be T+1, ubar T; we animate for min(T, len(xbar)-1)
        horizons.append(min(len(ubar), max(0, len(xbar)-1)))
    horizon = min(horizons)

    # Prepare figure: 4 rows x 3 cols
    nrows = len(algos)
    ncols = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 2.5 * nrows))

    # Ensure axes is 2D array
    if nrows == 1:
        axes = axes[np.newaxis, :]

    artists = []  # structure for update

    # Precompute time vector for inputs
    tvecs = []

    # Precompute extents for block animation
    all_positions = []
    all_vels = []
    all_inputs = []

    for (_, xbar, ubar) in algos:
        # assume state: [position, velocity]
        positions = np.asarray(xbar)[:horizon+1, 0]
        velocities = np.asarray(xbar)[:horizon+1, 1]
        inputs = np.asarray(ubar)[:horizon, 0]
        all_positions.append(positions)
        all_vels.append(velocities)
        all_inputs.append(inputs)
        tvecs.append(np.arange(len(inputs)) * dt)

    # Determine x limits across all algos
    pos_min = min(p.min() for p in all_positions)
    pos_max = max(p.max() for p in all_positions)
    xpad = max(0.5, 0.05 * (pos_max - pos_min + 1e-6))
    block_width = 0.15 * max(1.0, pos_max - pos_min + 1e-6)  # visual width

    # Determine input scaling for arrow visuals
    all_u_vals = np.concatenate(all_inputs)
    u_max = max(1.0, np.max(np.abs(all_u_vals)) + 1e-6)
    arrow_scale = 0.5 * (pos_max - pos_min + 1e-6) / u_max
    # compute a small threshold for arrow display (in position units)
    arrow_eps = ARROW_MIN * (pos_max - pos_min + 1e-6)

    # Compute shared phase plot limits across all algorithms
    all_pos_concat = np.concatenate(all_positions)
    all_vel_concat = np.concatenate(all_vels)
    p_min, p_max = float(all_pos_concat.min()), float(all_pos_concat.max())
    v_min, v_max = float(all_vel_concat.min()), float(all_vel_concat.max())
    # pad both lower and upper using PHASE_MARGIN fraction of range
    p_range = max(1e-6, p_max - p_min)
    v_range = max(1e-6, v_max - v_min)
    phase_xlim = (p_min - PHASE_MARGIN * p_range, p_max + PHASE_MARGIN * p_range)
    phase_ylim = (v_min - PHASE_MARGIN * v_range, v_max + PHASE_MARGIN * v_range)

    # Compute shared input plot limits across all algorithms
    all_u_concat = np.concatenate(all_inputs)
    u_min, u_max = float(all_u_concat.min()), float(all_u_concat.max())
    u_range = max(1e-6, u_max - u_min)
    input_ylim = (u_min - INPUT_MARGIN * u_range, u_max + INPUT_MARGIN * u_range)
    time_full_common = np.arange(horizon) * dt
    input_xlim = (0.0, float(time_full_common[-1]) if len(time_full_common) else 1.0)

    # Build rows
    for i, (alg_name, xbar, ubar) in enumerate(algos):
        positions = all_positions[i]
        velocities = all_vels[i]
        inputs = all_inputs[i]
        tvec = tvecs[i]

        # 1: block view
        ax0 = axes[i, 0]
        ax0.set_aspect('equal')
        ax0.set_xlim(pos_min - xpad, pos_max + xpad)
        ax0.set_ylim(-1.0, 1.0)
        ax0.get_yaxis().set_visible(False)
        ax0.set_xlabel('Position')
        ax0.set_title(DISPLAY_NAMES[i])

        # draw ground line (low zorder so other artists appear above it)
        ax0.plot([pos_min - xpad, pos_max + xpad], [ -0.2, -0.2], color='k', lw=2, zorder=1)

        # block as rectangle centered at position (above ground)
        block_y = -0.2
        block_h = 0.4
        block_center_y = block_y + block_h / 2.0
        block = Rectangle((positions[0] - block_width/2, block_y), block_width, block_h, facecolor='#2b7bba', edgecolor='k', zorder=5)
        ax0.add_patch(block)

        # trail (above ground, below block)
        trail_line, = ax0.plot([], [], color="#d84810", lw=2, alpha=0.9, zorder=4)

        # arrow placeholder - will be created in update; initialize to None
        arrow = None

        # 2: phase plot
        ax1 = axes[i, 1]
        ax1.set_xlabel('position')
        ax1.set_ylabel('velocity')
        # plot full curve
        phase_line, = ax1.plot(positions, velocities, color='tab:blue', lw=2)
        # red dot for current
        phase_dot, = ax1.plot([positions[0]], [velocities[0]], 'o', color='red')

        # plot start and goal markers if available
        # Use same Start/Goal styling as plot_compare_block_results in utils.py
        # smaller start/goal markers to match utility plot proportions
        if x0_data is not None:
            ax1.scatter(x0_data[0], x0_data[1], color="#1755b1", s=80, marker='o', zorder=6)
        if xg_data is not None:
            ax1.scatter(xg_data[0], xg_data[1], color="#48e750", s=100, marker='*', zorder=6)

        # set shared limits with margin
        ax1.set_xlim(phase_xlim)
        ax1.set_ylim(phase_ylim)

        # set overall column title for the top row
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
                    # place legend above the axes (just below the title)
                    ax1.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), bbox_transform=ax1.transAxes, ncol=len(legend_handles), fontsize='small')
                except Exception:
                    pass

        # 3: input vs time
        ax2 = axes[i, 2]
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Force')
        time_full = np.arange(len(inputs)) * dt
        input_line, = ax2.plot(time_full, inputs, color='tab:blue', lw=1.5)
        input_dot, = ax2.plot([0.0], [inputs[0]], 'o', color='red')
        ax2.set_xlim(input_xlim)
        ax2.set_ylim(input_ylim)

        # set overall column title for the top row
        if i == 0:
            ax2.set_title('Input vs Time', pad=24)

        # friction threshold line
        if SHOW_FRICTION_LINE:
            fr_line = ax2.axhline(FRICTION_THRESHOLD, color="#686565", linestyle='--', linewidth=3, alpha=0.5, zorder=3, label='Friction threshold')
            # show legend centered under the column title for the top row
            if i == 0:
                try:
                    # place friction legend above the axes (just below the title)
                    ax2.legend(handles=[fr_line], loc='lower center', bbox_to_anchor=(0.5, 1.0), bbox_transform=ax2.transAxes, fontsize='small')
                except Exception:
                    pass
            else:
                pass

        artists.append({
            'block': block,
            'trail': trail_line,
            'arrow': arrow,
            'block_center_y': block_center_y,
              'phase_dot': phase_dot,
              'input_dot': input_dot,
              'positions': positions,
              'velocities': velocities,
              'inputs': inputs,
              'tvec': tvec,
              'ax0': ax0,
              'ax1': ax1,
              'ax2': ax2,
          })

    plt.tight_layout()

    # Animation update
    def update(frame):
        # frame corresponds to time index into inputs (0..horizon-1)
        for i, art in enumerate(artists):
            pos = float(art['positions'][frame])
            vel = float(art['velocities'][frame])
            u = float(art['inputs'][frame])

            # update block rectangle position
            art['block'].set_x(pos - block_width/2)

            # update trail (past positions up to current frame) limited to TRAIL_LENGTH
            start_idx = max(0, frame - TRAIL_LENGTH + 1)
            xs = art['positions'][start_idx: frame + 1]
            ys = np.zeros_like(xs) + art.get('block_center_y', 0.0)
            art['trail'].set_data(xs, ys)

            # update arrow - compute end point
            dx = (u * arrow_scale)
            start = (pos, art.get('block_center_y', 0.0))
             # extend end by a small head length (in data units) so arrow tip is visible
            head_len = 0.02 * (pos_max - pos_min + 1e-6)
            end = (pos + dx + np.sign(dx) * head_len, art.get('block_center_y', 0.0))
            # remove previous arrow if present
            if art.get('arrow') is not None:
                try:
                    art['arrow'].remove()
                except Exception:
                    pass
                art['arrow'] = None
            # draw arrow if torque magnitude is non-negligible (use input u directly)
            if abs(u) > 1e-8:
                # annotate(xy=head, xytext=tail) -> head at 'end' (pos+dx), tail at 'start' (pos)
                art['arrow'] = art['ax0'].annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='red', lw=2), zorder=8)

            # update phase dot
            art['phase_dot'].set_data([pos], [vel])

            # update input dot
            art['input_dot'].set_data([art['tvec'][frame]], [u])

        return []

    interval = 1000 * dt
    anim = animation.FuncAnimation(fig, update, frames=horizon, interval=interval, blit=False)

    os.makedirs(out_dir, exist_ok=True)
    mp4_path = os.path.join(out_dir, 'block_compare_animation.mp4')
    gif_path = os.path.join(out_dir, 'block_compare_animation.gif')

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
    parser.add_argument('--npz', type=str, default='results/block_with_friction_solution.npz', help='Path to .npz results file')
    parser.add_argument('--out-dir', type=str, default=os.path.dirname(__file__), help='Directory to save animations')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for saved animations')
    parser.add_argument('--dt', type=float, default=None, help='Timestep (s). If not provided, the script will try to read from the npz file')

    args = parser.parse_args()
    build_animation(args.npz, args.out_dir, fps=args.fps, dt=args.dt)
