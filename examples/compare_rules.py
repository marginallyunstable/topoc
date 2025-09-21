#!/usr/bin/env python3
"""
Plot Cost vs Iteration for three experiments (Rule1, Rule2, Rule3).
Matches style used by plot_compare_cartpole_results in topoc/utils.py.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
from cycler import cycler
import seaborn as sns
import os

# --- Paths to the .npy cost files ---
BASE = "/workspace/topoc/results"
FILES = [
    ("PDDP Rule 1", os.path.join(BASE, "Rule1.npy")),
    ("PDDP Rule 2", os.path.join(BASE, "Rule2.npy")),
    ("PDDP Rule 3", os.path.join(BASE, "Rule3.npy")),
    ("SCS-DDP", os.path.join(BASE, "SCS-DDP.npy")),
    ("CS-DDP", os.path.join(BASE, "CS-DDP.npy")),
]

red = 99

# --- Appearance (copied from plot_compare_cartpole_results) ---
sns.set_context("notebook", font_scale=1.0)
n_series = max(1, len(FILES))
palette = sns.color_palette("flare", n_colors=max(4, n_series))
linestyles = [(0, (0.5, 0.5)), (0, (2, 1)), (0, (4, 1, 2, 1)), "-"]

mpl.rcParams.update({
    "mathtext.fontset": "stixsans",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "text.usetex": False,
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.figsize": (5, 5),
    "lines.linewidth": 2.0,
    "axes.labelsize": 24,
    "axes.titlesize": 26,
    "legend.fontsize": 15,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})
mpl.rcParams["axes.prop_cycle"] = cycler("color", palette) + cycler(
    "linestyle", [linestyles[i % len(linestyles)] for i in range(len(palette))]
)

# Marker / legend cycler to match subplot style
mfc = [(*c, 0.22) for c in palette]
mec = [c for c in palette]
markers = ["o"] * n_series
ms = [6] * n_series
mew = [1.2] * n_series

prop = (
    cycler("color", palette[:n_series])
    + cycler("linestyle", [linestyles[i % len(linestyles)] for i in range(n_series)])
    + cycler("marker", markers[:n_series])
    + cycler("markerfacecolor", mfc[:n_series])
    + cycler("markeredgecolor", mec[:n_series])
    + cycler("markersize", ms[:n_series])
    + cycler("markeredgewidth", mew[:n_series])
)

# --- Load data ---
series = []
labels = []
for label, path in FILES:
    if not os.path.isfile(path):
        print(f"Warning: file not found: {path}  (skipping)")
        continue
    try:
        V = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        continue
    # Ensure 1D numeric array
    V = np.asarray(V).ravel()
    if V.size == 0:
        print(f"Warning: empty array in {path} (skipping)")
        continue
    series.append(V.astype(float))
    labels.append(label)

if len(series) == 0:
    raise SystemExit("No valid series loaded. Exiting.")

# --- Compute offset to keep all values > 0 for log scale (same logic as function) ---
global_min = np.min([np.nanmin(V) for V in series])
offset = 0.0
if global_min <= 0:
    offset = 1e-12 - float(global_min)

# --- Metric: iterations to reach 90% reduction of initial cost ---
print("Iterations to reach 90% reduction (cost <= 10% of initial):")
for lab, V in zip(labels, series):
    Varr = np.asarray(V).ravel()
    if Varr.size == 0:
        print(f"  {lab}: empty series")
        continue
    V0 = float(Varr[0])
    # handle non-positive or NaN initial cost
    if np.isnan(V0) or V0 <= 0:
        print(f"  {lab}: initial cost = {V0}, metric undefined")
        continue
    thresh = V0 * (1-red/100)
    idx = np.where(Varr <= thresh)[0]
    if idx.size > 0:
        it = int(idx[0]) + 1  # 1-based iteration count
        print(f"  {lab}: {it} iterations (first index {idx[0]} -> cost={Varr[idx[0]]:.6g})")
    else:
        print(f"  {lab}: NOT reached within {Varr.size} iterations (last cost={Varr[-1]:.6g})")

# --- Plot ---
fig, ax = plt.subplots(figsize=(5, 9))
# We'll pick styles per-series so we can make SCS-DDP and CS-DDP faint while
# keeping PDDP Rule 1-3 pronounced. Colors are taken from `palette` so they
# match the same generator used in plot_compare_cartpole_results.

ax.xaxis.set_major_locator(LogLocator(numticks=3))

for idx, (lab, V) in enumerate(zip(labels, series)):
    color = palette[idx % len(palette)]
    ls = linestyles[idx % len(linestyles)]
    Varr = V + offset
    it = np.arange(1, Varr.shape[0] + 1)

    # Make the SCS-DDP and CS-DDP lines faint; others pronounced
    if lab in ("SCS-DDP", "CS-DDP"):
        alpha = 0.6
        lw = 3.0
        marker = None
        mface = None
        medge = None
        msize = None
        medw = None
    else:
        alpha = 1.0
        lw = 3.0
        marker = markers[idx % len(markers)]
        mface = mfc[idx % len(mfc)]
        medge = mec[idx % len(mec)]
        msize = ms[idx % len(ms)]
        medw = mew[idx % len(mew)]

    ax.plot(
        it,
        Varr,
        color=color,
        linestyle=ls,
        lw=lw,
        alpha=alpha,
        marker=marker,
        markersize=msize,
        markerfacecolor=mface,
        markeredgecolor=medge,
        markeredgewidth=medw,
    )

# Draw horizontal lines indicating the 90% reduction threshold (10% of initial)
for idx, (lab, V) in enumerate(zip(labels, series)):
    Varr = np.asarray(V).ravel()
    if Varr.size == 0:
        continue
    V0 = float(Varr[0])
    if np.isnan(V0) or V0 <= 0:
        continue
    thresh = V0 * (1-red/100)
    color = palette[idx % len(palette)]
    alpha_line = 0.28 if lab in ("SCS-DDP", "CS-DDP") else 0.8
    # add offset so the horizontal line matches plotted (offset applied to series)
    # draw threshold lines in a neutral dark-gray so legend entry matches
    ax.axhline(thresh + offset, color='0.2', linestyle='--', linewidth=1.5, alpha=alpha_line, zorder=0)

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"$\mathsf{Iteration}$")
ax.set_ylabel(r"$\mathsf{Cost\ (V)}$")
ax.set_title(r"$\mathsf{Cost\ vs\ Iteration}$")
ax.grid(which="both", alpha=0.25)

# --- Legend (matching style) ---
handles = []
for idx, lab in enumerate(labels):
    color = palette[idx % len(palette)]
    ls = linestyles[idx % len(linestyles)]
    if lab in ("SCS-DDP", "CS-DDP"):
        alpha = 0.6
        lw = 3.0
        marker = None
        mface = 'none'
    else:
        alpha = 1.0
        lw = 4.0
        marker = markers[idx % len(markers)]
        mface = mfc[idx % len(mfc)]

    h = Line2D(
        [0], [0],
        color=color,
        lw=lw,
        linestyle=ls,
        marker=marker,
        markerfacecolor=mface,
        markeredgecolor=mec[idx % len(mec)],
        markersize=ms[idx % len(ms)],
        markeredgewidth=mew[idx % len(mew)],
        alpha=alpha,
    )
    handles.append(h)

# Add a legend entry for the reduction threshold lines
threshold_label = f"{red}% cost reduction"
threshold_handle = Line2D([0], [0], color='0.2', lw=1.5, linestyle='--', alpha=0.9)
handles.append(threshold_handle)

legend_labels = list(labels) + [threshold_label]

leg = ax.legend(handles, legend_labels, loc="best", frameon=True, fancybox=True, edgecolor="0.2",
                handlelength=1.5, columnspacing=0.3, prop={"size": mpl.rcParams.get("legend.fontsize", 12)})
if leg is not None:
    for text in leg.get_texts():
        text.set_fontfamily("DejaVu Sans")

plt.tight_layout()
plt.show()

out_dir = BASE
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "compare_rules.svg")
try:
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    print(f"Saved figure to {out_path}")
except Exception as e:
    print(f"Failed to save figure: {e}")

