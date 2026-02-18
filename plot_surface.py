#!/usr/bin/env python3
"""3D surface plots: CSR vs MAGNUS sequential, CSR‖ vs MAGNUS‖.
Usage: python3 plot_surface.py [bench.csv]
  - Reads from stdin or file argument
  - Extracts benchmark CSV lines automatically from test output
"""

import sys
import csv
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def parse_csv(text):
    lines = []
    header_line = None
    for line in text.strip().splitlines():
        if line.startswith("side,nodes,"):
            header_line = line
            lines = []
        elif header_line and "," in line:
            parts = line.split(",")
            try:
                int(parts[0])
                lines.append(line)
            except ValueError:
                continue
    if not header_line or not lines:
        print("No benchmark CSV data found.", file=sys.stderr)
        sys.exit(1)
    reader = csv.DictReader(io.StringIO(header_line + "\n" + "\n".join(lines)))
    return list(reader)

LOG_Z = True  # Set to False for linear Z axis

def build_grid(rows, time_col):
    """Build meshgrid from rows. Returns log10(X), log10(Y), Z (log10 or linear), plus raw values for tick labels."""
    nodes_set = sorted(set(int(r["nodes"]) for r in rows))
    epn_set = sorted(set(float(r["e_per_n"]) for r in rows))

    X = np.array(nodes_set, dtype=float)
    Y = np.array(epn_set, dtype=float)
    Z = np.full((len(Y), len(X)), np.nan)

    nodes_idx = {n: i for i, n in enumerate(nodes_set)}
    epn_idx = {e: i for i, e in enumerate(epn_set)}

    for r in rows:
        xi = nodes_idx[int(r["nodes"])]
        yi = epn_idx[float(r["e_per_n"])]
        Z[yi, xi] = int(r[time_col]) / 1000.0  # µs → ms

    logX = np.log10(X)
    logY = np.log10(Y)
    logXm, logYm = np.meshgrid(logX, logY)
    Zout = np.log10(np.where(Z > 0, Z, np.nan)) if LOG_Z else Z

    return logXm, logYm, Zout, X, Y

def plot_pair(ax, rows, col_a, col_b, label_a, label_b, title):
    lXa, lYa, lZa, rawX, rawY = build_grid(rows, col_a)
    lXb, lYb, lZb, _, _ = build_grid(rows, col_b)

    ax.plot_surface(lXa, lYa, lZa, alpha=0.7, cmap=cm.Blues, edgecolor='steelblue', linewidth=0.5)
    ax.plot_surface(lXb, lYb, lZb, alpha=0.7, cmap=cm.Oranges, edgecolor='darkorange', linewidth=0.5)

    # Log-scale tick labels for X (nodes)
    ax.set_xticks(np.log10(rawX))
    ax.set_xticklabels([f"{int(v):,}" for v in rawX], fontsize=8)

    # Log-scale tick labels for Y (edges/node)
    ax.set_yticks(np.log10(rawY))
    ax.set_yticklabels([f"{v:g}" for v in rawY], fontsize=8)

    # Z-axis tick labels
    if LOG_Z:
        all_z = np.concatenate([lZa.ravel(), lZb.ravel()])
        all_z = all_z[np.isfinite(all_z)]
        zmin, zmax = np.floor(all_z.min()), np.ceil(all_z.max())
        zticks = np.arange(zmin, zmax + 1)
        ax.set_zticks(zticks)
        ax.set_zticklabels([f"{10**z:g}" for z in zticks], fontsize=8)

    ax.set_xlabel('Nodes', fontsize=10, labelpad=8)
    ax.set_ylabel('Edges/node', fontsize=10, labelpad=8)
    ax.set_zlabel('Time (ms)', fontsize=10, labelpad=8)
    ax.set_title(title, fontsize=12, pad=12)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='steelblue', alpha=0.7, label=label_a),
        Patch(facecolor='darkorange', alpha=0.7, label=label_b),
    ], loc='upper left', fontsize=9)

def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    rows = parse_csv(text)

    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    plot_pair(ax1, rows, "csr_us", "magnus_seq_us",
              "CSR (seq)", "MAGNUS (seq)", "Sequential: CSR vs MAGNUS")

    ax2 = fig.add_subplot(122, projection='3d')
    plot_pair(ax2, rows, "csr_par_us", "magnus_par_us",
              "CSR (par)", "MAGNUS (par)", "Parallel: CSR‖ vs MAGNUS‖")

    fig.suptitle("SpGEMM: CSR vs MAGNUS — Time by Nodes × Connectivity", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("surface_csr_vs_magnus.png", dpi=150, bbox_inches='tight')
    print("Saved surface_csr_vs_magnus.png")
    plt.show()

if __name__ == "__main__":
    main()
