import matplotlib.pyplot as plt
import numpy as np

# Data from bob_results_0..4
# Each tuple: (n_weights, ref_time_us, blas_time_us, [(density, attn_time_us), ...])
runs = [
    (6_291_456, 150_467, 20_099, [
        (0.0001, 50), (0.0002, 100), (0.0003, 215), (0.0006, 490),
        (0.0010, 874), (0.0018, 1769), (0.0032, 3709), (0.0056, 7874),
        (0.0100, 17906), (0.0178, 46280), (0.0316, 100203), (0.0562, 207270),
        (0.1000, 424316), (0.1778, 768154), (0.3162, 1355115),
        (0.5623, 2720411), (1.0000, 6234552),
    ]),
    (6_291_456, 142_731, 4_961, [
        (0.0001, 66), (0.0002, 111), (0.0003, 208), (0.0006, 421),
        (0.0010, 858), (0.0018, 1789), (0.0032, 3807), (0.0056, 7988),
        (0.0100, 15909), (0.0178, 44031), (0.0316, 97169), (0.0562, 195958),
    ]),
    (8_388_608, 257_030, 6_988, [
        (0.0001, 85), (0.0002, 151), (0.0003, 316), (0.0006, 652),
        (0.0010, 1294), (0.0018, 2732), (0.0032, 6151), (0.0056, 13555),
        (0.0100, 30897), (0.0178, 74214), (0.0316, 161262), (0.0562, 331050),
        (0.1000, 685938), (0.1778, 1222346), (0.3162, 2149019),
        (0.5623, 4664980), (1.0000, 11020754),
    ]),
    (10_485_760, 403_082, 71_937, [
        (0.0001, 147), (0.0002, 305), (0.0003, 625), (0.0006, 1202),
        (0.0010, 2297), (0.0018, 4617), (0.0032, 10025), (0.0056, 21685),
        (0.0100, 51581), (0.0178, 114948), (0.0316, 247047), (0.0562, 505262),
        (0.1000, 1076446), (0.1778, 1888211), (0.3162, 3352297),
        (0.5623, 7276583), (1.0000, 17245671),
    ]),
    (13_107_200, 639_979, 138_200, [
        (0.0001, 234), (0.0002, 443), (0.0003, 867), (0.0006, 1689),
        (0.0010, 3277), (0.0018, 6974), (0.0032, 15884), (0.0056, 35211),
        (0.0100, 81843), (0.0178, 177855), (0.0316, 363000), (0.0562, 814605),
        (0.1000, 1647120), (0.1778, 2997694), (0.3162, 5311837),
        (0.5623, 11512464), (1.0000, 26790027),
    ]),
]


def interpolate_crossover(points, threshold):
    """Find density where attn_time crosses threshold via linear interpolation."""
    for i in range(len(points) - 1):
        d0, t0 = points[i]
        d1, t1 = points[i + 1]
        if t0 <= threshold <= t1:
            frac = (threshold - t0) / (t1 - t0)
            return d0 + frac * (d1 - d0)
    return None


n_weights_list = []
blas_crossovers = []
ref_crossovers = []

for nw, ref_t, blas_t, pts in runs:
    cx_ref = interpolate_crossover(pts, ref_t)
    cx_blas = interpolate_crossover(pts, blas_t)
    if cx_blas is not None:
        n_weights_list.append(nw)
        blas_crossovers.append(cx_blas * 100)  # to percent
        ref_crossovers.append((cx_ref or 0) * 100)

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(
    [nw / 1e6 for nw in n_weights_list],
    blas_crossovers,
    "o-",
    color="steelblue",
    linewidth=2,
    markersize=8,
    label="vs BLAS dense",
)
ax.plot(
    [nw / 1e6 for nw in n_weights_list],
    ref_crossovers,
    "s-",
    color="coral",
    linewidth=2,
    markersize=8,
    label="vs naive dense",
)

ax.set_xlabel("Matrix size (millions of weights)", fontsize=12)
ax.set_ylabel("Crossover density (%)", fontsize=12)
ax.set_title("CSR vs Dense: Break-Even Density Threshold", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Annotate each point
for i, nw in enumerate(n_weights_list):
    ax.annotate(
        f"{blas_crossovers[i]:.2f}%",
        (nw / 1e6, blas_crossovers[i]),
        textcoords="offset points",
        xytext=(8, -12),
        fontsize=9,
        color="steelblue",
    )
    ax.annotate(
        f"{ref_crossovers[i]:.2f}%",
        (nw / 1e6, ref_crossovers[i]),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=9,
        color="coral",
    )

# Shade the "CSR always wins" region
ax.axhspan(0, min(blas_crossovers) * 0.95, alpha=0.08, color="green")
ax.text(
    7,
    min(blas_crossovers) * 0.4,
    "CSR always wins",
    fontsize=10,
    color="green",
    ha="center",
    style="italic",
)

# Shade the "dense always wins" region
ax.axhspan(max(ref_crossovers) * 1.02, ax.get_ylim()[1] * 1.1, alpha=0.08, color="red")

ax.set_ylim(0, max(ref_crossovers) * 1.3)

plt.tight_layout()
plt.savefig("crossover_threshold.png", dpi=150)
print("Saved crossover_threshold.png")
