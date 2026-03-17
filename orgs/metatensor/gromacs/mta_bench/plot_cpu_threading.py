#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Plot CPU threading benchmark results for the fix/cpu-threading-perf PR."""

import matplotlib.pyplot as plt
import numpy as np

# Ruhi colorscheme
TEAL = "#004D40"
CORAL = "#FF655D"
YELLOW = "#F1DB4B"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F5F5F5"

plt.rcParams.update({
    "font.family": "Jost",
    "font.size": 12,
    "axes.edgecolor": TEAL,
    "axes.labelcolor": TEAL,
    "xtick.color": TEAL,
    "ytick.color": TEAL,
    "text.color": TEAL,
    "figure.facecolor": WHITE,
    "axes.facecolor": LIGHT_GRAY,
})

# Benchmark data: 648 atoms (216 water), PET-MAD v1.0.2, CPU device
# cosmopc5 (Threadripper PRO 3955WX, 32 cores), real MPI build
# 30 MD steps, steady-state average (last 10 steps)

labels = [
    "Baseline\n4 rank x 4t",
    "Fix\n4 rank x 4t",
    "Baseline\n1 rank x 4t",
    "Fix\n1 rank x 4t",
    "LAMMPS\n1 rank x 1t",
]
# ms/step averages from benchmarks
values = [7400, 4190, 3030, 2930, 6900]
colors = [CORAL, TEAL, CORAL, TEAL, YELLOW]

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.barh(range(len(labels)), values, color=colors, edgecolor=TEAL, linewidth=0.5)

# Add value labels
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 80, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f} ms", va="center", fontsize=11, color=TEAL)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("calculateForces (ms/step, lower is better)", fontsize=12)
ax.set_title("CPU Inference: Threading Fix Impact\n648 atoms, PET-MAD, cosmopc5",
             fontsize=13, fontweight="bold", color=TEAL)

# Add speedup annotation for the 4-rank case
ax.annotate("1.77x faster", xy=(4190, 1), xytext=(5500, 1.5),
            fontsize=12, fontweight="bold", color=TEAL,
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))

ax.set_xlim(0, max(values) * 1.2)
ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("cpu_threading_benchmark.png", dpi=200, bbox_inches="tight")
plt.savefig("cpu_threading_benchmark.svg", bbox_inches="tight")
print("Saved cpu_threading_benchmark.png and .svg")
