"""Plot benchmark results from bench_compile.py and bench_ddp.py JSON output.

Usage:
  python plot_benchmarks.py --compile results_compile.json
  python plot_benchmarks.py --ddp results_ddp.json
  python plot_benchmarks.py --compile results_compile.json --ddp results_ddp.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Jost", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

TEAL = "#004D40"
CORAL = "#FF655D"
YELLOW = "#F1DB4B"
LIGHT_TEAL = "#80CBC4"
WHITE = "#FFFFFF"


def plot_compile_results(data: dict, output_dir: Path) -> None:
    """Bar chart: eager vs compiled step times per config."""
    configs = data["configs"]
    meta = data["metadata"]

    names = [c["name"] for c in configs]
    eager_ms = [c["eager_median"] * 1000 for c in configs]
    compiled_ms = [c["compiled_median"] * 1000 for c in configs]
    speedups = [e / c for e, c in zip(eager_ms, compiled_ms)]
    compile_overhead = [c["compile_time"] for c in configs]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars_eager = ax.bar(
        x - width / 2, eager_ms, width,
        label="Eager", color=LIGHT_TEAL, edgecolor=TEAL, linewidth=0.8,
    )
    bars_compiled = ax.bar(
        x + width / 2, compiled_ms, width,
        label="Compiled", color=CORAL, edgecolor=TEAL, linewidth=0.8,
    )

    # Speedup annotations
    for i, (e, c, s) in enumerate(zip(eager_ms, compiled_ms, speedups)):
        ax.annotate(
            f"{s:.2f}x",
            xy=(i + width / 2, c),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=TEAL,
        )

    ax.set_ylabel("Median step time (ms)")
    ax.set_title(
        f"PET torch.compile speedup\n"
        f"{meta.get('gpu', 'CPU')}, PyTorch {meta.get('pytorch', '?')}, "
        f"batch_size={meta.get('batch_size', '?')}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(eager_ms) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Compile overhead text
    overhead_text = ", ".join(
        f"{n}: {t:.1f}s" for n, t in zip(names, compile_overhead)
    )
    ax.text(
        0.02, 0.98, f"Compile overhead: {overhead_text}",
        transform=ax.transAxes, fontsize=8, va="top", color="gray",
    )

    out_path = output_dir / "compile_speedup.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_ddp_results(data: dict, output_dir: Path) -> None:
    """Grouped bar chart: 4 configs (eager/compiled x single/DDP)."""
    meta = data["metadata"]

    configs = {}
    for key in ["eager_single", "compiled_single", "eager_ddp", "compiled_ddp"]:
        if key in data:
            configs[key] = data[key]

    labels = []
    medians = []
    colors = []
    edge_colors = []

    config_display = [
        ("eager_single", "Eager\n1 GPU", LIGHT_TEAL),
        ("compiled_single", "Compiled\n1 GPU", CORAL),
        ("eager_ddp", "Eager\nDDP (2)", YELLOW),
        ("compiled_ddp", "Compiled\nDDP (2)", "#E53935"),
    ]

    for key, label, color in config_display:
        if key in configs:
            labels.append(label)
            medians.append(configs[key]["median_ms"])
            colors.append(color)
            edge_colors.append(TEAL)

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        x, medians, 0.6,
        color=colors, edgecolor=edge_colors, linewidth=0.8,
    )

    # Value labels on bars
    for bar, val in zip(bars, medians):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEAL,
        )

    # Speedup annotations
    if "eager_single" in configs and "compiled_single" in configs:
        es = configs["eager_single"]["median_ms"]
        cs = configs["compiled_single"]["median_ms"]
        ax.annotate(
            f"compile: {es/cs:.2f}x",
            xy=(0.5, max(es, cs) * 1.15),
            ha="center", fontsize=9, color=TEAL,
        )
    if "eager_ddp" in configs and "compiled_ddp" in configs:
        ed = configs["eager_ddp"]["median_ms"]
        cd = configs["compiled_ddp"]["median_ms"]
        ax.annotate(
            f"compile: {ed/cd:.2f}x",
            xy=(2.5, max(ed, cd) * 1.15),
            ha="center", fontsize=9, color=TEAL,
        )

    ax.set_ylabel("Median step time (ms)")
    ax.set_title(
        f"PET training: compile + DDP (single GPU, gloo backend)\n"
        f"{meta.get('gpu', 'CPU')}, PyTorch {meta.get('pytorch', '?')}, "
        f"batch_size={meta.get('batch_size', '?')}",
    )
    ax.text(
        0.5, -0.12,
        "Note: DDP on 1 GPU with gloo is a correctness test, not a performance target.\n"
        "Multi-GPU NCCL would show near-linear scaling.",
        transform=ax.transAxes, fontsize=8, ha="center", color="gray",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(medians) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = output_dir / "ddp_compile_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot PET benchmark results")
    parser.add_argument("--compile", type=str, help="bench_compile.py JSON output")
    parser.add_argument("--ddp", type=str, help="bench_ddp.py JSON output")
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory for output plots (default: .)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compile:
        with open(args.compile) as f:
            data = json.load(f)
        plot_compile_results(data, output_dir)

    if args.ddp:
        with open(args.ddp) as f:
            data = json.load(f)
        plot_ddp_results(data, output_dir)

    if not args.compile and not args.ddp:
        print("Provide --compile and/or --ddp JSON files")


if __name__ == "__main__":
    main()
