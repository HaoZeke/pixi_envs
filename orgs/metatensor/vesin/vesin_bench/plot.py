"""Plot neighbor list benchmark results comparing vesin versions.

Reads one or more results_*.json files (produced by bench_nl.py) and generates:
  1. scaling.png          -- wall time vs N atoms (log-log), all backends overlaid
  2. speedup_cpu.png      -- CPU speedup of feature branch over baseline
  3. gpu_scaling.png      -- GPU wall time comparison (if CUDA data present)
  4. verlet_saving.png    -- Verlet rebuild vs reuse vs stateless
  5. md_trajectory.png    -- per-step cost: stateless vs Verlet over MD trajectory
  6. md_speedup.png       -- MD trajectory speedup (Verlet / stateless)

Usage:
    python plot.py                           # auto-finds results*.json
    python plot.py results_baseline.json results_cluster-pair.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# Distinct colors per tag
TAG_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
]


TAG_RENAMES = {
    "baseline": "main",
    "gpu-cell-list": "cell-list-only",
    "gpu-optimized": "current",
}


def load_results(paths):
    """Load result files, return list of (tag, data) tuples."""
    datasets = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        tag = data.get("tag") or p.stem.replace("results_", "").replace("results", "default")
        tag = TAG_RENAMES.get(tag, tag)
        datasets.append((tag, data))
    return datasets


def plot_cpu_scaling(datasets, outdir):
    """Wall time vs N atoms for CPU, comparing all tags."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (tag, data) in enumerate(datasets):
        systems = data["systems"]
        n_atoms = np.array([s["n_atoms"] for s in systems])
        if not all("vesin_cpu_ms" in s for s in systems):
            continue
        vals = np.array([s["vesin_cpu_ms"] for s in systems])
        errs = np.array([s.get("vesin_cpu_std_ms", 0) for s in systems])
        color = TAG_COLORS[i % len(TAG_COLORS)]
        ax.errorbar(n_atoms, vals, yerr=errs, fmt="o-",
                    color=color, label=f"vesin CPU ({tag})", capsize=3)

    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Wall time (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.set_title("Neighbor list construction: CPU scaling")
    fig.tight_layout()

    fig.savefig(outdir / "scaling.png")
    plt.close(fig)
    print(f"  -> scaling.png")


def plot_cpu_speedup(datasets, outdir):
    """Speedup of later datasets relative to the first (baseline)."""
    if len(datasets) < 2:
        print("  (skipping speedup plot, need at least 2 result files)")
        return

    baseline_tag, baseline_data = datasets[0]
    baseline_sys = baseline_data["systems"]
    baseline_n = {s["n_atoms"]: s["vesin_cpu_ms"] for s in baseline_sys
                  if "vesin_cpu_ms" in s}

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (tag, data) in enumerate(datasets[1:], 1):
        systems = data["systems"]
        n_common = []
        speedups = []
        for s in systems:
            n = s["n_atoms"]
            if n in baseline_n and "vesin_cpu_ms" in s:
                n_common.append(n)
                speedups.append(baseline_n[n] / s["vesin_cpu_ms"])

        if n_common:
            color = TAG_COLORS[i % len(TAG_COLORS)]
            ax.plot(n_common, speedups, "s-", color=color,
                    label=f"{tag} vs {baseline_tag}")

    ax.axhline(1.0, ls="--", color="gray", alpha=0.5, label="parity")
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel(f"Speedup over {baseline_tag}")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.set_title("CPU speedup: feature branch vs baseline")
    fig.tight_layout()

    fig.savefig(outdir / "speedup_cpu.png")
    plt.close(fig)
    print(f"  -> speedup_cpu.png")


def plot_gpu_scaling(datasets, outdir):
    """GPU wall time comparison across tags, with CPU reference overlay."""
    has_gpu = any(
        "vesin_gpu_ms" in s
        for _, data in datasets
        for s in data["systems"]
    )
    if not has_gpu:
        print("  (skipping GPU plot, no GPU data)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    gpu_name = None

    # Plot CPU reference line (from the dataset that has GPU data, for fair comparison)
    cpu_plotted = False
    for i, (tag, data) in enumerate(datasets):
        systems = data["systems"]
        gpu_name = gpu_name or data.get("gpu_name")

        # GPU line
        mask = [j for j, s in enumerate(systems) if "vesin_gpu_ms" in s]
        if mask:
            n_atoms = np.array([systems[j]["n_atoms"] for j in mask])
            vals = np.array([systems[j]["vesin_gpu_ms"] for j in mask])
            errs = np.array([systems[j].get("vesin_gpu_std_ms", 0) for j in mask])
            color = TAG_COLORS[i % len(TAG_COLORS)]
            ax.errorbar(n_atoms, vals, yerr=errs, fmt="s-",
                        color=color, label=f"GPU ({tag})", capsize=3)

            # Also plot CPU from the same dataset for comparison
            if not cpu_plotted:
                cpu_mask = [j for j, s in enumerate(systems) if "vesin_cpu_ms" in s]
                if cpu_mask:
                    cpu_n = np.array([systems[j]["n_atoms"] for j in cpu_mask])
                    cpu_v = np.array([systems[j]["vesin_cpu_ms"] for j in cpu_mask])
                    ax.plot(cpu_n, cpu_v, "o--", color="gray", alpha=0.7,
                            label=f"CPU ({tag})")
                    cpu_plotted = True

    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Wall time (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    title = "Neighbor list: GPU vs CPU"
    if gpu_name:
        title += f"\n({gpu_name})"
    ax.set_title(title)
    fig.tight_layout()

    fig.savefig(outdir / "gpu_scaling.png")
    plt.close(fig)
    print(f"  -> gpu_scaling.png")


def plot_verlet(datasets, outdir):
    """Verlet rebuild vs reuse vs stateless cost."""
    # Find the dataset that has Verlet data
    verlet_data = None
    verlet_tag = None
    for tag, data in datasets:
        if any("verlet_rebuild_ms" in s for s in data["systems"]):
            verlet_data = data
            verlet_tag = tag
            break

    if verlet_data is None:
        print("  (skipping Verlet plot, no Verlet data)")
        return

    systems = verlet_data["systems"]
    n_atoms = np.array([s["n_atoms"] for s in systems])

    fig, ax = plt.subplots(figsize=(8, 5))

    cpu = np.array([s.get("vesin_cpu_ms", np.nan) for s in systems])
    rebuild = np.array([s.get("verlet_rebuild_ms", np.nan) for s in systems])
    reuse = np.array([s.get("verlet_reuse_ms", np.nan) for s in systems])

    m_c = ~np.isnan(cpu)
    m_r = ~np.isnan(rebuild)
    m_u = ~np.isnan(reuse)

    if m_c.any():
        ax.plot(n_atoms[m_c], cpu[m_c], "o-", color="#1f77b4",
                label="stateless (cutoff only)")
    if m_r.any():
        ax.plot(n_atoms[m_r], rebuild[m_r], "D-", color="#d62728",
                label="Verlet rebuild (cutoff+skin)")
    if m_u.any():
        ax.plot(n_atoms[m_u], reuse[m_u], "v-", color="#9467bd",
                label="Verlet reuse (vectors only)")

    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Wall time (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.set_title(f"Verlet caching cost breakdown ({verlet_tag})")
    fig.tight_layout()

    fig.savefig(outdir / "verlet_saving.png")
    plt.close(fig)
    print(f"  -> verlet_saving.png")


def plot_combined(datasets, outdir):
    """Combined CPU + GPU comparison across all backends and tags."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    gpu_name = None
    for i, (tag, data) in enumerate(datasets):
        systems = data["systems"]
        gpu_name = gpu_name or data.get("gpu_name")
        n_atoms = np.array([s["n_atoms"] for s in systems])
        color = TAG_COLORS[i % len(TAG_COLORS)]

        # CPU panel
        if all("vesin_cpu_ms" in s for s in systems):
            cpu_ms = np.array([s["vesin_cpu_ms"] for s in systems])
            ax1.plot(n_atoms, cpu_ms, "o-", color=color, label=f"CPU ({tag})")

        # GPU panel
        mask = [j for j, s in enumerate(systems) if "vesin_gpu_ms" in s]
        if mask:
            gpu_n = np.array([systems[j]["n_atoms"] for j in mask])
            gpu_ms = np.array([systems[j]["vesin_gpu_ms"] for j in mask])
            ax2.plot(gpu_n, gpu_ms, "s-", color=color, label=f"GPU ({tag})")

    for ax, title in [(ax1, "CPU"), (ax2, "GPU")]:
        ax.set_xlabel("Number of atoms")
        ax.set_ylabel("Wall time (ms)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.set_title(title)

    suptitle = "vesin neighbor list: baseline vs feature branch"
    if gpu_name:
        suptitle += f"  ({gpu_name})"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()

    fig.savefig(outdir / "combined.png")
    plt.close(fig)
    print(f"  -> combined.png")


def plot_md_trajectory(datasets, outdir):
    """Per-step cost comparison: stateless vs Verlet over an MD trajectory."""
    has_md = any(
        "md_stateless_per_step_ms" in s or "md_verlet_per_step_ms" in s
        for _, data in datasets
        for s in data["systems"]
    )
    if not has_md:
        print("  (skipping MD trajectory plot, no MD data)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    md_steps = None

    for i, (tag, data) in enumerate(datasets):
        systems = data["systems"]
        md_steps = md_steps or data.get("md_steps", "?")
        n_atoms = np.array([s["n_atoms"] for s in systems])
        color = TAG_COLORS[i % len(TAG_COLORS)]

        # Stateless per-step
        stat_mask = [j for j, s in enumerate(systems)
                     if "md_stateless_per_step_ms" in s]
        if stat_mask:
            n_s = np.array([systems[j]["n_atoms"] for j in stat_mask])
            v_s = np.array([systems[j]["md_stateless_per_step_ms"]
                            for j in stat_mask])
            ax.plot(n_s, v_s, "o-", color=color,
                    label=f"stateless ({tag})")

        # Verlet per-step
        vrl_mask = [j for j, s in enumerate(systems)
                    if "md_verlet_per_step_ms" in s]
        if vrl_mask:
            n_v = np.array([systems[j]["n_atoms"] for j in vrl_mask])
            v_v = np.array([systems[j]["md_verlet_per_step_ms"]
                            for j in vrl_mask])
            ax.plot(n_v, v_v, "v--", color=color, alpha=0.8,
                    label=f"Verlet ({tag})")

    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Per-step wall time (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.set_title(f"MD trajectory: per-step NL cost ({md_steps} steps)")
    fig.tight_layout()

    fig.savefig(outdir / "md_trajectory.png")
    plt.close(fig)
    print(f"  -> md_trajectory.png")


def plot_md_speedup(datasets, outdir):
    """MD trajectory speedup: Verlet amortized cost vs stateless."""
    has_both = any(
        "md_stateless_per_step_ms" in s and "md_verlet_per_step_ms" in s
        for _, data in datasets
        for s in data["systems"]
    )
    if not has_both:
        print("  (skipping MD speedup plot, need both stateless and Verlet MD data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, (tag, data) in enumerate(datasets):
        systems = data["systems"]
        md_steps = data.get("md_steps", "?")
        color = TAG_COLORS[i % len(TAG_COLORS)]

        both_mask = [j for j, s in enumerate(systems)
                     if "md_stateless_per_step_ms" in s
                     and "md_verlet_per_step_ms" in s]
        if not both_mask:
            continue

        n_atoms = np.array([systems[j]["n_atoms"] for j in both_mask])
        speedups = np.array([
            systems[j]["md_stateless_per_step_ms"] /
            systems[j]["md_verlet_per_step_ms"]
            for j in both_mask
        ])
        rebuilds = np.array([systems[j].get("md_verlet_rebuilds", 0)
                             for j in both_mask])
        reuses = np.array([systems[j].get("md_verlet_reuses", 0)
                           for j in both_mask])
        rebuild_frac = rebuilds / (rebuilds + reuses) * 100

        # Speedup panel
        ax1.plot(n_atoms, speedups, "s-", color=color,
                 label=f"{tag} ({md_steps} steps)")
        ax1.axhline(1.0, ls="--", color="gray", alpha=0.5)

        # Rebuild fraction panel
        ax2.plot(n_atoms, rebuild_frac, "D-", color=color,
                 label=f"{tag}")

    ax1.set_xlabel("Number of atoms")
    ax1.set_ylabel("Speedup (Verlet / stateless)")
    ax1.set_xscale("log")
    ax1.legend()
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.set_title("MD trajectory: Verlet speedup")

    ax2.set_xlabel("Number of atoms")
    ax2.set_ylabel("Rebuild fraction (%)")
    ax2.set_xscale("log")
    ax2.legend()
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.set_title("Verlet rebuild frequency")
    fig.tight_layout()

    fig.savefig(outdir / "md_speedup.png")
    plt.close(fig)
    print(f"  -> md_speedup.png")


def main():
    outdir = Path(__file__).parent

    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = sorted(outdir.glob("results*.json"))

    if not paths:
        print("No results files found. Run bench_nl.py first.")
        sys.exit(1)

    datasets = load_results(paths)
    print(f"Loaded {len(datasets)} result sets: {[t for t, _ in datasets]}")

    plot_cpu_scaling(datasets, outdir)
    plot_cpu_speedup(datasets, outdir)
    plot_gpu_scaling(datasets, outdir)
    plot_verlet(datasets, outdir)
    plot_md_trajectory(datasets, outdir)
    plot_md_speedup(datasets, outdir)
    if len(datasets) >= 2:
        plot_combined(datasets, outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
