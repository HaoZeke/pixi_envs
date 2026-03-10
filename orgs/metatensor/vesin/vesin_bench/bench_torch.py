"""Benchmark vesin-torch: GPU stateless vs CPU stateless vs CPU Verlet.

Measures wall-clock times for MD-like trajectories (small perturbations
per step) across all three modes to show where Verlet caching wins and
where GPU acceleration wins.
"""

import json
import time

import torch
from vesin.torch import NeighborList, VerletNeighborList


def make_fcc_system(n_cells, a=4.0, device="cpu"):
    """Generate FCC lattice with n_cells^3 unit cells."""
    basis = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ], dtype=torch.float64)

    positions = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                offset = torch.tensor([ix, iy, iz], dtype=torch.float64)
                for b in basis:
                    positions.append((b + offset) * a)

    points = torch.stack(positions).to(device)
    box = (torch.eye(3, dtype=torch.float64) * a * n_cells).to(device)
    return points, box


def perturb(points, amplitude=0.02):
    """Small random perturbation (MD-like)."""
    return points + amplitude * torch.randn_like(points)


def bench_stateless(points, box, cutoff, n_steps, warmup=3):
    """Time stateless neighbor list over n_steps."""
    # GPU doesn't support sorted output
    use_sorted = not points.is_cuda
    nl = NeighborList(cutoff=cutoff, full_list=True, sorted=use_sorted)
    current = points.clone()

    for _ in range(warmup):
        nl.compute(current, box, periodic=True, quantities="ijSDd")
        current = perturb(current)

    if points.is_cuda:
        torch.cuda.synchronize()

    current = points.clone()
    t0 = time.perf_counter()
    for step in range(n_steps):
        nl.compute(current, box, periodic=True, quantities="ijSDd")
        current = perturb(current)
    if points.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed


def bench_verlet(points, box, cutoff, skin, n_steps, warmup=3):
    """Time Verlet neighbor list over n_steps (CPU only)."""
    cpu_points = points.cpu()
    cpu_box = box.cpu()
    vl = VerletNeighborList(cutoff=cutoff, skin=skin, full_list=True, sorted=True)
    current = cpu_points.clone()

    for _ in range(warmup):
        vl.compute(current, cpu_box, periodic=True, quantities="ijSDd")
        current = perturb(current)

    current = cpu_points.clone()
    rebuilds = 0
    t0 = time.perf_counter()
    for step in range(n_steps):
        vl.compute(current, cpu_box, periodic=True, quantities="ijSDd")
        if vl.did_rebuild:
            rebuilds += 1
        current = perturb(current)
    elapsed = time.perf_counter() - t0
    return elapsed, rebuilds


def main():
    cutoff = 5.0
    skin = 1.0
    n_steps = 50
    has_cuda = torch.cuda.is_available()

    configs = [
        (2, "32"),
        (3, "108"),
        (4, "256"),
        (5, "500"),
        (6, "864"),
        (8, "2048"),
    ]
    if has_cuda:
        configs.append((10, "4000"))

    header = f"{'N':>6} {'CPU stat':>10} {'CPU Verlet':>10} {'VL speedup':>10} {'Rebuilds':>10}"
    if has_cuda:
        header += f" {'GPU stat':>10} {'GPU vs VL':>10}"
    print(header)
    print("-" * len(header))

    results = {}

    for n_cells, label in configs:
        points_cpu, box_cpu = make_fcc_system(n_cells, device="cpu")
        n_atoms = points_cpu.shape[0]

        t_cpu = bench_stateless(points_cpu, box_cpu, cutoff, n_steps)
        t_verlet, rebuilds = bench_verlet(points_cpu, box_cpu, cutoff, skin, n_steps)
        vl_speedup = t_cpu / t_verlet if t_verlet > 0 else float("inf")

        row = f"{n_atoms:>6} {t_cpu:>9.3f}s {t_verlet:>9.3f}s {vl_speedup:>9.2f}x {rebuilds:>5}/{n_steps}"

        entry = {
            "n_atoms": n_atoms,
            "cpu_stateless_s": round(t_cpu, 4),
            "cpu_verlet_s": round(t_verlet, 4),
            "verlet_speedup": round(vl_speedup, 2),
            "rebuilds": rebuilds,
            "n_steps": n_steps,
        }

        if has_cuda:
            points_gpu, box_gpu = make_fcc_system(n_cells, device="cuda")
            t_gpu = bench_stateless(points_gpu, box_gpu, cutoff, n_steps)
            gpu_vs_vl = t_verlet / t_gpu if t_gpu > 0 else float("inf")
            row += f" {t_gpu:>9.3f}s {gpu_vs_vl:>9.2f}x"
            entry["gpu_stateless_s"] = round(t_gpu, 4)
            entry["gpu_vs_verlet"] = round(gpu_vs_vl, 2)

        print(row)
        results[label] = entry

    with open("results_torch_verlet.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to vesin_bench/results_torch_verlet.json")
    if has_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Cutoff: {cutoff}, Skin: {skin}, Steps: {n_steps}")


if __name__ == "__main__":
    main()
