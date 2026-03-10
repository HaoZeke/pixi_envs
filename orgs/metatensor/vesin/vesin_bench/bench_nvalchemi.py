"""Compare GPU neighbor list strategies for metatomic MD.

Three paths measured:
1. "Old path" (CPU Verlet + D2H/H2D): copy GPU positions to CPU, run CPU
   Verlet, copy results back. This is what _verlet.py did before the fix.
2. "Nvalchemi" (GPU stateless): rebuild full neighbor list from scratch on
   GPU every step. No caching.
3. "GPU Verlet" (new path): GPU-native Verlet with NVRTC displacement check
   and pair recompute kernels. Only rebuilds when max displacement > skin/2.

Shows that GPU Verlet dominates at MD-relevant system sizes.
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
    return points + amplitude * torch.randn_like(points)


def bench_old_path(points_gpu, box_gpu, cutoff, skin, n_steps, warmup=3):
    """Old metatomic path: copy to CPU, run CPU Verlet, copy back."""
    vl = VerletNeighborList(cutoff=cutoff, skin=skin, full_list=True, sorted=True)
    current = points_gpu.clone()

    for _ in range(warmup):
        pts_cpu = current.detach().cpu().to(dtype=torch.float64)
        bx_cpu = box_gpu.detach().cpu().to(dtype=torch.float64)
        results = vl.compute(pts_cpu, bx_cpu, periodic=True, quantities="PSD", copy=True)
        # simulate copying results back to GPU
        for r in results:
            r.to(device=points_gpu.device)
        current = perturb(current)

    torch.cuda.synchronize()
    current = points_gpu.clone()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        pts_cpu = current.detach().cpu().to(dtype=torch.float64)
        bx_cpu = box_gpu.detach().cpu().to(dtype=torch.float64)
        results = vl.compute(pts_cpu, bx_cpu, periodic=True, quantities="PSD", copy=True)
        for r in results:
            r.to(device=points_gpu.device)
        current = perturb(current)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_nvalchemi(points_gpu, box_gpu, cutoff, n_steps, warmup=3):
    """GPU stateless: rebuild from scratch every step (nvalchemi)."""
    nl = NeighborList(cutoff=cutoff, full_list=True)
    current = points_gpu.clone()

    for _ in range(warmup):
        nl.compute(current, box_gpu, periodic=True, quantities="PSD")
        current = perturb(current)

    torch.cuda.synchronize()
    current = points_gpu.clone()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        nl.compute(current, box_gpu, periodic=True, quantities="PSD")
        current = perturb(current)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_gpu_verlet(points_gpu, box_gpu, cutoff, skin, n_steps, warmup=3):
    """GPU Verlet: NVRTC displacement check + pair recompute kernels."""
    vl = VerletNeighborList(cutoff=cutoff, skin=skin, full_list=True, sorted=False)
    current = points_gpu.clone()

    for _ in range(warmup):
        vl.compute(current, box_gpu, periodic=True, quantities="PSD")
        current = perturb(current)

    torch.cuda.synchronize()
    current = points_gpu.clone()
    rebuilds = 0
    t0 = time.perf_counter()
    for _ in range(n_steps):
        vl.compute(current, box_gpu, periodic=True, quantities="PSD")
        if vl.did_rebuild:
            rebuilds += 1
        current = perturb(current)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, rebuilds


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark requires a GPU.")
        return

    cutoff = 5.0
    skin = 1.0
    n_steps = 50

    configs = [
        (3, "108"),
        (4, "256"),
        (5, "500"),
        (6, "864"),
        (8, "2048"),
        (10, "4000"),
        (12, "6912"),
    ]

    header = (f"{'N':>6} {'Old(CPU+copy)':>14} {'Nvalchemi':>10} "
              f"{'GPU Verlet':>10} {'GV/NV':>8} {'GV/Old':>8} {'Rebuilds':>10}")
    print(header)
    print("-" * len(header))

    results = {}

    for n_cells, label in configs:
        points_gpu, box_gpu = make_fcc_system(n_cells, device="cuda")
        n_atoms = points_gpu.shape[0]

        t_old = bench_old_path(points_gpu, box_gpu, cutoff, skin, n_steps)
        t_nv = bench_nvalchemi(points_gpu, box_gpu, cutoff, n_steps)
        t_gv, rebuilds = bench_gpu_verlet(points_gpu, box_gpu, cutoff, skin, n_steps)

        gv_vs_nv = t_nv / t_gv if t_gv > 0 else float("inf")
        gv_vs_old = t_old / t_gv if t_gv > 0 else float("inf")

        row = (f"{n_atoms:>6} {t_old:>13.3f}s {t_nv:>9.3f}s "
               f"{t_gv:>9.3f}s {gv_vs_nv:>7.2f}x {gv_vs_old:>7.1f}x "
               f"{rebuilds:>5}/{n_steps}")
        print(row)

        results[label] = {
            "n_atoms": n_atoms,
            "old_cpu_copy_s": round(t_old, 4),
            "nvalchemi_s": round(t_nv, 4),
            "gpu_verlet_s": round(t_gv, 4),
            "gv_vs_nvalchemi": round(gv_vs_nv, 2),
            "gv_vs_old_path": round(gv_vs_old, 1),
            "rebuilds": rebuilds,
            "n_steps": n_steps,
        }

    with open("results_nvalchemi.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to vesin_bench/results_nvalchemi.json")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Cutoff: {cutoff}, Skin: {skin}, Steps: {n_steps}")
    print()
    print("Legend:")
    print("  Old(CPU+copy) = CPU Verlet + D2H/H2D copies (pre-fix metatomic path)")
    print("  Nvalchemi     = GPU stateless rebuild every step")
    print("  GPU Verlet    = GPU-native Verlet with NVRTC kernels (new path)")
    print("  GV/NV         = GPU Verlet speedup over nvalchemi")
    print("  GV/Old        = GPU Verlet speedup over old CPU+copy path")


if __name__ == "__main__":
    main()
