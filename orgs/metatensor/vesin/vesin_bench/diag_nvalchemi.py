"""Diagnostic: per-step timing for GPU Verlet at N=864 to find the spike."""

import time
import torch
from vesin.torch import NeighborList, VerletNeighborList


def make_fcc_system(n_cells, a=4.0, device="cpu"):
    basis = torch.tensor([
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
    ], dtype=torch.float64)
    positions = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                offset = torch.tensor([ix, iy, iz], dtype=torch.float64)
                for b in basis:
                    positions.append((b + offset) * a)
    return torch.stack(positions).to(device), (torch.eye(3, dtype=torch.float64) * a * n_cells).to(device)


def perturb(points, amplitude=0.02):
    return points + amplitude * torch.randn_like(points)


def diag_gpu_verlet(n_cells, n_steps=20, warmup=5):
    points, box = make_fcc_system(n_cells, device="cuda")
    n_atoms = points.shape[0]
    print(f"\n=== N={n_atoms} (n_cells={n_cells}) ===")

    vl = VerletNeighborList(cutoff=5.0, skin=1.0, full_list=True, sorted=False)
    current = points.clone()

    # Warmup with per-step timing
    print(f"Warmup ({warmup} steps):")
    for i in range(warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        vl.compute(current, box, periodic=True, quantities="PSD")
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        rb = "REBUILD" if vl.did_rebuild else "reuse"
        print(f"  warmup {i}: {dt:8.2f} ms  [{rb}]")
        current = perturb(current)

    # Reset and time
    current = points.clone()
    print(f"Timed ({n_steps} steps):")
    for i in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        vl.compute(current, box, periodic=True, quantities="PSD")
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        rb = "REBUILD" if vl.did_rebuild else "reuse"
        print(f"  step {i:2d}: {dt:8.2f} ms  [{rb}]")
        current = perturb(current)

    # Also time nvalchemi for comparison
    nl = NeighborList(cutoff=5.0, full_list=True)
    current = points.clone()
    for _ in range(warmup):
        nl.compute(current, box, periodic=True, quantities="PSD")
        current = perturb(current)
    current = points.clone()
    print(f"Nvalchemi ({n_steps} steps):")
    for i in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        nl.compute(current, box, periodic=True, quantities="PSD")
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        print(f"  step {i:2d}: {dt:8.2f} ms")
        current = perturb(current)


# Test the problematic size and its neighbors
for nc in [5, 6, 7, 8]:
    diag_gpu_verlet(nc, n_steps=20, warmup=5)
