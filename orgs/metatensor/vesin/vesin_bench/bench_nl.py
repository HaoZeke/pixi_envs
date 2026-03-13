"""Benchmark vesin neighbor list construction.

Measures wall time for neighbor list computation across system sizes.
Run twice (once per git checkout) to compare baseline vs feature branch.

Usage:
    python bench_nl.py                     # writes results.json
    python bench_nl.py --tag baseline      # writes results_baseline.json
    python bench_nl.py --tag cluster-pair  # writes results_cluster-pair.json

Backends tested (when available):
  - vesin CPU (stateless, auto algorithm)
  - vesin GPU (CUDA, auto algorithm)
  - vesin Verlet CPU (cached topology: rebuild vs reuse cost)
  - MD trajectory: stateless vs Verlet over N steps with thermal displacements
"""

import argparse
import ctypes
import json
import os
import time
import sys
from ctypes import POINTER
from pathlib import Path

import numpy as np

try:
    import sys
    from pathlib import Path
    # Work around vesin/ directory shadowing the installed package
    cwd = Path.cwd()
    # Remove ALL vesin-related paths from sys.path to avoid namespace shadowing
    sys.path = [p for p in sys.path if 'vesin' not in str(p) or 'site-packages' in str(p)]
    if 'vesin' in sys.modules:
        del sys.modules['vesin']
    import vesin
    if not hasattr(vesin, "NeighborList"):
        raise ImportError(
            "Got vesin namespace package (no NeighborList). "
            "Install vesin properly or run from a different directory.")
    HAS_VESIN = True
except ImportError as e:
    HAS_VESIN = False
    print(f"WARNING: {e}")

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

# Check if Verlet C API is available (feature branch only)
HAS_VERLET = False
if HAS_VESIN:
    try:
        from vesin._c_lib import _get_library
        from vesin._c_api import (
            VesinNeighborList, VesinOptions, VesinDevice, VesinCPU, VesinCUDA,
        )
        _lib = _get_library()
        _lib.vesin_verlet_new  # will AttributeError on upstream
        HAS_VERLET = True
    except (AttributeError, ImportError):
        pass

CUTOFF = 5.0
DATA_DIR = Path(__file__).parent / "data"
N_WARMUP = 3
N_ITER = 10
MD_STEPS = 200  # steps for MD trajectory benchmark
MD_DT_DISPLACEMENT = 0.02  # RMS displacement per step (Angstrom), ~1 fs MD step


def bench_vesin_cpu(positions, box, cutoff, full_list=True, skin=0.0):
    """Benchmark vesin CPU neighbor list (auto algorithm)."""
    periodic = [True, True, True]
    nl = vesin.NeighborList(cutoff=cutoff, full_list=full_list, skin=skin)

    for _ in range(N_WARMUP):
        nl.compute(points=positions, box=box.T, periodic=periodic, quantities="ij")

    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        nl.compute(points=positions, box=box.T, periodic=periodic, quantities="ij")
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.median(times), np.std(times)


def bench_vesin_gpu(positions_np, box_np, cutoff, full_list=True, skin=0.0):
    """Benchmark vesin GPU (CUDA) neighbor list.

    Returns (None, None) if CUDA not available or vesin lacks GPU support.
    """
    if not HAS_CUDA:
        return None, None

    positions_t = torch.tensor(positions_np, dtype=torch.float64, device="cuda")
    box_t = torch.tensor(box_np.T, dtype=torch.float64, device="cuda")
    periodic = [True, True, True]
    nl = vesin.NeighborList(cutoff=cutoff, full_list=full_list, skin=skin)

    # Check if this vesin build supports GPU tensors
    try:
        nl.compute(points=positions_t, box=box_t, periodic=periodic, quantities="ij")
        torch.cuda.synchronize()
    except (TypeError, RuntimeError):
        return None, None

    for _ in range(N_WARMUP):
        nl.compute(points=positions_t, box=box_t, periodic=periodic, quantities="ij")
        torch.cuda.synchronize()

    times = []
    for _ in range(N_ITER):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        nl.compute(points=positions_t, box=box_t, periodic=periodic, quantities="ij")
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.median(times), np.std(times)


def bench_verlet_cpu(positions_np, box_np, cutoff, skin=1.0):
    """Benchmark Verlet neighbor list via C API: rebuild vs reuse.

    Uses vesin_verlet_compute directly through ctypes. Simulates an MD loop:
    - rebuild: first call or large displacement (full spatial search)
    - reuse: subsequent calls with tiny perturbation (just recompute vectors)

    Returns (rebuild_median, rebuild_std, reuse_median, reuse_std) or Nones.
    """
    if not HAS_VERLET:
        return None, None, None, None

    lib = _get_library()
    n_points = len(positions_np)
    periodic = np.array([True, True, True], dtype=np.bool_)

    # Create Verlet handle
    error_message = ctypes.c_char_p()
    handle = lib.vesin_verlet_new(cutoff, skin, True, error_message)
    if handle is None:
        print(f"  WARNING: vesin_verlet_new failed: {error_message.value}")
        return None, None, None, None

    neighbors = VesinNeighborList()

    options = VesinOptions()
    options.cutoff = 0.0  # ignored by verlet
    options.full = False  # ignored by verlet
    options.sorted = False
    options.algorithm = 0
    options.return_shifts = True
    options.return_distances = True
    options.return_vectors = True

    points_ptr = positions_np.ctypes.data_as(POINTER(ctypes.c_double))
    box_ptr = box_np.ctypes.data_as(POINTER(ctypes.c_double))
    periodic_ptr = periodic.ctypes.data_as(POINTER(ctypes.c_bool))

    # Warmup
    for _ in range(N_WARMUP):
        # Force rebuild each time by creating new handle
        h = lib.vesin_verlet_new(cutoff, skin, True, error_message)
        n = VesinNeighborList()
        lib.vesin_verlet_compute(
            h, points_ptr, n_points, box_ptr, periodic_ptr,
            VesinDevice(VesinCPU, 0), options, ctypes.byref(n), error_message
        )
        lib.vesin_free(n)
        lib.vesin_verlet_free(h)

    # Measure rebuild time (each iteration creates a fresh handle -> forced rebuild)
    rebuild_times = []
    for _ in range(N_ITER):
        h = lib.vesin_verlet_new(cutoff, skin, True, error_message)
        n = VesinNeighborList()
        t0 = time.perf_counter()
        lib.vesin_verlet_compute(
            h, points_ptr, n_points, box_ptr, periodic_ptr,
            VesinDevice(VesinCPU, 0), options, ctypes.byref(n), error_message
        )
        t1 = time.perf_counter()
        rebuild_times.append(t1 - t0)
        lib.vesin_free(n)
        lib.vesin_verlet_free(h)

    # Measure reuse time: initial build, then small perturbation < skin/2
    handle = lib.vesin_verlet_new(cutoff, skin, True, error_message)
    neighbors = VesinNeighborList()

    # Initial build
    lib.vesin_verlet_compute(
        handle, points_ptr, n_points, box_ptr, periodic_ptr,
        VesinDevice(VesinCPU, 0), options, ctypes.byref(neighbors), error_message
    )
    assert lib.vesin_verlet_did_rebuild(handle), "first call should rebuild"

    rng = np.random.default_rng(12345)
    small_shift = rng.uniform(-skin * 0.1, skin * 0.1, size=positions_np.shape)
    shifted_pos = np.ascontiguousarray(positions_np + small_shift)
    shifted_ptr = shifted_pos.ctypes.data_as(POINTER(ctypes.c_double))

    # Warmup reuse
    for _ in range(N_WARMUP):
        lib.vesin_verlet_compute(
            handle, shifted_ptr, n_points, box_ptr, periodic_ptr,
            VesinDevice(VesinCPU, 0), options, ctypes.byref(neighbors), error_message
        )

    reuse_times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        lib.vesin_verlet_compute(
            handle, shifted_ptr, n_points, box_ptr, periodic_ptr,
            VesinDevice(VesinCPU, 0), options, ctypes.byref(neighbors), error_message
        )
        t1 = time.perf_counter()
        reuse_times.append(t1 - t0)
        did_rebuild = lib.vesin_verlet_did_rebuild(handle)
        if did_rebuild:
            print(f"  WARNING: unexpected rebuild on reuse iteration")

    lib.vesin_free(neighbors)
    lib.vesin_verlet_free(handle)

    return (
        np.median(rebuild_times), np.std(rebuild_times),
        np.median(reuse_times), np.std(reuse_times),
    )


def bench_md_integrated_verlet(positions_np, box_np, cutoff, skin=1.0, n_steps=MD_STEPS,
                               dt_disp=MD_DT_DISPLACEMENT):
    """Benchmark integrated Verlet NL (via NeighborList with skin) over MD trajectory.

    Uses the new integrated API: NeighborList(cutoff=X, full_list=Y, skin=Z).
    This is the recommended approach - no separate vesin_verlet_* handle needed.

    Returns (total_time_s, per_step_ms, n_rebuilds, n_reuses) or (None, ...).
    """
    if not HAS_VESIN:
        return None, None, None, None

    periodic = [True, True, True]
    positions = positions_np.copy()
    rng = np.random.default_rng(42)

    # Create NeighborList with Verlet caching enabled
    nl = vesin.NeighborList(cutoff=cutoff, full_list=True, skin=skin)

    total_time = 0.0
    n_rebuilds = 0
    n_reuses = 0

    for step in range(n_steps):
        # Small random displacement (thermal motion)
        positions += rng.normal(0, dt_disp, size=positions.shape)

        t0 = time.perf_counter()
        nl.compute(points=positions, box=box_np.T, periodic=periodic, quantities="ij")
        t1 = time.perf_counter()
        total_time += (t1 - t0)

        if nl.did_rebuild():
            n_rebuilds += 1
        else:
            n_reuses += 1

    per_step_ms = (total_time / n_steps) * 1000
    return total_time, per_step_ms, n_rebuilds, n_reuses


def bench_md_stateless(positions_np, box_np, cutoff, n_steps=MD_STEPS,
                       dt_disp=MD_DT_DISPLACEMENT):
    """Benchmark stateless NL over an MD-like trajectory.

    Each step applies a small random displacement (simulating thermal motion)
    and computes a fresh neighbor list from scratch. This is the baseline
    per-step cost that any caching scheme must beat.

    Returns (total_time_s, per_step_ms) or (None, None).
    """
    if not HAS_VESIN:
        return None, None

    periodic = [True, True, True]
    rng = np.random.default_rng(54321)
    pos = positions_np.copy()
    box_t = box_np.T.copy()

    # Warmup
    for _ in range(N_WARMUP):
        vesin.NeighborList(cutoff=cutoff, full_list=True).compute(
            points=pos, box=box_t, periodic=periodic, quantities="ij"
        )

    t_total = 0.0
    for step in range(n_steps):
        # Small displacement each step
        pos += rng.normal(0.0, dt_disp, size=pos.shape)
        pos_c = np.ascontiguousarray(pos)

        t0 = time.perf_counter()
        vesin.NeighborList(cutoff=cutoff, full_list=True).compute(
            points=pos_c, box=box_t, periodic=periodic, quantities="ij"
        )
        t1 = time.perf_counter()
        t_total += (t1 - t0)

    per_step = (t_total / n_steps) * 1000  # ms
    return t_total, per_step


def bench_md_verlet(positions_np, box_np, cutoff, skin=1.0,
                    n_steps=MD_STEPS, dt_disp=MD_DT_DISPLACEMENT):
    """Benchmark Verlet NL over an MD-like trajectory.

    Same trajectory as bench_md_stateless but uses the Verlet cached NL.
    The Verlet list auto-decides rebuild vs reuse based on displacements.

    Returns (total_time_s, per_step_ms, n_rebuilds, n_reuses) or Nones.
    """
    if not HAS_VERLET:
        return None, None, None, None

    lib = _get_library()
    n_points = len(positions_np)
    periodic = np.array([True, True, True], dtype=np.bool_)
    rng = np.random.default_rng(54321)  # same seed as stateless for same trajectory

    error_message = ctypes.c_char_p()
    handle = lib.vesin_verlet_new(cutoff, skin, True, error_message)
    if handle is None:
        print(f"  WARNING: vesin_verlet_new failed: {error_message.value}")
        return None, None, None, None

    neighbors = VesinNeighborList()

    options = VesinOptions()
    options.cutoff = 0.0
    options.full = False
    options.sorted = False
    options.algorithm = 0
    options.return_shifts = True
    options.return_distances = True
    options.return_vectors = True

    box_ptr = box_np.ctypes.data_as(POINTER(ctypes.c_double))
    periodic_ptr = periodic.ctypes.data_as(POINTER(ctypes.c_bool))

    pos = positions_np.copy()

    # Warmup (3 steps)
    for _ in range(N_WARMUP):
        pos_w = np.ascontiguousarray(pos + rng.normal(0, dt_disp, pos.shape))
        pts_ptr = pos_w.ctypes.data_as(POINTER(ctypes.c_double))
        lib.vesin_verlet_compute(
            handle, pts_ptr, n_points, box_ptr, periodic_ptr,
            VesinDevice(VesinCPU, 0), options, ctypes.byref(neighbors), error_message
        )

    # Reset for actual run
    lib.vesin_verlet_free(handle)
    handle = lib.vesin_verlet_new(cutoff, skin, True, error_message)
    neighbors = VesinNeighborList()
    pos = positions_np.copy()
    rng = np.random.default_rng(54321)  # reset to same trajectory

    n_rebuilds = 0
    n_reuses = 0
    t_total = 0.0

    for step in range(n_steps):
        pos += rng.normal(0.0, dt_disp, size=pos.shape)
        pos_c = np.ascontiguousarray(pos)
        pts_ptr = pos_c.ctypes.data_as(POINTER(ctypes.c_double))

        t0 = time.perf_counter()
        lib.vesin_verlet_compute(
            handle, pts_ptr, n_points, box_ptr, periodic_ptr,
            VesinDevice(VesinCPU, 0), options, ctypes.byref(neighbors), error_message
        )
        t1 = time.perf_counter()
        t_total += (t1 - t0)

        if lib.vesin_verlet_did_rebuild(handle):
            n_rebuilds += 1
        else:
            n_reuses += 1

    lib.vesin_free(neighbors)
    lib.vesin_verlet_free(handle)

    per_step = (t_total / n_steps) * 1000  # ms
    return t_total, per_step, n_rebuilds, n_reuses


def main():
    parser = argparse.ArgumentParser(description="Benchmark vesin NL construction")
    parser.add_argument("--tag", default="",
                        help="Tag for output file (e.g. 'baseline', 'cluster-pair')")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"Data directory {DATA_DIR} not found. Run prepare.py first.")
        sys.exit(1)

    system_files = sorted(DATA_DIR.glob("system_*.npz"),
                          key=lambda p: int(p.stem.split("_")[1]))

    if not system_files:
        print("No system files found. Run prepare.py first.")
        sys.exit(1)

    print(f"Benchmarking {len(system_files)} system sizes")
    print(f"Cutoff: {CUTOFF} A, warmup: {N_WARMUP}, iterations: {N_ITER}")
    print(f"vesin: {HAS_VESIN}, torch: {HAS_TORCH}, cuda: {HAS_CUDA}, "
          f"verlet: {HAS_VERLET}")
    if args.tag:
        print(f"Tag: {args.tag}")
    print("-" * 80)

    results = {
        "tag": args.tag,
        "cutoff": CUTOFF,
        "n_warmup": N_WARMUP,
        "n_iter": N_ITER,
        "md_steps": MD_STEPS,
        "md_dt_displacement": MD_DT_DISPLACEMENT,
        "has_cuda": HAS_CUDA,
        "has_verlet": HAS_VERLET,
        "systems": [],
    }

    if HAS_CUDA:
        results["gpu_name"] = torch.cuda.get_device_name(0)

    try:
        results["vesin_version"] = vesin.__version__
    except AttributeError:
        pass

    for sf in system_files:
        data = np.load(sf)
        positions = np.ascontiguousarray(data["positions"])
        box = np.ascontiguousarray(data["box"])
        n_atoms = len(positions)

        entry = {"n_atoms": n_atoms}
        print(f"\nN = {n_atoms}")

        # vesin CPU (stateless)
        if HAS_VESIN:
            med, std = bench_vesin_cpu(positions, box, CUTOFF)
            entry["vesin_cpu_ms"] = med * 1000
            entry["vesin_cpu_std_ms"] = std * 1000
            n_pairs = vesin.NeighborList(cutoff=CUTOFF, full_list=True).compute(
                points=positions, box=box.T, periodic=[True]*3, quantities="ij"
            )[0].shape[0]
            entry["n_pairs"] = int(n_pairs)
            print(f"  vesin CPU:         {med*1000:8.3f} ms  ({n_pairs} pairs)")

        # vesin GPU (stateless)
        if HAS_VESIN and HAS_CUDA:
            med, std = bench_vesin_gpu(positions, box, CUTOFF)
            if med is not None:
                entry["vesin_gpu_ms"] = med * 1000
                entry["vesin_gpu_std_ms"] = std * 1000
                print(f"  vesin GPU:         {med*1000:8.3f} ms")

        # vesin CPU with integrated Verlet (skin=1.0)
        if HAS_VESIN:
            med, std = bench_vesin_cpu(positions, box, CUTOFF, skin=1.0)
            if med is not None:
                entry["vesin_cpu_verlet_ms"] = med * 1000
                entry["vesin_cpu_verlet_std_ms"] = std * 1000
                print(f"  vesin CPU Verlet:  {med*1000:8.3f} ms")

        # vesin GPU with integrated Verlet (skin=1.0)
        if HAS_VESIN and HAS_CUDA:
            med, std = bench_vesin_gpu(positions, box, CUTOFF, skin=1.0)
            if med is not None:
                entry["vesin_gpu_verlet_ms"] = med * 1000
                entry["vesin_gpu_verlet_std_ms"] = std * 1000
                print(f"  vesin GPU Verlet:  {med*1000:8.3f} ms")

        # Verlet CPU (rebuild vs reuse)
        if HAS_VERLET:
            r_med, r_std, u_med, u_std = bench_verlet_cpu(
                positions, box, CUTOFF, skin=1.0
            )
            if r_med is not None:
                entry["verlet_rebuild_ms"] = r_med * 1000
                entry["verlet_rebuild_std_ms"] = r_std * 1000
                entry["verlet_reuse_ms"] = u_med * 1000
                entry["verlet_reuse_std_ms"] = u_std * 1000
                speedup = r_med / u_med if u_med > 0 else float("inf")
                print(f"  Verlet rebuild:    {r_med*1000:8.3f} ms")
                print(f"  Verlet reuse:      {u_med*1000:8.3f} ms  ({speedup:.1f}x faster)")

        # MD trajectory: stateless vs integrated Verlet (the real comparison)
        if HAS_VESIN:
            t_stat, ps_stat = bench_md_stateless(positions, box, CUTOFF)
            if t_stat is not None:
                entry["md_stateless_total_s"] = t_stat
                entry["md_stateless_per_step_ms"] = ps_stat
                print(f"  MD stateless:      {ps_stat:8.3f} ms/step  "
                      f"({t_stat:.3f}s total, {MD_STEPS} steps)")

            # Integrated Verlet (NEW API - recommended)
            t_vrl, ps_vrl, n_rb, n_ru = bench_md_integrated_verlet(
                positions, box, CUTOFF, skin=1.0
            )
            if t_vrl is not None:
                entry["md_integrated_verlet_total_s"] = t_vrl
                entry["md_integrated_verlet_per_step_ms"] = ps_vrl
                entry["md_integrated_verlet_rebuilds"] = n_rb
                entry["md_integrated_verlet_reuses"] = n_ru
                speedup = t_stat / t_vrl if t_stat and t_vrl > 0 else 0
                print(f"  MD Verlet (new):   {ps_vrl:8.3f} ms/step  "
                      f"({t_vrl:.3f}s total, {n_rb} rebuilds, {n_ru} reuses)")
                print(f"  MD speedup:        {speedup:.1f}x  "
                      f"(integrated Verlet vs stateless over {MD_STEPS} steps)")

        # Old standalone Verlet API (for comparison)
        if HAS_VERLET:
            t_vrl, ps_vrl, n_rb, n_ru = bench_md_verlet(
                positions, box, CUTOFF, skin=1.0
            )
            if t_vrl is not None:
                entry["md_verlet_total_s"] = t_vrl
                entry["md_verlet_per_step_ms"] = ps_vrl
                entry["md_verlet_rebuilds"] = n_rb
                entry["md_verlet_reuses"] = n_ru

        results["systems"].append(entry)

    # Save results
    suffix = f"_{args.tag}" if args.tag else ""
    outfile = Path(__file__).parent / f"results{suffix}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
