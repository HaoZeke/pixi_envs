"""Profile the GPU Verlet filter hot path.

This drives the integrated ``vesin.NeighborList(skin>0)`` on CUDA so that
``filter_verlet_compact_candidates_block`` and ``check_verlet_displacements``
get exercised by the same code paths that ncu was supposed to capture. Uses
CUDA events so it works without elevated profiling permissions.

Workflow per system size:

  1. Load data/system_<N>.npz (must be >= CUDA_VERLET_COMPACT_MIN_POINTS to
     trigger the block-compact path; default gate is 1024).
  2. Build once on GPU (rebuild path).
  3. Inner loop: perturb positions by < skin/2, call ``nl.compute`` again,
     time the *whole* compute call with cudaEventRecord.

Saves results to ``results_gpu_verlet_filter_<tag>.json`` containing per-N
wall-clock min/median/p95 (ms) for the rebuild call and the reuse calls,
along with cutoff/skin and the resolved GPU name + CUDA-runtime version.

Usage:

    python bench_gpu_verlet_filter.py --tag cuda-simd-current
    python bench_gpu_verlet_filter.py --tag baseline-pre-simd

Compare two runs with plot_gpu_verlet_filter.py.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Same vesin shadow guard as bench_nl.py.
sys.path = [p for p in sys.path if "vesin" not in str(p) or ".pixi" in str(p)]
if "vesin" in sys.modules:
    del sys.modules["vesin"]
import vesin  # noqa: E402

import torch  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
CUTOFF = 5.0
SKIN = 1.0
N_REUSE = 50          # cache-reuse iterations per system
N_WARMUP = 5
PERTURB_FRAC = 0.1    # multiplied by skin: |delta| < skin*PERTURB_FRAC < skin/2

SIZES = [1024, 2048, 4096, 8192, 16384, 32768]


def percentile(arr, p):
    return float(np.percentile(np.array(arr), p))


def time_compute_event(nl, points_t, box_t, periodic, n_iter):
    """Time n_iter compute() calls each with cuda events; return ms list."""
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(n_iter):
        start.record()
        nl.compute(points=points_t, box=box_t, periodic=periodic, quantities="ij")
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def bench_system(npz_path):
    data = np.load(npz_path)
    positions_np = np.ascontiguousarray(data["positions"])
    box_np = np.ascontiguousarray(data["box"])
    n_atoms = len(positions_np)

    box_t = torch.tensor(box_np.T, dtype=torch.float64, device="cuda")
    periodic = [True, True, True]

    rng = np.random.default_rng(12345)
    perturb_amp = SKIN * PERTURB_FRAC

    nl = vesin.NeighborList(cutoff=CUTOFF, full_list=True, skin=SKIN)

    # Initial rebuild on GPU.
    pts0 = torch.tensor(positions_np, dtype=torch.float64, device="cuda")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Warm up the JIT (first call compiles kernels).
    nl.compute(points=pts0, box=box_t, periodic=periodic, quantities="ij")
    torch.cuda.synchronize()

    # Now create a fresh nl for the *measured* rebuild so JIT is out of the path.
    nl = vesin.NeighborList(cutoff=CUTOFF, full_list=True, skin=SKIN)
    pts_rebuild = torch.tensor(positions_np, dtype=torch.float64, device="cuda")
    start.record()
    nl.compute(points=pts_rebuild, box=box_t, periodic=periodic, quantities="ij")
    end.record()
    torch.cuda.synchronize()
    rebuild_ms = start.elapsed_time(end)

    # Reuse loop. Each iter perturbs positions by < skin/2 (so the displacement
    # check accepts the cache and the filter kernels run on the cached
    # candidate list), then re-computes.
    perturbed_t = torch.empty_like(pts_rebuild)
    for _ in range(N_WARMUP):
        shift = rng.uniform(-perturb_amp, perturb_amp, size=positions_np.shape)
        perturbed_np = np.ascontiguousarray(positions_np + shift)
        perturbed_t.copy_(torch.from_numpy(perturbed_np))
        nl.compute(points=perturbed_t, box=box_t, periodic=periodic, quantities="ij")
    torch.cuda.synchronize()

    reuse_times = []
    for _ in range(N_REUSE):
        shift = rng.uniform(-perturb_amp, perturb_amp, size=positions_np.shape)
        perturbed_np = np.ascontiguousarray(positions_np + shift)
        perturbed_t.copy_(torch.from_numpy(perturbed_np))
        start.record()
        nl.compute(points=perturbed_t, box=box_t, periodic=periodic, quantities="ij")
        end.record()
        torch.cuda.synchronize()
        reuse_times.append(start.elapsed_time(end))

    n_pairs = nl.compute(
        points=perturbed_t, box=box_t, periodic=periodic, quantities="ij",
    )[0].shape[0]

    return {
        "n_atoms": n_atoms,
        "n_pairs": int(n_pairs),
        "rebuild_ms": rebuild_ms,
        "reuse_min_ms": float(np.min(reuse_times)),
        "reuse_median_ms": float(np.median(reuse_times)),
        "reuse_p95_ms": percentile(reuse_times, 95),
        "reuse_std_ms": float(np.std(reuse_times)),
        "reuse_samples": N_REUSE,
        "reuse_all_ms": [float(t) for t in reuse_times],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True,
                        help="Tag for output (e.g. cuda-simd-current, baseline-pre-simd)")
    parser.add_argument("--out-dir", default=str(Path(__file__).parent))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if not DATA_DIR.exists():
        raise SystemExit(f"{DATA_DIR} missing; run prepare.py first")

    print(f"Tag: {args.tag}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUTOFF={CUTOFF}  SKIN={SKIN}  N_REUSE={N_REUSE}")
    print(f"vesin: {getattr(vesin, '__version__', 'unknown')}")

    results = {
        "tag": args.tag,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "vesin_version": getattr(vesin, "__version__", None),
        "cutoff": CUTOFF,
        "skin": SKIN,
        "n_reuse": N_REUSE,
        "perturb_frac_of_skin": PERTURB_FRAC,
        "timestamp": time.time(),
        "systems": [],
    }

    for n in SIZES:
        npz = DATA_DIR / f"system_{n}.npz"
        if not npz.exists():
            print(f"  skip N={n} (no {npz.name})")
            continue
        print(f"\nN = {n}")
        entry = bench_system(npz)
        results["systems"].append(entry)
        print(f"  pairs={entry['n_pairs']}")
        print(f"  rebuild    : {entry['rebuild_ms']:.3f} ms")
        print(f"  reuse min  : {entry['reuse_min_ms']:.3f} ms")
        print(f"  reuse med  : {entry['reuse_median_ms']:.3f} ms")
        print(f"  reuse p95  : {entry['reuse_p95_ms']:.3f} ms")
        print(f"  speedup    : {entry['rebuild_ms']/entry['reuse_median_ms']:.1f}x")

    out = Path(args.out_dir) / f"results_gpu_verlet_filter_{args.tag}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
