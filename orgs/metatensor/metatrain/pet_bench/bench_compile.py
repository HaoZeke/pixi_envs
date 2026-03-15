"""Benchmark: torch.compile vs eager mode for PET training.

Measures wall-clock time per training step for both paths, reporting:
  - Compilation overhead (first step)
  - Steady-state step time (median of subsequent steps)
  - Speedup ratio

Supports both fixed cutoff and adaptive cutoff (num_neighbors_adaptive)
configurations via --config flag.

Usage:
  pixi run -e cuda bench-compile            # both configs
  pixi run -e cuda bench-compile-fixed      # fixed cutoff only
  pixi run -e cuda bench-compile-adaptive   # adaptive cutoff only
"""

import argparse
import gc
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

from metatrain.pet import PET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import (
    CollateFn,
    DatasetInfo,
    TargetInfo,
    get_atomic_types,
    get_dataset,
    unpack_batch,
)
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossAggregator, LossSpecification
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.per_atom import average_by_num_atoms
from metatrain.utils.transfer import batch_to


# Resolve dataset path relative to the metatrain repo
METATRAIN_ROOT = Path(__file__).resolve().parent.parent / "metatrain"
DATASET_PATH = str(METATRAIN_ROOT / "tests" / "resources" / "qm9_reduced_100.xyz")


def build_model_and_data(
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 8,
    with_forces: bool = True,
    with_stress: bool = True,
    num_neighbors_adaptive: Optional[int] = None,
) -> Tuple[PET, torch.utils.data.DataLoader, Dict[str, TargetInfo]]:
    """Set up a PET model, dataloader, and target info for benchmarking."""
    hypers = get_default_hypers("pet")
    model_hypers = hypers["model"]
    if num_neighbors_adaptive is not None:
        model_hypers["num_neighbors_adaptive"] = num_neighbors_adaptive

    energy_target = {
        "quantity": "energy",
        "read_from": DATASET_PATH,
        "reader": "ase",
        "key": "U0",
        "unit": "eV",
        "type": "scalar",
        "per_atom": False,
        "num_subtargets": 1,
        "forces": with_forces,
        "stress": with_stress,
        "virial": False,
    }
    dataset_targets = {"mtt::U0": energy_target}

    dataset, targets_info, _ = get_dataset(
        {
            "systems": {"read_from": DATASET_PATH, "reader": "ase"},
            "targets": dataset_targets,
        }
    )

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=get_atomic_types(dataset),
        targets=targets_info,
    )

    model = PET(model_hypers, dataset_info)
    model.to(device=device, dtype=dtype)

    requested_nls = get_requested_neighbor_lists(model)
    nl_transform = get_system_with_neighbor_lists_transform(requested_nls)

    collate_fn = CollateFn(
        target_keys=list(targets_info.keys()),
        callables=[nl_transform],
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return model, dataloader, targets_info


def _make_loss_fn(targets_info: Dict[str, TargetInfo]) -> LossAggregator:
    loss_conf = OmegaConf.create(
        {k: init_with_defaults(LossSpecification) for k in targets_info}
    )
    OmegaConf.resolve(loss_conf)
    return LossAggregator(targets=targets_info, config=loss_conf)


def bench_eager(
    model: PET,
    dataloader: torch.utils.data.DataLoader,
    targets_info: Dict[str, TargetInfo],
    device: torch.device,
    dtype: torch.dtype,
    n_steps: int = 50,
    warmup_steps: int = 5,
) -> List[float]:
    """Benchmark eager (non-compiled) training steps."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = _make_loss_fn(targets_info)

    times: List[float] = []
    step = 0
    while step < n_steps + warmup_steps:
        for batch in dataloader:
            if step >= n_steps + warmup_steps:
                break

            systems, batch_targets, extra_data = unpack_batch(batch)
            systems, batch_targets, extra_data = batch_to(
                systems, batch_targets, extra_data, dtype=dtype, device=device
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            optimizer.zero_grad()
            predictions = evaluate_model(
                model,
                systems,
                {key: targets_info[key] for key in batch_targets.keys()},
                is_training=True,
            )
            predictions = average_by_num_atoms(predictions, systems, [])
            batch_targets = average_by_num_atoms(batch_targets, systems, [])
            loss = loss_fn(predictions, batch_targets, extra_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            if step >= warmup_steps:
                times.append(t1 - t0)
            step += 1

    return times


def bench_compiled(
    model: PET,
    dataloader: torch.utils.data.DataLoader,
    targets_info: Dict[str, TargetInfo],
    device: torch.device,
    dtype: torch.dtype,
    n_steps: int = 50,
    warmup_steps: int = 3,
) -> Tuple[float, List[float]]:
    """Benchmark compiled training steps. Returns (compile_time, step_times)."""
    from metatrain.pet.modules.compile import compile_pet_model
    from metatrain.pet.modules.structures import systems_to_batch
    from metatrain.pet.trainer import _wrap_compiled_output

    has_gradients = any(
        len(ti.gradients) > 0 for ti in targets_info.values()
    )
    has_strain = any(
        "strain" in ti.gradients for ti in targets_info.values()
    )

    # --- Compilation phase ---
    torch._dynamo.reset()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_compile_start = time.perf_counter()
    compiled_fn, _, _ = compile_pet_model(
        model, dataloader, has_gradients, has_strain
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_compile_trace = time.perf_counter()

    # First call triggers Triton codegen
    batch = next(iter(dataloader))
    systems, batch_targets, extra_data = unpack_batch(batch)
    systems, batch_targets, extra_data = batch_to(
        systems, batch_targets, extra_data, dtype=dtype, device=device
    )
    (
        c_ein, c_eineigh, c_ev, _, c_pm, c_rni,
        c_cf, c_si, c_nai, c_sl,
    ) = systems_to_batch(
        systems,
        model.requested_nl,
        model.atomic_types,
        model.species_to_species_index,
        model.cutoff_function,
        model.cutoff_width,
        model.num_neighbors_adaptive,
    )
    if has_gradients:
        c_ev = c_ev.requires_grad_(True)
    n_structures = len(systems)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_first_start = time.perf_counter()
    compiled_fn(
        c_ev, c_ein, c_eineigh, c_pm, c_rni, c_cf, c_si, c_nai,
        n_structures,
        *list(model.parameters()),
        *list(model.buffers()),
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_first_end = time.perf_counter()
    compile_time = t_first_end - t_compile_start

    print(f"  FX trace time:        {t_compile_trace - t_compile_start:.2f}s")
    print(f"  First call (codegen): {t_first_end - t_first_start:.2f}s")
    print(f"  Total compile time:   {compile_time:.2f}s")

    # --- Steady-state benchmark ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = _make_loss_fn(targets_info)

    times: List[float] = []
    step = 0
    while step < n_steps + warmup_steps:
        for batch in dataloader:
            if step >= n_steps + warmup_steps:
                break

            systems, batch_targets, extra_data = unpack_batch(batch)
            systems, batch_targets, extra_data = batch_to(
                systems, batch_targets, extra_data, dtype=dtype, device=device
            )

            (
                c_ein, c_eineigh, c_ev, _, c_pm, c_rni,
                c_cf, c_si, c_nai, c_sl,
            ) = systems_to_batch(
                systems,
                model.requested_nl,
                model.atomic_types,
                model.species_to_species_index,
                model.cutoff_function,
                model.cutoff_width,
                model.num_neighbors_adaptive,
            )
            if has_gradients:
                c_ev = c_ev.requires_grad_(True)
            n_structures = len(systems)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            optimizer.zero_grad()
            energy, forces, stress, raw_preds = compiled_fn(
                c_ev, c_ein, c_eineigh, c_pm, c_rni, c_cf, c_si, c_nai,
                n_structures,
                *list(model.parameters()),
                *list(model.buffers()),
            )
            predictions = _wrap_compiled_output(
                energy, forces, stress, raw_preds,
                model, systems, c_sl, c_si, targets_info,
            )
            predictions = average_by_num_atoms(predictions, systems, [])
            batch_targets = average_by_num_atoms(batch_targets, systems, [])
            loss = loss_fn(predictions, batch_targets, extra_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            if step >= warmup_steps:
                times.append(t1 - t0)
            step += 1

    return compile_time, times


def _run_config(
    config_name: str,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    n_steps: int,
    num_neighbors_adaptive: Optional[int] = None,
) -> Dict:
    """Run eager + compiled benchmarks for a single configuration.

    Returns dict with timing results.
    """
    print(f"\n{'=' * 70}")
    print(f"  {config_name}")
    print(f"{'=' * 70}")

    # ---- Eager mode ----
    print(f"\n--- Eager mode ({config_name}) ---")
    model_eager, dl_eager, targets_eager = build_model_and_data(
        device, dtype, batch_size,
        num_neighbors_adaptive=num_neighbors_adaptive,
    )
    n_params = sum(p.numel() for p in model_eager.parameters())
    print(f"Model parameters: {n_params:,}")

    eager_times = bench_eager(
        model_eager, dl_eager, targets_eager, device, dtype, n_steps
    )
    eager_median = statistics.median(eager_times)
    eager_mean = statistics.mean(eager_times)
    eager_std = statistics.stdev(eager_times) if len(eager_times) > 1 else 0.0

    print(f"  Median step time: {eager_median*1000:.1f} ms")
    print(f"  Mean step time:   {eager_mean*1000:.1f} ms (+/- {eager_std*1000:.1f})")
    print(
        f"  Min / Max:        "
        f"{min(eager_times)*1000:.1f} / {max(eager_times)*1000:.1f} ms"
    )

    del model_eager, dl_eager
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- Compiled mode ----
    print(f"\n--- Compiled mode ({config_name}) ---")
    model_comp, dl_comp, targets_comp = build_model_and_data(
        device, dtype, batch_size,
        num_neighbors_adaptive=num_neighbors_adaptive,
    )

    compile_time, compiled_times = bench_compiled(
        model_comp, dl_comp, targets_comp, device, dtype, n_steps
    )
    compiled_median = statistics.median(compiled_times)
    compiled_mean = statistics.mean(compiled_times)
    compiled_std = (
        statistics.stdev(compiled_times) if len(compiled_times) > 1 else 0.0
    )

    print(f"  Median step time: {compiled_median*1000:.1f} ms")
    print(f"  Mean step time:   {compiled_mean*1000:.1f} ms (+/- {compiled_std*1000:.1f})")
    print(
        f"  Min / Max:        "
        f"{min(compiled_times)*1000:.1f} / {max(compiled_times)*1000:.1f} ms"
    )

    n_batches = len(dl_comp)

    del model_comp, dl_comp
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": config_name,
        "eager_median": eager_median,
        "compiled_median": compiled_median,
        "compile_time": compile_time,
        "n_batches": n_batches,
    }


def _print_summary(results: List[Dict]) -> None:
    """Print summary table for all benchmark configs."""
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for r in results:
        speedup = r["eager_median"] / r["compiled_median"]
        print(f"\n  [{r['name']}]")
        print(f"  Eager median step:     {r['eager_median']*1000:.1f} ms")
        print(f"  Compiled median step:  {r['compiled_median']*1000:.1f} ms")
        print(f"  Speedup:               {speedup:.2f}x")
        print(f"  Compile overhead:      {r['compile_time']:.1f} s")
        if speedup > 1.0:
            breakeven = r["compile_time"] / (
                r["eager_median"] - r["compiled_median"]
            )
            print(
                f"  Break-even after:      "
                f"{breakeven:.0f} steps "
                f"(~{breakeven/max(r['n_batches'],1):.0f} epochs)"
            )
        else:
            print("  (No speedup on this device/config)")
    print()


CONFIGS = {
    "fixed": {"name": "Fixed cutoff", "nna": None},
    "adaptive": {"name": "Adaptive cutoff (16)", "nna": 16},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="PET torch.compile benchmark")
    parser.add_argument(
        "--config",
        choices=["fixed", "adaptive", "all"],
        default="all",
        help="Which cutoff config to benchmark (default: all)",
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of steady-state steps to measure (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size (default: 8)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PET torch.compile Benchmark")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Batch size: {args.batch_size}")
    print(f"Steps: {args.steps}")

    if args.config == "all":
        configs_to_run = list(CONFIGS.values())
    else:
        configs_to_run = [CONFIGS[args.config]]

    results = []
    for cfg in configs_to_run:
        result = _run_config(
            cfg["name"], device, dtype, args.batch_size, args.steps,
            num_neighbors_adaptive=cfg["nna"],
        )
        results.append(result)

    _print_summary(results)


if __name__ == "__main__":
    main()
