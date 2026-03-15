"""Benchmark: DDP + torch.compile for PET training.

Launches 2 ranks on the same GPU (gloo backend) and measures per-step
wall-clock time for four configurations:
  1. Eager single-GPU (baseline)
  2. Compiled single-GPU
  3. Eager DDP (2 ranks)
  4. Compiled DDP (2 ranks)

Reports median step times and speedup ratios. Outputs JSON for plotting.

Usage:
  pixi run python pet_bench/bench_ddp.py [--steps N] [--output results.json]
"""

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


METATRAIN_ROOT = Path(__file__).resolve().parent.parent / "metatrain"
DATASET_PATH = str(METATRAIN_ROOT / "tests" / "resources" / "qm9_reduced_100.xyz")


def _build_model_and_data(
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 8,
) -> Tuple[Any, Any, Dict]:
    """Set up PET model, dataloader, and target info."""
    from metatrain.pet import PET
    from metatrain.utils.architectures import get_default_hypers
    from metatrain.utils.data import (
        CollateFn,
        DatasetInfo,
        get_atomic_types,
        get_dataset,
    )
    from metatrain.utils.neighbor_lists import (
        get_requested_neighbor_lists,
        get_system_with_neighbor_lists_transform,
    )

    hypers = get_default_hypers("pet")
    model_hypers = hypers["model"]

    energy_target = {
        "quantity": "energy",
        "read_from": DATASET_PATH,
        "reader": "ase",
        "key": "U0",
        "unit": "eV",
        "type": "scalar",
        "per_atom": False,
        "num_subtargets": 1,
        "forces": False,
        "stress": False,
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
        num_workers=0,
    )

    return model, dataloader, targets_info


def _make_loss_fn(targets_info):
    from omegaconf import OmegaConf

    from metatrain.utils.hypers import init_with_defaults
    from metatrain.utils.loss import LossAggregator, LossSpecification

    loss_conf = OmegaConf.create(
        {k: init_with_defaults(LossSpecification) for k in targets_info}
    )
    OmegaConf.resolve(loss_conf)
    return LossAggregator(targets=targets_info, config=loss_conf)


def _bench_single_eager(
    device, dtype, n_steps, warmup, batch_size,
) -> List[float]:
    """Benchmark eager single-GPU training."""
    from metatrain.utils.data import unpack_batch
    from metatrain.utils.evaluate_model import evaluate_model
    from metatrain.utils.per_atom import average_by_num_atoms
    from metatrain.utils.transfer import batch_to

    model, dl, targets_info = _build_model_and_data(device, dtype, batch_size)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = _make_loss_fn(targets_info)

    times = []
    step = 0
    while step < n_steps + warmup:
        for batch in dl:
            if step >= n_steps + warmup:
                break
            systems, batch_targets, extra_data = unpack_batch(batch)
            systems, batch_targets, extra_data = batch_to(
                systems, batch_targets, extra_data, dtype=dtype, device=device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            optimizer.zero_grad()
            preds = evaluate_model(
                model, systems,
                {k: targets_info[k] for k in batch_targets.keys()},
                is_training=True,
            )
            preds = average_by_num_atoms(preds, systems, [])
            batch_targets = average_by_num_atoms(batch_targets, systems, [])
            loss = loss_fn(preds, batch_targets, extra_data)
            loss.backward()
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            if step >= warmup:
                times.append(t1 - t0)
            step += 1

    del model, dl
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return times


def _bench_single_compiled(
    device, dtype, n_steps, warmup, batch_size,
) -> Tuple[float, List[float]]:
    """Benchmark compiled single-GPU training."""
    from metatrain.pet.modules.compile import compile_pet_model
    from metatrain.pet.modules.structures import systems_to_batch
    from metatrain.pet.trainer import _wrap_compiled_output
    from metatrain.utils.data import unpack_batch
    from metatrain.utils.per_atom import average_by_num_atoms
    from metatrain.utils.transfer import batch_to

    model, dl, targets_info = _build_model_and_data(device, dtype, batch_size)
    model.train()

    torch._dynamo.reset()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    compiled_fn, _, _ = compile_pet_model(model, dl, False, False)

    # First call triggers Triton codegen
    batch = next(iter(dl))
    systems, bt, ed = unpack_batch(batch)
    systems, bt, ed = batch_to(systems, bt, ed, dtype=dtype, device=device)
    c_ein, c_einb, c_ev, _, c_pm, c_rni, c_cf, c_si, c_nai, c_sl = systems_to_batch(
        systems, model.requested_nl, model.atomic_types,
        model.species_to_species_index, model.cutoff_function,
        model.cutoff_width, model.num_neighbors_adaptive,
    )
    compiled_fn(
        c_ev, c_ein, c_einb, c_pm, c_rni, c_cf, c_si, c_nai,
        len(systems), *list(model.parameters()), *list(model.buffers()),
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    compile_time = time.perf_counter() - t_start

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = _make_loss_fn(targets_info)

    times = []
    step = 0
    while step < n_steps + warmup:
        for batch in dl:
            if step >= n_steps + warmup:
                break
            systems, batch_targets, extra_data = unpack_batch(batch)
            systems, batch_targets, extra_data = batch_to(
                systems, batch_targets, extra_data, dtype=dtype, device=device,
            )
            c_ein, c_einb, c_ev, _, c_pm, c_rni, c_cf, c_si, c_nai, c_sl = (
                systems_to_batch(
                    systems, model.requested_nl, model.atomic_types,
                    model.species_to_species_index, model.cutoff_function,
                    model.cutoff_width, model.num_neighbors_adaptive,
                )
            )
            n_structures = len(systems)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            optimizer.zero_grad()
            energy, forces, stress, raw_preds = compiled_fn(
                c_ev, c_ein, c_einb, c_pm, c_rni, c_cf, c_si, c_nai,
                n_structures,
                *list(model.parameters()), *list(model.buffers()),
            )
            preds = _wrap_compiled_output(
                energy, forces, stress, raw_preds,
                model, systems, c_sl, c_si, targets_info,
            )
            preds = average_by_num_atoms(preds, systems, [])
            batch_targets = average_by_num_atoms(batch_targets, systems, [])
            loss = loss_fn(preds, batch_targets, extra_data)
            loss.backward()
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            if step >= warmup:
                times.append(t1 - t0)
            step += 1

    del model, dl
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return compile_time, times


def _ddp_worker(
    rank, world_size, compiled, n_steps, warmup, batch_size, results_dict,
):
    """DDP worker: run training steps and report timings from rank 0."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dtype = torch.float32

    from metatrain.utils.data import unpack_batch
    from metatrain.utils.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )
    from metatrain.utils.per_atom import average_by_num_atoms
    from metatrain.utils.transfer import batch_to

    model, dl, targets_info = _build_model_and_data(device, dtype, batch_size)
    model.train()

    compile_time = 0.0
    compiled_fn = None
    if compiled:
        from metatrain.pet.modules.compile import compile_pet_model

        torch._dynamo.reset()
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        compiled_fn, _, _ = compile_pet_model(model, dl, False, False)

        # Trigger codegen
        from metatrain.pet.modules.structures import systems_to_batch

        batch = next(iter(dl))
        systems, bt, ed = unpack_batch(batch)
        systems, bt, ed = batch_to(systems, bt, ed, dtype=dtype, device=device)
        c_ein, c_einb, c_ev, _, c_pm, c_rni, c_cf, c_si, c_nai, _ = systems_to_batch(
            systems, model.requested_nl, model.atomic_types,
            model.species_to_species_index, model.cutoff_function,
            model.cutoff_width, model.num_neighbors_adaptive,
        )
        compiled_fn(
            c_ev, c_ein, c_einb, c_pm, c_rni, c_cf, c_si, c_nai,
            len(systems), *list(model.parameters()), *list(model.buffers()),
        )
        torch.cuda.synchronize()
        compile_time = time.perf_counter() - t_start

    # Wrap with DDP
    ddp_model = DistributedDataParallel(model, device_ids=[device])
    raw_model = ddp_model.module

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-4)
    loss_fn = _make_loss_fn(targets_info)

    times = []
    step = 0
    while step < n_steps + warmup:
        for batch in dl:
            if step >= n_steps + warmup:
                break
            systems, batch_targets, extra_data = unpack_batch(batch)
            systems, batch_targets, extra_data = batch_to(
                systems, batch_targets, extra_data, dtype=dtype, device=device,
            )

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            optimizer.zero_grad()

            if compiled and compiled_fn is not None:
                from metatrain.pet.modules.structures import systems_to_batch
                from metatrain.pet.trainer import _wrap_compiled_output

                c_ein, c_einb, c_ev, _, c_pm, c_rni, c_cf, c_si, c_nai, c_sl = (
                    systems_to_batch(
                        systems, raw_model.requested_nl, raw_model.atomic_types,
                        raw_model.species_to_species_index,
                        raw_model.cutoff_function, raw_model.cutoff_width,
                        raw_model.num_neighbors_adaptive,
                    )
                )
                energy, forces, stress, raw_preds = compiled_fn(
                    c_ev, c_ein, c_einb, c_pm, c_rni, c_cf, c_si, c_nai,
                    len(systems),
                    *list(raw_model.parameters()), *list(raw_model.buffers()),
                )
                preds = _wrap_compiled_output(
                    energy, forces, stress, raw_preds,
                    raw_model, systems, c_sl, c_si, targets_info,
                )
                preds = average_by_num_atoms(preds, systems, [])
                batch_targets_avg = average_by_num_atoms(batch_targets, systems, [])
                loss = loss_fn(preds, batch_targets_avg, extra_data)
                loss.backward()
                # Explicit gradient sync for compiled path
                for param in raw_model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(
                            param.grad, op=dist.ReduceOp.AVG,
                        )
            else:
                from metatrain.utils.evaluate_model import evaluate_model

                preds = evaluate_model(
                    ddp_model, systems,
                    {k: targets_info[k] for k in batch_targets.keys()},
                    is_training=True,
                )
                preds = average_by_num_atoms(preds, systems, [])
                batch_targets_avg = average_by_num_atoms(batch_targets, systems, [])
                loss = loss_fn(preds, batch_targets_avg, extra_data)
                # DDP dummy touch for gradient sync
                for param in ddp_model.parameters():
                    loss += 0.0 * param.sum()
                loss.backward()

            optimizer.step()

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            if step >= warmup:
                times.append(t1 - t0)
            step += 1

    dist.destroy_process_group()

    # Only rank 0 reports
    if rank == 0:
        results_dict["compile_time"] = compile_time
        results_dict["times"] = times


def _bench_ddp(compiled, n_steps, warmup, batch_size, world_size=2):
    """Launch DDP benchmark with mp.spawn."""
    manager = mp.Manager()
    results_dict = manager.dict()
    mp.spawn(
        _ddp_worker,
        args=(world_size, compiled, n_steps, warmup, batch_size, results_dict),
        nprocs=world_size,
        join=True,
    )
    return dict(results_dict)


def main():
    parser = argparse.ArgumentParser(description="PET DDP + compile benchmark")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print("=" * 60)
    print("PET DDP + Compile Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Steps: {args.steps}, Warmup: {args.warmup}")
    print(f"Batch size: {args.batch_size}")

    results = {}

    # 1. Eager single-GPU
    print("\n--- Eager single-GPU ---")
    eager_times = _bench_single_eager(
        device, dtype, args.steps, args.warmup, args.batch_size,
    )
    results["eager_single"] = {
        "median_ms": statistics.median(eager_times) * 1000,
        "mean_ms": statistics.mean(eager_times) * 1000,
        "std_ms": statistics.stdev(eager_times) * 1000 if len(eager_times) > 1 else 0,
        "times": [t * 1000 for t in eager_times],
    }
    print(f"  Median: {results['eager_single']['median_ms']:.1f} ms")

    # 2. Compiled single-GPU
    print("\n--- Compiled single-GPU ---")
    ct, compiled_times = _bench_single_compiled(
        device, dtype, args.steps, args.warmup, args.batch_size,
    )
    results["compiled_single"] = {
        "median_ms": statistics.median(compiled_times) * 1000,
        "mean_ms": statistics.mean(compiled_times) * 1000,
        "std_ms": statistics.stdev(compiled_times) * 1000 if len(compiled_times) > 1 else 0,
        "compile_time_s": ct,
        "times": [t * 1000 for t in compiled_times],
    }
    print(f"  Median: {results['compiled_single']['median_ms']:.1f} ms")
    print(f"  Compile overhead: {ct:.1f} s")

    if device.type == "cuda":
        # 3. Eager DDP (2 ranks)
        print("\n--- Eager DDP (2 ranks, gloo) ---")
        ddp_eager = _bench_ddp(
            compiled=False, n_steps=args.steps, warmup=args.warmup,
            batch_size=args.batch_size,
        )
        results["eager_ddp"] = {
            "median_ms": statistics.median(ddp_eager["times"]) * 1000,
            "mean_ms": statistics.mean(ddp_eager["times"]) * 1000,
            "std_ms": statistics.stdev(ddp_eager["times"]) * 1000 if len(ddp_eager["times"]) > 1 else 0,
            "times": [t * 1000 for t in ddp_eager["times"]],
        }
        print(f"  Median: {results['eager_ddp']['median_ms']:.1f} ms")

        # 4. Compiled DDP (2 ranks)
        print("\n--- Compiled DDP (2 ranks, gloo) ---")
        ddp_compiled = _bench_ddp(
            compiled=True, n_steps=args.steps, warmup=args.warmup,
            batch_size=args.batch_size,
        )
        results["compiled_ddp"] = {
            "median_ms": statistics.median(ddp_compiled["times"]) * 1000,
            "mean_ms": statistics.mean(ddp_compiled["times"]) * 1000,
            "std_ms": statistics.stdev(ddp_compiled["times"]) * 1000 if len(ddp_compiled["times"]) > 1 else 0,
            "compile_time_s": ddp_compiled["compile_time"],
            "times": [t * 1000 for t in ddp_compiled["times"]],
        }
        print(f"  Median: {results['compiled_ddp']['median_ms']:.1f} ms")
        print(f"  Compile overhead: {ddp_compiled['compile_time']:.1f} s")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    e = results["eager_single"]["median_ms"]
    c = results["compiled_single"]["median_ms"]
    print(f"  Compile speedup (single GPU): {e/c:.2f}x ({e:.1f} -> {c:.1f} ms)")
    if "eager_ddp" in results:
        ed = results["eager_ddp"]["median_ms"]
        cd = results["compiled_ddp"]["median_ms"]
        print(f"  DDP overhead (eager):         {ed/e:.2f}x ({e:.1f} -> {ed:.1f} ms)")
        print(f"  DDP overhead (compiled):      {cd/c:.2f}x ({c:.1f} -> {cd:.1f} ms)")
        print(f"  Compile speedup (DDP):        {ed/cd:.2f}x ({ed:.1f} -> {cd:.1f} ms)")

    # Add metadata
    results["metadata"] = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A",
        "pytorch": torch.__version__,
        "steps": args.steps,
        "warmup": args.warmup,
        "batch_size": args.batch_size,
    }

    if args.output:
        # Strip per-step times for compact output
        compact = {}
        for k, v in results.items():
            if isinstance(v, dict) and "times" in v:
                compact[k] = {kk: vv for kk, vv in v.items() if kk != "times"}
            else:
                compact[k] = v
        with open(args.output, "w") as f:
            json.dump(compact, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
