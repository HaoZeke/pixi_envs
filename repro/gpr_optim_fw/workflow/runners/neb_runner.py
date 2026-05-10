"""GP-accelerated NEB runner using ``gpr_optim.neb.GPNEB``.

Reads reactant.con + product.con, builds an N-image chain, runs CI-NEB
against PET-MAD via metatomic. Writes a metrics JSON + the converged
chain (path.traj + per-image .con) into the rundir.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rundir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--n-intermediate", type=int, default=8)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--no-climb", action="store_true")
    parser.add_argument("--spring-constant", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--surrogate",
        default="laplace_slice_delta",
    )
    args = parser.parse_args()

    from ase.io import read, write
    from metatomic.torch.ase_calculator import MetatomicCalculator
    from gpr_optim.gp import GaussianProcess
    from gpr_optim.surrogate import Surrogate
    from gpr_optim.neb import GPNEB

    rundir = args.rundir
    reactant = read(str(rundir / "reactant.con"))
    product = read(str(rundir / "product.con"))
    images = [reactant] + [reactant.copy() for _ in range(args.n_intermediate)] + [product]

    oracle = MetatomicCalculator(str(args.model_path))
    chain = {
        "laplace_slice_delta": Surrogate.laplace_slice().delta(),
        "laplace_slice_delta_rff": Surrogate.laplace_slice().delta().rff(512),
        "laplace_slice": Surrogate.laplace_slice(),
    }.get(args.surrogate)
    if chain is None:
        raise ValueError(f"unknown surrogate preset: {args.surrogate}")

    gp = GaussianProcess(chain=chain)
    neb = GPNEB(
        images,
        oracle=oracle,
        gp=gp,
        climb=not args.no_climb,
        spring_constant=args.spring_constant,
        seed=args.seed,
    )
    neb.interpolate("idpp")

    t0 = time.monotonic()
    converged = neb.run(fmax=args.fmax, steps=args.steps)
    wall = time.monotonic() - t0

    for i, img in enumerate(neb.images):
        write(str(rundir / f"image_{i:02d}.con"), img)

    metrics = {
        "converged": bool(converged),
        "wall_s": wall,
        "elapsed_s": neb.elapsed_s,
        "oracle_calls": neb.oracle_calls,
        "outer_iterations": neb.outer_iterations,
        "max_energy_image": neb.max_energy_image,
        "barrier_eV": neb.barrier_eV,
        "n_images": neb.n_images,
        "fmax": args.fmax,
        "surrogate": args.surrogate,
    }
    (rundir / "metrics.gpneb.json").write_text(json.dumps(metrics, indent=2))
    print(f"converged={converged} wall={wall:.2f}s "
          f"oracle_calls={neb.oracle_calls} "
          f"barrier_eV={neb.barrier_eV:.4f}")
    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())
