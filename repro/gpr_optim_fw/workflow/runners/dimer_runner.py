"""GP-accelerated dimer search runner. Invoked by the FW dimer firetask
through the gpr_optim pixi env (which has the C++ extension built and
metatomic libs on LD_LIBRARY_PATH).

Reads pos.con + direction.dat from the rundir, calls
``gpr_optim.dimer.GPDimer`` against PET-MAD via metatomic, and writes
metrics + the converged saddle geometry alongside.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _load_orient(path: Path) -> np.ndarray:
    """Read eOn-style direction.dat (one `x y z` per atom)."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.split()])
    if not rows:
        raise ValueError(f"empty direction.dat at {path}")
    return np.asarray(rows, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rundir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--hyperparameter-mode", default="inla",
                        choices=("inla", "scg"))
    parser.add_argument("--fmax", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--surrogate",
        default="laplace_slice_delta",
        help="Surrogate chain preset; see gpr_optim.surrogate.Surrogate",
    )
    args = parser.parse_args()

    from ase.io import read, write
    from metatomic.torch.ase_calculator import MetatomicCalculator
    from gpr_optim.gp import GaussianProcess
    from gpr_optim.surrogate import Surrogate
    from gpr_optim.dimer import GPDimer

    rundir = args.rundir
    pos_path = rundir / "pos.con"
    direction_path = rundir / "direction.dat"

    atoms = read(str(pos_path))
    orient = _load_orient(direction_path) if direction_path.is_file() else None

    oracle = MetatomicCalculator(str(args.model_path))

    chain = {
        "laplace_slice_delta": Surrogate.laplace_slice().delta(),
        "laplace_slice_delta_rff": Surrogate.laplace_slice().delta().rff(512),
        "laplace_slice": Surrogate.laplace_slice(),
    }.get(args.surrogate)
    if chain is None:
        raise ValueError(f"unknown surrogate preset: {args.surrogate}")

    gp = GaussianProcess(chain=chain)
    # The hyperparameter mode flag mirrors gprd's --hyperparameter-mode
    # (INLA vs SCG). The Python class exposes this through the chain
    # config; here we toggle via the env var pattern the C++ side reads.
    import os as _os
    _os.environ["GPR_HYPERPARAMETER_MODE"] = args.hyperparameter_mode

    dimer = GPDimer(atoms, oracle=oracle, gp=gp, orient=orient, seed=args.seed)

    t0 = time.monotonic()
    converged = dimer.run(fmax=args.fmax, steps=args.steps)
    wall = time.monotonic() - t0

    out_atoms = dimer.atoms
    write(str(rundir / "saddle.con"), out_atoms)

    metrics = {
        "converged": bool(converged),
        "wall_s": wall,
        "elapsed_s": dimer.elapsed_s,
        "oracle_calls": dimer.oracle_calls,
        "outer_iterations": dimer.outer_iterations,
        "energy_eV": dimer.energy_eV,
        "fmax": args.fmax,
        "hyperparameter_mode": args.hyperparameter_mode,
        "surrogate": args.surrogate,
        "saddle_con": str(rundir / "saddle.con"),
    }
    (rundir / "metrics.gprd.json").write_text(json.dumps(metrics, indent=2))
    print(f"converged={converged} wall={wall:.2f}s "
          f"oracle_calls={dimer.oracle_calls} "
          f"outer_iterations={dimer.outer_iterations}")
    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())
