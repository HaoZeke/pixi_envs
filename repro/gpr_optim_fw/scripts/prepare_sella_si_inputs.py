"""Convert the Hermes/Sella `sella_si_data.zip` payload into per-system
input dirs the gpr_optim_fw Hermes dimer firetask expects.

Source: gprd_zbl/data/sella_si_data.zip
   sella_si/{singlets,doublets}/<NNN>.xyz
Target: $OUT_ROOT/{singlet,doublet}/<NNN>/
   pos.con          (initial geometry, .con format eOn reads)
   direction.dat    (unit dimer axis)
   displacement.con (small displacement applied along direction)

Default direction: random Gaussian unit vector seeded by system id, so
different runs reproduce the same starting axis. The dimer search will
rotate this to the lowest-curvature mode in the first few rotation
iterations regardless of starting choice.

Usage:
  python -m scripts.prepare_sella_si_inputs \
      --zip /path/to/sella_si_data.zip --out ~/gpr-optim-fw/inputs/hermes
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from ase.io import read, write


def _unit_vector_for_system(spin: str, idx: str) -> np.ndarray:
    seed = int(hashlib.sha256(f"{spin}_{idx}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def _write_direction_dat(path: Path, n_atoms: int, axis: np.ndarray):
    # eOn direction.dat: one line "x y z" per atom of the moving subsystem.
    # All atoms get the same axis projection (axis is global; per-atom
    # displacement amplitude is uniform when generated this way).
    with path.open("w") as f:
        for _ in range(n_atoms):
            f.write(f"{axis[0]:.16e} {axis[1]:.16e} {axis[2]:.16e}\n")


def _write_displacement_con(
    src_atoms, axis: np.ndarray, magnitude: float, path: Path
):
    disp = src_atoms.copy()
    pos = disp.get_positions()
    pos += magnitude * axis  # uniform displacement along axis
    disp.set_positions(pos)
    write(str(path), disp, format="eon")


def convert(zip_path: Path, out_root: Path, magnitude: float = 0.01) -> int:
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    out_root.mkdir(parents=True, exist_ok=True)

    n_systems = 0
    with zipfile.ZipFile(zip_path) as zf:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            zf.extractall(tmp)
            for spin_in, spin_out in (("singlets", "singlet"), ("doublets", "doublet")):
                src = tmp / "sella_si" / spin_in
                if not src.is_dir():
                    continue
                for xyz in sorted(src.glob("*.xyz")):
                    idx = xyz.stem
                    sys_dir = out_root / spin_out / idx
                    sys_dir.mkdir(parents=True, exist_ok=True)
                    atoms = read(str(xyz))
                    write(str(sys_dir / "pos.con"), atoms, format="eon")
                    axis = _unit_vector_for_system(spin_out, idx)
                    _write_direction_dat(
                        sys_dir / "direction.dat", len(atoms), axis
                    )
                    _write_displacement_con(
                        atoms, axis, magnitude, sys_dir / "displacement.con"
                    )
                    n_systems += 1

    return n_systems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        type=Path,
        required=True,
        help="Path to sella_si_data.zip",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "gpr-optim-fw" / "inputs" / "hermes",
    )
    parser.add_argument(
        "--magnitude",
        type=float,
        default=0.01,
        help="Displacement amplitude in Angstrom (default 0.01)",
    )
    args = parser.parse_args()
    n = convert(args.zip, args.out, args.magnitude)
    print(f"prepared {n} Hermes systems under {args.out}")


if __name__ == "__main__":
    main()
