#!/usr/bin/env python3
"""GROMACS vs LAMMPS metatomic benchmark comparison.

Runs both engines on the same water systems and reports timing.
Requires: gmx (metatomic), lmp (metatomic), python with ase + metatrain.

Usage:
    python run_comparison.py --sizes 216 --nsteps 100 --engines gmx_cpu,lmp
    python run_comparison.py --sizes 216,512 --nsteps 500 --engines gmx_cpu,lmp,gmx_gpu,lmp_kk
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

INPUT_DIR = Path(__file__).parent / "input"


def export_model(output_path):
    """Export PET-MAD model if not already present."""
    if output_path.exists():
        return
    from metatrain.cli.export import export_model as _export
    _export(
        path="https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-mad-s-v1.0.2.ckpt",
        output=output_path,
    )


def prepare_water_box(n_mol, workdir, gmx_bin="gmx"):
    """Create a water box with n_mol molecules using gmx solvate."""
    v_spce = 0.0304  # nm^3 per molecule
    L = (n_mol * v_spce) ** (1 / 3)

    shutil.copy2(INPUT_DIR / "topol.top", workdir / "topol.top")

    subprocess.run(
        [gmx_bin, "solvate", "-o", "conf.gro", "-p", "topol.top",
         "-box", str(L), str(L), str(L), "-maxsol", str(n_mol), "-scale", "0.48"],
        cwd=workdir, check=True, capture_output=True,
    )
    return workdir / "conf.gro"


def run_gmx(workdir, n_mol, nsteps, gmx_bin, nranks=1, gpu=False):
    """Run GROMACS metatomic and return ms/step."""
    shutil.copy2(INPUT_DIR / "grompp.mdp", workdir / "grompp.mdp")

    # Patch nsteps
    mdp = workdir / "grompp.mdp"
    text = mdp.read_text()
    text = re.sub(r"^nsteps\s*=.*$", f"nsteps = {nsteps}", text, flags=re.MULTILINE)
    mdp.write_text(text)

    # grompp
    result = subprocess.run(
        [gmx_bin, "grompp", "-f", "grompp.mdp", "-c", "conf.gro",
         "-p", "topol.top", "-maxwarn", "10"],
        cwd=workdir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  grompp FAILED: {result.stderr[-500:]}", file=sys.stderr)
        return None

    # mdrun
    env = os.environ.copy()
    env["GMX_METATOMIC_TIMER"] = "1"
    if gpu:
        env["GMX_METATOMIC_DEVICE"] = "cuda"
    else:
        env["GMX_METATOMIC_DEVICE"] = "cpu"

    cmd = [gmx_bin, "mdrun", "-ntomp", "4", "-ntmpi", str(nranks),
           "-g", "md.log", "-pin", "off", "-nsteps", str(nsteps)]

    t0 = time.time()
    result = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True)
    wall = time.time() - t0

    if result.returncode != 0:
        print(f"  GROMACS FAILED: {result.stderr[-500:]}", file=sys.stderr)
        return None

    # Parse timer from stderr (metatomic timer output)
    ms_per_step = wall * 1000 / nsteps
    # Try to get more accurate from timer log
    timer_file = workdir / "metatomic_timer.log"
    if timer_file.exists():
        for line in timer_file.read_text().splitlines():
            if "calculateForces" in line:
                m = re.search(r"avg=([\d.]+)", line)
                if m:
                    ms_per_step = float(m.group(1))
    return ms_per_step


def run_lmp(workdir, nsteps, kokkos=False):
    """Run LAMMPS metatomic and return ms/step."""
    src = INPUT_DIR / ("lmp.in" if kokkos else "lmp_cpu.in")
    shutil.copy2(src, workdir / "lmp.in")

    # Patch nsteps in lmp.in
    lmp_in = workdir / "lmp.in"
    text = lmp_in.read_text()
    text = re.sub(r"^run\s+\d+", f"run {nsteps}", text, flags=re.MULTILINE)
    lmp_in.write_text(text)

    env = os.environ.copy()
    env["LAMMPS_METATOMIC_PROFILE"] = "1"

    cmd = ["lmp", "-in", "lmp.in"]
    if kokkos:
        cmd = ["lmp", "-k", "on", "g", "1", "-sf", "kk", "-in", "lmp.in"]

    t0 = time.time()
    result = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True)
    wall = time.time() - t0

    if result.returncode != 0:
        print(f"  LAMMPS FAILED: {result.stderr[-500:]}", file=sys.stderr)
        return None

    ms_per_step = wall * 1000 / nsteps

    # Parse LAMMPS timer from log
    for line in result.stdout.splitlines():
        m = re.match(r"Loop time of ([\d.]+) on", line)
        if m:
            loop_time = float(m.group(1))
            ms_per_step = loop_time * 1000 / nsteps
    return ms_per_step


def main():
    parser = argparse.ArgumentParser(description="GROMACS vs LAMMPS metatomic benchmark")
    parser.add_argument("--sizes", default="216", help="Comma-separated water molecule counts")
    parser.add_argument("--nsteps", type=int, default=100, help="MD steps per run")
    parser.add_argument("--engines", default="gmx_cpu,lmp",
                        help="Comma-separated: gmx_cpu,gmx_gpu,lmp,lmp_kk")
    parser.add_argument("--gmx-bin", default=os.environ.get("GMX_BIN", "gmx"))
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each benchmark")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    engines = args.engines.split(",")

    # Export model once
    model_path = INPUT_DIR / "pet-mad-v1.0.2.pt"
    print(f"Exporting model to {model_path}...")
    export_model(model_path)

    print(f"\nBenchmark: {args.nsteps} steps, {args.repeats} repeat(s)")
    print(f"Engines: {engines}")
    print(f"Sizes: {sizes} water molecules\n")

    # Header
    header = f"{'Atoms':>8}"
    for eng in engines:
        header += f" | {eng:>12}"
    print(header)
    print("-" * len(header))

    for n_mol in sizes:
        n_atoms = n_mol * 3
        row = f"{n_atoms:>8}"

        for eng in engines:
            times = []
            for _ in range(args.repeats):
                tmpdir = Path(tempfile.mkdtemp(prefix=f"bench_{eng}_{n_mol}_"))
                try:
                    # Prepare water box
                    prepare_water_box(n_mol, tmpdir, gmx_bin=args.gmx_bin)

                    # Symlink model
                    (tmpdir / "pet-mad-v1.0.2.pt").symlink_to(model_path)

                    if eng == "gmx_cpu":
                        t = run_gmx(tmpdir, n_mol, args.nsteps, args.gmx_bin, nranks=1, gpu=False)
                    elif eng == "gmx_gpu":
                        t = run_gmx(tmpdir, n_mol, args.nsteps, args.gmx_bin, nranks=1, gpu=True)
                    elif eng == "gmx_dd4":
                        t = run_gmx(tmpdir, n_mol, args.nsteps, args.gmx_bin, nranks=4, gpu=False)
                    elif eng == "lmp":
                        # Convert gro to lammps data
                        from ase.io import read, write
                        atoms = read(tmpdir / "conf.gro")
                        write(tmpdir / "water.data", atoms, format="lammps-data", masses=True)
                        t = run_lmp(tmpdir, args.nsteps, kokkos=False)
                    elif eng == "lmp_kk":
                        from ase.io import read, write
                        atoms = read(tmpdir / "conf.gro")
                        write(tmpdir / "water.data", atoms, format="lammps-data", masses=True)
                        t = run_lmp(tmpdir, args.nsteps, kokkos=True)
                    else:
                        t = None
                        print(f"  Unknown engine: {eng}", file=sys.stderr)

                    if t is not None:
                        times.append(t)
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)

            if times:
                avg = sum(times) / len(times)
                row += f" | {avg:>9.1f} ms"
            else:
                row += f" | {'FAIL':>12}"

        print(row)

    print("\nms/step (lower is better)")


if __name__ == "__main__":
    main()
