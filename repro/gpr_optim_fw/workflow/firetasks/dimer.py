"""Dimer-search firetasks: GP-accelerated (gprd) and plain-eOn baseline.

Both record wall, image-eval count, final saddle position, and converged
force. The harvest firetask reads these to assemble the comparison CSV.

Run shape:
- ``GprdDimerFiretask`` invokes the gpr_optim ``gprd`` binary with
  ``--potential metatomic`` and ``--model <PETMAD_MODEL>``. SLURM rank
  count is taken from ``$SLURM_NTASKS`` (set by qadapter) so a single
  qadapter line gives both np=1 dev runs and np=8 production runs.
- ``EonDimerBaselineFiretask`` invokes the eOn ``eonclient`` binary with
  the same metatomic config. No GP, no acceleration -- every step calls
  the PET-MAD oracle.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from fireworks import FiretaskBase, FWAction
from fireworks.utilities.fw_utilities import explicit_serialize


_ELAPSED_RE = re.compile(r"Elapsed time:\s*([\d.]+)\s*s", re.IGNORECASE)
_IMAGE_EVALS_RE = re.compile(
    r"total number of image evaluations:\s*(\d+)", re.IGNORECASE
)
_PHASES_RE = re.compile(r"convergence obtained after\s+(\d+)\s+relaxation phases")


@explicit_serialize
class GprdDimerFiretask(FiretaskBase):
    """Run gprd on the prepared rundir."""

    required_params = ["system_id", "rundir", "model_path"]
    optional_params = [
        "gprd_binary",
        "extra_args",
        "hyperparameter_mode",
        "uncertainty_mode",
        "ranks",
    ]

    def run_task(self, fw_spec):
        rundir = Path(self["rundir"])
        gprd = self.get("gprd_binary") or os.environ.get(
            "GPR_OPTIM_GPRD",
            str(Path.home() / "gpr_optim" / "build-scalapack-real" / "gprd"),
        )
        if not Path(gprd).is_file():
            raise FileNotFoundError(f"gprd binary missing: {gprd}")

        ranks = int(self.get("ranks") or os.environ.get("SLURM_NTASKS", "1"))
        cmd = [
            "mpirun", "-np", str(ranks),
            gprd,
            "--pos", str(rundir / "pos.con"),
            "--orient", str(rundir / "direction.dat"),
            "--config", str(rundir / "config.ini"),
            "--potential", "metatomic",
            "--model", self["model_path"],
            "--hyperparameter-mode", self.get("hyperparameter_mode") or "inla",
            "--uncertainty-mode", self.get("uncertainty_mode") or "map",
        ]
        cmd += list(self.get("extra_args") or [])

        log_path = rundir / "gprd.log"
        t0 = time.monotonic()
        with log_path.open("w") as fout:
            proc = subprocess.run(
                cmd, cwd=str(rundir), stdout=fout, stderr=subprocess.STDOUT
            )
        wall = time.monotonic() - t0

        metrics = _parse_metrics(log_path, wall, proc.returncode, ranks)
        metrics_path = rundir / "metrics.gprd.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))

        if proc.returncode != 0:
            return FWAction(
                stored_data={"metrics": metrics},
                defuse_children=False,
                update_spec={
                    f"system_{self['system_id']}_gprd_metrics": str(metrics_path),
                    f"system_{self['system_id']}_gprd_returncode": proc.returncode,
                },
            )

        return FWAction(
            stored_data={"metrics": metrics},
            update_spec={
                f"system_{self['system_id']}_gprd_metrics": str(metrics_path),
                f"system_{self['system_id']}_gprd_returncode": 0,
            },
        )


@explicit_serialize
class EonDimerBaselineFiretask(FiretaskBase):
    """Run eonclient (no GP) on the same prepared rundir."""

    required_params = ["system_id", "rundir"]
    optional_params = ["eonclient_binary", "extra_args"]

    def run_task(self, fw_spec):
        rundir = Path(self["rundir"])
        # Copy inputs into a parallel baseline directory so the rundir's
        # results.dat / log files don't collide between gp and baseline.
        base_dir = rundir.parent / f"{rundir.name}__baseline"
        if base_dir.exists():
            shutil.rmtree(base_dir)
        shutil.copytree(rundir, base_dir)

        eon = self.get("eonclient_binary") or shutil.which("eonclient")
        if not eon:
            raise FileNotFoundError(
                "eonclient not on PATH; pixi run -e eon eonclient first"
            )

        cmd = [eon] + list(self.get("extra_args") or [])
        log_path = base_dir / "eon.log"
        t0 = time.monotonic()
        with log_path.open("w") as fout:
            proc = subprocess.run(
                cmd, cwd=str(base_dir), stdout=fout, stderr=subprocess.STDOUT
            )
        wall = time.monotonic() - t0

        metrics = _parse_metrics(log_path, wall, proc.returncode, 1)
        metrics_path = base_dir / "metrics.eon_baseline.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))

        return FWAction(
            stored_data={"metrics": metrics},
            update_spec={
                f"system_{self['system_id']}_eon_metrics": str(metrics_path),
                f"system_{self['system_id']}_eon_returncode": proc.returncode,
            },
        )


def _parse_metrics(log_path: Path, wall: float, returncode: int, ranks: int) -> dict:
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    elapsed = _first(_ELAPSED_RE, text)
    image_evals = _first_int(_IMAGE_EVALS_RE, text)
    phases = _first_int(_PHASES_RE, text)
    return {
        "wall_s": wall,
        "elapsed_s": elapsed,
        "image_evals": image_evals,
        "relaxation_phases": phases,
        "returncode": returncode,
        "ranks": ranks,
        "log_path": str(log_path),
    }


def _first(rx, text):
    m = rx.search(text)
    return float(m.group(1)) if m else None


def _first_int(rx, text):
    m = rx.search(text)
    return int(m.group(1)) if m else None
