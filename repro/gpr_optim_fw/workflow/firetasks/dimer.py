"""GP-accelerated dimer firetask.

Subprocesses into the gpr_optim project's pixi env (the one with the
nanobind C++ extension already built and metatomic libs on
LD_LIBRARY_PATH) and runs the GPDimer driver via
``workflow.runners.dimer_runner``. Reading the firetask in isolation:

  1. Worker pulls a READY firework off the LaunchPad.
  2. Pre-rocket has already cd'd into ~/gpr_optim_fw and activated the
     light gpr_optim_fw pixi env (fireworks + ase + rgpycrumbs).
  3. This firetask invokes ``pixi run -e scalapack python -m
     workflow.runners.dimer_runner ...`` from ``$GPR_OPTIM_REPO`` so the
     heavy env (metatomic, libtorch, vesin, capnp, scalapack) is picked
     up at the worker.
  4. The runner writes ``metrics.gprd.json`` + ``saddle.con`` to the
     rundir; the firetask reads those back, returns FWAction with the
     metrics path in update_spec for the harvest firetask to read.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from fireworks import FiretaskBase, FWAction
from fireworks.utilities.fw_utilities import explicit_serialize


@explicit_serialize
class GprdDimerFiretask(FiretaskBase):
    required_params = ["system_id", "rundir", "model_path"]
    optional_params = [
        "hyperparameter_mode",
        "surrogate",
        "fmax",
        "steps",
        "seed",
        "gpr_optim_repo",
    ]

    def run_task(self, fw_spec):
        rundir = Path(self["rundir"])
        rundir.mkdir(parents=True, exist_ok=True)
        repo = Path(
            self.get("gpr_optim_repo")
            or os.environ.get("GPR_OPTIM_REPO", str(Path.home() / "gpr_optim"))
        )

        cmd = [
            "pixi", "run", "-e", "scalapack",
            "python", "-m", "workflow.runners.dimer_runner",
            "--rundir", str(rundir),
            "--model-path", str(self["model_path"]),
            "--hyperparameter-mode", self.get("hyperparameter_mode") or "inla",
            "--surrogate", self.get("surrogate") or "laplace_slice_delta",
            "--fmax", str(self.get("fmax") or 0.005),
            "--steps", str(self.get("steps") or 2000),
            "--seed", str(self.get("seed") or 42),
        ]

        # The runner package lives in ~/gpr_optim_fw but the heavy pixi
        # env lives under ~/gpr_optim/.pixi/envs/scalapack. Point pixi
        # at the gpr_optim project and prepend the gpr_optim_fw root to
        # PYTHONPATH so `python -m workflow.runners.dimer_runner`
        # resolves either way (whether or not the package is pip-
        # installed into the scalapack env).
        env = dict(os.environ)
        gpr_optim_fw_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = (
            f"{gpr_optim_fw_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
        )

        log_path = rundir / "gprd.log"
        t0 = time.monotonic()
        with log_path.open("w") as fout:
            proc = subprocess.run(
                cmd, cwd=str(repo), env=env,
                stdout=fout, stderr=subprocess.STDOUT,
            )
        wall = time.monotonic() - t0

        metrics_path = rundir / "metrics.gprd.json"
        metrics = {}
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
        metrics["firetask_wall_s"] = wall
        metrics["firetask_returncode"] = proc.returncode
        metrics["log_path"] = str(log_path)
        metrics_path.write_text(json.dumps(metrics, indent=2))

        return FWAction(
            stored_data={"metrics": metrics},
            update_spec={
                f"system_{self['system_id']}_gprd_metrics": str(metrics_path),
                f"system_{self['system_id']}_gprd_returncode": proc.returncode,
            },
        )


@explicit_serialize
class EonDimerBaselineFiretask(FiretaskBase):
    """Plain eOn dimer baseline (every step calls the oracle).

    Currently a stub: the eonclient binary on elja is built without
    metatomic support, so the baseline branch is parked behind this
    firetask but not invoked from the production builders. When a
    metatomic-enabled eonclient is available, this firetask runs the
    same rundir through plain eOn dimer and writes
    ``metrics.eon_baseline.json``.
    """

    required_params = ["system_id", "rundir"]

    def run_task(self, fw_spec):
        rundir = Path(self["rundir"])
        metrics = {
            "skipped": True,
            "reason": "eonclient on elja lacks metatomic support; "
                      "build a metatomic-enabled eonclient to enable this branch",
        }
        metrics_path = rundir.parent / f"{rundir.name}__baseline_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        return FWAction(
            stored_data={"metrics": metrics},
            update_spec={
                f"system_{self['system_id']}_eon_metrics": str(metrics_path),
                f"system_{self['system_id']}_eon_returncode": 0,
            },
        )
