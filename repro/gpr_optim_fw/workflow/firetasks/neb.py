"""GP-accelerated NEB firetask.

Subprocesses into the gpr_optim project's pixi env and runs
``workflow.runners.neb_runner``. Mirrors the dimer firetask shape;
writes ``metrics.gpneb.json`` + per-image .con files to the rundir.

The previous EonNebFiretask invoked plain eonclient against
metatomic, but the elja eonclient build has metatomic disabled, so
that path was retired. The GP-accelerated NEB on the Baker set is
what the paper compares to literature DFT NEB barriers; a plain eOn
baseline can be re-added once a metatomic-enabled eonclient exists.
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
class EonNebFiretask(FiretaskBase):
    """GP-accelerated NEB via gpr_optim's GPNEB driver.

    Class name kept for backward compatibility with the existing
    builders; the implementation is GP-accelerated, not plain eOn.
    """

    required_params = ["system_id", "rundir"]
    optional_params = [
        "model_path",
        "n_intermediate",
        "fmax",
        "steps",
        "no_climb",
        "spring_constant",
        "seed",
        "surrogate",
        "gpr_optim_repo",
    ]

    def run_task(self, fw_spec):
        rundir = Path(self["rundir"])
        rundir.mkdir(parents=True, exist_ok=True)
        repo = Path(
            self.get("gpr_optim_repo")
            or os.environ.get("GPR_OPTIM_REPO", str(Path.home() / "gpr_optim"))
        )
        model_path = self.get("model_path") or fw_spec.get("petmad_model_path")
        if not model_path:
            raise RuntimeError("model_path missing on firetask + fw_spec")

        cmd = [
            "pixi", "run", "-e", "scalapack",
            "python", "-m", "workflow.runners.neb_runner",
            "--rundir", str(rundir),
            "--model-path", str(model_path),
            "--n-intermediate", str(self.get("n_intermediate") or 8),
            "--fmax", str(self.get("fmax") or 0.05),
            "--steps", str(self.get("steps") or 500),
            "--spring-constant", str(self.get("spring_constant") or 0.1),
            "--seed", str(self.get("seed") or 42),
            "--surrogate", self.get("surrogate") or "laplace_slice_delta",
        ]
        if self.get("no_climb"):
            cmd.append("--no-climb")

        env = dict(os.environ)
        gpr_optim_fw_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = (
            f"{gpr_optim_fw_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
        )

        log_path = rundir / "neb.log"
        t0 = time.monotonic()
        with log_path.open("w") as fout:
            proc = subprocess.run(
                cmd, cwd=str(repo), env=env,
                stdout=fout, stderr=subprocess.STDOUT,
            )
        wall = time.monotonic() - t0

        metrics_path = rundir / "metrics.gpneb.json"
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
                f"neb_{self['system_id']}_metrics": str(metrics_path),
                f"neb_{self['system_id']}_returncode": proc.returncode,
            },
        )
