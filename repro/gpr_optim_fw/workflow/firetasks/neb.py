"""NEB firetask: plain CI-NEB via eonclient against PET-MAD/metatomic.

GP-accelerated NEB is not in scope for this paper (gpr_optim's GP work
targets dimer / saddle search, not NEB). The NEB benchmark exists to
provide reference saddle barriers + paths against which the GP-dimer
results are compared on the Baker set.
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


_BARRIER_RE = re.compile(r"barrier:\s*([+-]?[\d.eE+-]+)\s*eV", re.IGNORECASE)


@explicit_serialize
class EonNebFiretask(FiretaskBase):
    required_params = ["system_id", "rundir"]
    optional_params = ["eonclient_binary", "extra_args"]

    def run_task(self, fw_spec):
        rundir = Path(self["rundir"])
        eon = self.get("eonclient_binary") or shutil.which("eonclient")
        if not eon:
            raise FileNotFoundError(
                "eonclient missing on PATH; activate the eon pixi env"
            )
        cmd = [eon] + list(self.get("extra_args") or [])
        log_path = rundir / "neb.log"
        t0 = time.monotonic()
        with log_path.open("w") as fout:
            proc = subprocess.run(
                cmd, cwd=str(rundir), stdout=fout, stderr=subprocess.STDOUT
            )
        wall = time.monotonic() - t0

        text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        m = _BARRIER_RE.search(text)
        barrier_eV = float(m.group(1)) if m else None
        metrics = {
            "wall_s": wall,
            "barrier_eV": barrier_eV,
            "returncode": proc.returncode,
            "log_path": str(log_path),
        }
        metrics_path = rundir / "metrics.neb.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        return FWAction(
            stored_data={"metrics": metrics},
            update_spec={
                f"neb_{self['system_id']}_metrics": str(metrics_path),
                f"neb_{self['system_id']}_returncode": proc.returncode,
            },
        )
