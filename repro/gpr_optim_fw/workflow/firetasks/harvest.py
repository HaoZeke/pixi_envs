"""Harvest firetasks: aggregate per-system metrics into one CSV per suite."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from fireworks import FiretaskBase, FWAction
from fireworks.utilities.fw_utilities import explicit_serialize


@explicit_serialize
class HarvestDimerFiretask(FiretaskBase):
    required_params = ["suite_name", "out_csv", "system_ids"]

    def run_task(self, fw_spec):
        rows = []
        for sys_id in self["system_ids"]:
            row = {"system_id": sys_id}
            for kind in ("gprd", "eon"):
                metrics_path = fw_spec.get(f"system_{sys_id}_{kind}_metrics")
                if not metrics_path:
                    continue
                metrics = json.loads(Path(metrics_path).read_text())
                for k, v in metrics.items():
                    row[f"{kind}_{k}"] = v
                row[f"{kind}_returncode"] = fw_spec.get(
                    f"system_{sys_id}_{kind}_returncode"
                )
            rows.append(row)

        out_csv = Path(self["out_csv"])
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            keys = sorted({k for r in rows for k in r.keys()})
            with out_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow(r)

        return FWAction(stored_data={"n_rows": len(rows), "out_csv": str(out_csv)})


@explicit_serialize
class HarvestNebFiretask(FiretaskBase):
    required_params = ["suite_name", "out_csv", "system_ids"]

    def run_task(self, fw_spec):
        rows = []
        for sys_id in self["system_ids"]:
            metrics_path = fw_spec.get(f"neb_{sys_id}_metrics")
            if not metrics_path:
                continue
            metrics = json.loads(Path(metrics_path).read_text())
            row = {"system_id": sys_id, **{f"neb_{k}": v for k, v in metrics.items()}}
            row["neb_returncode"] = fw_spec.get(f"neb_{sys_id}_returncode")
            rows.append(row)

        out_csv = Path(self["out_csv"])
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            keys = sorted({k for r in rows for k in r.keys()})
            with out_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow(r)

        return FWAction(stored_data={"n_rows": len(rows), "out_csv": str(out_csv)})
