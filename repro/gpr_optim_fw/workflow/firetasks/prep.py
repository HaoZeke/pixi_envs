"""Input-prep firetasks shared across the Baker NEB and Hermes dimer suites.

These run in single-shot mode on the elja login node (no SLURM) since they
are I/O-bound: model fetches, ASE-side endpoint alignment, configuration
template materialisation. The compute-heavy follow-ons live in
``dimer.py`` / ``neb.py``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from fireworks import FiretaskBase, FWAction
from fireworks.utilities.fw_utilities import explicit_serialize


@explicit_serialize
class FetchPetMadFiretask(FiretaskBase):
    """Resolve the PET-MAD model file under PETMAD_MODEL_DIR.

    The fworker env points PETMAD_MODEL_DIR at a NFS path. If the requested
    model file is already there with the right SHA256 we no-op; otherwise we
    download it via metatomic's HuggingFace helper. The file is host-shared
    so subsequent dimer / NEB firetasks read it without re-downloading.
    """

    required_params = ["model_name"]
    optional_params = ["sha256", "model_dir"]

    def run_task(self, fw_spec):
        model_name = self["model_name"]
        model_dir = Path(
            self.get("model_dir") or os.environ.get(
                "PETMAD_MODEL_DIR", str(Path.home() / "gpr-optim-fw" / "models")
            )
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        target = model_dir / f"{model_name}.pt"

        want_sha = self.get("sha256")
        if target.is_file():
            if want_sha:
                got = _sha256_of(target)
                if got != want_sha:
                    target.unlink()
                else:
                    return FWAction(
                        update_spec={"petmad_model_path": str(target)}
                    )
            else:
                # No SHA pinning: trust the staged file. This is the path
                # used when the model is pre-positioned by hand (no
                # compute-node internet on most HPC sites).
                return FWAction(update_spec={"petmad_model_path": str(target)})

        # Use the metatomic-bundled helper rather than raw HF, so the network
        # path matches what eonclient + gprd will use at runtime.
        from metatomic.torch import load_atomistic_model  # type: ignore

        # load_atomistic_model returns an in-memory model; we want it on disk.
        # Fallback: invoke the HF cache directly.
        from huggingface_hub import hf_hub_download  # type: ignore

        repo_id = "lab-cosmo/pet-mad"
        filename = f"{model_name}.pt"
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy2(downloaded, target)

        if want_sha:
            got = _sha256_of(target)
            if got != want_sha:
                raise RuntimeError(
                    f"PET-MAD model SHA mismatch: want={want_sha} got={got}"
                )

        return FWAction(update_spec={"petmad_model_path": str(target)})


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@explicit_serialize
class PrepareDimerInputsFiretask(FiretaskBase):
    """Materialise the per-system rundir for a dimer search.

    Inputs land at ``out_dir`` with filenames the gprd and eonclient binaries
    expect: ``pos.con`` (initial), ``direction.dat`` (initial dimer axis),
    optional ``displacement.con``, plus a ``config.ini`` for eonclient.

    For metatomic-backed runs we write a ``potential.metatomic.toml`` block
    pointing at PETMAD_MODEL_DIR/<model_name>.pt.
    """

    required_params = [
        "system_id",
        "reactant_con",
        "out_dir",
        "model_path",
    ]
    optional_params = ["initial_direction", "displacement_con", "config_overrides"]

    def run_task(self, fw_spec):
        out_dir = Path(self["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(self["reactant_con"], out_dir / "pos.con")

        direction = self.get("initial_direction") or fw_spec.get("initial_direction")
        if direction:
            shutil.copy2(direction, out_dir / "direction.dat")

        disp = self.get("displacement_con") or fw_spec.get("displacement_con")
        if disp:
            shutil.copy2(disp, out_dir / "displacement.con")

        model_path = self["model_path"]
        config_overrides = self.get("config_overrides", {}) or {}
        _write_eon_config(out_dir / "config.ini", model_path, config_overrides)

        return FWAction(
            update_spec={
                f"system_{self['system_id']}_dir": str(out_dir),
            }
        )


def _write_eon_config(path: Path, model_path: str, overrides: dict):
    """Minimal eOn config.ini for a metatomic-backed dimer search.

    Wired identically to gpr_optim's torch_native + dimer params; the worker's
    eonclient binary picks up the metatomic potential block.
    """
    base = {
        "Main": {
            "job": "saddle_search",
            "potential": "metatomic",
            "temperature": "300",
        },
        "Saddle Search": {
            "method": "min_mode",
            "min_mode_method": "dimer",
            "max_iterations": "5000",
            "displace_type": "load",
        },
        "Dimer": {
            "rotations_max": "12",
            "torque_min": "0.001",
        },
        "Optimizer": {
            "opt_method": "cg",
            "max_iterations": "1000",
            "converged_force": "0.005",
        },
        "Metatomic": {
            "model_path": model_path,
            "device": "cpu",
            "extensions_directory": "",
        },
    }
    for section, items in overrides.items():
        base.setdefault(section, {}).update(items)

    with path.open("w") as f:
        for section, items in base.items():
            f.write(f"[{section}]\n")
            for key, value in items.items():
                f.write(f"{key} = {value}\n")
            f.write("\n")


@explicit_serialize
class PrepareNebInputsFiretask(FiretaskBase):
    """Materialise per-system NEB rundir (reactant + product + IDPP path)."""

    required_params = [
        "system_id",
        "reactant_con",
        "product_con",
        "out_dir",
        "model_path",
    ]
    optional_params = ["n_images", "use_ira", "config_overrides"]

    def run_task(self, fw_spec):
        out_dir = Path(self["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(self["reactant_con"], out_dir / "reactant.con")
        shutil.copy2(self["product_con"], out_dir / "product.con")

        n_images = int(self.get("n_images", 8))
        use_ira = bool(self.get("use_ira", True))

        if use_ira:
            # IRA alignment via the same rgpycrumbs.geom.api.alignment
            # entry point eon_orchestrator's align_endpoints rule uses.
            # Standardise the cell to [25, 25, 25] + center first so the
            # potential evaluator (PET-MAD via metatomic) sees the same
            # box geometry every system.
            import ase.io
            from rgpycrumbs.geom.api.alignment import (
                IRAConfig,
                align_structure_robust,
            )

            reactant_atm = ase.io.read(str(out_dir / "reactant.con"))
            product_atm = ase.io.read(str(out_dir / "product.con"))
            for atm in (reactant_atm, product_atm):
                atm.set_cell([25, 25, 25])
                atm.center()

            cfg = IRAConfig(enabled=True, kmax=1.8)
            aligned = align_structure_robust(reactant_atm, product_atm, cfg)
            ase.io.write(str(out_dir / "reactant.con"), reactant_atm)
            ase.io.write(str(out_dir / "product.con"), aligned.atoms)

        model_path = self["model_path"]
        config_overrides = self.get("config_overrides", {}) or {}
        _write_neb_config(
            out_dir / "config.ini", model_path, n_images, config_overrides
        )

        return FWAction(
            update_spec={f"neb_{self['system_id']}_dir": str(out_dir)}
        )


def _write_neb_config(path: Path, model_path: str, n_images: int, overrides: dict):
    base = {
        "Main": {
            "job": "nudged_elastic_band",
            "potential": "metatomic",
        },
        "Nudged Elastic Band": {
            "images": str(n_images),
            "spring": "5.0",
            "max_iterations": "1000",
            "converged_force": "0.005",
            "climbing_image_method": "true",
        },
        "Optimizer": {
            "opt_method": "cg",
            "max_iterations": "2000",
            "converged_force": "0.005",
        },
        "Metatomic": {
            "model_path": model_path,
            "device": "cpu",
        },
    }
    for section, items in overrides.items():
        base.setdefault(section, {}).update(items)
    with path.open("w") as f:
        for section, items in base.items():
            f.write(f"[{section}]\n")
            for key, value in items.items():
                f.write(f"{key} = {value}\n")
            f.write("\n")
