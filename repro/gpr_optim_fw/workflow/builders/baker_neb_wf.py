"""Build + add the Baker 25-system NEB workflow.

Mirrors the eon_orchestrator + nebmmf_repro recipe: per system, prepare
inputs (IRA-aligned reactant/product, metatomic config) then run a
CI-NEB via eonclient against PET-MAD. A single harvest firetask
collects the per-system barrier + wall into ``data/baker_neb.csv``.

Sources Baker structures from the nebmmf_repro checkout.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from fireworks import Firework, LaunchPad, Workflow

from workflow.firetasks import (
    EonNebFiretask,
    FetchPetMadFiretask,
    HarvestNebFiretask,
    PrepareNebInputsFiretask,
)


# Baker 25 systems live under nebmmf_repro/eonRuns/resources/icFSM/baker/.
# Each has initial/ipath_000.con (reactant), initial/ipath_001.con
# (product), and ts.xyz (reference saddle for accuracy comparison).
DEFAULT_SOURCE = (
    Path.home()
    / "Git"
    / "Github"
    / "epfl"
    / "pixi_envs"
    / "repro"
    / "nebmmf_repro"
    / "eonRuns"
    / "resources"
    / "icFSM"
    / "baker"
)
DEFAULT_OUT_ROOT = Path.home() / "gpr-optim-fw" / "runs" / "baker_neb"


def _discover_systems(source_root: Path) -> list[tuple[str, Path, Path]]:
    out = []
    for sys_dir in sorted(source_root.iterdir()):
        if not sys_dir.is_dir():
            continue
        reactant = sys_dir / "initial" / "ipath_000.con"
        product = sys_dir / "initial" / "ipath_001.con"
        if reactant.is_file() and product.is_file():
            out.append((sys_dir.name, reactant, product))
    return out


def build(
    source_root: Path = DEFAULT_SOURCE,
    out_root: Path = DEFAULT_OUT_ROOT,
    model_name: str = "pet-mad-s-v1.5.0",
) -> Workflow:
    systems = _discover_systems(source_root)
    if not systems:
        raise FileNotFoundError(
            f"No Baker systems under {source_root}; expected ipath_000/001.con"
        )

    fetch_fw = Firework(
        FetchPetMadFiretask(model_name=model_name),
        name="fetch-petmad",
    )

    out_csv = out_root / "data" / "baker_neb.csv"
    prep_fws = []
    neb_fws = []
    for sys_id, reactant, product in systems:
        rundir = out_root / sys_id
        prep = Firework(
            PrepareNebInputsFiretask(
                system_id=sys_id,
                reactant_con=str(reactant),
                product_con=str(product),
                out_dir=str(rundir),
                model_path="{petmad_model_path}",
                use_ira=True,
                n_images=8,
            ),
            parents=[fetch_fw],
            name=f"prep-neb-{sys_id}",
        )
        # The FW spec interpolation `{petmad_model_path}` is filled by the
        # update_spec from FetchPetMadFiretask; FireWorks resolves these
        # tokens at task-launch time.
        run = Firework(
            EonNebFiretask(system_id=sys_id, rundir=str(rundir)),
            parents=[prep],
            name=f"neb-{sys_id}",
        )
        prep_fws.append(prep)
        neb_fws.append(run)

    harvest = Firework(
        HarvestNebFiretask(
            suite_name="baker_neb",
            out_csv=str(out_csv),
            system_ids=[s[0] for s in systems],
        ),
        parents=neb_fws,
        name="harvest-baker-neb",
    )

    return Workflow(
        [fetch_fw, *prep_fws, *neb_fws, harvest],
        name="baker_neb",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model-name", default="pet-mad-s-v1.5.0")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the workflow but don't add to LaunchPad",
    )
    args = parser.parse_args()

    wf = build(args.source, args.out, args.model_name)
    if args.dry_run:
        print(f"Built workflow: {wf.name}, {len(wf.fws)} fireworks")
        return

    lpad = LaunchPad.auto_load()
    fw_ids = lpad.add_wf(wf)
    print(f"Added baker_neb workflow; root fw_id={list(fw_ids.values())[0]}")


if __name__ == "__main__":
    main()
