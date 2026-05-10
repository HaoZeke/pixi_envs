"""Smoke-test workflow: d016 cyclopentadienyl dimer at a few rank counts.

Quickest way to validate the FW pipeline end-to-end before queueing the
full Baker / Hermes suites. Reuses gpr_optim's bench_data/petmad/d016_*
structures so the inputs match the wall numbers in the JCIM paper draft.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fireworks import Firework, LaunchPad, Workflow

from workflow.firetasks import (
    EonDimerBaselineFiretask,
    FetchPetMadFiretask,
    GprdDimerFiretask,
    HarvestDimerFiretask,
    PrepareDimerInputsFiretask,
)


DEFAULT_SOURCE = (
    Path.home() / "gpr_optim" / "bench_data" / "petmad"
)
DEFAULT_OUT_ROOT = Path.home() / "gpr-optim-fw" / "runs" / "d016_smoke"


def build(
    source_dir: Path = DEFAULT_SOURCE,
    out_root: Path = DEFAULT_OUT_ROOT,
    model_name: str = "pet-mad-s-v1.5.0",
    rank_sweep: tuple[int, ...] = (1, 8, 16),
    include_baseline: bool = True,
) -> Workflow:
    pos = source_dir / "d016_pos.con"
    direction = source_dir / "d016_direction.dat"
    if not pos.is_file() or not direction.is_file():
        raise FileNotFoundError(
            f"d016 inputs missing under {source_dir}; expected d016_pos.con + "
            "d016_direction.dat"
        )

    fetch_fw = Firework(
        FetchPetMadFiretask(model_name=model_name),
        name="fetch-petmad",
    )

    out_csv = out_root / "data" / "d016_smoke.csv"
    fws = [fetch_fw]
    terminals: list[Firework] = []
    sys_ids: list[str] = []
    for ranks in rank_sweep:
        sys_id = f"d016_np{ranks}"
        sys_ids.append(sys_id)
        rundir = out_root / sys_id
        prep = Firework(
            PrepareDimerInputsFiretask(
                system_id=sys_id,
                reactant_con=str(pos),
                initial_direction=str(direction),
                out_dir=str(rundir),
                model_path="{petmad_model_path}",
            ),
            parents=[fetch_fw],
            name=f"prep-{sys_id}",
        )
        gprd = Firework(
            GprdDimerFiretask(
                system_id=sys_id,
                rundir=str(rundir),
                model_path="{petmad_model_path}",
                ranks=ranks,
            ),
            parents=[prep],
            name=f"dimer-gprd-{sys_id}",
        )
        fws.extend([prep, gprd])
        terminals.append(gprd)

        if include_baseline and ranks == 1:
            # eOn baseline only meaningful at np=1 (eonclient is single-rank).
            eon = Firework(
                EonDimerBaselineFiretask(
                    system_id=sys_id, rundir=str(rundir)
                ),
                parents=[prep],
                name=f"dimer-eon-{sys_id}",
            )
            fws.append(eon)
            terminals.append(eon)

    harvest = Firework(
        HarvestDimerFiretask(
            suite_name="d016_smoke",
            out_csv=str(out_csv),
            system_ids=sys_ids,
        ),
        parents=terminals,
        name="harvest-d016",
    )
    fws.append(harvest)

    return Workflow(fws, name="d016_smoke")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model-name", default="pet-mad-s-v1.5.0")
    parser.add_argument("--ranks", default="1,8,16")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rank_sweep = tuple(int(r) for r in args.ranks.split(","))
    wf = build(
        source_dir=args.source,
        out_root=args.out,
        model_name=args.model_name,
        rank_sweep=rank_sweep,
        include_baseline=not args.no_baseline,
    )
    if args.dry_run:
        print(f"Built workflow: {wf.name}, {len(wf.fws)} fireworks")
        return
    lpad = LaunchPad.auto_load()
    fw_ids = lpad.add_wf(wf)
    print(f"Added d016 smoke workflow; root fw_id={list(fw_ids.values())[0]}")


if __name__ == "__main__":
    main()
