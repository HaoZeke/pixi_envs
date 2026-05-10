"""Hermes 500-system dimer workflow: GP-accelerated (gprd) vs eOn baseline.

Per system:
  1. fetch-petmad (shared root fw)
  2. prep-{system_id}: copy pos.con / direction.dat into the per-system
     rundir, write metatomic-flavoured config.ini
  3. dimer-gprd-{system_id}: gprd binary, INLA-mode hyperparams, np from
     SLURM_NTASKS
  4. dimer-eon-{system_id}: plain eonclient on the same rundir
  5. harvest-hermes: csv with per-system gprd vs baseline metrics

Source: gprd_zbl/runs/automated/<rundir>/<spin>/<index>/{pos.con,direction.dat}
which is the materialised Hermes/Sella benchmark (singlets 000..264 +
doublets 000..234 = 500 systems). For PET-MAD-only (this paper), spin
is single-tier; we bench the singlet set unless --include-doublet is set.
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
    Path.home()
    / "Git"
    / "Github"
    / "epfl"
    / "pixi_envs"
    / "repro"
    / "gprd_zbl"
    / "runs"
    / "automated"
    / "snake_runs"
)
DEFAULT_OUT_ROOT = Path.home() / "gpr-optim-fw" / "runs" / "hermes_dimer"


def _discover_systems(
    source_root: Path,
    rundir_name: str,
    include_doublet: bool,
    limit: int | None,
) -> list[tuple[str, Path, Path]]:
    """Return (system_id, pos.con, direction.dat) tuples."""
    rundir = source_root / rundir_name
    if not rundir.is_dir():
        raise FileNotFoundError(f"Hermes rundir not found: {rundir}")

    out = []
    spins = ["singlet"]
    if include_doublet:
        spins.append("doublet")
    for spin in spins:
        spin_dir = rundir / spin
        if not spin_dir.is_dir():
            continue
        for sys_dir in sorted(spin_dir.iterdir()):
            if not sys_dir.is_dir():
                continue
            pos = sys_dir / "pos.con"
            direction = sys_dir / "direction.dat"
            if pos.is_file() and direction.is_file():
                out.append((f"{spin}_{sys_dir.name}", pos, direction))
    if limit:
        out = out[:limit]
    return out


def build(
    source_root: Path = DEFAULT_SOURCE,
    out_root: Path = DEFAULT_OUT_ROOT,
    rundir_name: str = "default_run",
    model_name: str = "pet-mad-s-v1.5.0",
    include_doublet: bool = False,
    limit: int | None = None,
    skip_baseline: bool = False,
    skip_gprd: bool = False,
    ranks: int = 8,
) -> Workflow:
    systems = _discover_systems(source_root, rundir_name, include_doublet, limit)
    if not systems:
        raise FileNotFoundError(
            f"No Hermes systems under {source_root / rundir_name}; "
            "run gprd_zbl's prepare_files step first"
        )

    fetch_fw = Firework(
        FetchPetMadFiretask(model_name=model_name),
        name="fetch-petmad",
    )

    out_csv = out_root / "data" / "hermes_dimer.csv"
    final_dimer_fws: list[Firework] = []
    for sys_id, pos, direction in systems:
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
            name=f"prep-dimer-{sys_id}",
        )

        per_system_terminals: list[Firework] = []
        if not skip_gprd:
            gprd_fw = Firework(
                GprdDimerFiretask(
                    system_id=sys_id,
                    rundir=str(rundir),
                    model_path="{petmad_model_path}",
                    ranks=ranks,
                ),
                parents=[prep],
                name=f"dimer-gprd-{sys_id}",
            )
            per_system_terminals.append(gprd_fw)
        if not skip_baseline:
            eon_fw = Firework(
                EonDimerBaselineFiretask(
                    system_id=sys_id,
                    rundir=str(rundir),
                ),
                parents=[prep],
                name=f"dimer-eon-{sys_id}",
            )
            per_system_terminals.append(eon_fw)

        final_dimer_fws.extend(per_system_terminals)

    harvest = Firework(
        HarvestDimerFiretask(
            suite_name="hermes_dimer",
            out_csv=str(out_csv),
            system_ids=[s[0] for s in systems],
        ),
        parents=final_dimer_fws,
        name="harvest-hermes-dimer",
    )

    fws = [fetch_fw]
    fws += [fw for fw in final_dimer_fws if fw.name.startswith("prep")]
    fws += [fw for fw in final_dimer_fws]
    # Dedup while preserving order
    seen = set()
    unique = []
    for fw in [fetch_fw, *final_dimer_fws, harvest]:
        if id(fw) in seen:
            continue
        seen.add(id(fw))
        unique.append(fw)
    # Add the prep fws by walking parents.
    parents_map = {}
    for fw in final_dimer_fws:
        for p in fw.parents:
            parents_map[id(p)] = p
    for p in parents_map.values():
        if id(p) not in seen:
            seen.add(id(p))
            unique.append(p)

    return Workflow(unique, name="hermes_dimer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--rundir-name", default="default_run")
    parser.add_argument("--model-name", default="pet-mad-s-v1.5.0")
    parser.add_argument("--include-doublet", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-gprd", action="store_true")
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    wf = build(
        source_root=args.source,
        out_root=args.out,
        rundir_name=args.rundir_name,
        model_name=args.model_name,
        include_doublet=args.include_doublet,
        limit=args.limit,
        skip_baseline=args.skip_baseline,
        skip_gprd=args.skip_gprd,
        ranks=args.ranks,
    )
    if args.dry_run:
        print(f"Built workflow: {wf.name}, {len(wf.fws)} fireworks")
        return
    lpad = LaunchPad.auto_load()
    fw_ids = lpad.add_wf(wf)
    print(f"Added hermes_dimer workflow; root fw_id={list(fw_ids.values())[0]}")


if __name__ == "__main__":
    main()
