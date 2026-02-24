# Metatomic GROMACS

## Workflows
**1. Build** (`pixi r -e <ENV> <TASK> Release`)
- **CPU (Local)**: `metatomic-cpu` / `gromk-{mpi,tmpi}`
- **CUDA (Remote)**: `ssh rg.cosmolab` → `tmux -t gro` (window 2).
  - *Path*: `/home/goswami/Git/Github/epfl/lab-cosmo/pixi_envs/orgs/metatensor/gromacs`
  - *Sync*: Mutagen (auto) from local `pixi_envs`.
  - *Run*: `metatomic-cuda` / `gromk-tmpi`.

**2. Test**
- **Prep**: `cd mta_test && python create_model.py`
- **Run (exact commands)**:
  ```bash
  # Build first:
  pixi run --environment metatomic-cpu gromk-tmpi

  # Run tests:
  cd /home/rgoswami/Git/Github/epfl/pixi_envs/orgs/metatensor/gromacs/mta_test && \
  GMX_BIN=/home/rgoswami/Git/Github/epfl/pixi_envs/orgs/metatensor/gromacs/.pixi/envs/metatomic-cpu/bin/gmx \
  GMX_THREAD_MPI=1 \
  pixi run --environment metatomic-cpu -- python -m pytest test_metatomic.py -v

  # Run specific test:
  pixi run --environment metatomic-cpu -- python -m pytest test_metatomic.py::TestStep0Energy -v

  # With debug:
  GMX_METATOMIC_DEBUG=1 pixi run --environment metatomic-cpu -- python -m pytest test_metatomic.py::TestStep0Energy -v
  ```
  - *Local*: `gmx_mpi` / `gmx`; `-k "not dd8 and not dd12"`
  - *Remote*: CUDA enabled; run all tests in `tmux:cudarun`.

## Context
- **Source**: `gromacs/` (branch `realDomDec`) → `src/gromacs/applied_forces/metatomic/`
- **CI**: `gromacs/.github/workflows/metatomic-ci.yml`
- **Vault**: Run `pixi run seal` in `~/Git/Github/epfl/pixi_envs` after edits.

## Env Vars
- `GMX_BIN=gmx|gmx_mpi`
- `GMX_METATOMIC_DEBUG=1` (per-rank logs)
- `GMX_METATOMIC_{DEVICE=cpu|cuda}`
