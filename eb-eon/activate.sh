#!/usr/bin/env bash
# Clear any broken `ml` / `module` exported bash functions inherited from
# the parent shell (lmod sometimes leaves these in a bad state when the
# parent session didn't initialize lmod correctly).
unset -f ml module 2>/dev/null || true

# Source the lmod shell init that comes with the pixi-provided lmod.
# Without this, `ml`, `module`, `MODULEPATH` aren't set up and EasyBuild
# cannot emit usable module files.
_lmod_init="$CONDA_PREFIX/lmod/8.7.25/init/profile"
if [[ -f "$_lmod_init" ]]; then
  # shellcheck disable=SC1090
  source "$_lmod_init"
fi
unset _lmod_init

# Point lmod at the install tree EasyBuild writes into, so `module load`
# can find freshly built recipes without manual `module use`.
export MODULEPATH="${PIXI_PROJECT_ROOT}/install/modules/all${MODULEPATH:+:$MODULEPATH}"
