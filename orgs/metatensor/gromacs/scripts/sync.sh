#!/usr/bin/env bash
set -e

# --- Configuration ---
# The default host to use if none is provided
DEFAULT_HOST="rg.cosmolab"

# The local directory to sync (relative to the script)
LOCAL_DIR="gromacs"
LOCAL_PATH="${PWD}/${LOCAL_DIR}"

# The remote user and base path
# TODO(rg): pass this too
REMOTE_USER="goswami"
REMOTE_BASE_PATH="/home/${REMOTE_USER}/Git/Github/epfl/lab-cosmo/pixi_envs/orgs/metatensor/gromacs"
# --- End Configuration ---

# 1. Get the direction (to/from) and host from script arguments
DIRECTION=$1
HOST=${2:-$DEFAULT_HOST}
REMOTE_TARGET="${HOST}:${REMOTE_BASE_PATH}/${LOCAL_DIR}"

# Define explicit excludes
EXCLUDE_LIST=(
    --exclude=".pixi/*"
    --exclude="pixi.lock"
    --exclude=".gitignore"
    --exclude=".snakemake/*"
    --exclude="snake_runs/*"
    --exclude=".dvc/*"
    --exclude="build/*"
    --exclude="*/**/bbdir/*"
    --exclude="*.lock"
    --exclude="gprd_local/*"
    --exclude="gprd_hpc/*"
    --exclude=".rsync-exclude"
    --exclude="sync.sh"
)

# 3. Run the correct rsync command based on the direction
case $DIRECTION in
to)
    echo "==> Syncing ${LOCAL_PATH}/ TO ${REMOTE_TARGET} on $HOST..."
    # Note: The trailing slash on "$LOCAL_DIR/" syncs the *contents*
    # of the local dir into the remote dir.
    rsync -alPxz "${EXCLUDE_LIST[@]}" "${LOCAL_PATH}/" "$REMOTE_TARGET"
    echo "==> Sync 'to' complete."
    ;;
from)
    echo "==> Syncing ${REMOTE_TARGET}/ TO ${LOCAL_PATH} FROM $HOST..."
    # Ensure the local directory exists
    mkdir -p "$LOCAL_PATH"
    # Note: The trailing slash on "$REMOTE_TARGET/" syncs the *contents*
    # of the remote dir into the local dir.
    rsync -alPxz "${EXCLUDE_LIST[@]}" "${LOCAL_PATH}/" "$LOCAL_DIR"
    echo "==> Sync 'from' complete."
    ;;
*)
    echo "Error: Invalid direction. Use 'to' or 'from'."
    exit 1
    ;;
esac
