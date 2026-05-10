#!/usr/bin/env bash
# Bring up a single-tenant MongoDB instance for the gpr_optim_fw LaunchPad
# on the elja login node via apptainer (no root needed).
#
# Per the amsel-fireworks-orchestration internal doc: elja has no system
# mongo, so we use a per-user apptainer container bound to a NFS-mounted
# storage dir under $HOME. Compute-node FWorkers reach this via the elja
# internal network (port 27017 on the login node hostname).

set -uo pipefail

MONGO_HOME="${MONGO_HOME:-$HOME/gpr-optim-fw/mongodb}"
MONGO_LOG="${MONGO_LOG:-$HOME/gpr-optim-fw/logs/mongo.log}"
MONGO_PORT="${MONGO_PORT:-27017}"
MONGO_PIDFILE="${MONGO_PIDFILE:-$HOME/gpr-optim-fw/mongodb.pid}"
MONGO_SIF="${MONGO_SIF:-$HOME/gpr-optim-fw/mongo_7.sif}"

mkdir -p "$MONGO_HOME/data" "$(dirname "$MONGO_LOG")" "$(dirname "$MONGO_PIDFILE")"

if [[ ! -s "$MONGO_SIF" ]]; then
    echo "Pulling docker://mongo:7 -> $MONGO_SIF"
    apptainer pull --force "$MONGO_SIF" docker://mongo:7
fi

if [[ -s "$MONGO_PIDFILE" ]] && kill -0 "$(<"$MONGO_PIDFILE")" 2>/dev/null; then
    echo "mongo already running (pid=$(<"$MONGO_PIDFILE"))"
    exit 0
fi

nohup apptainer run \
    --bind "$MONGO_HOME/data":/data/db \
    "$MONGO_SIF" \
    mongod \
        --bind_ip 0.0.0.0 \
        --port "$MONGO_PORT" \
        --logpath /data/db/mongod.log \
        --logappend \
    >"$MONGO_LOG" 2>&1 &
echo $! > "$MONGO_PIDFILE"
echo "started mongo (pid=$(<"$MONGO_PIDFILE")), log: $MONGO_LOG"
sleep 2
echo "probe: nc -zv $(hostname) $MONGO_PORT"
nc -zv "$(hostname)" "$MONGO_PORT" || {
    echo "WARN: mongo probe failed; check $MONGO_LOG"
    exit 1
}
