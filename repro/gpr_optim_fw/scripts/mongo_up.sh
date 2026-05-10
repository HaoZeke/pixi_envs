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
# Default port: 27018, leaving the amsel project's existing 127.0.0.1:27017
# mongo (~/.pixi/envs/mongodb/bin/mongod, --bind_ip 127.0.0.1) untouched.
# A second instance on a separate port + cluster-internal interface lets the
# two LaunchPads coexist on the same elja login node.
MONGO_PORT="${MONGO_PORT:-27018}"
MONGO_PIDFILE="${MONGO_PIDFILE:-$HOME/gpr-optim-fw/mongodb.pid}"
MONGO_SIF="${MONGO_SIF:-$HOME/gpr-optim-fw/mongo_7.sif}"
MONGO_CREDS="${MONGO_CREDS:-$HOME/gpr-optim-fw/configs/mongo_creds.env}"
# Default to the elja cluster-internal interface (172.16.71.1, eno5 on
# elja-irhpc) plus loopback. Compute nodes reach the LaunchPad via that
# private subnet; the public eno1 (130.208.148.26) is intentionally
# excluded -- auth is on, but defence-in-depth says do not advertise
# mongo to the open internet.
MONGO_BIND_IP="${MONGO_BIND_IP:-127.0.0.1,172.16.71.1}"

if [[ ! -s "$MONGO_CREDS" ]]; then
    echo "missing creds file: $MONGO_CREDS" >&2
    echo "expected MONGO_ADMIN_USER + MONGO_ADMIN_PASS + MONGO_DB" >&2
    exit 1
fi
# shellcheck source=/dev/null
. "$MONGO_CREDS"
: "${MONGO_ADMIN_USER:?creds file missing MONGO_ADMIN_USER}"
: "${MONGO_ADMIN_PASS:?creds file missing MONGO_ADMIN_PASS}"
: "${MONGO_DB:?creds file missing MONGO_DB}"

# Prefer apptainer where present; elja login currently ships /usr/bin/singularity
# (Rocky 8.10), which speaks the same pull / run subcommands at the surface
# we use here.
if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_RUNTIME=singularity
else
    echo "no apptainer/singularity on PATH" >&2
    exit 1
fi

mkdir -p "$MONGO_HOME/data" "$(dirname "$MONGO_LOG")" "$(dirname "$MONGO_PIDFILE")"

if [[ ! -s "$MONGO_SIF" ]]; then
    echo "Pulling docker://mongo:7 -> $MONGO_SIF (using $CONTAINER_RUNTIME)"
    "$CONTAINER_RUNTIME" pull --force "$MONGO_SIF" docker://mongo:7
fi

if [[ -s "$MONGO_PIDFILE" ]] && kill -0 "$(<"$MONGO_PIDFILE")" 2>/dev/null; then
    echo "mongo already running (pid=$(<"$MONGO_PIDFILE"))"
    exit 0
fi

# First-launch detection: if /data/db is empty, mongod will run an init pass.
# Singularity does not respect mongo:7's MONGO_INITDB_ROOT_* env vars (those
# only fire under the docker-entrypoint), so we run a two-phase bring-up:
#  Phase 1: start mongod without --auth, create the admin user via mongosh,
#           shut down.
#  Phase 2: relaunch with --auth + the user from phase 1.
# Subsequent runs skip phase 1.
SINGULARITYENV_MONGO_INITDB_ROOT_USERNAME="$MONGO_ADMIN_USER" \
    SINGULARITYENV_MONGO_INITDB_ROOT_PASSWORD="$MONGO_ADMIN_PASS" \
    APPTAINERENV_MONGO_INITDB_ROOT_USERNAME="$MONGO_ADMIN_USER" \
    APPTAINERENV_MONGO_INITDB_ROOT_PASSWORD="$MONGO_ADMIN_PASS"

ADMIN_INIT_FLAG="$MONGO_HOME/.admin_initialized"
if [[ ! -f "$ADMIN_INIT_FLAG" ]]; then
    echo "first-run init: creating admin user '$MONGO_ADMIN_USER'"
    "$CONTAINER_RUNTIME" exec \
        --bind "$MONGO_HOME/data":/data/db \
        "$MONGO_SIF" \
        mongod --bind_ip 127.0.0.1 --port "$MONGO_PORT" --fork \
            --logpath /data/db/mongod-init.log --logappend \
        || { echo "phase-1 mongod failed; see $MONGO_HOME/data/mongod-init.log"; exit 1; }
    sleep 2
    "$CONTAINER_RUNTIME" exec "$MONGO_SIF" mongosh \
        --host 127.0.0.1 --port "$MONGO_PORT" --quiet --eval "
            db = db.getSiblingDB('admin');
            db.createUser({
                user: '$MONGO_ADMIN_USER',
                pwd: '$MONGO_ADMIN_PASS',
                roles: [
                    { role: 'userAdminAnyDatabase', db: 'admin' },
                    { role: 'readWriteAnyDatabase', db: 'admin' },
                    { role: 'dbAdminAnyDatabase', db: 'admin' }
                ]
            });
        " || { echo "createUser failed"; exit 1; }
    "$CONTAINER_RUNTIME" exec "$MONGO_SIF" mongosh \
        --host 127.0.0.1 --port "$MONGO_PORT" --quiet --eval "
            db.adminCommand({ shutdown: 1 });
        " 2>/dev/null || true
    sleep 2
    touch "$ADMIN_INIT_FLAG"
    echo "admin user created; container shut down"
fi

nohup "$CONTAINER_RUNTIME" run \
    --bind "$MONGO_HOME/data":/data/db \
    "$MONGO_SIF" \
    mongod \
        --bind_ip "$MONGO_BIND_IP" \
        --port "$MONGO_PORT" \
        --auth \
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
