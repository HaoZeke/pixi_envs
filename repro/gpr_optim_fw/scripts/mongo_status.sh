#!/usr/bin/env bash
set -uo pipefail
MONGO_PORT="${MONGO_PORT:-27017}"
MONGO_PIDFILE="${MONGO_PIDFILE:-$HOME/gpr-optim-fw/mongodb.pid}"
if [[ -s "$MONGO_PIDFILE" ]] && kill -0 "$(<"$MONGO_PIDFILE")" 2>/dev/null; then
    echo "mongo: running (pid=$(<"$MONGO_PIDFILE"))"
else
    echo "mongo: stopped"
fi
nc -zv "$(hostname)" "$MONGO_PORT" 2>&1 || true
