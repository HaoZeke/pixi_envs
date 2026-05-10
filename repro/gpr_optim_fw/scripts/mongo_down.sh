#!/usr/bin/env bash
set -uo pipefail
MONGO_PIDFILE="${MONGO_PIDFILE:-$HOME/gpr-optim-fw/mongodb.pid}"
if [[ -s "$MONGO_PIDFILE" ]]; then
    pid="$(<"$MONGO_PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid"
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "force-killing mongo (pid=$pid)"
            kill -KILL "$pid" || true
        fi
    fi
    rm -f "$MONGO_PIDFILE"
fi
echo "mongo stopped"
