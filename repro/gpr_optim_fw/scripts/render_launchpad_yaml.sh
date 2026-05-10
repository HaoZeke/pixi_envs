#!/usr/bin/env bash
# Render my_launchpad.yaml from the creds file. Lives outside git (writes
# to ~/gpr-optim-fw/configs/my_launchpad.yaml). Re-run whenever creds rotate.

set -uo pipefail
MONGO_CREDS="${MONGO_CREDS:-$HOME/gpr-optim-fw/configs/mongo_creds.env}"
TARGET="${TARGET:-$HOME/gpr-optim-fw/configs/my_launchpad.yaml}"
# Default to the elja cluster-internal IP since `elja-irhpc` resolves to
# the IPv6 loopback ::1 in /etc/hosts (mongo binds to v4 only). Compute
# nodes reach this address over the cluster-internal subnet (172.16.71.0/24).
HOST="${MONGO_HOST:-172.16.71.1}"
PORT="${MONGO_PORT:-27018}"

if [[ ! -s "$MONGO_CREDS" ]]; then
    echo "missing creds file: $MONGO_CREDS" >&2
    exit 1
fi
# shellcheck source=/dev/null
. "$MONGO_CREDS"

umask 077
cat > "$TARGET" <<EOF
host: $HOST
port: $PORT
name: $MONGO_DB
username: $MONGO_ADMIN_USER
password: $MONGO_ADMIN_PASS
authsource: admin
logdir: $HOME/gpr-optim-fw/logs/launchpad
strm_lvl: INFO
ssl: false
mongoclient_kwargs:
  serverSelectionTimeoutMS: 30000
EOF
chmod 600 "$TARGET"
echo "rendered $TARGET"
