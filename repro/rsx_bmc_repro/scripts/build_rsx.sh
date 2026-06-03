#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RSX_REPO="${RSX_REPO:-${ROOT}/../rsx-rs}"
RSX_BIN="${RSX_BIN:-${RSX_REPO}/target/release/rsx}"

if [[ -x "${RSX_BIN}" ]]; then
    echo "Using rsx binary: ${RSX_BIN}"
    exit 0
fi

if [[ ! -f "${RSX_REPO}/Cargo.toml" ]]; then
    echo "Set RSX_REPO to an rsx-rs checkout or RSX_BIN to an executable rsx binary." >&2
    exit 1
fi

cargo build --release --manifest-path "${RSX_REPO}/Cargo.toml" -p rsx-cli
test -x "${RSX_BIN}"
echo "Built rsx binary: ${RSX_BIN}"
