#!/usr/bin/env bash
set -euo pipefail

# Package this rsx_bmc_repro tree for Zenodo or Materials Cloud Archive deposition.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTDIR="${RSX_REPRO_OUTDIR:-${ROOT}}"
TS=$(date +%Y%m%d)
ARCHIVE="${OUTDIR}/rsx_bmc_repro_archive_${TS}.tar.xz"

echo "Preparing rsx_bmc_repro archive..."

tar \
    --exclude='.pixi' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='target' \
    --exclude='work' \
    -cJf "${ARCHIVE}" \
    -C "$(dirname "${ROOT}")" \
    "$(basename "${ROOT}")"

echo "Created ${ARCHIVE}"
echo "Size: $(du -h "${ARCHIVE}" | cut -f1)"
sha256sum "${ARCHIVE}"
