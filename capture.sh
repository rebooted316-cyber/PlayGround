#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
/usr/bin/time -p pwd
/usr/bin/time -p echo "capture start url=${CAPTURE_URL:-<unset>} dir=${CAPTURE_DIR:-<unset>}"
: "${CAPTURE_URL:?Set CAPTURE_URL}"
: "${CAPTURE_DIR:?Set CAPTURE_DIR}"
RUNTIME_DIR="${RUNTIME_DIR:-/home/runner/work/_temp/omgithub-runtime}"
/usr/bin/time -p mkdir -p "$CAPTURE_DIR"
/usr/bin/time -p node "$RUNTIME_DIR/scripts/default-capture.mjs"
/usr/bin/time -p ls -l "$CAPTURE_DIR/final-desktop.png" "$CAPTURE_DIR/final-mobile.png"
/usr/bin/time -p python3 -c "import os,sys; d=os.environ['CAPTURE_DIR']; assert os.path.getsize(os.path.join(d,'final-desktop.png'))>0, 'empty desktop'; assert os.path.getsize(os.path.join(d,'final-mobile.png'))>0, 'empty mobile'; print('capture artifacts verified')"
