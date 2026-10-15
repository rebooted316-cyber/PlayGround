#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
/usr/bin/time -p pwd
PROJECT_ROOT="$(pwd)"
PORT="${PORT:-${APP_PORT:-3000}}"
export PORT
export AGENTIC_OS_API_KEY="${AGENTIC_OS_API_KEY:-agentictemp}"
export AGENTIC_OS_DATA_DIR="${AGENTIC_OS_DATA_DIR:-$PROJECT_ROOT/.data}"
WEB_DIR="${OPENCODE_WEB_DIR:-/home/runner/work/_temp/omgithub-web}"
/usr/bin/time -p echo "AgenticOS start: project=$PROJECT_ROOT port=$PORT webdir=$WEB_DIR"
/usr/bin/time -p mkdir -p "$PROJECT_ROOT/dist" "$AGENTIC_OS_DATA_DIR" "$WEB_DIR"
/usr/bin/time -p pip install -e . --quiet
/usr/bin/time -p cp -f "$PROJECT_ROOT/agentic_os/server/static/index.html" "$PROJECT_ROOT/dist/index.html"
/usr/bin/time -p cp -f "$PROJECT_ROOT/agentic_os/server/static/app.js" "$PROJECT_ROOT/dist/app.js"
/usr/bin/time -p cp -f "$PROJECT_ROOT/agentic_os/server/static/app.css" "$PROJECT_ROOT/dist/app.css"
/usr/bin/time -p cp -f "$PROJECT_ROOT/agentic_os/server/static/manifest.webmanifest" "$PROJECT_ROOT/dist/manifest.webmanifest"
/usr/bin/time -p cp -f "$PROJECT_ROOT/agentic_os/server/static/sw.js" "$PROJECT_ROOT/dist/sw.js"
/usr/bin/time -p cp -f "$PROJECT_ROOT/agentic_os/server/static/icon.svg" "$PROJECT_ROOT/dist/icon.svg"
/usr/bin/time -p cp -f "$PROJECT_ROOT/agentic_os/server/static/icon-192.png" "$PROJECT_ROOT/dist/icon-192.png"
/usr/bin/time -p cp -f "$PROJECT_ROOT/agentic_os/server/static/icon-512.png" "$PROJECT_ROOT/dist/icon-512.png"
/usr/bin/time -p python3 -c "import json,os; root=os.getcwd(); web=os.environ.get('OPENCODE_WEB_DIR','/home/runner/work/_temp/omgithub-web'); open(os.path.join(web,'deployment-output.json'),'w').write(json.dumps({'project':root,'directory':os.path.join(root,'dist')}))"
/usr/bin/time -p cat "$WEB_DIR/deployment-output.json"
/usr/bin/time -p python3 -c "import os; assert os.path.isfile(os.path.join(os.getcwd(),'dist','index.html')), 'dist/index.html missing'"
exec /usr/bin/time -p agentic-os serve --host 0.0.0.0 --port "$PORT"
