#!/bin/bash
# Agentic OS: one-paste setup for an Oracle Cloud "Always Free" VM (Ubuntu 22.04/24.04, ARM or x86).
#
# Paste this whole file into: Create instance -> Advanced options -> Management -> "Paste cloud-init script".
# BEFORE pasting, change the settings in the block below (at least WAR_ROOM_PASSWORD).
#
# What it does on first boot (takes ~5-10 minutes):
#   * installs Docker, opens ports 80/443 in the VM firewall
#   * builds and runs Agentic OS (auto-restarts, data kept in /opt/agentic-os/data)
#   * runs Caddy in front of it with a free HTTPS certificate on https://<your-ip>.sslip.io
#   * checks GitHub every 15 minutes and redeploys when the branch changes
# Progress log: /var/log/agentic-os-setup.log
set -euo pipefail

# ============================ SETTINGS (edit these) ============================
WAR_ROOM_PASSWORD="agentictemp"   # War Room login password (temp password; use min. 12 characters in production)
GITHUB_REPO="JavsonOf/agentic-os"
BRANCH="main"
GITHUB_TOKEN=""   # only for a PRIVATE repo: fine-grained token with read-only "Contents" on this repo
# ===============================================================================

APP_DIR=/opt/agentic-os
exec > >(tee -a /var/log/agentic-os-setup.log) 2>&1
echo "=== Agentic OS setup started $(date -u) ==="

if [ "${#WAR_ROOM_PASSWORD}" -lt 8 ]; then
  echo "ERROR: set WAR_ROOM_PASSWORD (at least 8 characters; 12+ recommended for production) at the top of the script." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y docker.io git curl ca-certificates iptables-persistent
systemctl enable --now docker

# Oracle's Ubuntu images block everything except SSH with iptables: allow HTTP/HTTPS before the REJECT rule.
for port in 80 443; do
  iptables -C INPUT -p tcp --dport "$port" -m state --state NEW -j ACCEPT 2>/dev/null \
    || iptables -I INPUT 6 -p tcp --dport "$port" -m state --state NEW -j ACCEPT
done
netfilter-persistent save

mkdir -p "$APP_DIR/data"
chown 1000:1000 "$APP_DIR/data"   # the container runs as the unprivileged "agent" user (uid 1000)

umask 077
cat > "$APP_DIR/.env" <<EOF
AGENTIC_OS_API_KEY=$WAR_ROOM_PASSWORD
AGENTIC_OS_DATA_DIR=/home/agent/data
EOF
cat > "$APP_DIR/repo.conf" <<EOF
GITHUB_REPO=$GITHUB_REPO
BRANCH=$BRANCH
GITHUB_TOKEN=$GITHUB_TOKEN
EOF
umask 022

# ------------------------------------------------------------------ deploy script
cat > "$APP_DIR/deploy.sh" <<'DEPLOY'
#!/bin/bash
# Pull the branch; rebuild and restart Agentic OS only when it changed (or when forced with --force).
set -euo pipefail
APP_DIR=/opt/agentic-os
# shellcheck source=/dev/null
. "$APP_DIR/repo.conf"
if [ -n "$GITHUB_TOKEN" ]; then
  URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"
else
  URL="https://github.com/${GITHUB_REPO}.git"
fi
if [ ! -d "$APP_DIR/src/.git" ]; then
  git clone --depth 1 --branch "$BRANCH" "$URL" "$APP_DIR/src"
  CHANGED=1
else
  git -C "$APP_DIR/src" remote set-url origin "$URL"
  OLD=$(git -C "$APP_DIR/src" rev-parse HEAD)
  git -C "$APP_DIR/src" fetch --depth 1 origin "$BRANCH"
  git -C "$APP_DIR/src" reset --hard "origin/$BRANCH"
  NEW=$(git -C "$APP_DIR/src" rev-parse HEAD)
  [ "$OLD" != "$NEW" ] && CHANGED=1 || CHANGED=0
fi
if [ "${CHANGED}" = "1" ] || [ "${1:-}" = "--force" ] || ! docker ps --format '{{.Names}}' | grep -qx agentic-os; then
  docker build -t agentic-os:latest "$APP_DIR/src"
  docker rm -f agentic-os >/dev/null 2>&1 || true
  docker run -d --name agentic-os --restart unless-stopped \
    --env-file "$APP_DIR/.env" \
    -v "$APP_DIR/data:/home/agent/data" \
    -p 127.0.0.1:8000:8000 \
    agentic-os:latest
  docker image prune -f >/dev/null
  echo "$(date -u) deployed $(git -C "$APP_DIR/src" rev-parse --short HEAD)"
fi
DEPLOY
chmod 700 "$APP_DIR/deploy.sh"

"$APP_DIR/deploy.sh" --force

# --------------------------------------------------------------- HTTPS with Caddy
PUBLIC_IP=$(curl -fsS https://api.ipify.org || curl -fsS https://ifconfig.me)
DOMAIN="$(echo "$PUBLIC_IP" | tr . -).sslip.io"
cat > "$APP_DIR/Caddyfile" <<EOF
$DOMAIN {
  encode gzip
  reverse_proxy 127.0.0.1:8000
  header {
    Strict-Transport-Security "max-age=31536000"
    X-Frame-Options "DENY"
    Referrer-Policy "no-referrer"
  }
}
EOF
docker rm -f caddy >/dev/null 2>&1 || true
docker run -d --name caddy --restart unless-stopped --network host \
  -v "$APP_DIR/Caddyfile:/etc/caddy/Caddyfile:ro" \
  -v caddy_data:/data -v caddy_config:/config \
  caddy:2

# ------------------------------------------------------------ automatic updates
cat > /etc/cron.d/agentic-os <<'EOF'
*/15 * * * * root /opt/agentic-os/deploy.sh >> /var/log/agentic-os-deploy.log 2>&1
EOF

echo "https://$DOMAIN" > "$APP_DIR/URL.txt"
echo "=== Agentic OS is ready: https://$DOMAIN (log in with your WAR_ROOM_PASSWORD) ==="
