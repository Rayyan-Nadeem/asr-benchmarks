#!/bin/bash
# Cloud-init bootstrap for the model-playground GPU box.
# Installs NVIDIA driver + Docker + nvidia-container-toolkit + Caddy.
# Repo + app stack get rsync'd in by the operator post-bootstrap.

set -euo pipefail
exec > >(tee /var/log/bootstrap.log) 2>&1
echo "=== bootstrap start: $(date -u +%FT%TZ) ==="

# ---------------------------------------------------------------------------
# OS prep

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  build-essential git curl ca-certificates gnupg \
  python3-pip python3-venv python3-dev \
  linux-headers-$(uname -r) \
  ffmpeg jq

# ---------------------------------------------------------------------------
# NVIDIA driver (Tesla T4 → driver 535 is current stable)

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== installing NVIDIA driver 535 ==="
  apt-get install -y -qq nvidia-driver-535-server nvidia-utils-535-server
fi

# ---------------------------------------------------------------------------
# Docker + nvidia-container-toolkit

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin

usermod -aG docker ubuntu

distribution=ubuntu22.04
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -qq
apt-get install -y -qq nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# ---------------------------------------------------------------------------
# Caddy — TLS termination + basic auth + reverse proxy

curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
  | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt \
  | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
apt-get update -qq
apt-get install -y -qq caddy

# Hash the admin password for Caddy basicauth
ADMIN_HASH=$(caddy hash-password --plaintext '${admin_password}')

cat > /etc/caddy/Caddyfile <<CADDYEOF
${fqdn} {
    encode gzip

    # Strip identifying server header
    header -Server

    basicauth /* {
        admin $${ADMIN_HASH}
    }

    # WebSocket endpoint to the uvicorn server
    @ws {
        path /v2 /v2/*
        header Connection *Upgrade*
        header Upgrade websocket
    }
    reverse_proxy @ws 127.0.0.1:9000

    # Anything that looks like an orchestrator API call goes to control.py
    handle /engines* {
        reverse_proxy 127.0.0.1:9100
    }
    handle /diarizers* {
        reverse_proxy 127.0.0.1:9100
    }
    handle /switch* {
        reverse_proxy 127.0.0.1:9100
    }
    handle /current* {
        reverse_proxy 127.0.0.1:9100
    }
    handle /ready* {
        reverse_proxy 127.0.0.1:9000
    }

    # Everything else (live.html, assets) served by control.py too
    handle {
        reverse_proxy 127.0.0.1:9100
    }
}
CADDYEOF

# Caddy needs to be able to bind 80+443; allow it on its own systemd unit
systemctl enable caddy
# Don't start yet — wait until the app stack is on the box, otherwise basicauth
# guards an empty origin. Operator starts caddy after rsync+app boot.

# ---------------------------------------------------------------------------
# App user + workspace

mkdir -p /opt/model-playground
chown ubuntu:ubuntu /opt/model-playground

# Drop the admin password into a file ONLY readable by ubuntu so the operator
# can grab it. Same value is also a terraform output, so this is belt-and-
# suspenders.
echo '${admin_password}' > /home/ubuntu/.admin_password
chown ubuntu:ubuntu /home/ubuntu/.admin_password
chmod 600 /home/ubuntu/.admin_password

# Marker file so the operator can confirm bootstrap finished
echo "$(date -u +%FT%TZ) bootstrap complete" > /home/ubuntu/.bootstrap_done
chown ubuntu:ubuntu /home/ubuntu/.bootstrap_done

echo "=== bootstrap end: $(date -u +%FT%TZ) ==="

# NVIDIA driver may need a reboot to load the kernel module if it wasn't
# already present. If nvidia-smi fails, reboot. Cloud-init runs as root so
# this is fine.
if ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not ready — rebooting to load driver"
  reboot
fi
