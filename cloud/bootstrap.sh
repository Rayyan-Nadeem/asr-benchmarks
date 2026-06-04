#!/usr/bin/env bash
# Bootstrap script — runs ON the GPU instance once it's up.
# Cloud-agnostic: works on any Ubuntu 22.04 box with a CUDA-capable GPU.
# If NVIDIA drivers aren't already installed (e.g. GovCloud base Ubuntu AMI),
# installs them. If they ARE installed (DLAMI / DSVM), skips that step.
#
# Usage (from your Mac):
#   scp -i ~/.ssh/<keyname> cloud/bootstrap.sh ubuntu@<public-ip>:~/
#   ssh -i ~/.ssh/<keyname> ubuntu@<public-ip> \
#       'GITHUB_PAT=<token> GIT_REPO=Rayyan-Nadeem/asr-benchmarks \
#        GIT_BRANCH=parakeet-mac-adapters bash bootstrap.sh'
#
# What it does:
#   0. Installs NVIDIA driver + CUDA if missing (skipped on DLAMI)
#   1. Verifies GPU visible (nvidia-smi works)
#   2. Clones the asr-benchmarks repo (via PAT — passed via env var)
#   3. Creates a venv, installs NeMo + KenLM + base harness deps
#   4. Pre-downloads Parakeet TDT 0.6B v3 model weights to ~/.cache
#   5. Verifies everything imports cleanly
#
# Idempotent — safe to re-run. If the NVIDIA driver install requires a
# reboot, the script will tell you to reconnect and re-run.

set -euo pipefail

echo "=== asr-benchmarks GPU bootstrap ==="
echo "host: $(hostname)"
echo "user: $(whoami)"
echo "time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

# Step 0 — NVIDIA driver + CUDA install (only if not already present)
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "[0/5] NVIDIA driver not detected — installing (~10–15 min)..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq linux-headers-$(uname -r) build-essential dkms

  # Use Ubuntu's official packaged driver (no NVIDIA repo dance needed).
  # nvidia-driver-550-server is the current LTS server driver for U22.04
  # and supports CUDA 12.4 + L4 (Ada Lovelace) + T4 (Turing).
  sudo apt-get install -y -qq nvidia-driver-550-server nvidia-utils-550-server

  # Load module without reboot if possible. If this fails, a reboot is needed.
  if ! sudo modprobe nvidia 2>/dev/null; then
    echo ""
    echo "⚠ NVIDIA driver installed but module couldn't be loaded without reboot."
    echo "  Run:  sudo reboot"
    echo "  Then reconnect and re-run this script — it'll skip the install step"
    echo "  and continue from step 1."
    exit 0
  fi

  # CUDA toolkit (we install runtime libs only; NeMo brings its own torch wheels)
  sudo apt-get install -y -qq nvidia-cuda-toolkit
  echo "  driver install complete"
  echo
fi

# Step 1 — GPU sanity check
echo "[1/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
echo

# Step 2 — Clone repo. Requires GITHUB_PAT in env (or pre-cloned).
echo "[2/5] Cloning asr-benchmarks repo..."
if [ -d "asr-benchmarks" ]; then
  echo "  repo already cloned — pulling latest"
  cd asr-benchmarks
  git pull --rebase
  cd ..
else
  if [ -z "${GITHUB_PAT:-}" ] && [ -z "${GIT_REPO:-}" ]; then
    echo "ERROR: GITHUB_PAT + GIT_REPO env vars required (see cloud/.env.example)"
    exit 1
  fi
  git clone "https://${GITHUB_PAT}@github.com/${GIT_REPO}.git" asr-benchmarks
  cd asr-benchmarks
  git checkout "${GIT_BRANCH:-main}"
  cd ..
fi
cd asr-benchmarks
echo "  on branch: $(git branch --show-current), commit: $(git rev-parse --short HEAD)"
echo

# Step 3 — venv + install heavy deps. Plain Ubuntu may not have python3-venv;
# DLAMI / DSVM have it already. Either way the apt install is a no-op if present.
echo "[3/5] Setting up venv + installing dependencies..."
sudo apt-get install -y -qq python3-venv python3-pip ffmpeg git
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip

# Base harness deps
pip install --quiet -r requirements.txt

# Mac engines we ALSO run on GPU for cross-comparison
pip install --quiet onnx-asr[cpu,hub] openai-whisper

# Heavy GPU-specific deps
pip install --quiet 'nemo_toolkit[asr]==2.4.0'
pip install --quiet kenlm pyctcdecode

# Diarizer (will skip silently if HF_TOKEN isn't set; user can retry later)
pip install --quiet 'pyannote.audio>=4.0'

echo "  installed: $(.venv/bin/pip list --format=freeze | wc -l) packages"
echo

# Step 4 — pre-download model weights so the first measurement doesn't pay the network tax
echo "[4/5] Pre-downloading Parakeet TDT 0.6B v3 weights..."
python3 -c "
from nemo.collections.asr.models import ASRModel
m = ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')
print(f'Parakeet model loaded: {m.cfg.target}')
"
echo

# Step 5 — verify imports work
echo "[5/5] Verifying imports..."
python3 -c "
import torch
print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')
import nemo.collections.asr  # noqa
print('NeMo: ok')
from engines.parakeet_onnx import ParakeetOnnx
from engines.parakeet_nemo import ParakeetNemo
from engines._integration import transcribe_segment_first
print('harness: ok')
"
echo
echo "=== bootstrap complete — ready for run_experiments.sh ==="
