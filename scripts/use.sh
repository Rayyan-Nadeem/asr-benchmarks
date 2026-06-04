#!/usr/bin/env bash
# Switch the depodash-asr container between named stacks.
#
# Usage:
#   ./scripts/use.sh <stack-name>
#
# Stacks:
#   nemo-sortformer    Parakeet TDT FP16 + Sortformer        (GPU production pick)
#   nemo-pyannote      Parakeet TDT FP16 + pyannote          (GPU, alt diarizer)
#   onnx-sortformer    Parakeet TDT INT8 + Sortformer        (CPU+GPU mix)
#   onnx-pyannote      Parakeet TDT INT8 + pyannote          (best AMI DER)
#   onnx-passthrough   Parakeet TDT INT8 + passthrough       (CPU-only laptop)
#   whisper-sortformer Whisper Large v3 + Sortformer         (Whisper reference)
#   mlx-passthrough    MLX Whisper Large v3 + passthrough    (Apple Silicon, Mac only)
#   noop-passthrough   Noop engine + passthrough             (protocol smoke test)
#
# Anything beyond the production pick needs the right deps installed in the
# image — see server/Dockerfile + server/requirements.txt.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") <stack-name>

Stacks:
  nemo-sortformer      GPU production pick (best WER)
  nemo-pyannote        GPU, alt diarizer
  onnx-sortformer      INT8 ASR + GPU diarizer
  onnx-pyannote        Best AMI DER on the board
  onnx-passthrough     CPU-only laptop
  whisper-sortformer   Whisper Large v3 reference
  mlx-passthrough      Apple Silicon Whisper (Mac only)
  noop-passthrough     Protocol smoke test
EOF
    exit 1
}

[ $# -eq 1 ] || usage
stack="$1"

case "$stack" in
    nemo-sortformer)    ENGINE=parakeet-nemo  DIARIZER=sortformer-streaming ;;
    nemo-pyannote)      ENGINE=parakeet-nemo  DIARIZER=pyannote ;;
    onnx-sortformer)    ENGINE=parakeet-onnx  DIARIZER=sortformer-streaming ;;
    onnx-pyannote)      ENGINE=parakeet-onnx  DIARIZER=pyannote ;;
    onnx-passthrough)   ENGINE=parakeet-onnx  DIARIZER=passthrough ;;
    whisper-sortformer) ENGINE=whisper        DIARIZER=sortformer-streaming ;;
    mlx-passthrough)    ENGINE=mlx-whisper    DIARIZER=passthrough ;;
    noop-passthrough)   ENGINE=noop           DIARIZER=passthrough ;;
    *) echo "unknown stack: $stack"; echo; usage ;;
esac

export ENGINE DIARIZER

echo "→ switching to stack: $stack"
echo "  ENGINE=$ENGINE"
echo "  DIARIZER=$DIARIZER"
echo

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

echo "→ stopping current container (if any)…"
docker compose down --remove-orphans >/dev/null 2>&1 || true

echo "→ starting new container…"
docker compose up -d

echo "→ waiting for /ready…"
for i in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fsS http://localhost:9000/ready >/dev/null 2>&1; then
        echo
        echo "✓ stack '$stack' ready at ws://localhost:9000/v2"
        echo "  $(curl -fsS http://localhost:9000/ready)"
        exit 0
    fi
    sleep 2
done

echo "✗ /ready never came up — check 'docker compose logs depodash-asr'"
exit 1
