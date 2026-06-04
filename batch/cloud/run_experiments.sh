#!/usr/bin/env bash
# Run all the GPU-side measurements that the Mac path can't reach.
# Runs ON the box, after bootstrap.sh has set everything up.
#
# Cloud-agnostic — same script works on AWS DLAMI, Azure Ubuntu DSVM,
# GCP Deep Learning VM, etc.
#
# Usage:
#   ssh ubuntu@<ip> 'cd asr-benchmarks && source .venv/bin/activate && bash cloud/run_experiments.sh'
#
# Each measurement writes a run JSON to results/runs/. After all complete,
# tools/render_scoreboard.py refreshes SCOREBOARD.md.

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root, regardless of where invoked

if [ ! -d ".venv" ]; then
  echo "ERROR: .venv not found — run cloud/bootstrap.sh first"
  exit 1
fi
source .venv/bin/activate

GPU_TAG="gpu-$(date -u +%Y%m%d)"
echo "=== GPU experiment session — tag prefix: ${GPU_TAG} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# ─────────────────────────────────────────────────────────────────────
# Block A — Parakeet NeMo FP16 baselines on the cases we already measured on Mac.
# Compares: FP16 GPU vs INT8 ONNX Mac on the same audio.
# ─────────────────────────────────────────────────────────────────────
echo "[A.1] Parakeet NeMo FP16 — LibriSpeech (sanity baseline)"
python -m runner --engine parakeet_nemo --case librispeech-test-clean-mini --tag "${GPU_TAG}-fp16-baseline"

echo "[A.2] Parakeet NeMo FP16 — SCOTUS"
python -m runner --engine parakeet_nemo --case scotus-glossip-v-oklahoma --tag "${GPU_TAG}-fp16-baseline"

echo "[A.3] Parakeet NeMo FP16 — Deposition (Greg Erwin, customer audio)"
python -m runner --engine parakeet_nemo --case deposition-greg-erwin --tag "${GPU_TAG}-fp16-baseline"

# ─────────────────────────────────────────────────────────────────────
# Block B — Real decoder-time vocab biasing (the production version of what
# we emulated on Mac via post-processing).
# Expected: closes 2-3 pp WER on SCOTUS beyond what the emulation caught.
# ─────────────────────────────────────────────────────────────────────
echo "[B.1] Parakeet NeMo + additional_vocab (legal terms) — SCOTUS"
python -m runner --engine parakeet_nemo --case scotus-glossip-v-oklahoma --tag "${GPU_TAG}-vocab-bias"
# NOTE: engines/parakeet_nemo.py will need to accept an additional_vocab config
# value and route it through NeMo's decoder. See TODO in that file.

# ─────────────────────────────────────────────────────────────────────
# Block C — KenLM legal-corpus language model fusion.
# Expected: closes another 1-2 pp on sentence-level patterns ("the Honorable
# Justice X", "Federal Rule of Civil Procedure 12(b)(6)").
# ─────────────────────────────────────────────────────────────────────
echo "[C.1] Build KenLM 3-gram on legal corpus"
# TODO: this step needs a legal corpus. Stub for now — point at SCOTUS opinions
# downloaded from oyez.org or US Code text from law.cornell.edu.
# tools/build_legal_kenlm.sh ./corpus/legal.txt ./models/legal.kenlm.binary

echo "[C.2] Parakeet NeMo + KenLM legal — SCOTUS"
python -m runner --engine parakeet_nemo --case scotus-glossip-v-oklahoma --tag "${GPU_TAG}-kenlm"

# ─────────────────────────────────────────────────────────────────────
# Block D — Beam search vs greedy.
# Cheap accuracy lever Parakeet's onnx-asr doesn't expose.
# Expected: 0.5-1 pp WER reduction at the cost of ~3x decode time.
# ─────────────────────────────────────────────────────────────────────
echo "[D.1] Parakeet NeMo beam_size=4 — SCOTUS"
python -m runner --engine parakeet_nemo --case scotus-glossip-v-oklahoma --tag "${GPU_TAG}-beam4"

# ─────────────────────────────────────────────────────────────────────
# Block E — Streaming Sortformer diarizer (NVIDIA-only).
# This is the streaming-friendly Phase 4.5 alternative to pyannote.
# Pairs natively with Parakeet (both NeMo-family).
# ─────────────────────────────────────────────────────────────────────
echo "[E.1] Parakeet + Streaming Sortformer + segment_first on AMI"
python -m runner --engine parakeet_nemo --case ami-es2004a-5min \
    --diarizer streaming_sortformer --integration-mode segment_first \
    --tag "${GPU_TAG}-sortformer"
# NOTE: diarizers/streaming_sortformer.py needs to exist; will write on the box.

# ─────────────────────────────────────────────────────────────────────
# Block F — Multitalker Parakeet Streaming (joint ASR + diar, max 4 spk).
# Worth one measurement to bound what joint approaches can do.
# ─────────────────────────────────────────────────────────────────────
echo "[F.1] Multitalker Parakeet Streaming — AMI (4 speakers fits)"
python -m runner --engine multitalker_parakeet --case ami-es2004a-5min \
    --tag "${GPU_TAG}-multitalker"
# NOTE: engines/multitalker_parakeet.py needs to exist; will write on the box.

# ─────────────────────────────────────────────────────────────────────
# Block G — pyannote 3.1 / 4.x with HF_TOKEN, real (no oracle crutch)
# ─────────────────────────────────────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[G.1] Parakeet + pyannote + segment_first on AMI (real, no oracle)"
  python -m runner --engine parakeet_nemo --case ami-es2004a-5min \
      --diarizer pyannote --integration-mode segment_first \
      --tag "${GPU_TAG}-pyannote-real"
else
  echo "[G.1] SKIPPING pyannote — HF_TOKEN not set"
fi

# ─────────────────────────────────────────────────────────────────────
# Finalize — regenerate scoreboard, summarize what we collected.
# ─────────────────────────────────────────────────────────────────────
echo
echo "=== Regenerating scoreboard ==="
python tools/render_scoreboard.py

echo
echo "=== Session summary ==="
ls -lh results/runs/*${GPU_TAG}*.json 2>/dev/null | wc -l | xargs -I N echo "  N new run files written this session"
echo
echo "=== Done. Run cloud/push_results.sh to git-commit + push the new results. ==="
