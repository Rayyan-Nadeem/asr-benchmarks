#!/usr/bin/env bash
# Run the bimodal hot-swap matrix on a case.
# Usage:  bash tools/run_matrix.sh <case-name>
#
# 6 rows per case:
#   1. speechmatics + native            (Speechmatics' full stack)
#   2. speechmatics + pyannote          (SM ASR + external diarizer)
#   3. whisper      + pyannote          (open-source stack)
#   4. whisper      + speechmatics_diar (Whisper words + SM speaker labels)
#   5. whisper      + none              (pure WER baseline)
#   6. (pyannote-only DER scored externally — see tools/pyannote_only.sh)
#
# Requires:
#   - Speechmatics container running on ws://localhost:9000/v2 (rows 1, 2, 4)
#   - openai-whisper installed (rows 3, 4, 5)
#   - HF_TOKEN env var with pyannote model access (rows 2, 3)
set -e

CASE="${1:-ami-es2004a-5min}"
cd "$(dirname "$0")/.."

echo "=== matrix: $CASE ==="

echo
echo "--- 1/5: speechmatics + native (full SM stack) ---"
"$(dirname "$0")/../.venv/bin/python" -m runner --engine speechmatics_onprem --case "$CASE" --diarizer native --tag matrix_sm_native

echo
echo "--- 2/5: speechmatics + pyannote (SM ASR + pyannote diar) ---"
"$(dirname "$0")/../.venv/bin/python" -m runner --engine speechmatics_onprem --case "$CASE" --diarizer pyannote --tag matrix_sm_pyannote

echo
echo "--- 3/5: whisper + pyannote (open-source stack) ---"
"$(dirname "$0")/../.venv/bin/python" -m runner --engine whisper --case "$CASE" --diarizer pyannote --tag matrix_whisper_pyannote

echo
echo "--- 4/5: whisper + speechmatics_diar (Whisper words + SM diar) ---"
"$(dirname "$0")/../.venv/bin/python" -m runner --engine whisper --case "$CASE" --diarizer speechmatics_diar --tag matrix_whisper_smdiar

echo
echo "--- 5/5: whisper + none (pure WER baseline) ---"
"$(dirname "$0")/../.venv/bin/python" -m runner --engine whisper --case "$CASE" --diarizer none --tag matrix_whisper_only

echo
echo "=== matrix complete for $CASE ==="
echo "render scoreboard: python3 tools/render_scoreboard.py"
