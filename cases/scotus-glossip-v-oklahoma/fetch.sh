#!/usr/bin/env bash
# Reproducibly fetch the Oyez audio + parsed reference for SCOTUS 22-7466.
# Run from the repo root:  bash cases/scotus-glossip-v-oklahoma/fetch.sh
set -euo pipefail

CASE_DIR="cases/scotus-glossip-v-oklahoma"
mkdir -p "$CASE_DIR"

curl -fsSL "https://s3.amazonaws.com/oyez.case-media.mp3/case_data/2024/22-7466/22-7466_20241009-argument.delivery.mp3" \
    -o "$CASE_DIR/audio.mp3"

curl -fsSL "https://api.oyez.org/cases/2024/22-7466" -A "Mozilla/5.0" \
    -o "$CASE_DIR/oyez_case.json"

curl -fsSL "https://api.oyez.org/case_media/oral_argument_audio/25657" -A "Mozilla/5.0" \
    -o "$CASE_DIR/oyez_argument.json"

python3 tools/oyez_to_reference.py "$CASE_DIR/oyez_argument.json"

echo "done. cases/$(basename $CASE_DIR)/ now has audio.mp3, reference.txt, reference.rttm"
