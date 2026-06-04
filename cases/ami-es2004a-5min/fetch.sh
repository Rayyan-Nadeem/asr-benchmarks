#!/usr/bin/env bash
# Reproducibly fetch + build the AMI ES2004a 5-min diarization stress case.
# Run from repo root:  bash cases/ami-es2004a-5min/fetch.sh
set -euo pipefail

CASE_DIR="cases/ami-es2004a-5min"
mkdir -p "$CASE_DIR"

curl -fsSL "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/ES2004a.Mix-Headset.wav" \
    -o "$CASE_DIR/source.wav"

ffmpeg -hide_banner -loglevel error -y \
    -ss 360 -t 300 -i "$CASE_DIR/source.wav" \
    -ac 1 -ar 16000 -sample_fmt s16 \
    "$CASE_DIR/audio.wav"

TMP_RTTM=$(mktemp)
curl -fsSL "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/only_words/rttms/test/ES2004a.rttm" \
    -o "$TMP_RTTM"

python3 - "$TMP_RTTM" "$CASE_DIR/reference.rttm" <<'PY'
import sys
inp, out = sys.argv[1], sys.argv[2]
WIN_START, WIN_END = 360, 660
lines = []
for line in open(inp):
    parts = line.strip().split()
    if not parts or parts[0] != 'SPEAKER':
        continue
    s, dur, spk = float(parts[3]), float(parts[4]), parts[7]
    e = s + dur
    if e <= WIN_START or s >= WIN_END:
        continue
    es = max(s, WIN_START) - WIN_START
    ee = min(e, WIN_END) - WIN_START
    lines.append(f"SPEAKER ami-es2004a-5min 1 {es:.3f} {ee-es:.3f} <NA> <NA> {spk} <NA> <NA>")
open(out, 'w').write('\n'.join(lines) + '\n')
print(f"wrote {out} ({len(lines)} turns)")
PY

rm -f "$TMP_RTTM"
echo "done. $CASE_DIR/ now has audio.wav + reference.rttm"
