#!/usr/bin/env bash
# Reproducibly build the librispeech-test-clean-mini case.
# Run from repo root:  bash cases/librispeech-test-clean-mini/fetch.sh
set -euo pipefail

CASE_DIR="cases/librispeech-test-clean-mini"
TMP=$(mktemp -d)

# Download the test-clean tarball (~346 MB) if we don't have it cached
TARBALL="$TMP/test-clean.tar.gz"
curl -fsSL https://www.openslr.org/resources/12/test-clean.tar.gz -o "$TARBALL"
tar -xzf "$TARBALL" -C "$TMP"

SRC="$TMP/LibriSpeech/test-clean/1089/134686"

# Concat 10 utterances → mono 16k wav
INPUTS=()
for i in 0000 0001 0002 0003 0004 0005 0006 0007 0008 0009; do
    INPUTS+=("-i" "$SRC/1089-134686-$i.flac")
done
ffmpeg -hide_banner -loglevel error -y \
    "${INPUTS[@]}" \
    -filter_complex "concat=n=10:v=0:a=1" \
    -ac 1 -ar 16000 -sample_fmt s16 \
    "$CASE_DIR/audio.wav"

# Build reference.txt by joining the canonical transcripts
python3 - "$SRC/1089-134686.trans.txt" "$CASE_DIR/reference.txt" <<'PY'
import sys
trans_path, out_path = sys.argv[1], sys.argv[2]
ids = ["1089-134686-" + str(i).zfill(4) for i in range(10)]
text = " ".join(
    line.split(" ", 1)[1].strip()
    for line in open(trans_path)
    if line.split(" ", 1)[0] in ids
)
open(out_path, "w").write(text.lower().strip() + "\n")
PY

rm -rf "$TMP"
echo "done. cases/librispeech-test-clean-mini/ now has audio.wav + reference.txt"
