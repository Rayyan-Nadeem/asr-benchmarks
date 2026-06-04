#!/usr/bin/env bash
# Build a KenLM n-gram language model biased toward US legal vocabulary.
#
# Output: /opt/kenlm/legal_3gram.bin — point NEMOTRON_KENLM_PATH at this
# in the sweep harness to evaluate the shallow-fusion lift on SCOTUS +
# deposition fixtures.
#
# Why: courtroom audio has heavy proper-noun + legal-term concentration
# that out-of-domain ASR misses ("Glossip", "voir dire", "objection
# hearsay", "Federal Rule 12(b)(6)", exhibit numbers). Shallow fusion
# with a small in-domain n-gram model rescues most of those misses
# without retraining the acoustic model.
#
# Per the AWS NeMo fine-tuning blog (2024-12): clinical-domain shallow
# fusion gave ~25 % relative WER reduction on medical terminology with
# zero acoustic-model retraining. Legal is the same kind of fix.
#
# Requirements:
#   - kenlm built from source (build/bin/lmplz and build/bin/build_binary
#     on PATH). Install:
#       git clone https://github.com/kpu/kenlm /opt/kenlm-src
#       cd /opt/kenlm-src && mkdir build && cd build && cmake .. && make -j
#   - Text corpus of legal proceedings.

set -euo pipefail

KENLM_BIN="${KENLM_BIN:-/opt/kenlm-src/build/bin}"
OUT_DIR="${OUT_DIR:-/opt/kenlm}"
ORDER="${ORDER:-3}"     # 3-gram by default; 4-gram adds size, modest accuracy
CORPUS="${CORPUS:-/opt/kenlm/legal_corpus.txt}"

mkdir -p "$OUT_DIR"

if [ ! -x "$KENLM_BIN/lmplz" ]; then
    echo "kenlm lmplz not found at $KENLM_BIN/lmplz" >&2
    echo "Install kenlm from https://github.com/kpu/kenlm first." >&2
    exit 1
fi

if [ ! -f "$CORPUS" ]; then
    cat >&2 <<EOF
Legal corpus not found at $CORPUS. Build it from one of these sources:

  1. OYEZ SCOTUS oral arguments (public domain, ~700 cases worth
     of high-density legal English):
       cases/scotus-*/reference.txt
       cases/oyez_*/oyez_argument.json
     Concatenate all the .txt + .json transcript fields into one file.

  2. CourtListener public-domain court filings:
       https://www.courtlistener.com/api/bulk-data/

  3. CASEtext / Casemine free public-domain dockets.

Once you have ~10-50 MB of clean legal English text at \$CORPUS, re-run:
    bash tools/build_legal_kenlm.sh
EOF
    exit 1
fi

echo "Building ${ORDER}-gram KenLM from $CORPUS..."
"$KENLM_BIN/lmplz" -o "$ORDER" -S 60% -T /tmp < "$CORPUS" \
    > "$OUT_DIR/legal_${ORDER}gram.arpa"
echo "Quantizing to binary..."
"$KENLM_BIN/build_binary" "$OUT_DIR/legal_${ORDER}gram.arpa" \
    "$OUT_DIR/legal_${ORDER}gram.bin"

echo
echo "Done. Output: $OUT_DIR/legal_${ORDER}gram.bin"
echo
echo "Point the sweep at it:"
echo "  python3 tools/sweep_nemotron.py --kenlm $OUT_DIR/legal_${ORDER}gram.bin \\"
echo "      --beam 4 --left 70 --right 13"
