"""Compute streaming DER for a result JSON against a reference RTTM.

Takes the same result-JSON format that `measure_via_ws.py` writes, extracts
per-word speaker labels from the `finals[*].results[*].alternatives[0].speaker`
path, builds a hypothesis pyannote Annotation by merging adjacent same-speaker
words into segments, and scores against a reference RTTM with
`pyannote.metrics.diarization.DiarizationErrorRate(collar=0.25)` — the
standard NIST RT-style collar that matches the SM baseline measurements
already in `results/SCOREBOARD.md`.

Also computes:
  - speaker count (hypothesis vs reference)
  - per-speaker word totals
  - label-stability proxy: per-speaker max contiguous run length /
    total words for that speaker (proxy for how "chunky" labels are
    vs. flip-flop noisy)

Usage:
    python3 tools/score_der.py <result.json> <reference.rttm>
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate


def _words_from_result(result_json: Path):
    d = json.loads(result_json.read_text())
    finals = d.get("measured", {}).get("finals", []) or []
    audio_duration = d.get("measured", {}).get("audio_duration_s") or 0.0
    words = []
    for f in finals:
        for w in f.get("results", []) or []:
            if w.get("type") != "word":
                continue
            alt = (w.get("alternatives") or [{}])[0]
            spk = w.get("speaker") or alt.get("speaker")
            if not spk:
                continue
            st = w.get("start_time") or alt.get("start_time")
            en = w.get("end_time") or alt.get("end_time")
            if st is None or en is None or en <= st:
                continue
            words.append((float(st), float(en), str(spk)))
    return words, audio_duration


def _hyp_annotation(words, pad: float = 0.1):
    """Merge adjacent same-speaker words into segments for DER scoring.

    Pad each word by `pad` seconds on both sides before merging — inter-
    word gaps in streaming hypotheses are natural prosody silence, not
    "missed speech." Without padding, every fluent pause counts against
    DER inflating the missed-detection component by 50–90 pp. 100 ms
    aligns roughly with how the pyannote-eval collar treats boundaries.
    """
    ann = Annotation()
    if not words:
        return ann
    words = sorted(words, key=lambda w: w[0])
    cur_spk = words[0][2]
    cur_st = max(0.0, words[0][0] - pad)
    cur_en = words[0][1] + pad
    seg_idx = 0
    for st, en, spk in words[1:]:
        st_p = max(0.0, st - pad)
        en_p = en + pad
        # Merge if same speaker and the padded segments touch/overlap.
        if spk == cur_spk and st_p <= cur_en:
            cur_en = max(cur_en, en_p)
        else:
            ann[Segment(cur_st, cur_en), seg_idx] = cur_spk
            seg_idx += 1
            cur_spk, cur_st, cur_en = spk, st_p, en_p
    ann[Segment(cur_st, cur_en), seg_idx] = cur_spk
    return ann


def _label_stability(words) -> dict[str, float]:
    """For each speaker, fraction of the speaker's words that are part of
    the LONGEST contiguous run (in time order). 1.0 = the speaker's words
    are one continuous block; 0.0 means every word flip-flops with another
    speaker. Higher = more coherent label, less editor pain."""
    if not words:
        return {}
    words = sorted(words, key=lambda w: w[0])
    runs_by_spk: dict[str, list[int]] = defaultdict(list)
    cur_spk = words[0][2]
    cur_run = 1
    for _, _, spk in words[1:]:
        if spk == cur_spk:
            cur_run += 1
        else:
            runs_by_spk[cur_spk].append(cur_run)
            cur_spk, cur_run = spk, 1
    runs_by_spk[cur_spk].append(cur_run)
    spk_totals = Counter(w[2] for w in words)
    return {
        spk: max(runs_by_spk[spk]) / spk_totals[spk]
        for spk in spk_totals
        if spk_totals[spk] > 0
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    result_path = Path(sys.argv[1])
    ref_path = Path(sys.argv[2])

    words, audio_duration = _words_from_result(result_path)
    if not words:
        print(json.dumps({"error": "no words with speaker labels in result"}))
        sys.exit(1)

    hyp = _hyp_annotation(words)
    ref_rttms = load_rttm(str(ref_path))
    if not ref_rttms:
        print(json.dumps({"error": f"no annotations in {ref_path}"}))
        sys.exit(1)
    ref = next(iter(ref_rttms.values()))

    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    der = metric(ref, hyp, detailed=True)

    spk_totals = Counter(w[2] for w in words)
    stability = _label_stability(words)

    total = der["total"]
    def _pct(key: str) -> float | None:
        return round(der[key] / total * 100, 2) if total > 0 else None

    report = {
        "result": str(result_path),
        "reference": str(ref_path),
        "audio_duration_s": audio_duration,
        "DER": round(der["diarization error rate"] * 100, 2),
        "false_alarm_pct": _pct("false alarm"),
        "missed_pct": _pct("missed detection"),
        "confusion_pct": _pct("confusion"),
        "n_ref_speakers": len(ref.labels()),
        "n_hyp_speakers": len(hyp.labels()),
        "ref_speakers": sorted(ref.labels()),
        "hyp_speakers": sorted(hyp.labels()),
        "hyp_words_per_speaker": dict(spk_totals.most_common()),
        "label_stability_per_speaker": {
            spk: round(stab, 3) for spk, stab in stability.items()
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
