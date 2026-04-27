"""
Run pyannote.audio standalone on a case's audio and score DER directly.
This is the "diarizer ceiling" — best DER any pyannote-based stack could
achieve on this audio, with no ASR involved.

Writes a synthetic run JSON so the result shows up in SCOREBOARD as a
`(engine=_pyannote_only_, diarizer=pyannote)` row.

Usage:  HF_TOKEN=... python3 tools/pyannote_only.py <case-name>
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from diarizers.pyannote import PyannoteDiarizer
from metrics.diarization import score_der


async def main():
    if len(sys.argv) < 2:
        sys.exit("usage: HF_TOKEN=... python3 tools/pyannote_only.py <case-name>")
    case_name = sys.argv[1]
    case_dir = REPO_ROOT / "cases" / case_name
    audio = case_dir / "audio.wav"
    rttm = case_dir / "reference.rttm"
    if not audio.exists() or not rttm.exists():
        sys.exit(f"case {case_name} missing audio.wav or reference.rttm")

    print(f"[pyannote-only] case: {case_name}")
    diar = PyannoteDiarizer()
    t0 = time.monotonic()
    segments = await diar.diarize(audio)
    wall = time.monotonic() - t0
    print(f"[pyannote-only] {len(segments)} turns, {wall:.1f}s wall-clock")

    der = score_der(rttm, segments)
    print(f"[pyannote-only] DER: {der.der*100:.2f}%  speakers ref/hyp: {der.speaker_count_ref}/{der.speaker_count_hyp}")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = f"{timestamp}___pyannote_only___{case_name}__diar-pyannote"
    out = REPO_ROOT / "results" / "runs" / f"{run_id}.json"
    out.write_text(json.dumps({
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "engine": "_pyannote_only_",
        "case": case_name,
        "diarizer": "pyannote",
        "tag": "ceiling",
        "scores": {
            "der": {"der": der.der, "speaker_count_ref": der.speaker_count_ref, "speaker_count_hyp": der.speaker_count_hyp},
            "diarizer": "pyannote",
            "latency": {"ttft_ms": None, "ttfp_ms": None, "final_lag_p50_ms": None,
                        "final_lag_p95_ms": None, "final_lag_p99_ms": None,
                        "rtf": wall / (segments[-1][2] if segments else 1), "finals_count": 0, "partials_count": 0},
            "resources": {},
            "confidence": {"words_scored": 0, "mean": None, "p10": None, "low_conf_rate": None},
        },
        "result": {"transcript": "", "words": [], "raw_dump": [
            {"_direction": "in", "message": "PyannoteRTTM", "speaker": s, "start": st, "end": e}
            for s, st, e in segments
        ]},
    }, indent=2))
    print(f"[pyannote-only] → {out}")


if __name__ == "__main__":
    asyncio.run(main())
