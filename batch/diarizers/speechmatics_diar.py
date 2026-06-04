"""
Speechmatics diarizer — runs the Speechmatics on-prem container with
diarization=speaker, then extracts only the speaker timeline (RTTM segments)
from the resulting word stream. Discards the actual transcript.

Why: lets us pair Speechmatics' speaker labels with a different engine's
words (e.g. Whisper ASR + Speechmatics diar), to isolate which side of
the pipeline is the weak link.

Caches per-audio segments to /tmp/asr-benchmarks-diar-cache/ so a single
extraction can be reused across multiple Whisper runs without keeping the
Speechmatics container running. Critical for VRAM management — Whisper
needs ~15.7 GB and the SM container holds ~7.7 GB pinned, so they cannot
coexist on a 16 GB card.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from diarizers._base import DiarSegment
from engines.speechmatics_onprem import SpeechmaticsOnPrem


CACHE_DIR = Path("/tmp/asr-benchmarks-diar-cache")


class SpeechmaticsDiarizer:
    name = "speechmatics_diar"
    requires_token = False

    def __init__(self, url: str = "ws://localhost:9000/v2"):
        self._sm = SpeechmaticsOnPrem(url=url)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, audio: Path) -> Path:
        h = hashlib.sha256(str(audio.resolve()).encode()).hexdigest()[:16]
        return CACHE_DIR / f"sm_diar_{audio.stem}_{h}.json"

    async def diarize(self, audio: Path) -> list[DiarSegment]:
        cache = self._cache_path(audio)
        if cache.exists():
            return [tuple(seg) for seg in json.loads(cache.read_text())]

        # Cache miss — run Speechmatics with diarization on, harvest RTTM only.
        result = await self._sm.transcribe(
            audio,
            config={"transcription_config": {"diarization": "speaker"}},
        )
        segs: list[list[float | str]] = []
        last_speaker: str | None = None
        for w in result.words:
            if w.speaker is None:
                continue
            if last_speaker == w.speaker and segs:
                segs[-1][2] = w.end_time
            else:
                segs.append([w.speaker, w.start_time, w.end_time])
                last_speaker = w.speaker
        out = [(str(s[0]), float(s[1]), float(s[2])) for s in segs if s[2] > s[1]]
        cache.write_text(json.dumps(out))
        return out
