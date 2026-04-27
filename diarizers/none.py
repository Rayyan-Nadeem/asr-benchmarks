"""
None diarizer — strip speaker labels entirely. Used for pure-WER runs where
diarization isn't meaningful (e.g. single-speaker audio) or for ASR-only
ceiling measurements.
"""
from __future__ import annotations

from pathlib import Path

from diarizers._base import DiarSegment


class NoneDiarizer:
    name = "none"
    requires_token = False

    async def diarize(self, audio: Path) -> list[DiarSegment]:
        # Returning [{(speaker, 0, +inf)}] would force everyone to one label;
        # returning [] tells the runner to skip overlay. The engine's words
        # keep whatever speaker (or None) they had.
        return []
