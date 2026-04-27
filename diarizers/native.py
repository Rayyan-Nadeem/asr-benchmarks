"""
Native diarizer — passthrough. The engine already labelled the words during
ASR; we don't touch them. Used implicitly when no --diarizer flag is passed.
"""
from __future__ import annotations

from pathlib import Path

from diarizers._base import DiarSegment


class NativeDiarizer:
    name = "native"
    requires_token = False

    async def diarize(self, audio: Path) -> list[DiarSegment]:
        # No-op: native diarization is already in the engine's word stream.
        # Returning [] signals "do not overlay; leave engine speakers alone".
        return []
