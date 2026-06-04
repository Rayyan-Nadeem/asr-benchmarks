"""StreamingDiarizer protocol — assigns speaker labels to engine word output."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from server.engines._base import StreamWord


@dataclass
class SpeakerTurn:
    speaker: str
    start_time: float
    end_time: float


class StreamingDiarizer(Protocol):
    name: str

    async def label(self, words: list[StreamWord], audio_chunk: bytes | None = None) -> list[StreamWord]:
        """Return the same words with `speaker` populated."""
        ...

    async def turns_to_now(self, audio_chunk: bytes) -> list[SpeakerTurn]:
        """Streaming diarizer's view of speaker turns up to this point in audio."""
        ...
