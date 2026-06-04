"""StreamingEngine protocol — the contract every engine in `server/engines/` honors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterable, Protocol


@dataclass
class StreamPartial:
    """Best-current-hypothesis text; replaceable by a later partial or a final."""
    transcript: str
    start_time: float
    end_time: float


@dataclass
class StreamWord:
    content: str
    start_time: float
    end_time: float
    confidence: float | None = None
    speaker: str | None = None
    is_punctuation: bool = False


@dataclass
class StreamFinal:
    """Committed segment with per-word timestamps."""
    transcript: str
    words: list[StreamWord]
    start_time: float
    end_time: float


StreamEvent = StreamPartial | StreamFinal


class StreamingEngine(Protocol):
    name: str

    async def stream(
        self,
        pcm_frames: AsyncIterable[bytes],
        transcription_config: dict,
    ) -> AsyncIterable[StreamEvent]:
        """Consume 16 kHz mono PCM S16LE frames; yield partials + finals as they emit."""
        ...
