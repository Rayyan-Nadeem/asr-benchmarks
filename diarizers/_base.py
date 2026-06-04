"""
Diarizer protocol — the bimodal hot-swap contract.

Any diarizer takes audio in, returns a speaker timeline. The runner pairs an
engine (which produces words + word timestamps) with a diarizer (which produces
the speaker timeline) and overlays the two by timestamp lookup. This means any
ASR can be benchmarked against any diarizer without each engine knowing about
each diarizer.

Tuple shape: (speaker_label, start_seconds, end_seconds) — same shape simpleder
expects for DER, same shape RTTM uses on disk (modulo the file format).
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

DiarSegment = tuple[str, float, float]  # (speaker, start, end)


@runtime_checkable
class Diarizer(Protocol):
    """Returns a list of speaker turns for the given audio file."""
    name: str
    requires_token: bool

    async def diarize(self, audio: Path) -> list[DiarSegment]:
        ...


def overlay_speakers(words, segments: list[DiarSegment]) -> None:
    """
    Mutate `words` in place — set each Word.speaker to whichever segment covers
    its midpoint. Words outside any segment get speaker=None.

    Cheap O(W * S) lookup since W (~hundreds) and S (~tens) for typical 5-min
    benchmarks. If we ever scale this to long audio, switch to interval tree.
    """
    for w in words:
        mid = (w.start_time + w.end_time) / 2.0
        match: str | None = None
        for spk, s, e in segments:
            if s <= mid < e:
                match = spk
                break
        w.speaker = match
