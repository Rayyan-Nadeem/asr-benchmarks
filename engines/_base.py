"""
Common interface every engine adapter implements.

The contract: same `transcribe(audio, config)` signature, same canonical
`Result` shape out, regardless of whether the engine speaks WebSockets,
gRPC, runs in-process via Python, or shells out to a binary.

Scorers consume the normalized `words` list. The original engine-native
output is preserved on `Result.raw_dump` so we never lose information.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class Word:
    """One word from the engine output, normalized across vendors."""
    content: str
    start_time: float           # seconds from start of audio
    end_time: float
    speaker: str | None = None  # e.g. "S1", or None if engine didn't diarize
    confidence: float | None = None
    is_punctuation: bool = False
    language: str | None = None


@dataclass
class FrameTimestamp:
    """Wall-clock receipt of one engine output frame, used for latency metrics."""
    audio_end_time: float       # the audio time covered up to this frame (seconds)
    wall_time: float            # time.monotonic() when frame arrived at the harness
    kind: str                   # "RecognitionStarted", "AddTranscript", "AddPartialTranscript", etc.


@dataclass
class RunMetadata:
    """Per-run book-keeping used by latency + RTF scorers."""
    audio_duration_s: float
    wall_clock_start: float     # time.monotonic() when first audio chunk was sent
    wall_clock_end: float       # time.monotonic() when EndOfTranscript was received
    first_audio_send: float     # time.monotonic() of first audio frame out
    first_final_recv: float | None = None   # first AddTranscript arrival
    first_partial_recv: float | None = None # first AddPartialTranscript arrival
    frames: list[FrameTimestamp] = field(default_factory=list)


@dataclass
class Result:
    """What every engine adapter returns."""
    engine: str                 # e.g. "speechmatics_onprem"
    audio_path: Path
    config: dict[str, Any]      # snapshot of the actual config used (for reproducibility)
    words: list[Word]           # normalized, scorer-ready
    transcript: str             # joined plain text (convenience)
    raw_dump: list[dict]        # engine-native frames, one dict per frame
    metadata: RunMetadata
    error: str | None = None    # set if the engine raised mid-stream


@dataclass
class Capabilities:
    """Static description of what an engine can do. Surfaced in the scoreboard."""
    streaming: bool                 # can it emit results before audio ends
    partials: bool                  # does it emit AddPartialTranscript-style mid-utterance updates
    diarization: bool               # native speaker labels
    speaker_fingerprinting: bool    # cross-session speaker identification
    custom_vocab: bool              # additional_vocab / phrase boosts
    word_timing: bool               # per-word start/end timestamps
    confidence_scores: bool         # per-word confidence
    languages: list[str]            # ISO codes the engine supports for this build
    on_prem: bool                   # runs without phoning home
    notes: str = ""


@runtime_checkable
class Engine(Protocol):
    """Every engine adapter implements this."""
    name: str
    capabilities: Capabilities

    async def transcribe(self, audio: Path, config: dict[str, Any]) -> Result:
        ...
