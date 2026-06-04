"""Speechmatics realtime protocol — message dataclasses + (de)serialization.

Canonical reference: /Users/rayyannadeem/speechmaticsContainer/INTEGRATION_GUIDE.md.
Reference client: /Users/rayyannadeem/speechmaticsContainer/scripts/rt_test_client.py.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Client → server


@dataclass
class AudioFormat:
    type: str = "raw"
    encoding: str = "pcm_s16le"
    sample_rate: int = 16000


@dataclass
class StartRecognition:
    audio_format: AudioFormat
    transcription_config: dict[str, Any]
    message: str = "StartRecognition"

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "StartRecognition":
        af = raw.get("audio_format") or {}
        return cls(
            audio_format=AudioFormat(
                type=af.get("type", "raw"),
                encoding=af.get("encoding", "pcm_s16le"),
                sample_rate=int(af.get("sample_rate", 16000)),
            ),
            transcription_config=raw.get("transcription_config") or {},
        )


@dataclass
class EndOfStream:
    last_seq_no: int
    message: str = "EndOfStream"


# ---------------------------------------------------------------------------
# Server → client


_DEFAULT_LANGUAGE_PACK_INFO = {
    "adapted": False,
    "itn": True,
    "language_description": "English",
    "word_delimiter": " ",
    "writing_direction": "left-to-right",
}

# Honest version string — distinguishable from real SM in logs while keeping
# the field present so consumers that read it don't trip on KeyError.
ORCHESTRATOR_VERSION = "2026.06.01+depodash-asr+0.1.0"

# Pinned schema string. Matches what Speechmatics v15 ships so any client that
# asserts on schema version doesn't fail. We follow the same schema.
ENVELOPE_FORMAT = "2.9"


@dataclass
class RecognitionStarted:
    id: str
    message: str = "RecognitionStarted"
    orchestrator_version: str = ORCHESTRATOR_VERSION
    language_pack_info: dict[str, Any] = field(
        default_factory=lambda: dict(_DEFAULT_LANGUAGE_PACK_INFO)
    )


@dataclass
class Info:
    type: str
    reason: str
    quality: str
    message: str = "Info"


@dataclass
class AudioAdded:
    seq_no: int
    message: str = "AudioAdded"


@dataclass
class Word:
    start_time: float
    end_time: float
    alternatives: list[dict[str, Any]]
    type: str = "word"  # "word" | "punctuation"
    # Punctuation-only result fields. SM emits these on punctuation rows; on
    # word rows they are absent (not null) — to_dict respects that.
    attaches_to: str | None = None
    is_eos: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "alternatives": self.alternatives,
        }
        if self.attaches_to is not None:
            d["attaches_to"] = self.attaches_to
        if self.is_eos:
            d["is_eos"] = True
        return d


@dataclass
class AddPartialTranscript:
    transcript: str
    start_time: float
    end_time: float
    results: list[Word] = field(default_factory=list)
    message: str = "AddPartialTranscript"

    def to_json(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "format": ENVELOPE_FORMAT,
            "metadata": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "transcript": self.transcript,
            },
            "results": [w.to_dict() for w in self.results],
        }


@dataclass
class AddTranscript:
    transcript: str
    start_time: float
    end_time: float
    results: list[Word] = field(default_factory=list)
    message: str = "AddTranscript"

    def to_json(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "format": ENVELOPE_FORMAT,
            "metadata": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "transcript": self.transcript,
            },
            "results": [w.to_dict() for w in self.results],
        }


@dataclass
class EndOfTranscript:
    message: str = "EndOfTranscript"


@dataclass
class Error:
    # SM enum: invalid_message | invalid_model | invalid_config |
    # invalid_audio_type | not_authorised | quota_exceeded |
    # timelimit_exceeded | job_error
    type: str
    reason: str
    message: str = "Error"


@dataclass
class Warning_:
    reason: str
    message: str = "Warning"


# ---------------------------------------------------------------------------
# Helpers


def encode(msg: Any) -> str:
    if hasattr(msg, "to_json"):
        return json.dumps(msg.to_json())
    return json.dumps(asdict(msg))


def decode(raw: str) -> dict[str, Any]:
    return json.loads(raw)


def validate_audio_format(af: AudioFormat) -> Error | None:
    if af.type not in {"raw", "file"}:
        return Error("invalid_config", f"unsupported audio_format.type: {af.type!r}")
    if af.encoding != "pcm_s16le":
        return Error("invalid_config", f"only pcm_s16le supported; got {af.encoding!r}")
    if af.sample_rate != 16000:
        return Error("invalid_config", f"only 16000 Hz supported; got {af.sample_rate}")
    return None


def validate_transcription_config(cfg: dict[str, Any]) -> Error | None:
    diar = cfg.get("diarization", "none")
    if diar not in {"none", "speaker", "channel"}:
        return Error("invalid_config", f"invalid diarization mode: {diar!r}")
    if diar != "none":
        marks = (cfg.get("punctuation_overrides") or {}).get("permitted_marks") or []
        if not marks:
            return Error(
                "invalid_config",
                "punctuation_overrides.permitted_marks must be non-empty when diarization is enabled",
            )
    return None
