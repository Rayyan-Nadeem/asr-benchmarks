"""
Speechmatics on-prem engine adapter — talks to the rt-transcriber container's
WebSocket on `ws://localhost:9000/v2`.

Defaults to DepoDash's production cloud `transcription_config` (per Ed's C++
struct, 2026-04-22) so this benchmark measures the engine *as DepoDash uses
it*, not some out-of-the-box config nobody runs.

Lifted from speechmaticsContainer/scripts/dump_ws.py and extended with
per-frame wall-clock timestamps so the latency scorers have something to
chew on.
"""
from __future__ import annotations

import asyncio
import json
import time
import wave
from pathlib import Path
from typing import Any

import websockets

from engines._base import (
    Capabilities,
    FrameTimestamp,
    Result,
    RunMetadata,
    Word,
)


SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_BYTES = (SAMPLE_RATE * 2 * CHUNK_MS) // 1000
DEFAULT_URL = "ws://localhost:9000/v2"


# ---- DepoDash production config (Ed's C++ struct, 2026-04-22) ----
# Lifted verbatim from /mnt/c/Users/Admin/speechmaticsContainer/INTEGRATION_GUIDE.md §4.
# Every field is what their cloud client sends. Override per-call by passing a
# `transcription_config` overlay in the engine config.
DEPODASH_TRANSCRIPTION_CONFIG: dict[str, Any] = {
    "language": "en",
    "operating_point": "enhanced",
    "diarization": "speaker",
    "speaker_diarization_config": {"max_speakers": 10},
    "enable_partials": False,
    "max_delay": 5.0,           # solo would be 3.0; default to diarized
    "max_delay_mode": "flexible",
    # Required for diarization to emit correct speaker labels — see §5.2.
    "punctuation_overrides": {"permitted_marks": [",", ".", "?"]},
    "output_locale": "en-US",
    "enable_entities": False,
    # additional_vocab is per-case, never set globally.
}


CAPABILITIES = Capabilities(
    streaming=True,
    partials=True,
    diarization=True,
    speaker_fingerprinting=False,  # Speechmatics does not maintain cross-session ID
    custom_vocab=True,
    word_timing=True,
    confidence_scores=True,
    languages=["en"],  # this build; English-only language pack
    on_prem=True,
    notes="Speechmatics rt-asr-transcriber-en:15.0.0 + sm-gpu-inference-server-en:15.0.0",
)


def _load_pcm_s16le_16k(path: Path) -> tuple[bytes, float]:
    """Load a wav, return (pcm_bytes, duration_seconds). Engine sees only PCM."""
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SAMPLE_RATE:
            raise ValueError(
                f"{path}: must be mono PCM S16LE @ {SAMPLE_RATE} Hz "
                f"(got ch={w.getnchannels()} width={w.getsampwidth()} rate={w.getframerate()})"
            )
        pcm = w.readframes(w.getnframes())
        duration = w.getnframes() / SAMPLE_RATE
    return pcm, duration


def _normalize_words(raw_dump: list[dict]) -> list[Word]:
    """Walk the WS frames and pull every word/punct out of AddTranscript results."""
    words: list[Word] = []
    for frame in raw_dump:
        if frame.get("message") != "AddTranscript":
            continue
        for r in frame.get("results", []):
            alt = (r.get("alternatives") or [{}])[0]
            content = alt.get("content", "")
            if not content:
                continue
            words.append(
                Word(
                    content=content,
                    start_time=float(r.get("start_time", 0.0)),
                    end_time=float(r.get("end_time", 0.0)),
                    speaker=r.get("speaker") or alt.get("speaker"),
                    confidence=alt.get("confidence"),
                    is_punctuation=r.get("type") == "punctuation",
                    language=alt.get("language"),
                )
            )
    return words


def _join_transcript(words: list[Word]) -> str:
    """Concatenate normalized words with proper punctuation attachment."""
    out: list[str] = []
    for w in words:
        if w.is_punctuation and out:
            out[-1] = out[-1] + w.content
        else:
            out.append(w.content)
    return " ".join(out)


class SpeechmaticsOnPrem:
    name = "speechmatics_onprem"
    capabilities = CAPABILITIES

    def __init__(self, url: str = DEFAULT_URL):
        self.url = url

    async def transcribe(
        self,
        audio: Path,
        config: dict[str, Any] | None = None,
    ) -> Result:
        """
        config schema (all optional):
            url: str                    — override ws endpoint
            transcription_config: dict  — overlay on DEPODASH_TRANSCRIPTION_CONFIG
            fast: bool                  — send audio as fast as WS allows (default: 1× realtime)
        """
        config = config or {}
        url = config.get("url", self.url)
        fast = bool(config.get("fast", False))

        # Build effective transcription_config: DepoDash defaults + caller overlay
        tc = {**DEPODASH_TRANSCRIPTION_CONFIG, **(config.get("transcription_config") or {})}
        # If diarization turned off, drop the per-diarization sub-config
        if tc.get("diarization") == "none":
            tc.pop("speaker_diarization_config", None)
            # When solo, max_delay should default to 3.0 unless caller said otherwise
            if "max_delay" not in (config.get("transcription_config") or {}):
                tc["max_delay"] = 3.0

        pcm, duration_s = _load_pcm_s16le_16k(audio)

        raw_dump: list[dict] = []
        meta = RunMetadata(
            audio_duration_s=duration_s,
            wall_clock_start=0.0,
            wall_clock_end=0.0,
            first_audio_send=0.0,
        )
        error: str | None = None

        # Disable client keepalive — in --fast mode we send all audio quickly then
        # idle waiting for finals, easily exceeding the default 20s ping timeout.
        async with websockets.connect(url, max_size=None, ping_interval=None) as ws:
            start_msg = {
                "message": "StartRecognition",
                "audio_format": {
                    "type": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": SAMPLE_RATE,
                },
                "transcription_config": tc,
            }
            raw_dump.append({"_direction": "out", **start_msg})
            await ws.send(json.dumps(start_msg))

            async def reader() -> None:
                nonlocal error
                async for message in ws:
                    now = time.monotonic()
                    if isinstance(message, (bytes, bytearray)):
                        raw_dump.append(
                            {"_direction": "in", "_t": now, "message": "_binary", "bytes": len(message)}
                        )
                        continue
                    try:
                        msg = json.loads(message)
                    except json.JSONDecodeError:
                        raw_dump.append({"_direction": "in", "_t": now, "message": "_unparsed", "raw": message})
                        continue

                    raw_dump.append({"_direction": "in", "_t": now, **msg})
                    kind = msg.get("message")

                    if kind == "AddPartialTranscript":
                        if meta.first_partial_recv is None:
                            meta.first_partial_recv = now
                        end_t = float(msg.get("metadata", {}).get("end_time", 0.0))
                        meta.frames.append(FrameTimestamp(end_t, now, kind))
                    elif kind == "AddTranscript":
                        if meta.first_final_recv is None:
                            meta.first_final_recv = now
                        end_t = float(msg.get("metadata", {}).get("end_time", 0.0))
                        meta.frames.append(FrameTimestamp(end_t, now, kind))
                    elif kind == "Error":
                        error = f"{msg.get('type')}: {msg.get('reason')}"
                        return
                    elif kind == "EndOfTranscript":
                        return

            reader_task = asyncio.create_task(reader())

            meta.wall_clock_start = time.monotonic()
            seq = 0
            for i in range(0, len(pcm), CHUNK_BYTES):
                if seq == 0:
                    meta.first_audio_send = time.monotonic()
                await ws.send(pcm[i : i + CHUNK_BYTES])
                seq += 1
                if not fast:
                    await asyncio.sleep(CHUNK_MS / 1000)

            eos = {"message": "EndOfStream", "last_seq_no": seq}
            raw_dump.append({"_direction": "out", **eos})
            await ws.send(json.dumps(eos))
            await reader_task
            meta.wall_clock_end = time.monotonic()

        words = _normalize_words(raw_dump)
        return Result(
            engine=self.name,
            audio_path=audio,
            config={"url": url, "fast": fast, "transcription_config": tc},
            words=words,
            transcript=_join_transcript(words),
            raw_dump=raw_dump,
            metadata=meta,
            error=error,
        )
