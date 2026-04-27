"""
Whisper engine adapter — OpenAI's `openai-whisper` package, model `large-v3`.
The "better, not faster" choice: full-precision PyTorch, not quantized
faster-whisper, not the smaller turbo variant.

This is **batch / offline** transcription: the entire audio is processed at
once, then the result is post-hoc bucketed back into per-segment timestamps.
That means:
  - WER is comparable to streaming engines.
  - DER (when paired with a diarizer) is comparable.
  - Latency metrics (TTFT, per-final lag) are NOT comparable to streaming engines.
    Whisper has no "first token" — it returns everything in one shot. The
    LatencyReport will report `first_final_recv` once at end-of-audio, so
    TTFT effectively equals "wall-clock time to transcribe the full file".

Use this engine when you care about accuracy + the diarization-from-pyannote
combination, not when comparing live-feel responsiveness against streaming
engines.

Default config:
  model = "large-v3"
  language = "en"
  word_timestamps = True   (required so diarizers can overlay speakers)
  fp16 = auto-detect from CUDA availability
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from engines._base import (
    Capabilities,
    FrameTimestamp,
    Result,
    RunMetadata,
    Word,
)


CAPABILITIES = Capabilities(
    streaming=False,
    partials=False,
    diarization=False,             # not built-in; pair with a diarizer
    speaker_fingerprinting=False,
    custom_vocab=False,            # initial_prompt is loosely similar; out of scope today
    word_timing=True,              # via word_timestamps=True
    confidence_scores=False,       # whisper exposes avg_logprob per segment, not per word; we don't surface
    languages=["en", "*"],         # multilingual; we run en here
    on_prem=True,
    notes="openai-whisper large-v3, batch (offline). Pair with a diarizer for DER.",
)


_MODEL = None  # cached; loading is expensive (~3 GB weights)


def _get_model(name: str = "large-v3"):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import whisper  # imported lazily; heavy
    _MODEL = whisper.load_model(name)
    return _MODEL


class WhisperOpenai:
    name = "whisper"
    capabilities = CAPABILITIES

    def __init__(self, model: str = "large-v3"):
        self._model_name = model

    async def transcribe(self, audio: Path, config: dict[str, Any] | None = None) -> Result:
        """
        config schema (all optional):
            model: str        — override default "large-v3"
            language: str     — default "en"
            initial_prompt: str — optional context to bias decoding
        """
        config = config or {}
        model_name = config.get("model", self._model_name)
        language = config.get("language", "en")
        initial_prompt = config.get("initial_prompt")

        # Compute audio duration (mono PCM 16k WAV by harness convention)
        import wave
        with wave.open(str(audio), "rb") as w:
            duration_s = w.getnframes() / w.getframerate()

        meta = RunMetadata(
            audio_duration_s=duration_s,
            wall_clock_start=time.monotonic(),
            wall_clock_end=0.0,
            first_audio_send=time.monotonic(),
        )

        # Run Whisper in a thread (model.transcribe is sync + blocking)
        result_dict = await asyncio.to_thread(
            self._transcribe_sync, audio, model_name, language, initial_prompt
        )
        meta.wall_clock_end = time.monotonic()
        meta.first_final_recv = meta.wall_clock_end  # batch: result arrives once at end

        words = self._extract_words(result_dict)
        # Single synthetic frame so the latency scorer has something to chew on
        if words:
            meta.frames.append(FrameTimestamp(
                audio_end_time=words[-1].end_time,
                wall_time=meta.wall_clock_end,
                kind="AddTranscript",
            ))

        return Result(
            engine=self.name,
            audio_path=audio,
            config={
                "model": model_name,
                "language": language,
                "initial_prompt": initial_prompt,
                "_mode": "batch",
            },
            words=words,
            transcript=result_dict.get("text", "").strip(),
            raw_dump=[{"_direction": "in", "message": "WhisperResult", **{k: v for k, v in result_dict.items() if k != "segments"}},
                      *[{"_direction": "in", "message": "WhisperSegment", **s} for s in result_dict.get("segments", [])]],
            metadata=meta,
            error=None,
        )

    def _transcribe_sync(self, audio: Path, model_name: str, language: str, initial_prompt: str | None) -> dict:
        import torch
        model = _get_model(model_name)
        return model.transcribe(
            str(audio),
            language=language,
            word_timestamps=True,
            fp16=torch.cuda.is_available(),
            initial_prompt=initial_prompt,
            verbose=False,
        )

    def _extract_words(self, result_dict: dict) -> list[Word]:
        """Pull per-word entries out of Whisper's segment list."""
        out: list[Word] = []
        for seg in result_dict.get("segments", []):
            for w in seg.get("words", []) or []:
                content = (w.get("word") or "").strip()
                if not content:
                    continue
                out.append(Word(
                    content=content,
                    start_time=float(w.get("start", seg.get("start", 0.0))),
                    end_time=float(w.get("end", seg.get("end", 0.0))),
                    speaker=None,                    # set by diarizer overlay
                    confidence=w.get("probability"), # whisper exposes a per-word probability
                    is_punctuation=False,
                ))
        return out
