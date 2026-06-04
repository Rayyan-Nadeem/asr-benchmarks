"""
MLX-Whisper engine adapter — Apple's MLX framework port of Whisper.

Native Apple Silicon. Uses unified memory + Metal (and the Neural Engine where
possible). Typically 2-3× faster than CTranslate2 / openai-whisper on the same
M-series chip, with similar accuracy.

This adapter is a Mac-specific comparison data point, not a candidate for the
final locked recommendation — we want a single binary that runs on all four
deployment shapes, not separate Mac-only and non-Mac code paths. Parakeet ONNX
already serves that role hardware-agnostically.

Install (macOS arm64 only):
    pip install mlx-whisper

Default model: mlx-community/whisper-large-v3-mlx (~3 GB weights, auto-downloaded
on first use to ~/.cache/huggingface/hub/).
"""
from __future__ import annotations

import asyncio
import platform
import time
from pathlib import Path
from typing import Any

from engines._audio import audio_duration_s
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
    diarization=False,
    speaker_fingerprinting=False,
    custom_vocab=False,
    word_timing=True,
    confidence_scores=True,        # via word.probability
    languages=["en", "*"],
    on_prem=True,
    notes="MLX-Whisper large-v3 on Apple Silicon (Metal + ANE). macOS arm64 only.",
)


def _platform_ok() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64")


class MlxWhisper:
    name = "mlx_whisper"
    capabilities = CAPABILITIES

    def __init__(self, model: str = "mlx-community/whisper-large-v3-mlx"):
        self._model = model

    async def transcribe(self, audio: Path, config: dict[str, Any] | None = None) -> Result:
        """
        config schema (all optional):
            model: str          — HF repo for an MLX-converted Whisper model
            language: str       — default "en"
            initial_prompt: str — optional context for biasing decoding
        """
        if not _platform_ok():
            raise RuntimeError(
                f"mlx_whisper requires macOS arm64. Current platform: "
                f"{platform.system()} {platform.machine()}"
            )

        config = config or {}
        model = config.get("model", self._model)
        language = config.get("language", "en")
        initial_prompt = config.get("initial_prompt")

        duration_s = audio_duration_s(audio)

        meta = RunMetadata(
            audio_duration_s=duration_s,
            wall_clock_start=time.monotonic(),
            wall_clock_end=0.0,
            first_audio_send=time.monotonic(),
        )

        result_dict = await asyncio.to_thread(
            self._transcribe_sync, audio, model, language, initial_prompt
        )
        meta.wall_clock_end = time.monotonic()
        meta.first_final_recv = meta.wall_clock_end  # batch: result arrives once at end

        words = self._extract_words(result_dict)
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
                "model": model,
                "language": language,
                "initial_prompt": initial_prompt,
                "_mode": "batch_mlx",
            },
            words=words,
            transcript=(result_dict.get("text") or "").strip(),
            raw_dump=[
                {"_direction": "in", "message": "MlxWhisperResult", "text": result_dict.get("text", "")},
                *[{"_direction": "in", "message": "MlxWhisperSegment", **s}
                  for s in result_dict.get("segments", [])],
            ],
            metadata=meta,
            error=None,
        )

    def _transcribe_sync(
        self,
        audio: Path,
        model: str,
        language: str,
        initial_prompt: str | None,
    ) -> dict:
        import mlx_whisper
        return mlx_whisper.transcribe(
            str(audio),
            path_or_hf_repo=model,
            language=language,
            word_timestamps=True,
            initial_prompt=initial_prompt,
            verbose=False,
        )

    @staticmethod
    def _extract_words(result_dict: dict) -> list[Word]:
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
                    speaker=None,
                    confidence=w.get("probability"),
                    is_punctuation=False,
                ))
        return out
