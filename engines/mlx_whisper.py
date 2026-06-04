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

    # ------------------------------------------------------------------
    # warm() — pay model-load + first-call JIT before any session arrives.

    def warm(self) -> None:
        import tempfile, wave as _wave
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = Path(f.name)
        try:
            with _wave.open(str(tmp), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                w.writeframes(b"\x00\x00" * 8000)  # 0.5 s silence
            try:
                self._transcribe_sync(tmp, self._model, "en", None)
            except Exception:
                pass  # warmup ran even if silent clip yields nothing
        finally:
            try: tmp.unlink()
            except OSError: pass

    # ------------------------------------------------------------------
    # Streaming path — LocalAgreement-2 rolling finals.
    # Sliding window (default 8 s) transcribed every commit_delay; words past
    # the commit watermark are committed when the new chunk agrees with the
    # previous chunk's hypothesis on them. Uncommitted tail emits as a
    # StreamPartial so the live pane fills word-by-word between commits.

    async def stream(self, pcm_frames, transcription_config):
        from io import BytesIO
        import wave as _wave

        from server.engines._base import StreamFinal, StreamPartial, StreamWord

        SAMPLE_RATE = 16000
        BYTES_PER_SEC = SAMPLE_RATE * 2

        config = transcription_config or {}
        model_name = config.get("model", self._model)
        language = config.get("language", "en")
        diar_mode = config.get("diarization", "none")
        default_commit = 2.0 if diar_mode != "none" else 1.0
        commit_delay = float(config.get("max_delay", default_commit))
        chunk_window_s = float(config.get("chunk_window_s", 8.0))
        single_final = config.get("stream_mode") == "single-final"

        buffer = bytearray()

        def _transcribe_chunk(pcm: bytes, offset_s: float):
            if not pcm or len(pcm) < BYTES_PER_SEC // 2:
                return []
            wav_bytes = BytesIO()
            with _wave.open(wav_bytes, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
                w.writeframes(pcm)
            tmp = Path(f"/tmp/mlx_whisper_la_{id(pcm)}.wav")
            tmp.write_bytes(wav_bytes.getvalue())
            try:
                try:
                    result_dict = self._transcribe_sync(tmp, model_name, language, None)
                except Exception:
                    return []
                words = self._extract_words(result_dict)
                for w in words:
                    w.start_time += offset_s
                    w.end_time += offset_s
                return words
            finally:
                try: tmp.unlink()
                except OSError: pass

        def _to_stream_words(ws):
            return [
                StreamWord(
                    content=w.content,
                    start_time=w.start_time,
                    end_time=w.end_time,
                    confidence=w.confidence,
                    speaker=None,
                    is_punctuation=False,
                )
                for w in ws
            ]

        if single_final:
            async for frame in pcm_frames:
                buffer.extend(frame)
            words = await asyncio.to_thread(_transcribe_chunk, bytes(buffer), 0.0)
            if words:
                sw = _to_stream_words(words)
                yield StreamFinal(
                    transcript=" ".join(w.content for w in sw),
                    words=sw,
                    start_time=sw[0].start_time,
                    end_time=sw[-1].end_time,
                )
            return

        last_transcribe_at_s = 0.0
        committed_end_s = 0.0
        prev_uncommitted: list = []

        def _norm(s: str) -> str:
            return "".join(c.lower() for c in s if c.isalnum())

        def _common_prefix(a, b):
            agreed = []
            for x, y in zip(a, b):
                if _norm(x.content) and _norm(x.content) == _norm(y.content):
                    agreed.append(y)
                else:
                    break
            return agreed

        async for frame in pcm_frames:
            buffer.extend(frame)
            audio_so_far_s = len(buffer) / BYTES_PER_SEC
            if audio_so_far_s - last_transcribe_at_s < commit_delay:
                continue
            last_transcribe_at_s = audio_so_far_s
            window_start_s = max(0.0, audio_so_far_s - chunk_window_s)
            window_start_s = min(window_start_s, max(0.0, committed_end_s - 1.0))
            window_pcm = bytes(buffer[int(window_start_s * BYTES_PER_SEC):])
            chunk_words = await asyncio.to_thread(_transcribe_chunk, window_pcm, window_start_s)
            new_uncommitted = [w for w in chunk_words if w.start_time > committed_end_s]
            agreed = _common_prefix(prev_uncommitted, new_uncommitted)
            if agreed:
                sw = _to_stream_words(agreed)
                yield StreamFinal(
                    transcript=" ".join(w.content for w in sw),
                    words=sw,
                    start_time=sw[0].start_time,
                    end_time=sw[-1].end_time,
                )
                committed_end_s = agreed[-1].end_time
                prev_uncommitted = [w for w in new_uncommitted if w.start_time > committed_end_s]
            else:
                prev_uncommitted = new_uncommitted

            if prev_uncommitted:
                partial_text = " ".join(w.content for w in prev_uncommitted)
                if partial_text.strip():
                    yield StreamPartial(
                        transcript=partial_text,
                        start_time=prev_uncommitted[0].start_time,
                        end_time=prev_uncommitted[-1].end_time,
                    )

        # Drain at EOS
        audio_total_s = len(buffer) / BYTES_PER_SEC
        window_start_s = max(0.0, audio_total_s - chunk_window_s)
        window_start_s = min(window_start_s, max(0.0, committed_end_s - 1.0))
        window_pcm = bytes(buffer[int(window_start_s * BYTES_PER_SEC):])
        final_chunk = await asyncio.to_thread(_transcribe_chunk, window_pcm, window_start_s)
        final_new = [w for w in final_chunk if w.start_time > committed_end_s]
        if final_new:
            sw = _to_stream_words(final_new)
            yield StreamFinal(
                transcript=" ".join(w.content for w in sw),
                words=sw,
                start_time=sw[0].start_time,
                end_time=sw[-1].end_time,
            )
