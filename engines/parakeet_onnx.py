"""
Parakeet TDT 0.6B v3 — INT8 ONNX adapter.

Backed by istupakov's `onnx-asr` package which wraps the ONNX models with
a TDT inference loop, VAD-based long-audio segmentation, and word-level
timestamp extraction. Pure Python, no PyTorch / CUDA stack required.

**Hardware-agnostic by design.** The same model file dispatches to whatever
ONNX Runtime Execution Provider is available:
  - CoreMLExecutionProvider on macOS arm64 (Apple Neural Engine + GPU)
  - CUDAExecutionProvider on Linux/Windows with NVIDIA + CUDA
  - DmlExecutionProvider on Windows with DirectX-compatible GPU
  - CPUExecutionProvider as the always-works fallback

~670 MB on disk (encoder INT8) + 18 MB (decoder INT8), ~2 GB RAM per worker.

Install (per-engine venv recommended):
    pip install "onnx-asr[cpu,hub]"             # base + CPU EP + HF Hub downloader
    pip install "onnx-asr[cpu,hub,coreml]"      # adds CoreML EP on macOS arm64
    pip install "onnx-asr[gpu,hub]"             # adds CUDA EP on NVIDIA systems

Model is auto-downloaded on first use to `~/.cache/huggingface/hub/`.
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
    streaming=False,                # onnx-asr exposes batch; TDT can stream, but we use the VAD-segmented offline path
    partials=False,
    diarization=False,              # not built-in; pair with a diarizer
    speaker_fingerprinting=False,
    custom_vocab=False,             # add_decoder_lm_weights / KenLM not exposed by onnx-asr today
    word_timing=True,
    confidence_scores=True,         # via with_timestamps() log_probs
    languages=[
        "en", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr",
        "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro",
        "ru", "sk", "sl", "sv", "uk",
    ],
    on_prem=True,
    notes="Parakeet TDT 0.6B v3 INT8 ONNX (istupakov). CPU/CoreML/CUDA/DirectML via ONNX Runtime.",
)


_MODEL = None  # cached; loading is expensive
_VAD = None    # cached Silero VAD for long-audio chunking

# CoreML EP fails on long encoder sequences (`error: -7` from dynamic resize).
# For any audio over this threshold, route through VAD chunking which keeps
# each individual ASR pass to a manageable length.
LONG_AUDIO_THRESHOLD_S = 25.0


def _detect_providers() -> list:
    """Pick ONNX Runtime Execution Providers in priority order for this host."""
    sys = platform.system()
    machine = platform.machine().lower()
    providers: list = []

    if sys == "Darwin" and machine in ("arm64", "aarch64"):
        providers.append("CoreMLExecutionProvider")
    if sys == "Linux" or sys == "Windows":
        # CUDA tried first if available; falls through to CPU silently if not
        providers.append("CUDAExecutionProvider")
        if sys == "Windows":
            providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _get_model(model_name: str, quantization: str | None, providers: list | None):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import onnx_asr
    kwargs: dict[str, Any] = {}
    if quantization:
        kwargs["quantization"] = quantization
    if providers:
        kwargs["providers"] = providers
    _MODEL = onnx_asr.load_model(model_name, **kwargs)
    return _MODEL


def _get_vad(providers: list | None):
    """Silero VAD for long-audio chunking. CPU EP is fine — VAD is tiny."""
    global _VAD
    if _VAD is not None:
        return _VAD
    import onnx_asr
    # Force CPU EP for VAD to avoid spawning extra CoreML sessions
    _VAD = onnx_asr.load_vad("silero", providers=["CPUExecutionProvider"])
    return _VAD


class ParakeetOnnx:
    name = "parakeet_onnx"
    capabilities = CAPABILITIES

    def __init__(
        self,
        model_name: str = "nemo-parakeet-tdt-0.6b-v3",
        quantization: str | None = "int8",
    ):
        self._model_name = model_name
        self._quantization = quantization
        self._providers = _detect_providers()

    def warm(self) -> None:
        """Pay the first-call JIT/compile cost before any session arrives.
        Loads the model and runs a tiny silent transcribe so the next real
        call (from a WebSocket session) hits the warm path."""
        import tempfile
        import wave
        # Force the singleton model load.
        _get_model(self._model_name, self._quantization, self._providers)
        # Run one silent inference so CoreML compiles its graph upfront.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = Path(f.name)
        try:
            with wave.open(str(tmp), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(b"\x00\x00" * 8000)  # 0.5 s of silence
            try:
                self._transcribe_sync(tmp, self._model_name, self._quantization, self._providers, vad_kwargs={})
            except (IndexError, RuntimeError):
                pass  # silent clip may produce no tokens — fine, warmup still happened
        finally:
            try: tmp.unlink()
            except OSError: pass

    async def transcribe(self, audio: Path, config: dict[str, Any] | None = None) -> Result:
        """
        config schema (all optional):
            model: str         — override default "nemo-parakeet-tdt-0.6b-v3"
            quantization: str  — "int8" (default) or None for FP32
            providers: list    — override auto-detected ONNX Runtime EPs
        """
        config = config or {}
        model_name = config.get("model", self._model_name)
        quantization = config.get("quantization", self._quantization)
        providers = config.get("providers", self._providers)

        duration_s = audio_duration_s(audio)

        meta = RunMetadata(
            audio_duration_s=duration_s,
            wall_clock_start=time.monotonic(),
            wall_clock_end=0.0,
            first_audio_send=time.monotonic(),
        )

        # Pull VAD knobs out of the config; remaining VAD-tunables propagate via vad_kwargs.
        vad_kwargs: dict[str, Any] = {}
        for k in ("threshold", "neg_threshold", "speech_pad_ms",
                  "min_silence_duration_ms", "min_speech_duration_ms",
                  "max_speech_duration_s", "batch_size"):
            if k in config:
                vad_kwargs[k] = config[k]

        words, transcript, raw_dump = await asyncio.to_thread(
            self._transcribe_sync, audio, model_name, quantization, providers, vad_kwargs
        )

        meta.wall_clock_end = time.monotonic()
        meta.first_final_recv = meta.wall_clock_end  # offline: all results arrive at end
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
                "quantization": quantization,
                "providers_requested": providers,
                "_mode": "batch_vad_segmented",
            },
            words=words,
            transcript=transcript,
            raw_dump=raw_dump,
            metadata=meta,
            error=None,
        )

    def _transcribe_sync(
        self,
        audio: Path,
        model_name: str,
        quantization: str | None,
        providers: list | None,
        vad_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list[Word], str, list[dict]]:
        """
        Synchronous core: load model, run with timestamps, normalize output.

        onnx-asr TimestampedResult shape (verified empirically against
        onnx-asr 0.11.0, nemo-parakeet-tdt-0.6b-v3):
          - text:       str (fully detokenized, ready-to-use)
          - tokens:     list[str] of subword pieces; leading space marks
                        word boundary (NOT the SentencePiece "▁" U+2581)
          - timestamps: list[float], one per token, START time in seconds
                        (end of token i is implicitly start of token i+1)
          - logprobs:   list[float], one per token

        Long audio path: iterating .recognize() yields SegmentResult; only
        triggered when audio > ~30s. We treat both shapes uniformly.
        """
        from engines._audio import audio_duration_s
        model = _get_model(model_name, quantization, providers)
        duration_s = audio_duration_s(audio)

        vad_kwargs = vad_kwargs or {}

        # Short audio (< LONG_AUDIO_THRESHOLD_S): single encoder pass with timestamps.
        # Long audio: VAD-chunked iteration so each encoder pass fits CoreML's
        # dynamic-resize budget (and is faster anyway — VAD trims silence).
        if duration_s <= LONG_AUDIO_THRESHOLD_S and not vad_kwargs:
            ts_model = model.with_timestamps()
            result = ts_model.recognize(str(audio))
            segments = [result]
        else:
            vad = _get_vad(providers)
            ts_vad_model = model.with_timestamps().with_vad(vad, **vad_kwargs)
            # SegmentResultsAsrAdapter.recognize() returns an iterable of SegmentResult.
            segments = list(ts_vad_model.recognize(str(audio)))

        all_words: list[Word] = []
        transcript_parts: list[str] = []
        raw_dump: list[dict] = []

        import math

        for seg in segments:
            seg_offset = float(getattr(seg, "start", 0.0) or 0.0)
            text = getattr(seg, "text", None) or ""
            tokens = getattr(seg, "tokens", None) or []
            timestamps = getattr(seg, "timestamps", None) or []
            logprobs = getattr(seg, "logprobs", None) or []
            seg_end = float(getattr(seg, "end", 0.0)) or (seg_offset + duration_s)

            if tokens and timestamps:
                # Path 1: TimestampedResult — full token-level info available.
                # Pre-compute per-token (start, end): each token's start is timestamps[i];
                # end is timestamps[i+1] or seg_end for the last token.
                n = len(tokens)
                token_starts = [seg_offset + float(timestamps[i]) for i in range(n)]
                token_ends = [token_starts[i + 1] if i + 1 < n else seg_end for i in range(n)]

                # Group tokens into words by leading-space (or "▁") boundary.
                groups: list[list[int]] = []
                for i, tok in enumerate(tokens):
                    is_word_start = (i == 0) or tok.startswith(" ") or tok.startswith("▁")
                    if is_word_start or not groups:
                        groups.append([i])
                    else:
                        groups[-1].append(i)

                for grp in groups:
                    content = "".join(tokens[i] for i in grp).lstrip("▁").lstrip(" ")
                    if not content:
                        continue
                    grp_logprobs = [float(logprobs[i]) for i in grp if i < len(logprobs)]
                    conf = math.exp(min(sum(grp_logprobs) / len(grp_logprobs), 0.0)) if grp_logprobs else None
                    all_words.append(Word(
                        content=content,
                        start_time=token_starts[grp[0]],
                        end_time=token_ends[grp[-1]],
                        speaker=None,
                        confidence=conf,
                        is_punctuation=False,
                    ))
            elif text:
                # Path 2: SegmentResult (VAD-chunked path) — only text + segment-level
                # start/end available. Distribute words evenly across the segment.
                # Good enough for WER scoring; diarization overlay loses some precision
                # at speaker-boundary words but still works for segment-based diarizers.
                seg_words = [w for w in text.split() if w]
                if seg_words:
                    span = max(seg_end - seg_offset, 0.001)
                    per_word = span / len(seg_words)
                    for i, wtxt in enumerate(seg_words):
                        all_words.append(Word(
                            content=wtxt,
                            start_time=seg_offset + i * per_word,
                            end_time=seg_offset + (i + 1) * per_word,
                            speaker=None,
                            confidence=None,
                            is_punctuation=False,
                        ))

            if text:
                transcript_parts.append(text)
            raw_dump.append({
                "_direction": "in",
                "message": "ParakeetSegment",
                "text": text,
                "start": seg_offset,
                "end": seg_end,
                "tokens": tokens,
                "timestamps": timestamps,
            })

        transcript = " ".join(transcript_parts).strip() or " ".join(w.content for w in all_words)
        return all_words, transcript, raw_dump

    # ------------------------------------------------------------------
    # Streaming path — LocalAgreement-2 rolling finals (same scheme as
    # parakeet_nemo). Every commit_delay seconds we transcribe a bounded
    # sliding window (commit_delay + left_context, NOT the whole growing
    # buffer) and commit any word the new chunk agrees on with the previous
    # chunk. Bounded compute per step + finals stream as you speak, so the
    # transcript fills in live instead of all-at-once at EOS.
    #
    # `stream_mode: single-final` falls back to one transcribe at EOS.

    async def stream(self, pcm_frames, transcription_config):
        from io import BytesIO
        import wave

        from server.engines._base import StreamFinal, StreamWord

        SAMPLE_RATE = 16000
        BYTES_PER_SEC = SAMPLE_RATE * 2
        # Hard cap on the re-transcribe window. Parakeet-TDT was trained on
        # ~30s segments; beyond that the encoder's word-level output destabilizes
        # and the post-committed region tends to come back empty. That empty
        # return is what kills the streaming loop past ~138s on SCOTUS-class
        # audio — once `new_uncommitted` is empty, LA2 has nothing to agree on
        # and committed_end never moves.
        MAX_WINDOW_S = 30.0
        # If committed_end falls more than MAX_LAG_S behind real-time audio
        # despite the window cap, LA2 is stalling on word-boundary jitter.
        # Flush `prev_uncommitted` as a single final to break the stall and
        # let the next iteration anchor close to real-time.
        MAX_LAG_S = 20.0

        config = transcription_config or {}
        model_name = config.get("model", self._model_name)
        quantization = config.get("quantization", self._quantization)
        providers = config.get("providers", self._providers)
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
            with wave.open(wav_bytes, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(pcm)
            tmp = Path(f"/tmp/parakeet_onnx_la_{id(pcm)}.wav")
            tmp.write_bytes(wav_bytes.getvalue())
            try:
                try:
                    words, _, _ = self._transcribe_sync(
                        tmp, model_name, quantization, providers, vad_kwargs={}
                    )
                except (IndexError, RuntimeError):
                    return []
                for w in words:
                    w.start_time += offset_s
                    w.end_time += offset_s
                return words
            finally:
                try:
                    tmp.unlink()
                except OSError:
                    pass

        def _to_stream_words(ws):
            return [
                StreamWord(
                    content=w.content,
                    start_time=w.start_time,
                    end_time=w.end_time,
                    confidence=w.confidence,
                    speaker=None,
                    is_punctuation=w.is_punctuation,
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

            # Safety valve: if LA2 has stalled (committed_end hasn't advanced
            # in a while), flush prev_uncommitted as a single final. Without
            # this, prev_uncommitted keeps refreshing each iteration but never
            # commits, and we lose live transcript output.
            if audio_so_far_s - committed_end_s > MAX_LAG_S and prev_uncommitted:
                sw = _to_stream_words(prev_uncommitted)
                yield StreamFinal(
                    transcript=" ".join(w.content for w in sw),
                    words=sw,
                    start_time=sw[0].start_time,
                    end_time=sw[-1].end_time,
                )
                committed_end_s = prev_uncommitted[-1].end_time
                prev_uncommitted = []

            window_start_s = max(0.0, audio_so_far_s - chunk_window_s)
            window_start_s = min(window_start_s, max(0.0, committed_end_s - 1.0))
            # Hard cap on window size to keep parakeet inside its training
            # context. When this clips past committed_end the audio in that
            # gap is dropped — but only happens when the safety valve had
            # nothing to flush (model produced no post-committed words),
            # which means the audio in the gap was already unrecoverable.
            window_start_s = max(window_start_s, audio_so_far_s - MAX_WINDOW_S)
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

            # Emit a partial of the still-uncommitted tail so the live pane
            # fills word-by-word between commits.
            if prev_uncommitted:
                partial_text = " ".join(w.content for w in prev_uncommitted)
                if partial_text.strip():
                    from server.engines._base import StreamPartial
                    yield StreamPartial(
                        transcript=partial_text,
                        start_time=prev_uncommitted[0].start_time,
                        end_time=prev_uncommitted[-1].end_time,
                    )

        # Drain at EOS — emit everything still uncommitted from a final pass
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
