"""
Parakeet TDT 0.6B v3 — NeMo (FP16 / CUDA) adapter.

This is the GPU-baseline path. Use it on the Alienware (or any NVIDIA host)
to establish the FP16 accuracy ceiling that the INT8 ONNX path is judged
against. Both adapters share the same model weights (`nvidia/parakeet-tdt-0.6b-v3`
HF Hub); the difference is the runtime + precision + EP.

For hardware-agnostic production deployment we prefer `parakeet_onnx` — it
runs on Mac, Windows, Linux, CPU, CUDA, CoreML, DirectML all from one binary.
This NeMo adapter is the calibration tool.

Install (NVIDIA + CUDA host only):
    pip install -r requirements/parakeet-nemo.txt

Model is auto-downloaded on first use to `~/.cache/huggingface/hub/`.

Two transcription paths:
  - **batch** (default): `model.transcribe([wav])` with `timestamps=True`.
    Whole-file inference. Best accuracy. Use for accuracy benchmarks.
  - **streaming**: chunked inference via NeMo's cache-aware streaming API.
    Use for latency benchmarks. Word timestamps require careful frame-rate
    bookkeeping (TDT outputs every 80 ms post-subsampling).
"""
from __future__ import annotations

import asyncio
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
    streaming=True,                 # NeMo cache-aware streaming API
    partials=False,                 # not exposed for the TDT decoder via the simple API
    diarization=False,              # pair with Sortformer or pyannote
    speaker_fingerprinting=False,
    custom_vocab=True,              # KenLM / additional vocab via decoder LM weights
    word_timing=True,
    confidence_scores=False,        # NeMo's word-level confidence is not directly exposed
    languages=[
        "en", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr",
        "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro",
        "ru", "sk", "sl", "sv", "uk",
    ],
    on_prem=True,
    notes="NVIDIA Parakeet TDT 0.6B v3 via NeMo. FP16 on CUDA, CPU fallback. Batch by default.",
)


_MODEL = None  # cached; ~600M params on GPU


def _get_model(model_name: str):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import nemo.collections.asr as nemo_asr  # lazy: heavy stack
    import torch
    _MODEL = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    if torch.cuda.is_available():
        _MODEL = _MODEL.cuda().half()
    else:
        _MODEL = _MODEL.cpu()
    _MODEL.eval()
    return _MODEL


class ParakeetNemo:
    name = "parakeet_nemo"
    capabilities = CAPABILITIES

    def __init__(self, model: str = "nvidia/parakeet-tdt-0.6b-v3"):
        self._model = model

    def warm(self) -> None:
        """Pay model load + first-call JIT before any user session arrives.

        Without warm, the first session's first transcribe call has to
        download (if not cached), load to GPU, and JIT-compile — easily 30+ s
        on cold cache. That blocks long enough to trip the WebSocket
        keepalive ping timeout and kill the session.
        """
        from io import BytesIO
        import wave
        model = _get_model(self._model)
        # Quick no-op transcribe on 0.5 s of silence.
        wav_bytes = BytesIO()
        with wave.open(wav_bytes, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 8000)
        tmp = Path("/tmp/parakeet_nemo_warm.wav")
        tmp.write_bytes(wav_bytes.getvalue())
        try:
            try:
                model.transcribe([str(tmp)], batch_size=1)
            except Exception:
                pass  # warmup ran even if silence yields nothing
        finally:
            try: tmp.unlink()
            except OSError: pass

    async def transcribe(self, audio: Path, config: dict[str, Any] | None = None) -> Result:
        """
        config schema (all optional):
            model: str         — HF repo override
            mode: str          — "batch" (default) or "streaming"
            chunk_secs: float  — streaming chunk size (default 2.0)
            left_context_secs: float   — default 10.0
            right_context_secs: float  — default 2.0 (the lookahead-cost knob)
        """
        config = config or {}
        model_name = config.get("model", self._model)
        # DEFAULT IS STREAMING. Courtroom transcription is live; batch is academic.
        # Batch path is kept as opt-in (mode="batch") for accuracy-only sanity checks.
        mode = config.get("mode", "streaming")

        duration_s = audio_duration_s(audio)

        meta = RunMetadata(
            audio_duration_s=duration_s,
            wall_clock_start=time.monotonic(),
            wall_clock_end=0.0,
            first_audio_send=time.monotonic(),
        )

        if mode == "streaming":
            words, transcript, raw_dump = await asyncio.to_thread(
                self._transcribe_streaming, audio, model_name, config
            )
        else:
            words, transcript, raw_dump = await asyncio.to_thread(
                self._transcribe_batch, audio, model_name
            )

        meta.wall_clock_end = time.monotonic()
        meta.first_final_recv = meta.wall_clock_end  # batch: result arrives at end
        if words:
            meta.frames.append(FrameTimestamp(
                audio_end_time=words[-1].end_time,
                wall_time=meta.wall_clock_end,
                kind="AddTranscript",
            ))

        return Result(
            engine=self.name,
            audio_path=audio,
            config={"model": model_name, "_mode": mode},
            words=words,
            transcript=transcript,
            raw_dump=raw_dump,
            metadata=meta,
            error=None,
        )

    def _transcribe_batch(
        self,
        audio: Path,
        model_name: str,
        config: dict | None = None,
    ) -> tuple[list[Word], str, list[dict]]:
        """
        Batch (offline) path with optional decoder tuning.

        config keys (all optional):
            beam_size       — int > 1 switches to alsd (Alignment Length
                              Synchronous Decoding) beam search. Default 1 = greedy.
            kenlm_path      — path to a KenLM .binary file. Switches to maes
                              (Modified Adaptive Expansion Search) with N-gram LM
                              fusion. Requires beam_size > 1.
            kenlm_alpha     — KenLM LM weight (default 0.3).
            kenlm_beta      — KenLM word-insertion bonus (default 1.0).

        We restore the default greedy decoding after each call so consecutive
        runs with different configs don't bleed into each other.
        """
        config = config or {}
        model = _get_model(model_name)

        beam_size = int(config.get("beam_size", 1))
        kenlm_path = config.get("kenlm_path")
        applied_strategy = "greedy_batch"

        if kenlm_path or beam_size > 1:
            from omegaconf import OmegaConf

            if kenlm_path:
                # KenLM fusion path — requires maes strategy
                applied_strategy = "maes"
                decoding_cfg = OmegaConf.create({
                    "strategy": "maes",
                    "beam": {
                        "beam_size": max(beam_size, 4),
                        "ngram_lm_model": str(kenlm_path),
                        "ngram_lm_alpha": float(config.get("kenlm_alpha", 0.3)),
                        "maes_num_steps": 2,
                        "maes_prefix_alpha": 1,
                        "maes_expansion_gamma": 2.3,
                    },
                })
            else:
                # Beam search without LM
                applied_strategy = "alsd"
                decoding_cfg = OmegaConf.create({
                    "strategy": "alsd",
                    "beam": {
                        "beam_size": beam_size,
                    },
                })

            try:
                model.change_decoding_strategy(decoding_cfg)
            except Exception as e:
                # Fall back to greedy if the strategy can't be applied
                # (e.g., maes incompatible with TDT in this NeMo version)
                applied_strategy = f"greedy_batch (fallback; tried {applied_strategy}: {e!r})"

        try:
            outputs = model.transcribe([str(audio)], timestamps=True)
        finally:
            # Restore default greedy_batch for next call (idempotency)
            try:
                from omegaconf import OmegaConf
                model.change_decoding_strategy(
                    OmegaConf.create({"strategy": "greedy_batch"})
                )
            except Exception:
                pass

        # NeMo's transcribe() returns a list (one element per input wav)
        hyp = outputs[0]
        text = (getattr(hyp, "text", None) or "").strip()
        word_stamps = (getattr(hyp, "timestamp", {}) or {}).get("word", [])

        words: list[Word] = []
        for ws in word_stamps:
            content = (ws.get("word") or "").strip()
            if not content:
                continue
            words.append(Word(
                content=content,
                start_time=float(ws.get("start", 0.0)),
                end_time=float(ws.get("end", 0.0)),
                speaker=None,
                confidence=None,
                is_punctuation=False,
            ))

        raw_dump = [
            {"_direction": "in", "message": "ParakeetHypothesis",
             "text": text, "decoding_strategy": applied_strategy,
             "beam_size": beam_size, "kenlm_path": str(kenlm_path) if kenlm_path else None},
            {"_direction": "in", "message": "ParakeetTimestamps",
             "word": word_stamps,
             "segment": (getattr(hyp, "timestamp", {}) or {}).get("segment", []),
             "char": (getattr(hyp, "timestamp", {}) or {}).get("char", [])},
        ]
        return words, text, raw_dump

    def _transcribe_streaming(self, audio: Path, model_name: str, config: dict) -> tuple[list[Word], str, list[dict]]:
        """
        Cache-aware streaming path. Used for latency measurement comparisons.

        Calls into NeMo's `transcribe_simulate_chunked_inference` or the script
        at `examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py`
        depending on what's exposed by the installed NeMo version.

        NOTE: This path is unimplemented in the first cut — the batch path covers
        the accuracy comparison we need against `parakeet_onnx`. Streaming numbers
        are a Phase 4 deliverable (tuning chunk_secs, left/right context).
        """
        raise NotImplementedError(
            "parakeet_nemo streaming mode is not yet implemented. "
            "Use mode='batch' (default) for accuracy benchmarks; streaming will be "
            "added in Phase 4 (parameter sweep harness)."
        )

    # ------------------------------------------------------------------
    # Streaming path for the M2 WebSocket server.
    #
    # LocalAgreement-2 (whisper-streaming pattern): every commit_delay
    # seconds transcribe a sliding window of audio; a word past the last
    # commit boundary is committed only when the new chunk's hypothesis
    # AGREES with the previous chunk's hypothesis on it (content match,
    # case-insensitive). Eliminates the chunk-boundary duplicate ("mutton.
    # mutton") and drop failures of naive sliding. Per-final lag tracks
    # `max_delay` instead of audio duration.
    #
    # Set `stream_mode: single-final` in transcription_config to fall back
    # to the legacy "one final at EOS" path (highest accuracy, worst lag).

    async def stream(self, pcm_frames, transcription_config):
        from io import BytesIO
        import wave

        from server.engines._base import StreamFinal, StreamWord

        SAMPLE_RATE = 16000
        BYTES_PER_SEC = SAMPLE_RATE * 2

        config = transcription_config or {}
        model_name = config.get("model", self._model)
        diar_mode = config.get("diarization", "none")
        default_commit = 5.0 if diar_mode != "none" else 3.0
        commit_delay = float(config.get("max_delay", default_commit))
        chunk_window_s = float(config.get("chunk_window_s", 30.0))
        single_final = config.get("stream_mode") == "single-final"

        buffer = bytearray()

        def _transcribe_chunk(pcm: bytes, offset_s: float):
            if not pcm or len(pcm) < BYTES_PER_SEC // 2:
                return []
            wav_bytes = BytesIO()
            with wave.open(wav_bytes, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm)
            tmp = Path(f"/tmp/parakeet_nemo_la_{id(pcm)}.wav")
            tmp.write_bytes(wav_bytes.getvalue())
            try:
                try:
                    words, _, _ = self._transcribe_batch(tmp, model_name, config={})
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

        # --- Legacy single-final-at-EOS path -----------------------------
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

        # --- LocalAgreement-2 path ---------------------------------------
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
            # Always include some left context past committed boundary
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

        # Drain at EOS — one more transcription pass, then emit ALL remaining uncommitted
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
