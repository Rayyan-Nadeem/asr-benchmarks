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
