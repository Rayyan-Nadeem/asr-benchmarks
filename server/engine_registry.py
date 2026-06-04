"""Engine registry — picks an engine by name (env var ENGINE)."""
from __future__ import annotations

import os

from server.engines._base import StreamingEngine
from server.engines.noop import NoopEngine
from server.engines.speechmatics import SpeechmaticsEngine


def _load_parakeet_onnx():
    from engines.parakeet_onnx import ParakeetOnnx
    return ParakeetOnnx()


def _load_parakeet_nemo():
    from engines.parakeet_nemo import ParakeetNemo
    return ParakeetNemo()


def _load_nemotron_nemo():
    # Native NeMo FP16 cache-aware streaming. Reuses the ParakeetNemo
    # adapter (same RNN-T inference path); only the loaded model changes.
    # LibriSpeech offline WER measured 0.00% on the 70 s test clip
    # (vs sherpa-onnx INT8 export's 18.93% — that tax was real).
    from engines.parakeet_nemo import ParakeetNemo
    return ParakeetNemo(model="nvidia/nemotron-speech-streaming-en-0.6b")


def _load_nemotron_native():
    # PROPER cache-aware streaming via conformer_stream_step. Per-chunk
    # processing, constant memory, sub-second TTFT. Use this for the
    # live-mic demo — the sliding-window LA2 adapter (used by parakeet-nemo
    # and nemotron-nemo) eventually OOMs on long sessions because the
    # diarizer overlay re-runs on the full growing buffer per emitted final.
    from engines.nemotron_native_streaming import NemotronNativeStreaming
    return NemotronNativeStreaming()


def _load_fastconformer_hybrid():
    # 114M-param multi-latency cache-aware streaming. CC-BY-4.0.
    # Card claims 5.4% LS-test-other at 1040ms, 7.0% at 0ms. Smaller +
    # cheaper than Nemotron 0.6B; the right pick for 8 GB GPU + CPU tiers.
    from engines.parakeet_nemo import ParakeetNemo
    return ParakeetNemo(model="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi")


def _load_multitalker_parakeet():
    # Joint ASR + speaker-kernel diarization. Per brief §7 escalation for
    # overlap-heavy DER: speaker info is injected into the encoder layers,
    # producing per-speaker parallel transcripts instead of one mixed
    # stream. Eliminates the per-word speaker-attribution ambiguity that
    # sortformer-overlay can't solve when two voices speak simultaneously.
    # Bundles its own diarization — set diarizer=passthrough.
    from engines.multitalker_parakeet import MultitalkerParakeet
    return MultitalkerParakeet()


def _load_auto_multispeaker():
    # Production architecture (docs/ARCHITECTURE.md): routes per session
    # to multitalker-parakeet for ≤4 speakers (the modal deposition shape,
    # where multitalker's speaker-kernel architecture wins on confusion%)
    # else nemotron-native + sortformer-streaming v2.1 for >4-spk sessions
    # (multi-party depos, mediations, oral arguments) that exceed the
    # multitalker 4-channel hardware ceiling. The routing reads
    # transcription_config.max_speakers from each StartRecognition frame.
    from engines.auto_multispeaker import AutoMultispeakerEngine
    return AutoMultispeakerEngine()


def _load_whisper():
    from engines.whisper import WhisperOpenai
    return WhisperOpenai()


def _load_nemotron_streaming():
    from engines.nemotron_streaming import NemotronStreaming
    return NemotronStreaming()


# `mlx-whisper` removed from the registry: Apple-Silicon-only via the MLX
# framework. The M2 container deploys to customer Windows/Linux hosts, so
# Mac-only engines aren't shippable. Keep the file in `engines/mlx_whisper.py`
# for local Mac dev exploration but don't expose it as a selectable engine.

_LAZY: dict[str, callable] = {
    "auto-multispeaker": _load_auto_multispeaker,
    "parakeet-onnx": _load_parakeet_onnx,
    "parakeet-nemo": _load_parakeet_nemo,
    "nemotron-nemo": _load_nemotron_nemo,
    "nemotron-native": _load_nemotron_native,
    "fastconformer-hybrid": _load_fastconformer_hybrid,
    "multitalker-parakeet": _load_multitalker_parakeet,
    "whisper": _load_whisper,
    "nemotron-streaming": _load_nemotron_streaming,
}


_REGISTRY: dict[str, type[StreamingEngine]] = {
    "noop": NoopEngine,
    "speechmatics": SpeechmaticsEngine,
}


def load_engine() -> StreamingEngine:
    name = os.environ.get("ENGINE", "noop")
    if name in _LAZY:
        return _LAZY[name]()
    cls = _REGISTRY.get(name)
    if cls is None:
        available = sorted(set(_REGISTRY) | set(_LAZY))
        raise ValueError(f"unknown ENGINE={name!r}; available: {available}")
    return cls()
