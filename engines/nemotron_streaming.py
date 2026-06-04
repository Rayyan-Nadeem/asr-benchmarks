"""Nemotron Speech Streaming 0.6B — true cache-aware streaming ASR.

Cache-aware FastConformer-RNNT. The encoder maintains per-layer KV/conv
cache state across audio chunks, so per-chunk inference is O(chunk),
not O(buffer-so-far) — the missing piece for real-time on Mac CPU.

Backed by sherpa-onnx's OnlineRecognizer, which threads the cache state
internally between accept_waveform calls. Pre-exported INT8 ONNX weights
from huggingface.co/csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25.

160 ms chunk size — the latency/accuracy sweet spot for this model.
Other variants exist (80 / 560 / 1120 ms) under the same author.

Install:
    pip install sherpa-onnx
Model:
    huggingface_hub.snapshot_download(repo_id=
        "csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25",
        local_dir="models/nemotron-160ms")
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np


_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "nemotron-160ms"

_RECOGNIZER = None


def _get_recognizer():
    global _RECOGNIZER
    if _RECOGNIZER is not None:
        return _RECOGNIZER
    import sherpa_onnx
    if not _MODEL_DIR.exists():
        raise FileNotFoundError(
            f"nemotron sherpa-onnx model not found at {_MODEL_DIR}. "
            f"Download via huggingface_hub.snapshot_download — see module docstring."
        )
    # Pick CUDA if a GPU is around; otherwise CPU. The 8 GB cap is enforced
    # uniformly across engines via server._gpu_cap.
    provider = os.environ.get("SHERPA_PROVIDER", "cpu")
    provider_kwargs: dict = {"num_threads": 4}
    if provider == "cuda":
        from server._gpu_cap import sherpa_cuda_provider_options
        provider_kwargs["provider_config"] = sherpa_onnx.ProviderConfig(
            provider="cuda",
            cuda_config=sherpa_onnx.CudaConfig(cudnn_conv_algo_search=1),
        ) if hasattr(sherpa_onnx, "ProviderConfig") else None
    _RECOGNIZER = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(_MODEL_DIR / "tokens.txt"),
        encoder=str(_MODEL_DIR / "encoder.int8.onnx"),
        decoder=str(_MODEL_DIR / "decoder.int8.onnx"),
        joiner=str(_MODEL_DIR / "joiner.int8.onnx"),
        num_threads=4,
        provider=provider,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20,
    )
    return _RECOGNIZER


class NemotronStreaming:
    name = "nemotron-streaming"

    def warm(self) -> None:
        """Pay model load + ONNX session init before any user audio arrives."""
        r = _get_recognizer()
        s = r.create_stream()
        # Push 0.5 s of silence to force the encoder ONNX session to initialize
        # all its cache tensors.
        s.accept_waveform(16000, np.zeros(8000, dtype=np.float32))
        while r.is_ready(s):
            r.decode_streams([s])

    async def stream(self, pcm_frames, transcription_config):
        from server.engines._base import StreamFinal, StreamPartial, StreamWord

        SAMPLE_RATE = 16000
        BYTES_PER_SEC = SAMPLE_RATE * 2

        recognizer = _get_recognizer()
        s = recognizer.create_stream()

        last_partial = ""
        bytes_seen = 0
        # When sherpa fires an endpoint we commit what's there as a final.
        # Track the audio-time offset of each commit so words on subsequent
        # turns have monotonic timestamps.
        segment_start_s = 0.0

        # decode_streams is ~10ms per 160ms chunk on Mac CPU — fast enough to
        # run on the event loop thread directly. The threadpool roundtrip cost
        # more than the call itself.
        def _drain_sync():
            while recognizer.is_ready(s):
                recognizer.decode_streams([s])

        async for frame in pcm_frames:
            bytes_seen += len(frame)
            audio_so_far_s = bytes_seen / BYTES_PER_SEC
            samples = (
                np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            )
            s.accept_waveform(SAMPLE_RATE, samples)
            _drain_sync()

            text = recognizer.get_result(s).strip()

            if recognizer.is_endpoint(s):
                if text:
                    words = text.split()
                    span = max(audio_so_far_s - segment_start_s, 0.001)
                    per_word = span / max(1, len(words))
                    sw = [
                        StreamWord(
                            content=w,
                            start_time=segment_start_s + i * per_word,
                            end_time=segment_start_s + (i + 1) * per_word,
                            confidence=None,
                            speaker=None,
                            is_punctuation=False,
                        )
                        for i, w in enumerate(words)
                    ]
                    yield StreamFinal(
                        transcript=text,
                        words=sw,
                        start_time=sw[0].start_time if sw else segment_start_s,
                        end_time=sw[-1].end_time if sw else audio_so_far_s,
                    )
                recognizer.reset(s)
                segment_start_s = audio_so_far_s
                last_partial = ""
            elif text and text != last_partial:
                last_partial = text
                yield StreamPartial(
                    transcript=text,
                    start_time=segment_start_s,
                    end_time=audio_so_far_s,
                )

        # EOS: flush whatever's still buffered as a final.
        s.input_finished()
        _drain_sync()
        text = recognizer.get_result(s).strip()
        if text:
            words = text.split()
            audio_total_s = bytes_seen / BYTES_PER_SEC
            span = max(audio_total_s - segment_start_s, 0.001)
            per_word = span / max(1, len(words))
            sw = [
                StreamWord(
                    content=w,
                    start_time=segment_start_s + i * per_word,
                    end_time=segment_start_s + (i + 1) * per_word,
                    confidence=None,
                    speaker=None,
                    is_punctuation=False,
                )
                for i, w in enumerate(words)
            ]
            yield StreamFinal(
                transcript=text,
                words=sw,
                start_time=sw[0].start_time if sw else segment_start_s,
                end_time=sw[-1].end_time if sw else audio_total_s,
            )
