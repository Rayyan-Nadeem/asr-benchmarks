"""Native cache-aware streaming via NeMo's `conformer_stream_step`.

This is the ARCHITECTURAL FIX for the live-mic issues the sliding-window LA2
adapter in `engines/parakeet_nemo.py` was causing — disconnects past ~5 min
of audio (CUDA OOM on growing buffer), word-skipping, latency drift.

How it actually works:
  - Maintain ONE encoder-cache tuple + ONE RNN-T decoder hypothesis across the
    whole session.
  - For each new audio frame: append to a NeMo `CacheAwareStreamingAudioBuffer`,
    pull all ready chunks, call `model.conformer_stream_step(...)` per chunk.
  - Each chunk is processed exactly once. Memory is constant in session length.
  - Words emit as the decoder hypothesis grows.

Latency knob: `att_context_size = [past, lookahead]` at 80 ms/frame.
  - Default for nemotron-streaming: `[70, 13]` → 1.12 s lookahead.
  - We switch to `[70, 6]` → 560 ms lookahead at runtime: the live-mic sweet
    spot per the model card (7.07 % WER vs 6.93 % at 1.12 s — ~0.1 pp tax for
    half the latency).

Use this engine for the live demo. Use `parakeet_nemo` for highest accuracy on
recorded clips; use `parakeet_onnx` for CPU-laptop deployment.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

import numpy as np


_MODEL = None
_MODEL_NAME: str | None = None


def _get_model(model_name: str):
    """Load the model once per process, switch to live-mic latency settings."""
    global _MODEL, _MODEL_NAME
    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL
    import nemo.collections.asr as nemo_asr
    import torch
    m = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    # Keep in FP32 — the buffer's preprocessor returns float32 mel features,
    # so the encoder needs to accept float32. Casting model to half() breaks
    # this with "Input type (float) and bias type (c10::Half)" errors.
    # FP32 nemotron 0.6B ≈ 2.4 GB VRAM, well within the 8 GB cap.
    if torch.cuda.is_available():
        m = m.cuda()
    else:
        m = m.cpu()
    m.eval()
    # Att context — [left, right] in subsampled frames. Defaults to
    # [70, 13] = 1.12 s lookahead (accuracy-optimal per model card,
    # 6.93 % LS-clean). Override via env for sweeps:
    #   NEMOTRON_LEFT_CTX, NEMOTRON_RIGHT_CTX
    # Smaller right ctx → lower latency, worse partial accuracy.
    # Bigger left ctx → better partial accuracy, more VRAM.
    left = int(os.environ.get("NEMOTRON_LEFT_CTX", "70"))
    right = int(os.environ.get("NEMOTRON_RIGHT_CTX", "13"))
    try:
        m.encoder.set_default_att_context_size([left, right])
    except Exception:
        pass
    m.encoder.setup_streaming_params()

    # Decoder tuning — beam search and optional KenLM shallow fusion.
    # Both close the streaming-WER gap to SM on domain-specific audio.
    # Defaults stay greedy (matches existing measurements) so toggling
    # the env var is what activates them.
    #   NEMOTRON_BEAM_SIZE  (default 1 = greedy)
    #   NEMOTRON_KENLM_PATH (default unset = no shallow fusion)
    #   NEMOTRON_KENLM_ALPHA (default 0.6 = standard literature value)
    #   NEMOTRON_KENLM_BETA  (default 1.0 = standard literature value)
    beam_size = int(os.environ.get("NEMOTRON_BEAM_SIZE", "1"))
    kenlm_path = os.environ.get("NEMOTRON_KENLM_PATH") or None
    if beam_size > 1 or kenlm_path:
        try:
            from omegaconf import OmegaConf, open_dict
            dec_cfg = m.cfg.decoding
            with open_dict(dec_cfg):
                if beam_size > 1:
                    dec_cfg.strategy = "beam"
                    dec_cfg.beam = OmegaConf.create({
                        "beam_size": beam_size,
                        "return_best_hypothesis": True,
                        "score_norm": True,
                    })
                if kenlm_path:
                    dec_cfg.beam.kenlm_path = kenlm_path
                    dec_cfg.beam.ngram_lm_alpha = float(
                        os.environ.get("NEMOTRON_KENLM_ALPHA", "0.6"))
                    dec_cfg.beam.ngram_lm_beta = float(
                        os.environ.get("NEMOTRON_KENLM_BETA", "1.0"))
            m.change_decoding_strategy(dec_cfg)
        except Exception as e:
            import sys
            print(
                f"nemotron-native: decoder tuning failed ({beam_size=}, "
                f"{kenlm_path=}): {e}; falling back to greedy.",
                file=sys.stderr, flush=True,
            )

    _MODEL = m
    _MODEL_NAME = model_name
    return m


class NemotronNativeStreaming:
    name = "nemotron-native"

    def __init__(self, model: str = "nvidia/nemotron-speech-streaming-en-0.6b"):
        self._model_name = model

    def warm(self) -> None:
        """Pay model-load + first-call JIT before any user session arrives."""
        import torch
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )
        m = _get_model(self._model_name)
        with torch.inference_mode():
            cache_lc, cache_lt, cache_lc_len = m.encoder.get_initial_cache_state(batch_size=1)
            buf = CacheAwareStreamingAudioBuffer(model=m, online_normalization=False)
            silence = np.zeros(16000, dtype=np.float32)
            buf.append_audio(silence, stream_id=-1)
            prev_hyp = None
            pred_out = None
            for chunk_idx, (chunk_audio, chunk_lens) in enumerate(buf):
                drop_extra = (
                    0
                    if chunk_idx == 0
                    else m.encoder.streaming_cfg.drop_extra_pre_encoded
                )
                result = m.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lens,
                    cache_last_channel=cache_lc,
                    cache_last_time=cache_lt,
                    cache_last_channel_len=cache_lc_len,
                    keep_all_outputs=buf.is_buffer_empty(),
                    previous_hypotheses=prev_hyp,
                    previous_pred_out=pred_out,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )
                pred_out, _, cache_lc, cache_lt, cache_lc_len, prev_hyp = result
                break

    async def stream(self, pcm_frames, transcription_config):
        """Per-chunk cache-aware streaming. Emits a StreamFinal per chunk-delta."""
        import torch
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )
        from server.engines._base import StreamFinal, StreamPartial, StreamWord

        SAMPLE_RATE = 16000
        BYTES_PER_SAMPLE = 2  # int16

        m = _get_model(self._model_name)

        # Build streaming buffer + initial encoder cache. The buffer is created
        # empty; first append uses stream_id=-1 to create stream 0, subsequent
        # appends use stream_id=0 to EXTEND stream 0's time dim. (stream_id=-1
        # on later calls adds a new batch entry — wrong for single-mic streaming.)
        buf = CacheAwareStreamingAudioBuffer(model=m, online_normalization=False)
        cache_lc, cache_lt, cache_lc_len = m.encoder.get_initial_cache_state(batch_size=1)
        prev_hyp = None
        pred_out = None
        chunk_idx = 0
        committed_text = ""
        audio_seconds = 0.0
        stream_initialized = False
        # Accumulate raw PCM until we have at least one full encoder chunk
        # of audio. NeMo's streaming buffer doesn't enforce a minimum chunk
        # length — it'll happily iterate over short slices and produce empty
        # hypotheses. We bound the buffer ourselves: feed at least
        # `chunk_seconds` of audio per `append_audio` call.
        cfg = m.encoder.streaming_cfg
        chunk_size = cfg.chunk_size[1] if isinstance(cfg.chunk_size, list) else cfg.chunk_size
        # Mel hop is 10 ms for the FastConformer preprocessor, so each chunk
        # of `chunk_size` mel frames = chunk_size * 10 ms of audio.
        chunk_seconds = chunk_size * 0.01
        MIN_APPEND_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * chunk_seconds)
        sample_accum = bytearray()

        def _emit_words(new_text: str, t_start: float, t_end: float):
            """Tokenize a text delta into StreamWord objects with timestamps
            interpolated across the chunk's audio window."""
            tokens = new_text.split()
            if not tokens:
                return []
            span = max(0.0, t_end - t_start)
            per = span / len(tokens)
            out: list[StreamWord] = []
            for i, tok in enumerate(tokens):
                start = t_start + i * per
                end = t_start + (i + 1) * per
                out.append(
                    StreamWord(
                        content=tok,
                        start_time=start,
                        end_time=end,
                        confidence=None,
                        speaker=None,
                        is_punctuation=tok in {".", ",", "?", "!"},
                    )
                )
            return out

        async for frame in pcm_frames:
            if not frame:
                continue
            sample_accum.extend(frame)
            if len(sample_accum) < MIN_APPEND_BYTES:
                continue
            # Convert raw PCM s16le to float32 normalized to [-1, 1].
            samples = (
                np.frombuffer(bytes(sample_accum), dtype=np.int16).astype(np.float32)
                / 32768.0
            )
            audio_seconds += len(samples) / SAMPLE_RATE
            sample_accum.clear()

            # Append to the streaming buffer. First call creates stream 0
            # (stream_id=-1); subsequent calls extend stream 0's time dim
            # (stream_id=0). Using -1 on later calls would silently add a
            # new batch entry and the new audio would never reach the iterator.
            sid = -1 if not stream_initialized else 0
            try:
                buf.append_audio(samples, stream_id=sid)
            except Exception as e:
                import sys
                print(f"nemotron-native append_audio error: {e}", file=sys.stderr, flush=True)
                continue
            stream_initialized = True

            # Pull all chunks that are ready and process them. The iterator
            # advances buf.buffer_idx so subsequent calls resume from where
            # we left off.
            # Important: yield must NOT happen inside torch.inference_mode()
            # context manager — that combination breaks async generator
            # state. Collect deltas into a list, exit the inference mode,
            # then yield.
            pending_finals = []
            with torch.inference_mode():
                # Only pull FULL chunks. CacheAwareStreamingAudioBuffer's
                # iterator will happily yield partial slices when buffer is
                # close to exhausted; passing a 4-frame slice to
                # conformer_stream_step produces garbage hypothesis updates.
                while (
                    buf.buffer is not None
                    and buf.buffer_idx + chunk_size <= buf.buffer.size(-1)
                ):
                    # Fetch one chunk manually with the SAME logic the
                    # iterator uses (so cache offsets stay consistent).
                    iterator = iter(buf)
                    try:
                        chunk_audio, chunk_lens = next(iterator)
                    except StopIteration:
                        break
                    drop_extra = (
                        0
                        if chunk_idx == 0
                        else m.encoder.streaming_cfg.drop_extra_pre_encoded
                    )
                    try:
                        result = m.conformer_stream_step(
                            processed_signal=chunk_audio,
                            processed_signal_length=chunk_lens,
                            cache_last_channel=cache_lc,
                            cache_last_time=cache_lt,
                            cache_last_channel_len=cache_lc_len,
                            keep_all_outputs=False,
                            previous_hypotheses=prev_hyp,
                            previous_pred_out=pred_out,
                            drop_extra_pre_encoded=drop_extra,
                            return_transcription=True,
                        )
                    except Exception as e:
                        import sys
                        print(f"nemotron-native chunk {chunk_idx} error: {e}", file=sys.stderr, flush=True)
                        break
                    pred_out, transcribed, cache_lc, cache_lt, cache_lc_len, prev_hyp = result
                    chunk_idx += 1

                    current = ""
                    if transcribed:
                        current = transcribed[0].text if hasattr(transcribed[0], "text") else str(transcribed[0])
                    if current and current != committed_text:
                        delta = current[len(committed_text):].strip()
                        if delta:
                            t_end = audio_seconds
                            t_start = max(0.0, t_end - 0.56)
                            words = _emit_words(delta, t_start, t_end)
                            if words:
                                pending_finals.append((delta, words))
                        committed_text = current
            # Now safely yield the pending finals outside inference_mode
            for delta, words in pending_finals:
                yield StreamFinal(
                    transcript=delta,
                    words=words,
                    start_time=words[0].start_time,
                    end_time=words[-1].end_time,
                )

        # EOS: flush whatever's still in the decoder hypothesis.
        if prev_hyp is not None:
            try:
                current = prev_hyp[0].text if hasattr(prev_hyp[0], "text") else ""
            except Exception:
                current = ""
            if current and current != committed_text:
                delta = current[len(committed_text):].strip()
                if delta:
                    words = _emit_words(delta, max(0.0, audio_seconds - 0.56), audio_seconds)
                    if words:
                        yield StreamFinal(
                            transcript=delta,
                            words=words,
                            start_time=words[0].start_time,
                            end_time=words[-1].end_time,
                        )
