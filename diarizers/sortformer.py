"""NVIDIA Streaming Sortformer v2 — 4-speaker streaming-native diarizer.

CUDA-only (the NeMo SortformerEncLabelModel needs a GPU). Loads
`nvidia/diar_streaming_sortformer_4spk-v2` and uses the AOSC online mode.

Used by the M2 server: caller passes the audio bytes that produced the
engine's words; we write a temp WAV, run sortformer, parse the RTTM
output, overlay speakers onto each word by start_time midpoint.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import wave
from io import BytesIO
from pathlib import Path


_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    # v2 is the AOSC streaming model. `model.diarize()` calls
    # `forward_streaming` internally which chunks the audio with
    # `streaming_feat_loader` and runs `forward_streaming_step` per chunk —
    # proper AOSC speaker cache + FIFO, stable arrival-order indices, real
    # streaming numbers (DIHARD III ~20%, AMI IHM ~17%, CALLHOME ~11% per
    # the model card). v1 was offline-only with a 5–10 pp DER tax.
    # Requires NeMo 2.5+ for the SortformerModules config kwargs.
    model_id = os.environ.get(
        "SORTFORMER_MODEL", "nvidia/diar_streaming_sortformer_4spk-v2"
    )

    # NeMo 2.4/2.5 SortformerModules.__init__ rejects `spkcache_update_period`
    # which appears in v2/v2.1 model configs. Wrap to drop unknown kwargs.
    from nemo.collections.asr.modules import sortformer_modules as _smod
    import inspect as _inspect
    if not getattr(_smod.SortformerModules.__init__, "_depodash_compat", False):
        _orig_init = _smod.SortformerModules.__init__
        _accepted = set(_inspect.signature(_orig_init).parameters.keys())
        def _patched(self, *args, **kwargs):
            for k in [x for x in kwargs if x not in _accepted]:
                kwargs.pop(k, None)
            return _orig_init(self, *args, **kwargs)
        _patched._depodash_compat = True
        _smod.SortformerModules.__init__ = _patched

    model = SortformerEncLabelModel.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    # Live-mic latency config (~1.04 s buffer) from the v2.1 model card.
    # These attributes set the streaming chunk window.
    try:
        model.sortformer_modules.chunk_len = 340
        model.sortformer_modules.chunk_right_context = 40
        model.sortformer_modules.fifo_len = 40
    except Exception:
        pass
    _MODEL = model
    return model


class SortformerStreaming:
    name = "sortformer-streaming"

    async def label(self, words, audio_chunk=None):
        if not audio_chunk or not words:
            return words
        return await asyncio.to_thread(self._label_sync, words, audio_chunk)

    def _label_sync(self, words, pcm: bytes):
        SAMPLE_RATE = 16000
        BYTES_PER_SEC = SAMPLE_RATE * 2
        # v2 streaming sortformer internally chunks audio via streaming_feat_loader
        # and uses AOSC (fixed-size speaker cache + FIFO) — constant memory per
        # call regardless of input length. So we feed it the full buffer rather
        # than a sliding window: AOSC needs sufficient per-speaker history to
        # keep embeddings distinct, and a 30 s window starves it when speakers
        # alternate slowly (one person stops talking for >30 s → their embedding
        # falls off → AOSC re-anchors → speaker IDs swap).
        window_start_s = 0.0

        wav_bytes = BytesIO()
        with wave.open(wav_bytes, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = Path(f.name)
            f.write(wav_bytes.getvalue())
        try:
            model = _get_model()
            predictions = model.diarize(str(tmp_path), batch_size=1)
            segments = _parse_rttm_lines(predictions[0] if predictions else [])
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

        # Segments are in the WINDOW's local time (0 = start of window).
        # Shift to global audio time so word timestamps line up.
        segments = [(spk, s + window_start_s, e + window_start_s) for (spk, s, e) in segments]

        for w in words:
            midpoint = 0.5 * (w.start_time + w.end_time)
            # Only assign a speaker if the word falls within the window we
            # actually analyzed; for words older than that, leave .speaker
            # alone (the session layer's speaker remap is sticky per session
            # so previous chunks' labels persist).
            if midpoint < window_start_s:
                continue
            for speaker, start, end in segments:
                if start <= midpoint <= end:
                    w.speaker = speaker
                    break
        return words

    async def turns_to_now(self, audio_chunk):
        return []


def _parse_rttm_lines(lines: list[str]) -> list[tuple[str, float, float]]:
    """NeMo Sortformer emits `start end speaker_X` (space-separated, 3 fields).
    Standard RTTM (9 fields, leading `SPEAKER`) is also handled as fallback."""
    segs = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "SPEAKER" and len(parts) >= 8:
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segs.append((speaker, start, start + duration))
        elif len(parts) == 3:
            try:
                start = float(parts[0])
                end = float(parts[1])
                speaker = parts[2]
                segs.append((speaker, start, end))
            except ValueError:
                continue
    return segs
