"""Streaming Sortformer v2 diarizer per brief §4 — per-chunk inference,
persistent streaming_state across all calls in a session.

What the previous diarizer (`diarizers/sortformer.py`) does wrong: it calls
`model.diarize()` per emission, which internally re-creates `streaming_state`
on every call via `forward_streaming(processed_signal)`. AOSC's speaker cache
is therefore wiped between calls, so the model has to re-anchor speaker IDs
each emission. Result: S1/S2 swap as speakers' relative arrival order shifts
between input slices. The brief calls this out — you NEED state continuity
across calls for AOSC's arrival-order guarantee to hold.

What this diarizer does right:
  1. Initializes `streaming_state` + `total_preds` ONCE per session (`__init__`).
  2. Maintains a growing mel-feature buffer per session. `streaming_feat_loader`
     needs prior frames for left-context padding when chunking new audio,
     which is why we keep the full buffer (it's small — ~15 MB for an hour
     at 128 mels × 10 ms hop × 4 bytes).
  3. On each `label(words, audio_so_far)` call:
       a. Compute the audio delta since the last call.
       b. Preprocess delta → new mel features, append to buffer.
       c. Use `streaming_feat_loader` from the prior frame offset.
       d. For each chunk yielded, call `forward_streaming_step` — this
          updates `streaming_state` (AOSC speaker cache, FIFO, encoder
          cache) AND `total_preds` (cumulative frame-level speaker
          probabilities).
  4. After processing, assign words to speakers by max-overlap of each
     word's [start, end] interval against `total_preds`, with `prefer
     current speaker` tie-break (matches Speechmatics behavior).
  5. AOSC indices 0..3 map directly to S1..S4 — no remap, because AOSC
     guarantees the same voice → same index for the whole session.
  6. Punctuation-boundary smoothing: a lone minority-speaker word between
     same-speaker neighbors (without a sentence boundary in between) gets
     corrected.

Model config matches the v2 live-mic sweet spot from the model card:
  chunk_len = 340, chunk_right_context = 40, fifo_len = 40
  → ~1.04 s input-buffer latency, ~16% DER on AMI-IHM (collar 0 s).
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

import numpy as np


_MODEL = None
_MODEL_ID: str | None = None
_PATCHED = False


def _patch_sortformer_modules():
    """NeMo 2.4/2.5 SortformerModules.__init__ rejects `spkcache_update_period`
    which appears in v2/v2.1 model configs. Wrap to drop unknown kwargs."""
    global _PATCHED
    if _PATCHED:
        return
    import inspect
    from nemo.collections.asr.modules import sortformer_modules as _smod

    _orig_init = _smod.SortformerModules.__init__
    _accepted = set(inspect.signature(_orig_init).parameters.keys())

    def _patched(self, *args, **kwargs):
        for k in [x for x in kwargs if x not in _accepted]:
            kwargs.pop(k, None)
        return _orig_init(self, *args, **kwargs)

    _smod.SortformerModules.__init__ = _patched
    _PATCHED = True


def _get_model():
    global _MODEL, _MODEL_ID
    # v2.1 is the live-mic deployment target per brief §3/§7. The key
    # delta from v2: `spkcache_update_period` — controls how often the
    # long-term speaker cache (separate from the FIFO) gets refreshed.
    # Without this knob (v2 doesn't have it), the speaker cache reverts
    # to FIFO-only memory, and after >fifo_len * chunk_len of silence the
    # voice fingerprints decay and returning speakers get re-labeled as
    # new (the "I came back after 3 min and got relabeled S3" symptom).
    model_id = os.environ.get(
        "SORTFORMER_STREAMING_MODEL",
        "nvidia/diar_streaming_sortformer_4spk-v2.1",
    )
    if _MODEL is not None and _MODEL_ID == model_id:
        return _MODEL
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    _patch_sortformer_modules()  # tolerated as no-op on 2.5.3+
    m = SortformerEncLabelModel.from_pretrained(model_id)
    if torch.cuda.is_available():
        m = m.cuda()
    m.eval()
    # Live-mic config from v2.1 model card (~1.04 s buffer, longer-term
    # speaker memory). spkcache_update_period=300 is what makes voices
    # persist across multi-minute silences.
    m.sortformer_modules.chunk_len = 340
    m.sortformer_modules.chunk_right_context = 40
    m.sortformer_modules.fifo_len = 40
    if hasattr(m.sortformer_modules, "spkcache_update_period"):
        m.sortformer_modules.spkcache_update_period = 300
    _MODEL = m
    _MODEL_ID = model_id
    return m


class SortformerStreaming:
    name = "sortformer-streaming"

    def __init__(self):
        # State persists across all label() calls in this session.
        self._streaming_state = None
        self._total_preds = None  # (1, total_diar_frames, n_spk)
        # Growing mel-feature buffer (1, n_mels, total_mel_frames).
        # Needed because streaming_feat_loader pads each new chunk with
        # `chunk_left_context * subsampling_factor` frames from before it.
        self._mel_features = None
        # How many mel frames have been ingested by forward_streaming_step.
        # Next call to streaming_feat_loader uses this as feat_seq_offset.
        self._processed_mel_frames = 0
        # How many raw audio bytes have been consumed from the session
        # buffer. Each label() call processes only the delta since last.
        self._processed_audio_bytes = 0

    def warm(self) -> None:
        """Pay model load + first-call JIT during server startup."""
        import torch

        m = _get_model()
        with torch.inference_mode():
            # Tiny warmup pass — don't init the per-session state here; that
            # happens lazily in label() so each session is a clean slate.
            silence = np.zeros(16000, dtype=np.float32)
            audio_signal = torch.from_numpy(silence).unsqueeze(0).to(m.device)
            audio_len = torch.tensor([silence.shape[0]], device=m.device)
            m.preprocessor(input_signal=audio_signal, length=audio_len)

    async def label(self, words, audio_chunk=None):
        if not audio_chunk or not words:
            return words
        return await asyncio.to_thread(self._label_sync, words, audio_chunk)

    def _label_sync(self, words, pcm: bytes):
        import torch

        SAMPLE_RATE = 16000
        BYTES_PER_SAMPLE = 2
        MIN_DELTA_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE // 4  # 250 ms

        m = _get_model()
        device = m.device

        new_bytes = pcm[self._processed_audio_bytes:]
        if len(new_bytes) < MIN_DELTA_BYTES:
            # Not enough new audio to be worth a streaming step; just label
            # words against existing total_preds and return.
            return self._assign_speakers(words)
        self._processed_audio_bytes = len(pcm)

        with torch.inference_mode():
            # 1) Preprocess the audio delta to mel features.
            new_samples = (
                np.frombuffer(new_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )
            audio_signal = torch.from_numpy(new_samples).unsqueeze(0).to(device)
            audio_len = torch.tensor([new_samples.shape[0]], device=device)
            new_features, new_feat_len = m.preprocessor(
                input_signal=audio_signal, length=audio_len
            )
            # new_features shape: (1, n_mels, T_new)

            # 2) Append to the per-session mel buffer.
            if self._mel_features is None:
                self._mel_features = new_features
            else:
                self._mel_features = torch.cat(
                    [self._mel_features, new_features], dim=2
                )
            total_feat_len = self._mel_features.shape[2]

            # 3) Lazy-init streaming_state + total_preds the first time we
            # actually have features to push (warm() is process-wide, not
            # session-wide).
            if self._streaming_state is None:
                self._streaming_state = m.sortformer_modules.init_streaming_state(
                    batch_size=1,
                    async_streaming=getattr(m, "async_streaming", False),
                    device=device,
                )
                self._total_preds = torch.zeros(
                    (1, 0, m.sortformer_modules.n_spk), device=device
                )

            # 4) Use streaming_feat_loader from where we left off. The loader
            # reads backward by chunk_left_context * subsampling_factor for
            # encoder-cache padding, which is why the full buffer must stay.
            feat_seq_offset = torch.tensor(
                [self._processed_mel_frames], device=device, dtype=torch.long
            )
            feat_seq_length = torch.tensor(
                [total_feat_len], device=device, dtype=torch.long
            )
            # Patch the loader to start from our offset by slicing the seq.
            # streaming_feat_loader's internal `stt_feat` starts at 0, so to
            # resume from `_processed_mel_frames` we slice the buffer and
            # adjust offset accordingly.
            tail = self._mel_features[:, :, self._processed_mel_frames :]
            tail_len = torch.tensor(
                [tail.shape[2]], device=device, dtype=torch.long
            )
            tail_offset = torch.tensor([0], device=device, dtype=torch.long)

            for (
                _chunk_idx,
                chunk_feat_seq_t,
                chunk_feat_lengths,
                left_offset,
                right_offset,
            ) in m.sortformer_modules.streaming_feat_loader(
                feat_seq=tail,
                feat_seq_length=tail_len,
                feat_seq_offset=tail_offset,
            ):
                # Loader emits chunk_feat_seq_t already transposed to
                # (batch, time, features) — the format pre_encode wants.
                # Pass through directly (don't re-transpose).
                self._streaming_state, self._total_preds = m.forward_streaming_step(
                    processed_signal=chunk_feat_seq_t,
                    processed_signal_length=chunk_feat_lengths,
                    streaming_state=self._streaming_state,
                    total_preds=self._total_preds,
                    left_offset=left_offset,
                    right_offset=right_offset,
                )
            self._processed_mel_frames = total_feat_len

        return self._assign_speakers(words)

    def _assign_speakers(self, words):
        """Max-overlap assignment of each word's [start, end] interval to
        the AOSC speaker with the highest total activity in that window.
        Prefers the current (last-assigned) speaker on near-ties, per
        Speechmatics `prefer_current_speaker` behavior."""
        if self._total_preds is None or self._total_preds.shape[1] == 0:
            return words

        # Sortformer output frame rate is 80 ms per frame.
        FRAME_S = 0.08
        preds = self._total_preds[0].detach().cpu().numpy()  # (frames, n_spk)
        n_frames, n_spk = preds.shape

        last_speaker_idx = -1
        for w in words:
            if w.start_time is None or w.end_time is None:
                continue
            f0 = max(0, int(w.start_time / FRAME_S))
            f1 = min(n_frames, int(w.end_time / FRAME_S) + 1)
            if f1 <= f0:
                continue
            activity = preds[f0:f1].sum(axis=0)  # (n_spk,)
            max_act = float(activity.max())
            if max_act < 0.3:
                # Below threshold — likely silence between speakers. Don't
                # overwrite; let the engine's default (None) stand.
                continue
            top = int(activity.argmax())
            # Tie-break: stick with the current speaker if their activity
            # is within 10 % of the top — reduces single-word spurious
            # speaker flips (matches Speechmatics behavior).
            if 0 <= last_speaker_idx < n_spk:
                if activity[last_speaker_idx] >= 0.9 * activity[top]:
                    top = last_speaker_idx
            w.speaker = f"S{top + 1}"
            last_speaker_idx = top

        _smooth_singletons(words)
        return words

    async def turns_to_now(self, audio_chunk):
        return []


_EOS_MARKS = {".", "?", "!"}


def _smooth_singletons(words):
    """`S1 S1 [S2] S1 S1` within a sentence → `S1 S1 S1 S1 S1`.

    Speechmatics corrects a lone minority-speaker word inside a sentence
    to the majority. Local-neighbor version: a word whose speaker differs
    from BOTH neighbors gets re-labeled to the neighbor speaker, unless
    there's a sentence boundary in between.
    """
    nonpunct = [
        i for i, w in enumerate(words)
        if not getattr(w, "is_punctuation", False) and w.speaker is not None
    ]
    for k in range(1, len(nonpunct) - 1):
        i_prev, i_cur, i_next = nonpunct[k - 1], nonpunct[k], nonpunct[k + 1]
        sp_prev = words[i_prev].speaker
        sp_cur = words[i_cur].speaker
        sp_next = words[i_next].speaker
        if sp_prev == sp_next and sp_cur != sp_prev:
            if not _has_eos_between(words, i_prev, i_cur):
                if not _has_eos_between(words, i_cur, i_next):
                    words[i_cur].speaker = sp_prev


def _has_eos_between(words, i_left, i_right) -> bool:
    for j in range(i_left + 1, i_right):
        if getattr(words[j], "is_punctuation", False):
            if words[j].content in _EOS_MARKS:
                return True
    return False
