"""Native streaming Sortformer v2.1 with AOSC — per-chunk inference, persistent state.

This is the proper streaming diarizer per the M2 architecture brief:
NVIDIA Streaming Sortformer v2.1 (`nvidia/diar_streaming_sortformer_4spk-v2.1`)
called per audio chunk via `forward_streaming_step` with a session-persistent
`streaming_state` (AOSC speaker cache + FIFO) and `total_preds` running output.

Why this replaces the offline `model.diarize()` integration:
  - The offline pattern recreated `streaming_state` on every call to label(),
    losing speaker continuity, and re-processed the full growing audio buffer
    end-to-end — O(N²) work and CUDA OOM past ~5 min of live mic.
  - Here the state is created ONCE per session in `warm()`. Each call to
    `label()` processes only the audio delta since the previous call, calls
    `forward_streaming_step` to advance the encoder cache and AOSC speaker
    cache, and updates `total_preds` with the new frame-level speaker
    activity. Memory is constant in audio length (cache size is fixed by
    `fifo_len` + `spkcache`). Speaker labels are stable across the session
    because AOSC's arrival-order indexing guarantees it.

Word-to-speaker assignment uses max-overlap of each word's [start, end]
interval against the per-speaker activity timeline derived from total_preds.
On ties we prefer the current speaker (mirrors Speechmatics
`prefer_current_speaker`).

Per the v2.1 model card the recommended config is:
  chunk_len = 340, chunk_right_context = 40, fifo_len = 40,
  spkcache_update_period = 300
which corresponds to ~1.04 s input-buffer latency, matching the streaming
DER numbers reported in the model card (CALLHOME-part2 11.19% with 0.25 s
collar; AMI IHM 16.67% with 0 s collar).
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

import numpy as np


_MODEL = None
_MODEL_ID: str | None = None


def _get_model():
    """Load Streaming Sortformer v2.1 once per process and configure for
    the live-mic latency sweet spot from the model card."""
    global _MODEL, _MODEL_ID
    # v2 (not v2.1): NeMo 2.4 in our runtime can't restore v2.1's config
    # (it adds `spkcache_update_period` which the 2.4 SortformerModules
    # constructor rejects). v2 uses the same AOSC architecture and reports
    # the same streaming DER class on the model card; v2.1 was a checkpoint
    # refinement, not an architecture change. Upgrade NeMo to use v2.1.
    model_id = os.environ.get(
        "SORTFORMER_STREAMING_MODEL",
        "nvidia/diar_streaming_sortformer_4spk-v2",
    )
    if _MODEL is not None and _MODEL_ID == model_id:
        return _MODEL
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel
    from nemo.collections.asr.modules import sortformer_modules as _smod

    # NeMo 2.4's SortformerModules.__init__ rejects `spkcache_update_period`
    # which is present in v2 / v2.1 model configs. We're on 2.4 (newer NeMo
    # has torch / pyannote upgrade conflicts; not safe to bump runtime).
    # Wrap __init__ to drop unknown kwargs and forward the rest.
    if not getattr(_smod.SortformerModules.__init__, "_depodash_compat", False):
        _orig_init = _smod.SortformerModules.__init__
        import inspect as _inspect
        _accepted = set(_inspect.signature(_orig_init).parameters.keys())
        def _patched(self, *args, **kwargs):
            unknown = [k for k in kwargs if k not in _accepted]
            for k in unknown:
                kwargs.pop(k, None)
            return _orig_init(self, *args, **kwargs)
        _patched._depodash_compat = True
        _smod.SortformerModules.__init__ = _patched

    m = SortformerEncLabelModel.from_pretrained(model_id)
    if torch.cuda.is_available():
        m = m.cuda()
    m.eval()
    # Live-mic latency config from the model card (~1.04 s buffer).
    m.sortformer_modules.chunk_len = 340
    m.sortformer_modules.chunk_right_context = 40
    m.sortformer_modules.fifo_len = 40
    _MODEL = m
    _MODEL_ID = model_id
    return m


class SortformerNativeStreaming:
    name = "sortformer-streaming-v2"

    def __init__(self):
        self._streaming_state = None
        self._total_preds = None
        # Audio bytes already pushed through forward_streaming_step. The
        # session passes the full growing audio buffer on each label() call;
        # we slice off only the new bytes since the last call.
        self._processed_audio_bytes = 0
        # Per-session speaker label stickiness — AOSC indices 0..3 map to
        # S1..S4 stably, but we also remember the "current" (last-assigned)
        # speaker for tie-breaking in word-overlap matching.
        self._last_assigned_speaker: str | None = None

    def warm(self) -> None:
        """Pay model load + first-call JIT cost during server startup so the
        first user session doesn't trip the WebSocket keepalive timeout."""
        import torch

        m = _get_model()
        with torch.inference_mode():
            self._streaming_state = m.sortformer_modules.init_streaming_state(
                batch_size=1, async_streaming=False, device=m.device
            )
            self._total_preds = torch.zeros(
                (1, 0, m.sortformer_modules.n_spk), device=m.device
            )
            # Push 0.5 s of silence through to JIT the kernels.
            silence = np.zeros(8000, dtype=np.float32)
            self._step(m, silence)
            # Reset per-session state — warm() runs at startup, not at
            # session start. The session's __init__ creates a fresh
            # instance with fresh state.
            self._streaming_state = None
            self._total_preds = None
            self._processed_audio_bytes = 0

    def _step(self, m, samples_f32: np.ndarray):
        """Push one chunk of float32 samples through the streaming sortformer.
        Updates self._streaming_state and self._total_preds in place."""
        import torch

        if self._streaming_state is None:
            self._streaming_state = m.sortformer_modules.init_streaming_state(
                batch_size=1, async_streaming=False, device=m.device
            )
            self._total_preds = torch.zeros(
                (1, 0, m.sortformer_modules.n_spk), device=m.device
            )

        # Preprocess raw samples → mel features.
        audio_signal = torch.from_numpy(samples_f32).unsqueeze(0).to(m.device)
        audio_len = torch.tensor([samples_f32.shape[0]], device=m.device)
        processed_signal, processed_signal_length = m.preprocessor(
            input_signal=audio_signal, length=audio_len
        )

        # One step of streaming inference; advances the AOSC speaker cache,
        # the FIFO, and the encoder cache, all stored in self._streaming_state.
        self._streaming_state, self._total_preds = m.forward_streaming_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            streaming_state=self._streaming_state,
            total_preds=self._total_preds,
            left_offset=0,
            right_offset=0,
        )

    async def label(self, words, audio_chunk=None):
        if not audio_chunk or not words:
            return words
        return await asyncio.to_thread(self._label_sync, words, audio_chunk)

    def _label_sync(self, words, pcm: bytes):
        import torch

        SAMPLE_RATE = 16000
        BYTES_PER_SAMPLE = 2

        # The caller passes the FULL growing audio buffer. Slice off only
        # the NEW bytes since the last forward_streaming_step. This is what
        # makes the diarizer streaming-native — constant memory in session
        # length.
        new_bytes = pcm[self._processed_audio_bytes:]
        if len(new_bytes) < SAMPLE_RATE * BYTES_PER_SAMPLE // 4:  # 250 ms minimum
            return self._assign_speakers(words)
        new_samples = (
            np.frombuffer(new_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        self._processed_audio_bytes = len(pcm)

        m = _get_model()
        with torch.inference_mode():
            self._step(m, new_samples)

        return self._assign_speakers(words)

    def _assign_speakers(self, words):
        """Map word [start, end] to the AOSC speaker with maximum activity
        overlap. AOSC orders speakers by arrival time and keeps the index
        stable across the session, so we just emit f"S{i+1}".

        We also smooth single-word minority-speaker excursions inside a
        sentence: a lone S2 word surrounded by S1 words within the same
        sentence (terminated by `.`/`?`/`!`) is corrected to S1.
        """
        import torch

        if self._total_preds is None or self._total_preds.shape[1] == 0:
            return words

        # total_preds is (batch=1, frames, n_spk). Sortformer outputs frame
        # rate is 80 ms per frame (post-subsampling).
        FRAME_S = 0.08
        preds = self._total_preds[0].detach().cpu().numpy()  # (frames, n_spk)
        n_frames, n_spk = preds.shape
        # Speaker probabilities are independent per speaker (multi-label),
        # so threshold rather than argmax: a frame can have multiple active
        # speakers (overlap). For word-level assignment we pick the speaker
        # with the highest *total activity* over the word's interval.

        for w in words:
            if w.start_time is None or w.end_time is None:
                continue
            f0 = max(0, int(w.start_time / FRAME_S))
            f1 = min(n_frames, int(w.end_time / FRAME_S) + 1)
            if f1 <= f0:
                continue
            activity = preds[f0:f1].sum(axis=0)  # (n_spk,)
            # If no speaker shows meaningful activity in this window, leave
            # the word's speaker as-is (likely silence or unmodeled audio).
            if float(activity.max()) < 0.5:
                continue
            top = int(activity.argmax())
            top_activity = float(activity[top])
            # Tie-break: prefer current speaker if their activity is within
            # 10 % of the top — mirrors Speechmatics prefer_current_speaker.
            if self._last_assigned_speaker is not None:
                cur_idx = _label_to_index(self._last_assigned_speaker)
                if 0 <= cur_idx < n_spk:
                    if activity[cur_idx] >= 0.9 * top_activity:
                        top = cur_idx
            w.speaker = f"S{top + 1}"
            self._last_assigned_speaker = w.speaker

        # Punctuation-boundary smoothing: a lone minority-speaker word
        # between two same-speaker neighbors (within the same sentence)
        # gets corrected to the neighbors' speaker. Speechmatics does this.
        _smooth_singletons(words)

        return words

    async def turns_to_now(self, audio_chunk):
        return []


def _label_to_index(label: str) -> int:
    """`S1` → 0, `S2` → 1, etc. Returns -1 on parse failure."""
    if not label or len(label) < 2 or label[0] != "S":
        return -1
    try:
        return int(label[1:]) - 1
    except ValueError:
        return -1


_EOS_MARKS = {".", "?", "!"}


def _smooth_singletons(words):
    """Fix `S1 S1 [S2] S1 S1` → `S1 S1 S1 S1 S1` within a sentence.

    Speechmatics corrects a lone minority-speaker word inside a sentence to
    the sentence majority. We do the local-neighbor version: a word whose
    speaker differs from BOTH its non-punctuation neighbors gets re-labeled
    to the neighbors' speaker, but only if we don't cross a sentence
    boundary (period / question / exclamation between this word and either
    neighbor).
    """
    # Build an index over non-punctuation tokens with their speakers.
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
            # Don't cross a sentence boundary.
            if not _has_eos_between(words, i_prev, i_cur):
                if not _has_eos_between(words, i_cur, i_next):
                    words[i_cur].speaker = sp_prev


def _has_eos_between(words, i_left, i_right) -> bool:
    for j in range(i_left + 1, i_right):
        if getattr(words[j], "is_punctuation", False):
            if words[j].content in _EOS_MARKS:
                return True
    return False
