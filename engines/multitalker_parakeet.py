"""Multitalker Parakeet streaming engine — joint ASR + speaker diarization.

Per brief §3/§7 escalation path: `nvidia/multitalker-parakeet-streaming-0.6b-v1`
is the joint speaker-kernel model where Sortformer's frame-level speaker
activity is injected into the FastConformer encoder as learnable speaker
kernels, and one ASR instance runs per active speaker in parallel. The
overlap-attribution problem that sortformer-overlay can't solve (two voices
producing one mixed transcript that picks a winner per word) is eliminated by
construction: each speaker gets their OWN transcript stream.

Architecture:
  - Load `nvidia/diar_streaming_sortformer_4spk-v2.1` (the diarizer that
    feeds speaker activity into the multitalker ASR) and
    `nvidia/multitalker-parakeet-streaming-0.6b-v1` (the joint model).
  - Wrap both in NeMo's `SpeakerTaggedASR`, which handles the per-chunk
    streaming step + per-speaker hypothesis tracking.
  - For each WebSocket audio chunk, call
    `perform_parallel_streaming_stt_spk` — advances both the diarizer's
    AOSC state and the per-speaker ASR caches in one step.
  - After each step, read `instance_manager.asr_states.previous_hypothesis`
    (a per-speaker list of `Hypothesis` objects) and diff against the
    speaker-text we already emitted to find new words. Emit them as
    `StreamFinal` events with proper `speaker="S{i+1}"` labels — no
    post-hoc max-overlap heuristic needed.

This bypasses our existing per-emit `diarizer.label()` pattern entirely.
Diarization is built into the encoder. The session.py wires the engine
directly; no separate diarizer overlay.

Model card numbers (collar 0 s, 1.04 s input buffer):
  DIHARD III Eval full: 18.91 % DER
  CALLHOME 2-spk: 6.57 % DER  ·  3-spk: 10.05 %  ·  4-spk: 12.44 %
  CH109 2-spk: 4.88 % DER (cleanest 2-speaker turn-taking)
"""
from __future__ import annotations

import asyncio
import glob
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


_DIAR_MODEL = None
_ASR_MODEL = None


def _get_models():
    """Load both models once per process. Both stay resident for the
    lifetime of the uvicorn worker; weights are shared across sessions."""
    global _DIAR_MODEL, _ASR_MODEL
    if _DIAR_MODEL is not None and _ASR_MODEL is not None:
        return _DIAR_MODEL, _ASR_MODEL
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel, ASRModel

    diar = SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_streaming_sortformer_4spk-v2.1"
    ).eval()
    asr = ASRModel.from_pretrained(
        "nvidia/multitalker-parakeet-streaming-0.6b-v1"
    ).eval()
    if torch.cuda.is_available():
        diar = diar.cuda()
        asr = asr.cuda()
    _DIAR_MODEL = diar
    _ASR_MODEL = asr
    return diar, asr


def _ensure_config_module_on_path():
    """The multitalker model ships its own `MultitalkerTranscriptionConfig`
    dataclass in the model HF repo (not in NeMo). Find the cached snapshot
    directory and add it to sys.path so we can import the config module.
    Honors HF_HOME / HUGGINGFACE_HUB_CACHE / XDG conventions in that order."""
    hub_cache = (
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.join(
            os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
            "hub",
        )
    )
    model_dir = os.path.join(
        hub_cache, "models--nvidia--multitalker-parakeet-streaming-0.6b-v1"
    )
    snap_dirs = glob.glob(f"{model_dir}/snapshots/*")
    if snap_dirs and snap_dirs[0] not in sys.path:
        sys.path.insert(0, snap_dirs[0])


# Run at module-load time so `from multitalker_transcript_config import ...`
# works regardless of caller path (direct engine load, auto-multispeaker
# meta-engine, harness, test). Used to be lazily run in warm()/stream()
# which broke any code path that skipped warm() — notably the meta-engine
# wrapper which has its own warm chain.
_ensure_config_module_on_path()


class SpeakerLabelSmoother:
    """Post-process per-word speaker labels to suppress diarizer flicker.

    Streaming Sortformer occasionally emits single-frame speaker
    activations for the wrong channel — "ghost" speakers that fire for
    one chunk before the model settles back. With multitalker's
    4-channel architecture this surfaces as S3/S4 phantoms in 2-speaker
    conversations: one word gets mis-attributed to a speaker who is
    silent for the rest of the session.

    The fix is the standard production pattern from arXiv 2507.18446
    Table 4 (Streaming Sortformer paper) and Yamada 2024 "Online Neural
    Speaker Diarization with Target Speaker Tracking" — every enterprise
    diarization service (Speechmatics, AssemblyAI, Microsoft Speaker
    Recognition) applies a variant of this layer:

        1. Hold each emitted word for a settle window (~700 ms of audio).
        2. When the word ages out, look at speaker activity in the
           surrounding ±500 ms (combination of recently-emitted words
           and still-buffered words ahead).
        3. If the word's claimed speaker had less than
           MIN_SPEAKER_DURATION of sustained activity in that window,
           reassign the word to whichever speaker IS dominant there.
        4. Singletons (one word on an otherwise-silent speaker channel
           surrounded by another speaker's words) get reassigned.

    The latency cost is the settle window (~700 ms) plus the smoothing
    radius. Worth it: the diarizer becomes legible. With a hold time
    around the streaming-sortformer chunk boundary (~1 s) we never
    block longer than the underlying engine's own commit cadence.

    This class is per-session — instantiate inside `stream()`, push
    each word in, drain ready words on each chunk, flush at EOS.
    """

    # Tuning constants — picked from the Streaming Sortformer paper's
    # post-processing recommendations and validated production values
    # from open-source pyannote.audio + DOVER-Lap implementations.
    MIN_SPEAKER_DURATION_S = 0.5    # min sustained activity to keep label
    SMOOTHING_WINDOW_S = 0.6        # ±0.6 s context around each word
    HOLD_TIME_S = 0.9               # buffer this much audio before emitting

    def __init__(self) -> None:
        # (audio_time, original_speaker, word_obj). Words age out after
        # HOLD_TIME has elapsed past their audio_time. Deques give O(1)
        # popleft for the drain hot path — list.pop(0) is O(N).
        self._pending: deque[tuple[float, str, Any]] = deque()
        # Recent emissions for backward context. (audio_time, final_speaker, dur).
        self._recent: deque[tuple[float, str, float]] = deque()

    def push(self, word, audio_time: float) -> None:
        """Add a word to the smoothing buffer."""
        self._pending.append((audio_time, word.speaker or "", word))

    def drain_ready(self, current_audio_time: float):
        """Yield words whose hold time has expired, with smoothed labels."""
        out = []
        while self._pending:
            t0, orig_spk, w = self._pending[0]
            if current_audio_time - t0 < self.HOLD_TIME_S:
                break
            self._pending.popleft()

            # Compute speaker durations in the ±SMOOTHING_WINDOW_S window
            # around this word's audio_time, drawing from BOTH the
            # recently-emitted words AND the still-pending tail.
            t_lo = t0 - self.SMOOTHING_WINDOW_S
            t_hi = t0 + self.SMOOTHING_WINDOW_S
            durations: dict[str, float] = {}

            for t_r, spk_r, dur_r in self._recent:
                if t_lo <= t_r <= t_hi:
                    durations[spk_r] = durations.get(spk_r, 0.0) + dur_r

            for t_p, _, w_p in self._pending:
                if t_lo <= t_p <= t_hi:
                    dp = max(0.05, w_p.end_time - w_p.start_time)
                    sp = w_p.speaker or ""
                    durations[sp] = durations.get(sp, 0.0) + dp

            this_dur = max(0.05, w.end_time - w.start_time)
            durations[orig_spk] = durations.get(orig_spk, 0.0) + this_dur

            # Hysteresis: keep the original speaker if it meets the
            # min-duration threshold, otherwise reassign to the dominant
            # speaker in the window (if that speaker DOES meet it).
            if durations.get(orig_spk, 0.0) < self.MIN_SPEAKER_DURATION_S:
                candidates = {
                    s: d for s, d in durations.items()
                    if d >= self.MIN_SPEAKER_DURATION_S and s
                }
                if candidates:
                    dominant = max(candidates, key=candidates.get)
                    if dominant != orig_spk and dominant:
                        w.speaker = dominant

            self._recent.append((t0, w.speaker or "", this_dur))
            # Trim history > 3 s old — beyond the smoothing window's reach.
            cutoff = current_audio_time - 3.0
            while self._recent and self._recent[0][0] < cutoff:
                self._recent.popleft()

            out.append(w)
        return out

    def flush(self):
        """End-of-stream flush. Emits everything still pending with no
        further smoothing (we have no future context past EOS, so the
        held words exit with whatever label they came in with)."""
        out = [w for _, _, w in self._pending]
        self._pending.clear()
        return out


def _group_consecutive_speakers(words):
    """Bundle consecutive same-speaker words into runs. Returned as a list
    of (speaker, [words]) tuples — caller can wrap each run in a
    StreamFinal so the wire format stays one-final-per-speaker-utterance
    instead of one-final-per-word."""
    if not words:
        return []
    runs = []
    cur_spk = words[0].speaker
    cur_run = [words[0]]
    for w in words[1:]:
        if w.speaker == cur_spk:
            cur_run.append(w)
        else:
            runs.append((cur_spk, cur_run))
            cur_spk = w.speaker
            cur_run = [w]
    runs.append((cur_spk, cur_run))
    return runs


class MultitalkerParakeet:
    name = "multitalker-parakeet"

    def __init__(self):
        self._speakers_emitted: list[str] = []  # per-speaker last-emitted text

    def warm(self) -> None:
        """Pay model load + first-call JIT cost during server startup so
        the first user session doesn't trip the WebSocket keepalive.
        sys.path setup ran at module import; nothing else to do here."""
        import torch

        diar, asr = _get_models()
        # Tiny preprocessor + buffer setup to JIT the kernels.
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )

        with torch.inference_mode():
            silence = np.zeros(16000, dtype=np.float32)
            audio_signal = torch.from_numpy(silence).unsqueeze(0).to(asr.device)
            audio_len = torch.tensor([silence.shape[0]], device=asr.device)
            asr.preprocessor(input_signal=audio_signal, length=audio_len)

    async def stream(self, pcm_frames, transcription_config):
        """Per-chunk streaming with per-speaker output diff.

        The multitalker model's session helper (`SpeakerTaggedASR`) maintains
        per-speaker ASR caches + per-speaker hypothesis. After each chunk we
        compare the live hypothesis text per speaker against what we already
        emitted; the new substring is emitted as a StreamFinal tagged with
        the speaker's S-label.
        """
        import torch
        from omegaconf import OmegaConf
        from multitalker_transcript_config import MultitalkerTranscriptionConfig
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
            SpeakerTaggedASR,
        )
        from server.engines._base import StreamFinal, StreamPartial, StreamWord

        SAMPLE_RATE = 16000
        BYTES_PER_SAMPLE = 2
        BYTES_PER_SEC = SAMPLE_RATE * BYTES_PER_SAMPLE

        diar, asr = _get_models()

        # Build the session config. streaming_mode=True is critical.
        cfg = OmegaConf.structured(MultitalkerTranscriptionConfig())
        cfg.streaming_mode = True
        cfg.real_time_mode = True
        cfg.audio_file = "/tmp/multitalker_dummy.wav"  # unused (we feed chunks)
        cfg.output_path = "/tmp/multitalker_out.json"  # unused
        # Bundled cfg defaults masked_asr=True — that path multiplies
        # audio by the diar mask before encoding and never calls
        # set_speaker_targets. For a speaker-kernel checkpoint
        # (this one ships spk_kernel_layers + add_bg_spk_kernel) that
        # leaves self.spk_targets=None inside the encoder, so
        # solve_length_mismatch fires "Mask is None" per chunk and
        # everything collapses to single-speaker output. Flip the
        # branch.
        cfg.masked_asr = False

        diar_session = MultitalkerTranscriptionConfig.init_diar_model(cfg, diar)

        # The streaming buffer expects to be appended to with audio. We
        # feed it our session's incoming PCM by calling append_audio
        # (numpy, no file) — but per the buffer code that adds a NEW batch
        # entry for each call unless stream_id is reused. We track that
        # via a session-local accumulator: when enough audio is buffered
        # we call append_audio with a fresh array, then iterate available
        # chunks from buffer_idx forward.
        streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=asr,
            online_normalization=cfg.online_normalization,
            pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
        )

        multispk = SpeakerTaggedASR(cfg, asr, diar_session)

        # Per-speaker text we've already emitted. Indexed by speaker id (0..3).
        # Multitalker stops at max_num_of_spks (default 4).
        n_spk = multispk.instance_manager.max_num_of_spks
        emitted_text: list[str] = ["" for _ in range(n_spk)]

        # The multitalker model needs FULL chunks (112 mel features each =
        # 1.12 s of audio at the ASR's 10 ms hop). The buffer's iterator
        # will happily yield partial chunks at the tail of the buffer; if
        # we pass a partial chunk to perform_parallel_streaming_stt_spk
        # the diarizer mask (shape [n_spk, 112]) won't align with the
        # audio chunk (shape [n_spk, 128, <112]) and multitalker_asr_mixins
        # falls back to single-speaker mode — that's why every word came
        # out as S1 before this gate.
        chunk_size_feats = (
            asr.encoder.streaming_cfg.chunk_size[1]
            if isinstance(asr.encoder.streaming_cfg.chunk_size, list)
            else asr.encoder.streaming_cfg.chunk_size
        )
        stream_id = -1  # first call creates stream 0
        step_num = 0
        audio_seconds = 0.0
        # 250 ms append granularity — fine-grained enough that the buffer
        # accumulates ~112 features within 1-2 appends.
        MIN_APPEND_BYTES = BYTES_PER_SEC // 4
        sample_accum = bytearray()

        # Speaker-label smoother — see SpeakerLabelSmoother docstring.
        # Production-grade post-process that suppresses single-chunk
        # diarizer flicker (the S3/S4 ghost-speakers that fired in 2-spk
        # live testing). Adds ~0.9 s of buffering latency in exchange for
        # ~order-of-magnitude reduction in phantom-speaker emissions.
        smoother = SpeakerLabelSmoother()

        # Time-based flush for held partial-word tails. The word-boundary
        # commit logic holds the trailing word until whitespace appears
        # after it, which fragments emissions when a sentence ends without
        # a trailing space ("speakers right now" → "now" gets held
        # forever, eventually flushed as a standalone fragment when
        # something else triggers a poll). Track when each speaker's text
        # last grew; force-flush a held tail after STALE_HOLD_S of audio
        # has elapsed without growth.
        STALE_HOLD_S = 1.5
        last_hyp_change_at: list[float] = [-1.0] * n_spk
        prev_hyp_text: list[str] = [""] * n_spk

        # Word-boundary set — whitespace closes a word, so do sentence-end
        # and mid-sentence punctuation. The decoder never inserts whitespace
        # inside a word, so this is a safe commit signal.
        _WORD_BOUNDARY_CHARS = set(" \t\n.!?,;:")

        def _last_boundary_index(s: str) -> int:
            """Index of the last char in `s` that closes a word, or -1 if
            none. Lets us split a streaming hypothesis into 'committed
            words' (everything through the last boundary) and 'pending tail'
            (the unfinished last word, if any)."""
            for i in range(len(s) - 1, -1, -1):
                if s[i] in _WORD_BOUNDARY_CHARS:
                    return i
            return -1

        def _emit_speaker_delta(
            spk_idx: int, full_text: str, force_flush: bool = False,
        ):
            """Emit only the substring up to the last word boundary.

            Before this commit, we emitted whatever substring had grown
            since the last poll. That fragmented words across speaker
            boundaries — when speaker A's instance was mid-word ('mo-') at
            the moment a chunk closed and speaker B's instance had a new
            word emerging, A's hypothesis flushed 'mo' as a standalone
            word and the 've' tail showed up in B's stream at the next
            poll. The proper fix: never emit a partial word. Hold the tail
            until the next poll either completes it (whitespace appears
            after) or end-of-stream forces a flush.

            Cost: every word lags one poll. Worth it for clean
            attribution in overlap scenarios.
            """
            prev = emitted_text[spk_idx]
            if not full_text or full_text == prev:
                return None
            # The hypothesis usually grows monotonically (cache-aware
            # streaming = greedy decoder). On the rare model rewrite,
            # fall back to treating the entire hypothesis as fresh.
            if full_text.startswith(prev):
                new_text = full_text[len(prev):]
            else:
                new_text = full_text
                prev = ""
            if not new_text:
                return None

            if force_flush:
                # EOS flush — emit everything we've been holding, including
                # the trailing partial word.
                committed = new_text
            else:
                boundary = _last_boundary_index(new_text)
                if boundary < 0:
                    # No complete word in the new text yet; hold it.
                    return None
                committed = new_text[:boundary + 1]
            # prev + committed == full_text on the flush path
            # (committed = new_text = full_text[len(prev):]) and == the
            # prefix-through-boundary on the gated path; either way it's
            # what we've now emitted.
            next_emitted = prev + committed

            committed = committed.strip()
            if not committed:
                # Boundary character only (e.g. lone space). Don't emit
                # but DO advance `emitted_text` so we don't reprocess it.
                emitted_text[spk_idx] = next_emitted
                return None
            tokens = committed.split()
            if not tokens:
                emitted_text[spk_idx] = next_emitted
                return None

            t_end = audio_seconds
            # The chunk window is ~1.12 s of audio at the model's
            # streaming chunk size; the emitted words should span at most
            # that. Heuristic timing — proper per-word timing would need
            # to map hyp.timestep to character offsets via the tokenizer,
            # which is a bigger refactor.
            span = min(0.56, max(0.05, t_end))
            t_start = max(0.0, t_end - span)
            per = span / max(1, len(tokens))
            speaker_label = f"S{spk_idx + 1}"
            words = [
                StreamWord(
                    content=tok,
                    start_time=t_start + i * per,
                    end_time=t_start + (i + 1) * per,
                    confidence=None,
                    speaker=speaker_label,
                    is_punctuation=tok in {".", ",", "?", "!"},
                )
                for i, tok in enumerate(tokens)
            ]
            emitted_text[spk_idx] = next_emitted
            return StreamFinal(
                transcript=committed,
                words=words,
                start_time=words[0].start_time,
                end_time=words[-1].end_time,
            )

        async for frame in pcm_frames:
            if not frame:
                continue
            sample_accum.extend(frame)
            if len(sample_accum) < MIN_APPEND_BYTES:
                continue
            # bytearray supports the buffer protocol; bytes() would copy.
            samples = (
                np.frombuffer(sample_accum, dtype=np.int16).astype(np.float32)
                / 32768.0
            )
            audio_seconds += len(samples) / SAMPLE_RATE
            sample_accum.clear()
            try:
                streaming_buffer.append_audio(samples, stream_id=stream_id)
            except Exception as e:
                print(f"multitalker append_audio: {e}", file=sys.stderr, flush=True)
                continue
            stream_id = 0  # subsequent appends extend stream 0

            # Pull FULL chunks only. The buffer's iterator advances
            # buf.buffer_idx by shift_size and yields whatever audio is
            # in [buffer_idx, buffer_idx + chunk_size]; near the buffer
            # tail that's a partial slice. Multitalker single-speaker
            # fallback fires on partial chunks, so gate iteration on
            # having a full chunk's worth of NEW features.
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=True):
                    while (
                        streaming_buffer.buffer is not None
                        and streaming_buffer.buffer_idx + chunk_size_feats
                        <= streaming_buffer.buffer.size(-1)
                    ):
                        try:
                            chunk_audio, chunk_lengths = next(iter(streaming_buffer))
                        except StopIteration:
                            break
                        drop_extra = (
                            0
                            if step_num == 0 and not cfg.pad_and_drop_preencoded
                            else asr.encoder.streaming_cfg.drop_extra_pre_encoded
                        )
                        try:
                            multispk.perform_parallel_streaming_stt_spk(
                                step_num=step_num,
                                chunk_audio=chunk_audio,
                                chunk_lengths=chunk_lengths,
                                is_buffer_empty=streaming_buffer.is_buffer_empty(),
                                drop_extra_pre_encoded=drop_extra,
                            )
                        except Exception as e:
                            print(
                                f"multitalker step {step_num} err: {e}",
                                file=sys.stderr, flush=True,
                            )
                            break
                        step_num += 1

            # After all available chunks processed, peek per-speaker
            # hypothesis text and feed any new words into the smoother.
            # The ASRState lives at instance_manager.batch_asr_states[0]
            # (we're batch_size=1).
            batch_states = getattr(multispk.instance_manager, "batch_asr_states", None) or []
            prev_hyps = batch_states[0].previous_hypothesis if batch_states else None
            if prev_hyps:
                for spk_idx, hyp in enumerate(prev_hyps):
                    if hyp is None:
                        continue
                    text = getattr(hyp, "text", None) or ""

                    # Detect stale held tails — when the hypothesis hasn't
                    # changed for STALE_HOLD_S of audio but we have buffered
                    # text waiting for a boundary, force-flush it so the
                    # last word doesn't pile up indefinitely.
                    if text != prev_hyp_text[spk_idx]:
                        prev_hyp_text[spk_idx] = text
                        last_hyp_change_at[spk_idx] = audio_seconds
                        final = _emit_speaker_delta(spk_idx, text)
                    elif (
                        last_hyp_change_at[spk_idx] >= 0
                        and audio_seconds - last_hyp_change_at[spk_idx]
                        >= STALE_HOLD_S
                        and emitted_text[spk_idx] != text
                    ):
                        # Same hypothesis, but emitted_text != text means
                        # we're holding a partial-word tail. Flush it.
                        final = _emit_speaker_delta(spk_idx, text, force_flush=True)
                        # Reset so we don't keep flushing on every poll.
                        last_hyp_change_at[spk_idx] = -1.0
                    else:
                        final = None

                    if final is not None:
                        for word in final.words:
                            smoother.push(word, word.start_time)

            # Drain smoothed words whose hold time has expired, group
            # consecutive same-speaker words into one StreamFinal per
            # run, and yield. The grouping keeps the wire format clean
            # (one final = one speaker's utterance) instead of
            # one-final-per-word.
            ready_words = smoother.drain_ready(audio_seconds)
            for spk, run in _group_consecutive_speakers(ready_words):
                transcript = " ".join(w.content for w in run).strip()
                if not transcript:
                    continue
                yield StreamFinal(
                    transcript=transcript,
                    words=run,
                    start_time=run[0].start_time,
                    end_time=run[-1].end_time,
                )

        # End-of-stream flush — generate the final seglst and emit any
        # remaining per-speaker delta. Don't error if instance_manager has
        # no entries (very short session).
        try:
            multispk.generate_seglst_dicts_from_parallel_streaming(
                samples=[{"audio_filepath": "/tmp/eos.wav"}]
            )
        except Exception:
            pass
        batch_states = getattr(multispk.instance_manager, "batch_asr_states", None) or []
        prev_hyps = batch_states[0].previous_hypothesis if batch_states else None
        if prev_hyps:
            for spk_idx, hyp in enumerate(prev_hyps):
                if hyp is None:
                    continue
                text = getattr(hyp, "text", None) or ""
                # force_flush=True so any held partial-word tail emits now.
                final = _emit_speaker_delta(spk_idx, text, force_flush=True)
                if final is not None:
                    for word in final.words:
                        smoother.push(word, word.start_time)

        # EOS: drain everything left in the smoother. Words that were
        # still inside the hold window get emitted without further
        # smoothing — we have no future context past EOS, so we accept
        # whatever label they came in with.
        tail_words = smoother.drain_ready(audio_seconds + 999.0)
        tail_words.extend(smoother.flush())
        for spk, run in _group_consecutive_speakers(tail_words):
            transcript = " ".join(w.content for w in run).strip()
            if not transcript:
                continue
            yield StreamFinal(
                transcript=transcript,
                words=run,
                start_time=run[0].start_time,
                end_time=run[-1].end_time,
            )
