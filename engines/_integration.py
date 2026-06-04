"""
Integration patterns for combining an Engine (ASR) with a Diarizer.

Three patterns:

  * **overlay** (default, legacy)
      Engine runs on full audio → diarizer runs on full audio → for each word,
      look up which diarizer segment covers the word's midpoint and assign that
      speaker. Cheap. Loses information at speaker boundaries — short turns get
      misattributed, overlapping speech goes to whichever speaker dominates the
      centroid. The 15–20 pp DER tax we measured on AMI ES2004a.

  * **segment_first** (the proposed Phase 3 default)
      Diarizer runs on full audio → for each speaker segment, slice the audio
      and run the engine on that slice → assign the segment's speaker to all
      words in the slice. Eliminates boundary ambiguity by construction at the
      cost of N engine calls instead of one. For AMI 5-min audio with ~50
      segments and an engine RTF of ~0.3, the overhead is ~3× wall-clock — a
      reasonable trade for cutting DER by 5–15 pp.

  * **force_align** (future, WhisperX-style)
      Engine runs on full audio → diarizer runs on full audio → wav2vec2
      forced alignment produces *true* word-level timestamps → boundary-aware
      assignment. Implementation deferred until we measure whether
      segment_first alone closes the gap.
"""
from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Any

from engines._audio import audio_duration_s, load_pcm_as_float32
from engines._base import (
    Capabilities,
    FrameTimestamp,
    Result,
    RunMetadata,
    Word,
)


def overlay_speakers_at_midpoint(words: list[Word], segments: list[tuple[str, float, float]]) -> None:
    """
    Naive overlay: assign each word the speaker whose segment covers its midpoint.
    This is the existing behavior in diarizers/_base.py::overlay_speakers; we
    re-export here for symmetry with the alternative integration modes.
    """
    for w in words:
        mid = (w.start_time + w.end_time) / 2.0
        match: str | None = None
        for spk, s, e in segments:
            if s <= mid < e:
                match = spk
                break
        w.speaker = match


async def transcribe_segment_first(
    engine,
    audio: Path,
    segments: list[tuple[str, float, float]],
    config: dict[str, Any] | None = None,
) -> Result:
    """
    Segment-first integration. Returns a single Result aggregating per-segment ASR.

    For each (speaker, start, end) in segments:
      1. Write a temp WAV of just that audio slice.
      2. Run engine.transcribe() on the slice.
      3. Rebase word timestamps to global time (add `start` offset).
      4. Set every word's `.speaker` to the segment's speaker.
    Merge results into one Result with concatenated transcript, all words, and
    raw_dump containing every per-segment dump.
    """
    if not segments:
        # No diarizer output — fall back to a single full-audio call.
        return await engine.transcribe(audio, config=config)

    # Load full audio once; we slice numpy → soundfile.write per segment.
    samples, full_duration_s = load_pcm_as_float32(audio)
    sample_rate = 16000  # harness boundary

    all_words: list[Word] = []
    transcript_parts: list[str] = []
    raw_dump: list[dict] = []
    seg_errors: list[str] = []

    meta = RunMetadata(
        audio_duration_s=full_duration_s,
        wall_clock_start=time.monotonic(),
        wall_clock_end=0.0,
        first_audio_send=time.monotonic(),
    )

    with tempfile.TemporaryDirectory(prefix="asr_segfirst_") as tmpdir:
        tmp_root = Path(tmpdir)
        for i, (speaker, start, end) in enumerate(segments):
            if end <= start:
                continue
            # Slice the numpy array — clamp to bounds defensively.
            s_idx = max(0, int(start * sample_rate))
            e_idx = min(len(samples), int(end * sample_rate))
            if e_idx <= s_idx:
                continue
            seg_samples = samples[s_idx:e_idx]

            # Write the slice to a temp WAV so the engine.transcribe(Path)
            # contract works unchanged.
            seg_path = tmp_root / f"seg_{i:04d}_{speaker}.wav"
            _write_pcm_s16le_wav(seg_path, seg_samples, sample_rate)

            try:
                seg_result = await engine.transcribe(seg_path, config=config)
            except Exception as exc:
                seg_errors.append(f"segment {i} ({speaker} {start:.2f}-{end:.2f}): {exc!r}")
                continue

            # Rebase word timestamps; force speaker assignment.
            for w in seg_result.words:
                w.start_time += start
                w.end_time += start
                w.speaker = speaker
                all_words.append(w)

            if seg_result.transcript:
                transcript_parts.append(seg_result.transcript)

            # Tag segment frame so latency aggregation has something to read.
            for fr in seg_result.metadata.frames:
                fr.audio_end_time += start
            raw_dump.append({
                "_direction": "in",
                "message": "SegmentASRResult",
                "segment_index": i,
                "speaker": speaker,
                "start": start,
                "end": end,
                "transcript": seg_result.transcript,
                "word_count": len(seg_result.words),
                "engine_raw": seg_result.raw_dump,
            })

    meta.wall_clock_end = time.monotonic()
    meta.first_final_recv = meta.wall_clock_end  # batch-equivalent: arrives at end

    return Result(
        engine=engine.name,
        audio_path=audio,
        config={**(config or {}), "_integration_mode": "segment_first"},
        words=all_words,
        transcript=" ".join(transcript_parts).strip(),
        raw_dump=raw_dump,
        metadata=meta,
        error="; ".join(seg_errors) if seg_errors else None,
    )


def _write_pcm_s16le_wav(path: Path, samples_f32, sample_rate: int) -> None:
    """Write a float32 numpy array to mono PCM S16LE @ 16 kHz WAV."""
    import numpy as np
    import wave
    pcm16 = (samples_f32.clip(-1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
