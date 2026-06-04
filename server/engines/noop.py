"""Echo-back stub engine. Emits a partial every 1 s of audio, a final on stream end.

Useful for validating the protocol shim end-to-end without standing up a real
engine. The transcripts are placeholders — accuracy is meaningless here.
"""
from __future__ import annotations

from typing import AsyncIterable

from server.engines._base import StreamFinal, StreamingEngine, StreamPartial, StreamWord


SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
SECONDS_PER_PARTIAL = 1.0


class NoopEngine(StreamingEngine):
    name = "noop"

    async def stream(self, pcm_frames, transcription_config):
        bytes_received = 0
        last_partial_at_seconds = 0.0
        partial_count = 0

        async for frame in pcm_frames:
            bytes_received += len(frame)
            audio_seconds = bytes_received / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            if audio_seconds - last_partial_at_seconds >= SECONDS_PER_PARTIAL:
                partial_count += 1
                last_partial_at_seconds = audio_seconds
                yield StreamPartial(
                    transcript=f"[noop partial {partial_count}]",
                    start_time=0.0,
                    end_time=audio_seconds,
                )

        total_seconds = bytes_received / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        yield StreamFinal(
            transcript=f"[noop final: {total_seconds:.2f}s of audio received]",
            words=[
                StreamWord(
                    content=f"noop",
                    start_time=0.0,
                    end_time=total_seconds,
                    confidence=1.0,
                    speaker=None,
                    is_punctuation=False,
                ),
            ],
            start_time=0.0,
            end_time=total_seconds,
        )
