"""Single-speaker no-op diarizer. Stamps every word with speaker 'S1'."""
from __future__ import annotations

from server.diarizers._base import SpeakerTurn, StreamingDiarizer
from server.engines._base import StreamWord


class PassthroughDiarizer(StreamingDiarizer):
    name = "passthrough"

    async def label(self, words, audio_chunk=None):
        # Don't clobber engine-set speaker labels. Multitalker Parakeet
        # emits per-speaker (S1/S2/S3/S4) directly because its speaker
        # kernels run one ASR instance per speaker — overwriting those
        # back to a flat "S1" would erase the entire point of selecting
        # passthrough as the diarizer for that engine.
        for w in words:
            if not w.speaker:
                w.speaker = "S1"
        return words

    async def turns_to_now(self, audio_chunk):
        return []
