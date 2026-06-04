"""
pyannote.audio 3.1 diarizer.

Token-gated: requires a HuggingFace read token AND acceptance of the model
terms at https://huggingface.co/pyannote/speaker-diarization-3.1 and
https://huggingface.co/pyannote/segmentation-3.0.

Set the token via the HF_TOKEN env var (or HUGGINGFACEHUB_API_TOKEN, which
the pyannote loader also recognises).

This is offline batch diarization — the entire audio file is processed at
once. Speechmatics' diarization runs in streaming mode, so direct
comparison favors pyannote here.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from diarizers._base import DiarSegment


_PIPELINE = None  # cached across calls; loading the model is expensive


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    if not token:
        raise RuntimeError(
            "pyannote.audio requires a HuggingFace token. "
            "Set HF_TOKEN env var. Get one at https://huggingface.co/settings/tokens "
            "and accept the model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    import torch
    # PyTorch 2.6+ flipped torch.load default to weights_only=True. pyannote.audio
    # 3.4 ships pickled checkpoints that need weights_only=False. Patch the
    # default — safe because we're loading models we just downloaded from HF.
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        # Hard-override: pyannote/lightning explicitly pass weights_only=True
        # in some code paths. Their checkpoints aren't weights-only-safe
        # (contain pickled Python objects), so we have to force False.
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load

    from pyannote.audio import Pipeline  # imported lazily; heavy

    # huggingface_hub renamed use_auth_token → token in 0.23+. Pass via env so
    # the underlying Pipeline.from_pretrained → hf_hub_download chain picks it up.
    import os as _os
    _os.environ.setdefault("HF_TOKEN", token)
    _os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)
    try:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    except TypeError:
        # very old pyannote / hub combo
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
    _PIPELINE = pipe
    return pipe


class PyannoteDiarizer:
    name = "pyannote"
    requires_token = True

    async def diarize(self, audio: Path) -> list[DiarSegment]:
        # Pipeline call is sync + blocking; run in a thread so we don't wedge asyncio.
        return await asyncio.to_thread(self._diarize_sync, audio)

    def _diarize_sync(self, audio: Path) -> list[DiarSegment]:
        pipe = _get_pipeline()
        out = pipe(str(audio))
        # pyannote 4.x returns DiarizeOutput; 3.x returns Annotation directly.
        diarization = getattr(out, "speaker_diarization", out)
        segs: list[DiarSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.end > turn.start:
                segs.append((str(speaker), float(turn.start), float(turn.end)))
        return segs

    # ------------------------------------------------------------------
    # Streaming-shaped wrapper for the M2 WebSocket server.
    #
    # Overlay semantics: caller supplies the audio bytes that produced the
    # words; we run pyannote on that audio and stamp each word with the
    # speaker whose segment contains the word's midpoint. Leaves the 15-20 pp
    # integration tax vs the pyannote-alone ceiling (see FINDINGS.md);
    # segment_first integration in server/session.py is the Phase 4 fix.

    async def label(self, words, audio_chunk=None):
        if not audio_chunk or not words:
            return words
        return await asyncio.to_thread(self._label_sync, words, audio_chunk)

    def _label_sync(self, words, pcm):
        from io import BytesIO
        import wave

        SAMPLE_RATE = 16000
        BYTES_PER_SEC = SAMPLE_RATE * 2
        # Sliding-window cap: pyannote runs on the most recent WINDOW_S
        # seconds only. Caller passes the full growing audio buffer; without
        # this cap pyannote crashes (OOM or pipeline timeout) past ~5 min of
        # live mic. 30 s gives 2-3 speaker-turn windows of context.
        WINDOW_S = 30.0
        WINDOW_BYTES = int(WINDOW_S * BYTES_PER_SEC)
        if len(pcm) > WINDOW_BYTES:
            window_start_byte = len(pcm) - WINDOW_BYTES
            pcm = pcm[window_start_byte:]
        else:
            window_start_byte = 0
        window_start_s = window_start_byte / BYTES_PER_SEC

        wav_bytes = BytesIO()
        with wave.open(wav_bytes, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)
        tmp = Path(f"/tmp/pyannote_stream_{id(pcm)}.wav")
        tmp.write_bytes(wav_bytes.getvalue())
        try:
            segments = self._diarize_sync(tmp)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

        # Shift segment times back into global audio coordinates.
        segments = [(spk, s + window_start_s, e + window_start_s) for (spk, s, e) in segments]

        for w in words:
            midpoint = 0.5 * (w.start_time + w.end_time)
            # Only assign speakers to words that fall inside the analyzed window.
            if midpoint < window_start_s:
                continue
            for speaker, start, end in segments:
                if start <= midpoint <= end:
                    w.speaker = speaker
                    break
        return words

    async def turns_to_now(self, audio_chunk):
        return []
