"""
pyannote.audio 3.1 diarizer.

Token-gated: requires a HuggingFace read token AND acceptance of the model
terms at https://huggingface.co/pyannote/speaker-diarization-3.1 and
https://huggingface.co/pyannote/segmentation-3.0.

Set the token via the HF_TOKEN env var (or HUGGINGFACEHUB_API_TOKEN, which
the pyannote loader also recognises).

This is offline batch diarization — the entire audio file is processed at
once. Compare to Speechmatics' streaming diarization with the unfair-
comparison caveat noted in METHODOLOGY.md.
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
        diarization = pipe(str(audio))
        segs: list[DiarSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.end > turn.start:
                segs.append((str(speaker), float(turn.start), float(turn.end)))
        return segs
