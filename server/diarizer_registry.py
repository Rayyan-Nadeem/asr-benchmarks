"""Diarizer registry — picks a diarizer by name (env var DIARIZER)."""
from __future__ import annotations

import os

from server.diarizers._base import StreamingDiarizer
from server.diarizers.passthrough import PassthroughDiarizer


def _load_pyannote():
    from diarizers.pyannote import PyannoteDiarizer
    return PyannoteDiarizer()


def _load_sortformer():
    # PROPER streaming sortformer: persistent streaming_state across all
    # label() calls in a session, per-chunk forward_streaming_step. AOSC
    # speaker cache survives between emissions, so S1/S2/S3 stay locked to
    # the same voices for the whole session (no per-emit re-anchoring).
    # See diarizers/sortformer_streaming.py for the architecture rationale.
    from diarizers.sortformer_streaming import SortformerStreaming
    return SortformerStreaming()


def _load_sortformer_per_emit():
    # Legacy offline-per-emit pattern — kept for back-compat / A-B testing.
    # Re-initializes streaming_state every call; S1/S2 swap is possible.
    from diarizers.sortformer import SortformerStreaming
    return SortformerStreaming()


def _load_sortformer_streaming_v2():
    # Native streaming Sortformer v2.1 with AOSC. Per-chunk inference,
    # persistent (streaming_state, total_preds) per session. Constant memory.
    from diarizers.sortformer_native_streaming import SortformerNativeStreaming
    return SortformerNativeStreaming()


_REGISTRY: dict[str, type[StreamingDiarizer]] = {
    "passthrough": PassthroughDiarizer,
    "none": PassthroughDiarizer,
}


_LAZY: dict[str, callable] = {
    "pyannote": _load_pyannote,
    "sortformer-streaming": _load_sortformer,
    "sortformer": _load_sortformer,  # alias used by orchestrator presets
    "sortformer-per-emit": _load_sortformer_per_emit,  # legacy
    "sortformer-streaming-v2": _load_sortformer_streaming_v2,
    "sortformer-v2": _load_sortformer_streaming_v2,
}


def load_diarizer() -> StreamingDiarizer:
    name = os.environ.get("DIARIZER", "passthrough")
    if name in _LAZY:
        return _LAZY[name]()
    cls = _REGISTRY.get(name)
    if cls is None:
        available = sorted(set(_REGISTRY) | set(_LAZY))
        raise ValueError(f"unknown DIARIZER={name!r}; available: {available}")
    return cls()
