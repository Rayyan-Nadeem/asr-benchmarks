"""Punctuator registry — picks a punctuator by name (env var PUNCTUATOR).

Mirrors engine_registry.py + diarizer_registry.py. Default is
'passthrough' so the system defaults to the engine's raw output —
turn the BERT post-process on per session by switching to 'distilbert'.
"""
from __future__ import annotations

import os

from server.punctuators._base import Punctuator
from server.punctuators.passthrough import PassthroughPunctuator


def _load_distilbert():
    from engines.punctuation_distilbert import DistilbertPunctuator
    return DistilbertPunctuator()


_REGISTRY: dict[str, type[Punctuator]] = {
    "passthrough": PassthroughPunctuator,
    "none": PassthroughPunctuator,
}

_LAZY: dict[str, callable] = {
    "distilbert": _load_distilbert,
}


def load_punctuator() -> Punctuator:
    name = os.environ.get("PUNCTUATOR", "passthrough")
    if name in _LAZY:
        return _LAZY[name]()
    cls = _REGISTRY.get(name)
    if cls is None:
        available = sorted(set(_REGISTRY) | set(_LAZY))
        raise ValueError(f"unknown PUNCTUATOR={name!r}; available: {available}")
    return cls()
