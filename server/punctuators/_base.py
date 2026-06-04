"""Punctuator interface — restores punctuation and capitalization after
the diarizer step. Selectable per session via the orchestrator's /switch
endpoint or the PUNCTUATOR env var, so users can A-B raw vs punctuated.

The interface is intentionally a single method that takes a list of
words and returns a list of words. Implementations may:

  - Modify the .content field of existing words to add trailing
    punctuation or fix capitalization
  - Insert new StreamWord entries with is_punctuation=True for standalone
    punctuation marks (preferred for protocol compliance — Speechmatics
    treats punctuation as its own result row with attaches_to='previous')

Implementations are stateless across calls — they receive each finalized
batch of words for the same speaker run as a single list.
"""
from __future__ import annotations

from typing import Protocol

from server.engines._base import StreamWord


class Punctuator(Protocol):
    name: str

    async def punctuate(self, words: list[StreamWord]) -> list[StreamWord]:
        """Return the words list with punctuation + capitalization applied."""
        ...
