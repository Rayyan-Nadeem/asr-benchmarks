"""No-op punctuator — leaves engine output untouched. Used to A-B the
BERT post-process against raw model emission on the live demo."""
from __future__ import annotations

from server.engines._base import StreamWord
from server.punctuators._base import Punctuator


class PassthroughPunctuator(Punctuator):
    name = "passthrough"

    async def punctuate(self, words: list[StreamWord]) -> list[StreamWord]:
        return words
