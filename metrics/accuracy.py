"""
Accuracy metrics: WER, CER, S/D/I breakdown, entity preservation.

Uses jiwer with a Whisper-style normalizer stack so engines aren't penalized
for casing or punctuation differences when those don't carry meaning.
"""
from __future__ import annotations

from dataclasses import dataclass

import jiwer


# Whisper-style normalization. Order matters.
NORMALIZER = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


@dataclass
class WERReport:
    wer: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_word_count: int
    hyp_word_count: int


def score_wer(reference: str, hypothesis: str) -> WERReport:
    """Compute WER + S/D/I + CER between a reference and hypothesis transcript."""
    out = jiwer.process_words(
        reference,
        hypothesis,
        reference_transform=NORMALIZER,
        hypothesis_transform=NORMALIZER,
    )
    cer = jiwer.cer(reference, hypothesis)
    return WERReport(
        wer=out.wer,
        cer=cer,
        substitutions=out.substitutions,
        deletions=out.deletions,
        insertions=out.insertions,
        hits=out.hits,
        ref_word_count=sum(len(s) for s in out.references),
        hyp_word_count=sum(len(s) for s in out.hypotheses),
    )


@dataclass
class EntityReport:
    total: int
    preserved: int
    missing: list[str]

    @property
    def preservation_rate(self) -> float:
        return self.preserved / self.total if self.total else 1.0


def score_entity_preservation(hypothesis: str, key_terms: list[str]) -> EntityReport:
    """
    For each key term (case-insensitive substring), check if it appears in the
    hypothesis. Useful for legal/courtroom: "did it get the proper nouns".
    """
    if not key_terms:
        return EntityReport(0, 0, [])
    hyp_lower = hypothesis.lower()
    preserved = 0
    missing: list[str] = []
    for term in key_terms:
        if term.lower() in hyp_lower:
            preserved += 1
        else:
            missing.append(term)
    return EntityReport(total=len(key_terms), preserved=preserved, missing=missing)
