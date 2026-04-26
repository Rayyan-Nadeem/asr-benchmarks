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


def score_confidence(words: list) -> dict:
    """
    Per-word confidence aggregates (mean / p10 / low-confidence-rate).
    Useful as a calibration signal — if an engine is consistently 1.0 on words
    it gets wrong, it's poorly calibrated even if WER looks fine.
    """
    confs = [w.confidence for w in words if w.confidence is not None and not w.is_punctuation]
    if not confs:
        return {"words_scored": 0, "mean": None, "p10": None, "low_conf_rate": None}
    confs_sorted = sorted(confs)
    p10 = confs_sorted[len(confs_sorted) // 10] if len(confs_sorted) >= 10 else confs_sorted[0]
    low_conf = sum(1 for c in confs if c < 0.7) / len(confs)
    return {
        "words_scored": len(confs),
        "mean": sum(confs) / len(confs),
        "p10": p10,
        "low_conf_rate": low_conf,  # fraction of words below 0.7
    }


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
