"""
Diarization metrics: DER (and JER if reference supports it).

Reference: RTTM file (one line per speaker turn).
Hypothesis: derived from engine words via `words_to_rttm`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import simpleder

from engines._base import Word


@dataclass
class DERReport:
    der: float                     # diarization error rate, [0, 1]
    speaker_count_ref: int
    speaker_count_hyp: int


def words_to_rttm_segments(words: list[Word]) -> list[tuple[float, float, str]]:
    """
    Collapse contiguous same-speaker words into (start, end, speaker) segments.
    Punctuation is attached to the preceding speaker without breaking a segment.
    Returns a list usable by simpleder (treats it as ground truth via tuples).
    """
    segs: list[list[float | str]] = []
    last_speaker: str | None = None

    for w in words:
        if w.speaker is None:
            continue
        if last_speaker == w.speaker and segs:
            segs[-1][1] = w.end_time   # extend end
        else:
            segs.append([w.start_time, w.end_time, w.speaker])
            last_speaker = w.speaker

    return [(float(s[0]), float(s[1]), str(s[2])) for s in segs if s[1] > s[0]]


def parse_rttm(path: Path) -> list[tuple[float, float, str]]:
    """
    Parse an RTTM file. Each "SPEAKER" line:
      SPEAKER <file_id> <chnl> <onset> <duration> <NA> <NA> <speaker> <NA> <NA>
    Returns a list of (start, end, speaker) tuples.
    """
    segments: list[tuple[float, float, str]] = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            try:
                onset = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
            except (IndexError, ValueError):
                continue
            segments.append((onset, onset + duration, speaker))
    return segments


def write_rttm(path: Path, file_id: str, segments: list[tuple[float, float, str]]) -> None:
    """Write segments as a valid RTTM file."""
    with open(path, "w") as f:
        for start, end, speaker in segments:
            duration = end - start
            f.write(
                f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
            )


def score_der(
    reference_rttm: Path,
    hypothesis_segments: list[tuple[float, float, str]],
) -> DERReport:
    """Compute DER using simpleder."""
    ref = parse_rttm(reference_rttm)
    hyp = hypothesis_segments
    der = simpleder.DER(ref, hyp)
    return DERReport(
        der=float(der),
        speaker_count_ref=len({s for _, _, s in ref}),
        speaker_count_hyp=len({s for _, _, s in hyp}),
    )
