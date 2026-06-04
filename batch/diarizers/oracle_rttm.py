"""
Oracle diarizer: reads ground-truth `reference.rttm` from the case directory.

Pretends to be a diarizer but returns the human-annotated speaker timeline
verbatim. Used to test integration patterns (`overlay` vs `segment_first`
vs `force_align`) with *perfect* speaker boundaries — isolates the
integration tax from the diarizer's own quality.

If `segment_first` integration with this diarizer doesn't push AMI DER
near 0%, there's a bug in the integration code. If it does, the remaining
gap when using `pyannote` or `sortformer` is purely diarizer error, not
integration error.

Requires `reference.rttm` to exist in the case directory; raises
FileNotFoundError otherwise.
"""
from __future__ import annotations

from pathlib import Path

from diarizers._base import DiarSegment
from metrics.diarization import parse_rttm


class OracleRttmDiarizer:
    name = "oracle_rttm"
    requires_token = False

    async def diarize(self, audio: Path) -> list[DiarSegment]:
        # Audio is at cases/<name>/audio.wav (or similar); the RTTM sits next to it.
        case_dir = audio.parent
        rttm_path = case_dir / "reference.rttm"
        if not rttm_path.exists():
            raise FileNotFoundError(
                f"oracle_rttm diarizer needs {rttm_path} (ground-truth speaker timeline). "
                f"Not all cases ship one; check case.yaml for ground_truth.reference_rttm."
            )
        # parse_rttm already returns (speaker, start, end) tuples in the shape we need.
        return parse_rttm(rttm_path)
