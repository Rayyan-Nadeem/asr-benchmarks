"""
Latency metrics computed from RunMetadata.frames timestamps.

- TTFT: time to first AddTranscript arrival, measured from first audio send
- per-final lag: arrival wall_time - audio_end_time, percentiles over the run
- RTF: total wall time / audio duration. <1 = faster than realtime.
- partial-final stability: edit distance between last partial and matching final
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from engines._base import RunMetadata


@dataclass
class LatencyReport:
    ttft_ms: float | None              # time to first final
    ttfp_ms: float | None              # time to first partial (None if partials disabled)
    final_lag_p50_ms: float | None
    final_lag_p95_ms: float | None
    final_lag_p99_ms: float | None
    rtf: float                          # realtime factor
    finals_count: int
    partials_count: int


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def score_latency(meta: RunMetadata, fast_mode: bool = False) -> LatencyReport:
    finals = [f for f in meta.frames if f.kind == "AddTranscript"]
    partials = [f for f in meta.frames if f.kind == "AddPartialTranscript"]

    # per-final lag = wall arrival relative to start of streaming - the audio time it covers
    # i.e. (frame.wall_time - meta.first_audio_send) - frame.audio_end_time
    # Only meaningful in 1×-realtime mode — in --fast mode all audio is sent before
    # finals arrive, so the formula gives spurious negative numbers.
    final_lags_ms: list[float] = []
    if not fast_mode:
        final_lags_ms = [
            ((f.wall_time - meta.first_audio_send) - f.audio_end_time) * 1000.0
            for f in finals
            if f.audio_end_time > 0
        ]

    ttft_ms: float | None = None
    if meta.first_final_recv is not None:
        ttft_ms = (meta.first_final_recv - meta.first_audio_send) * 1000.0

    ttfp_ms: float | None = None
    if meta.first_partial_recv is not None:
        ttfp_ms = (meta.first_partial_recv - meta.first_audio_send) * 1000.0

    rtf = (meta.wall_clock_end - meta.wall_clock_start) / max(meta.audio_duration_s, 1e-9)

    return LatencyReport(
        ttft_ms=ttft_ms,
        ttfp_ms=ttfp_ms,
        final_lag_p50_ms=_percentile(final_lags_ms, 50),
        final_lag_p95_ms=_percentile(final_lags_ms, 95),
        final_lag_p99_ms=_percentile(final_lags_ms, 99),
        rtf=rtf,
        finals_count=len(finals),
        partials_count=len(partials),
    )
