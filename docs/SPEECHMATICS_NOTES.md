# Speechmatics-specific notes

Engine-specific gotchas observed while building the on-prem adapter and
running the first benchmarks. Useful both for this repo and for the
companion `speechmaticsContainer` repo.

## `operating_point` must match the server

The `rt-transcriber` defaults a session's `operating_point` to `"standard"`
when the client omits it. The `triton` GPU server only loads the model for
the value of `SM_OPERATING_POINT` it was *started with* (e.g.
`enhanced` loads `ensemble_en_enhanced` only).

**Symptom:** session rejected at handshake time with
`Error / invalid_config / "Triton model: ensemble_en_standard is not available"`.

**Fix:** every client must set `operating_point` in `transcription_config`
to match the Triton container's `SM_OPERATING_POINT`. The Speechmatics
adapter in this repo defaults to `"enhanced"` to match the
`speechmaticsContainer` deployment.

## `punctuation_overrides.permitted_marks` required for diarization

If `punctuation_overrides.permitted_marks` is missing or empty, **speaker
diarization silently misbehaves** — no error, just wrong labels or merged
speakers. Speechmatics anchors diarization decisions on punctuation
boundaries, so the engine needs at least one permitted mark to emit
correctly-labeled results.

**Source:** Ed's C++ struct + a 2023 Speechmatics support interaction
(ticket era of #94866). DepoDash production sends `[",", ".", "?"]`.

**Fix:** baked into `engines/speechmatics_onprem.py:DEPODASH_TRANSCRIPTION_CONFIG`.
Don't strip it.

## `SM_MAX_CONCURRENT_CONNECTIONS=1` is the silent default

Out of the box the rt-transcriber accepts **exactly one** WebSocket
connection at a time. A second connection is rejected. Any reconnect race
(client closes + immediately reconnects before the server finalizes
session teardown) trips this.

**Fix:** set `SM_MAX_CONCURRENT_CONNECTIONS` explicitly in the .env. The
companion `speechmaticsContainer/.env.example` ships with `=2`. Production
sizing is a license conversation with Speechmatics — concurrency is
gated on the contract.

**Relevance to this repo:** the concurrency ramp test ramps to 4 streams.
On the current Alienware setup, N=3 and N=4 fail closed because the
configured cap is 2. To benchmark higher concurrency, bump the env var,
restart the speechmaticsContainer stack, then re-ramp.

## RTX 5080 (Blackwell, sm_120) + `diar_enhanced` — likely fine, was previously suspect

**History:** on 2026-04-18 we observed `kaldi::KaldiFatalError /
"Failed to record the event"` from libtorch when streaming a 30-s audio
with `diarization=speaker` on this hardware. At the time we suspected a
Blackwell sm_120 incompatibility in the bundled libtorch.

**Update 2026-04-22 / 2026-04-25:** the original failing client did NOT
send `permitted_marks`. The post-meeting live HTML demo *and* tonight's
SCOTUS benchmark (5 min, diarization=speaker, all DepoDash defaults) ran
clean — DER 2.0 %, no engine errors. Likely root cause was the missing
`permitted_marks`, not Blackwell.

**Open question:** does this hold for *long* diarized streams (30+
minutes)? The 5-min run is clean; full-length SCOTUS at 67 min is
untested. Worth a single 1× realtime full-SCOTUS run when there's
spare hardware time — it'll take ~67 min wall-clock and tell us if there
is a separate long-stream stability concern.

**The minimal CUDA event diagnostic** (`nvcc -arch=sm_120` +
`cudaEventRecord` + `cudaEventElapsedTime`) succeeded on this exact
hardware/WSL/Docker setup, so the underlying GPU pathway is healthy.
Whatever broke earlier was inside the engine, not the platform.

## Latency observations from this benchmark

Numbers as of 2026-04-25 (Speechmatics on-prem v15.0.0 on RTX 5080,
DepoDash production config, 1× realtime):

| Case | TTFT | per-final p95 | RTF | Notes |
| --- | --- | --- | --- | --- |
| LibriSpeech mini (70 s, 1 spk) | 4019 ms | 6348 ms | 1.05 | Diarization off |
| SCOTUS 5min (5 spk, real audio) | 5131 ms | 18.7 s | 1.05 | Diarization on, max_delay 5.0 flexible |

**Per-final p95 of 18.7 s on the diarized SCOTUS case** is the number to
flag for DepoDash. With `max_delay: 5.0` + `max_delay_mode: flexible`, the
server will exceed `max_delay` when it needs more context for accurate
diarization. The `flexible` mode trades tail latency for accuracy. This
is by design and matches DepoDash's production cloud config; the lower
TTFT on the LibriSpeech case (no diarization, max_delay 3.0 fixed) shows
the cost.

## Concurrency observations

| N | Outcome | TTFT p95 | RTF p95 | GPU peak |
| --- | --- | --- | --- | --- |
| 1 | clean | 5602 ms | 1.053 | 7684 MiB |
| 2 | clean | 5232 ms | 1.046 | 7684 MiB |
| 4 | 2 successes / 2 rejections | 5070 ms | 1.049 | 7782 MiB |

At N=2 vs N=1: VRAM identical, TTFT slightly *lower* (within noise), no
WER drift. So adding a second concurrent stream costs ~zero on this
hardware. N=4 hits the configured concurrency cap (2), not a hardware
limit. Bump the cap and re-ramp to find the actual hardware ceiling
before the laptop goes back.

## Cross-session speaker fingerprinting

Speechmatics does **not** support cross-session speaker identification.
Speaker IDs (`S1`, `S2`, ...) are reset every session — same person in
two consecutive recordings gets two different labels.

This is a confirmed gap (Keanan, 2026-04-22). Phase 2 in-house engines
based on NeMo / pyannote.audio embeddings *do* support it. When the
Phase 2 engines land, this becomes a capability column in the scoreboard
and a meaningful feature differentiator.
