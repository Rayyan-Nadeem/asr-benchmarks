# Methodology

How this harness measures things, and what counts as "good" or "broken". All
choices are defensible against a senior engineer asking *"how did you measure
that?"*. Thresholds are defined here, before any run, so results are not
rationalized post-hoc.

## Audio handling

- All engines see the **same audio bytes**: mono PCM S16LE @ 16 kHz, derived
  from the case's `audio.<ext>` via ffmpeg if not already in that format.
- The 16 kHz constraint is universal (all production ASR engines downsample to
  16 kHz internally); standardizing at the harness boundary means we measure
  the engine, not the resampler.
- Audio is streamed at **1× realtime** for streaming/latency benchmarks (the
  production scenario) and as fast as the WS allows for offline accuracy-only
  runs (`--fast` flag). The two are different tests; never mix.

## Engine config

The engine adapter ships with the **production config** for that engine baked
in as the default. For Speechmatics on-prem this means DepoDash's exact cloud
`transcription_config` (per Ed's C++ struct, 2026-04-22): `permitted_marks`,
`max_delay_mode: flexible`, `speaker_diarization_config.max_speakers: 10`,
`output_locale: en-US`, `enable_entities: false`. CLI overrides exist but the
default is "what production sends."

Rationale: a benchmark that uses non-production config is benchmarking a
fiction. Default-of-default-of-default lazy configs systematically overstate
or understate engines depending on which knobs they expose.

## Accuracy

### Word Error Rate (WER) and Character Error Rate (CER)

`jiwer` 4.0+ with a Whisper-style normalizer stack:

```
ToLowerCase → RemovePunctuation → ExpandCommonEnglishContractions →
RemoveMultipleSpaces → RemoveEmptyStrings
```

Both `wer()` (the score) and `process_words()` (S/D/I breakdown) are
recorded. S/D/I breakdown is the diagnostic — it tells you *why* the WER is
what it is (substitution-heavy = acoustic confusion; insertion-heavy =
hallucination; deletion-heavy = silence cuts).

### Entity preservation

Per case, a list of "key terms" lives in `case.yaml.key_terms` — proper
nouns, dates, case numbers, money amounts. The metric is:
`(distinct key terms preserved in hypothesis) / (distinct key terms in reference)`.
Reported alongside WER because legal/courtroom users care more about
"did it get the names right" than the global WER number.

## Diarization

### Diarization Error Rate (DER)

`simpleder` (no torch dep). Inputs are RTTM files: hypothesis (engine output
converted via `metrics/diarization.py:words_to_rttm`) vs. reference
(`case.reference.rttm`). DER decomposes into Speaker Confusion + Missed Speech
+ False Alarm.

For engines that don't natively diarize, the harness can run a separate
diarizer (e.g., `pyannote.audio`) on the same audio and report DER for that
combination — surfaced in the scoreboard as `<engine> + pyannote-diar`. This
mirrors how DepoDash's existing stack works: one ASR, separate diarizer.

### Composite cpWER

`meeteval` for the composite "who said what" metric. This is the metric that
matters most for multi-speaker meetings — high WER + low DER doesn't help if
the speaker labels are attached to the wrong words. Reported only for cases
with both `reference.txt` and `reference.rttm`.

## Latency

Every WS frame in/out is timestamped at the harness boundary (`time.monotonic()`).

| Metric | Definition |
| --- | --- |
| **TTFT** (time to first token) | First `AddTranscript` arrival − first audio chunk send |
| **Per-final lag** (p50, p95, p99) | For each final: arrival timestamp − `metadata.end_time` (audio time of the last word) |
| **RTF** (realtime factor) | Total wall-clock time / audio duration. `<1.0` = faster than realtime. |
| **Partial-final stability** | When `enable_partials: true`, edit distance between the last partial and the corresponding final. Lower = more stable. |

Per-final lag is the metric that matches "perceived live-ness" — TTFT is good
for first-impression but per-final p95 is what end users actually feel during
a long meeting.

## Resources

Background sampler at 1 Hz writes to `results/runs/<run-id>__resources.jsonl`.

- **GPU**: `nvidia-smi --query-gpu=memory.used,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader` against the host (works inside WSL2 via `/usr/lib/wsl/lib/nvidia-smi`).
- **Container CPU/memory**: `docker stats --no-stream --format json <container>` for each Speechmatics container (`sm-triton`, `sm-rt-transcriber`).
- **Host process**: `psutil.Process(self_pid)` for the harness itself, as a sanity check that the client isn't the bottleneck.

Per-container GPU memory attribution is **not measurable** on consumer GPUs
(no MIG on the RTX 5080). GPU numbers are aggregate across all containers
sharing the GPU; per-engine cost is inferred from the delta between idle
baseline and load.

Reported per run: peak, p50, p95 of each metric over the run window
(excluding the first 30 s warmup).

## Concurrency

Ramp test: 1 → 2 → 4 → (8 → 16 if stable) concurrent streams. Each level
holds for 180 s with the first 30 s discarded as warmup. Streams run the same
audio at 1× realtime via `asyncio.gather`.

### Quantitative thresholds (defined upfront)

Compared to the single-stream baseline at the same case:

| Metric | Threshold | Verdict |
| --- | --- | --- |
| WER drift (absolute Δ) | ≤ 1 pp | OK |
| WER drift | 1–2 pp | degraded |
| WER drift | > 2 pp | broken |
| TTFT p95 multiplier | ≤ 1.5× | OK |
| TTFT p95 multiplier | 1.5–2× | degraded |
| TTFT p95 multiplier | > 2× | broken |
| Per-final p95 multiplier | ≤ 1.5× | OK |
| Per-final p95 multiplier | 1.5–2× | degraded |
| Per-final p95 multiplier | > 2× | broken |
| Connection reject rate | ≤ 1% | OK |
| Connection reject rate | 1–5% | degraded |
| Connection reject rate | > 5% | cap reached |

The reported `max_stable_concurrency` is the highest N where **all** metrics
stay in OK or degraded for the full hold period.

### Pitfalls and what we do about them

- **Client bottleneck**: confirmed by running the same ramp from a second
  process; if numbers don't change, the client wasn't the bottleneck.
- **Thermal throttling** on a laptop under sustained load: GPU clock
  frequency captured alongside utilization; > 10% drop is flagged in the run.
- **Cold-start bias**: discard first 30 s of every concurrency level.
- **Audio diversity**: identical audio across streams for the first pass
  (removes content as a variable); diverse audio in a follow-up sensitivity
  run if the first pass shows degradation.

## Reproducibility

Every run writes a `results/runs/<timestamp>__<engine>__<case>.json` file
containing:

- engine + case + config (full snapshot)
- raw engine dump (the WS jsonl from a Speechmatics-style engine, or whatever
  the native shape is for an offline engine)
- normalized words list
- all computed metrics
- pointer to the resource sampler JSONL for the same run

`tools/render_scoreboard.py` is deterministic over the run files — given the
same files in `results/runs/`, the same `SCOREBOARD.md` is produced.

## What's NOT measured (and why)

- **Cost per minute / per session** — varies by license tier and not directly
  comparable between cloud and on-prem. Surfaced separately in business
  conversations, not the technical scoreboard.
- **Subjective quality** ("does it sound smart") — explicitly out of scope.
  WER + DER + entity preservation cover the cases that matter for legal use.
- **Hallucination rate as a separate metric** — captured indirectly through
  the I (insertion) component of the WER S/D/I breakdown. A standalone
  hallucination metric requires constructing audio with known silent regions
  and is Phase 2 work.
- **Custom vocabulary effectiveness** — capability flag only for now;
  scoring custom-vocab efficacy needs a domain-specific term list paired with
  audio that exercises those terms. Phase 2.
