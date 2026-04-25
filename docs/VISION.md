# Vision

## What this repo is

A vendor-agnostic real-time ASR benchmark harness. It takes any audio case
(audio + ground truth) and runs it through any engine adapter, producing a
comparable scorecard across **accuracy** (WER, CER, S/D/I, entity preservation),
**diarization** (DER, JER, cpWER), **latency** (TTFT, per-final lag, RTF),
**resources** (GPU VRAM, util %, container CPU/RAM), and **concurrency**
(degradation curve as N streams increases).

Engines are pluggable: one file per engine in `engines/`, all implementing the
same `transcribe(audio, config) -> Result` interface. Cases are pluggable:
one folder per case in `cases/`, all containing `audio.<ext>` + `case.yaml` +
optional `reference.txt` / `reference.rttm`.

## Why it exists

This was built for a specific business decision the **DepoDash** team
(Bob Bakva / ProCat) is making in real time:

> DepoDash currently uses Speechmatics for the on-prem transcription product
> being delivered as Phase 1 MVP of the parent engagement. **Roughly 25 % of
> their revenue goes to Speechmatics.** Bob has explicitly said he wants to
> evaluate replacing it with an in-house engine. He's funding Phase 2 (sending
> a dedicated i7 + GPU box) to make that evaluation possible.

The strategic question is: **which ASR engine becomes DepoDash's long-term
in-house ASR?** Today there is no rigorous way to answer it. The previous
benchmark artifact (a one-off transcript dump in the `speechmaticsContainer`
repo) proves the engine works; it does not measure how it compares to
alternatives. This repo is the comparison framework.

## Why DepoDash isn't actually vendor-locked

Quoting Keanan from the 2026-04-22 sync:

> "We've actually got multiple models implemented on our server side. We've
> got a connection to Speechmatics, we've got one to AssemblyAI. We've made
> our own in-house Whisper... our wrapper grabs the packets off of the
> WebSocket and passes it through a parser and then gets a generic object
> that we use to actually do all our work on. So if we need to parse a
> different kind of JSON, then we can do that."

History: Watson → Speechmatics → AssemblyAI → in-house Whisper attempt
(shelved due to old hardware + limited diarization). They've replaced
engines before. The middleware tolerates it. **They are looking for the
next move**, and a defensible benchmark is what unlocks it.

## The metrics that matter to DepoDash specifically

In the same call, Keanan made one priority crisp:

> "Diarization accuracy is the metric we care about most, and it gets
> skipped a lot. Just keep that in mind."

Translation for the scoreboard:

- **DER** (or its modern composite, **cpWER**) is weighted as heavily as WER
  for any multi-speaker case. A scoreboard that hides diarization behind a
  single WER number would be the wrong answer.
- **Entity preservation** for legal proper nouns, dates, case numbers — the
  things that, when wrong, cost the customer billable accuracy. Tracked
  per-case via `case.yaml.key_terms`.
- **Latency p95** matters more than median. End users feel the worst
  finals, not the typical ones.

## Strategic lens

This benchmark unlocks three downstream conversations:

1. **Stay with Speechmatics, paid license.** Phase 1 MVP ships, eval license
   converts to paid, status quo continues. Benchmark numbers justify the
   spend or expose where it doesn't pencil.
2. **Replace Speechmatics with an in-house Parakeet-based container.** Bob's
   stated direction. Benchmark numbers establish the bar Parakeet has to
   clear. Becomes Phase 2 of the engagement (~$5–10 k separate scope).
3. **Hybrid — different engines for different products / SLA tiers.** Cloud
   engine for the web app, on-prem Parakeet for the courtroom desk app,
   etc. Benchmark numbers inform which engine for which surface.

The benchmark itself stays neutral. It scores; it doesn't pick.

## Non-goals (explicitly)

- **Not a UI / dashboard.** Numbers go in `SCOREBOARD.md`. A web frontend on
  top is a separately scoped Phase 3+ thing.
- **Not a replacement for production monitoring.** This is a one-shot
  comparison harness, not an SLO observability stack.
- **Not a hosting / serving solution.** Engines being benchmarked are
  expected to already be running (Speechmatics container up, cloud API key
  available, etc). The harness connects to them; it doesn't deploy them.
- **Not a research framework.** Existing tools (NeMo benchmarks, Open ASR
  Leaderboard) cover offline academic-style comparisons better. This is
  application-shaped: real cases, real config, real engines.
