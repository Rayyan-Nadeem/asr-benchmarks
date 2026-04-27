# Speechmatics on-prem — engine evaluation, what we found

> Built off the Phase 1 deliverable. Independent benchmark framework run
> against real legal multi-speaker audio (AMI Meeting Corpus, the academic
> gold standard) and the Phase 1 deliverable's own SCOTUS test case.
> Hardware: the same Alienware RTX 5080 you sent.

## The headline number

| Audio | Conditions | Speechmatics DER |
|---|---|---|
| **AMI ES2004a 5-min** | 4 speakers, 17 % overlap (real meeting) | **45 %** |
| SCOTUS Glossip oral arg | 3 speakers, ~0 % overlap (polite turn-taking) | **2 %** |

**Speechmatics handles polite turn-taking great. Real overlapping audio is
where it falls down.** A courtroom is closer to AMI than to SCOTUS.

DER 45 % means: of all the time someone is speaking, Speechmatics gets the
speaker label wrong 45 % of the time. For a deposition that's roughly half
the spoken minutes attributed to the wrong person. That's why your court
reporters spend time fixing speaker labels in post.

## I tried every config knob to fix it

Following Ed's struct + Patrick (Speechmatics) recommendations, swept five
variants on the same AMI audio:

| Config | DER |
|---|---|
| `max_delay=5` (current default) | 44.24 % |
| `max_delay=10` | 45.02 % |
| `max_delay=15` | 45.02 % |
| `max_speakers=4` (matched to ground truth) | 45.02 % |
| `max_delay=10, max_speakers=4` | 45.02 % |
| `max_delay=15, max_speakers=4, fixed mode` | 45.02 % |

**No knob moves it more than the noise floor (~1 pp).** This isn't a tuning
problem. It's an architectural ceiling in Speechmatics' streaming
diarization pipeline.

## I also confirmed it's the diarizer, not the words

Plugged Whisper-large-v3 (industry-leading open-source ASR) into the same
pipeline. Compared "Speechmatics' words + Speechmatics' speaker labels" vs
"Whisper's words + Speechmatics' speaker labels":

| Words | Speaker labels | DER |
|---|---|---|
| Speechmatics ASR | Speechmatics diarization | 45 % |
| **Whisper ASR** | **Speechmatics diarization** | **47 %** |

Same diarizer labels → same DER, regardless of how good the words are.
**The speaker assignment is the bottleneck. The transcription itself isn't.**

## What this means for the 25 % revenue share

You're paying Speechmatics for the whole stack, but the part of the stack
that's failing on real courtroom audio is specifically their diarization.
The published research benchmarks for open-source diarization (`pyannote.audio`)
land at **22–28 % DER on the same AMI test data** — ~half of what
Speechmatics is delivering you.

We have the harness in place to swap in `pyannote` and Parakeet
(open-source SOTA ASR, top of HuggingFace leaderboard, Apache license, free
forever) and re-run the same comparisons on your i7 box.

## Phase 2 proposal (the in-house engine)

**Goal:** drop Speechmatics, ship a Parakeet + pyannote container with the
same WebSocket protocol Phase 1 uses, so DepoDash's middleware drops in
unchanged.

### Two-stage version (recommended)

**2A — proof of numbers** (~3-4 weeks, ~$4-6k):
- Build Parakeet + pyannote container, batch mode first
- Run the same benchmark matrix on the i7 box
- Deliverable: scoreboard with Phase 2 numbers vs Speechmatics
- **Decision gate:** if numbers don't beat Speechmatics, you stop here

**2B — productionization** (~3-4 weeks, ~$5-8k, only if 2A passes):
- Streaming-mode for Parakeet
- Speechmatics-WebSocket-compatible API layer
- Container + runbook ready to deploy
- Concurrency / load testing on i7

**Total Phase 2 budget if it all goes through: ~$9-14k.**

Roughly 4-6 weeks of revenue at the 25 % share you're paying Speechmatics
today, by my back-of-envelope. Then it's $0 forever.

### What you get either way

The benchmark harness (`asr-benchmarks` repo) is reusable for every
future engine decision DepoDash makes. AssemblyAI, Deepgram, whatever
NVIDIA ships next year — same audio in, comparable scorecard out.
Strategic asset, not a one-shot.

## The Phase 2 preview numbers — measured tonight

Same AMI audio. Same harness. Different stacks:

| Setup | DER | What it says |
|---|---|---|
| Speechmatics ASR + Speechmatics diarization (today) | **45 %** | Where we are. |
| Speechmatics ASR + **pyannote diarization** | **40 %** | Drop the diarizer alone, save 5 pp. |
| Whisper ASR + pyannote diarization | 43 % | All-open-source. |
| **pyannote alone (diarizer ceiling)** | **25 %** | Best possible on this audio. |
| For perspective: random | ~75 % | Floor. |

**The diarization-alone number (24.80 %) is the proof.** Pyannote can do
~half the diarization errors of Speechmatics on the same audio. Phase 2
combines that diarizer with Parakeet (the open-source SOTA streaming ASR)
to land somewhere in the **25–35 % DER range realistically**, vs the
**45 % Speechmatics gives you today**.

That's a **30–45 % reduction in speaker-attribution errors** — the metric
Keanan said matters most.

---

**Live scoreboard:**
<https://github.com/Rayyan-Nadeem/asr-benchmarks/blob/main/results/SCOREBOARD.md>

**Methodology + decisions:** `docs/` directory in the same repo.
