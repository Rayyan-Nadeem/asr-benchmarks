# Benchmark cases

Each subdirectory is one audio case. The harness picks the case by folder
name (`--case <folder>`), reads `case.yaml` for metadata, and runs every
configured engine against `audio.<ext>`.

## Layout

```
cases/<case-name>/
├── case.yaml            # required — see schema below
├── audio.<ext>          # required — mp3/wav/flac/m4a/ogg, auto-converted to wav at run time
├── audio.wav            # optional — auto-generated from audio.<ext> on first run
├── reference.txt        # optional — ground-truth transcript for WER scoring
├── reference.rttm       # optional — speaker timeline for DER scoring
└── fetch.sh             # optional — script that produced audio.* (reproducibility)
```

## case.yaml schema

```yaml
name: my-case
description: |
  What the audio is, where it came from, why it's useful.
source: "URL or attribution"
license: "Public domain / CC-BY 4.0 / Customer-provided / etc."
duration_s: 300                 # approximate seconds, for sizing
speakers: 1                     # speaker count (1 if mono)
domain: legal | meeting | interview | broadcast | read
language: en

key_terms:                      # for entity preservation scoring
  - "Proper Noun"
  - "Case Number"

ground_truth: null              # populated when a transcript is added
```

## Adding a new case

1. `mkdir cases/<case-name>/`
2. Write `case.yaml` from the schema above.
3. Drop the audio in as `audio.mp3` (or any other ffmpeg-readable format).
4. If you have a transcript, write it as `reference.txt` (plain text, one sentence per line is fine).
5. If multi-speaker and you want DER scoring, write speaker turns to `reference.rttm` (RTTM format).
6. Optional: write a `fetch.sh` so the next person can reproduce.
7. Run the case: `python tools/measure_via_ws.py --case <case-name> --url ws://127.0.0.1:9000/v2 --engine auto-multispeaker --diarizer passthrough --max-speakers 4 --tag baseline`

## Day-1 cases (what's currently in the repo)

| Case | Speakers | License | Why |
|---|---|---|---|
| `librispeech-test-clean-mini` | 1 (per utterance, ~10 different across the set) | Public domain (LibriVox) + CC-BY 4.0 metadata | Industry-standard ASR sanity baseline. An engine that can't hit <5 % WER here is broken. Tiny — runs in seconds. |
| `scotus-glossip-v-oklahoma` | 3 (Roberts, Thomas, Waxman in our 5-min excerpt; full case has 9) | Public domain (US government work) | Domain-perfect for legal ASR + diarization. Official PDF transcripts at supremecourt.gov. |
| `ami-es2004a-5min` | 4 (meeting) | CC-BY 4.0 | Overlap-heavy multi-speaker stress test. The metric that matters for cross-examination. |
| `deposition-greg-erwin` | 1 | Customer-provided, internal only | Bob Bakva's CSM reference file — the audio he uses to evaluate engines. No official transcript so latency + resources only. |

## Selection criteria — apply to any candidate

A case earns inclusion if it:

1. **Has a permissive license** for the audio (public domain, CC-BY, or
   customer-provided where redistribution isn't required).
2. **Has high-quality ground truth** (word-level or sentence-level reference;
   for multi-speaker, an RTTM file or convertible equivalent).
3. **Adds something the existing cases don't cover** — a new domain, a
   harder diarization scenario, a longer-form test.
4. **Is reproducible** — anyone should be able to re-fetch the audio from a
   stable URL using the included `fetch.sh`.

## Skipped sources (don't re-research these)

| Source | Reason |
|---|---|
| NPR (Fresh Air etc.) | Transcripts copyrighted, non-commercial license. Cannot redistribute fetch script legally. |
| This American Life | Transcripts not licensed for redistribution; not diarized or time-aligned at speaker level. |
| Lex Fridman Podcast | Licensing unclear; YouTube ToS issues for fetch scripts. |
| TED-LIUM v3 | CC-BY-NC-ND 3.0 — non-commercial, no derivatives. Disqualified. |
| ICSI Meeting Corpus | LDC paywall, restricted research-only license. |
| DIHARD II/III | NIST OpenSAT registration required, not openly redistributable. |
| VoxConverse | Audio files not hosted by the project; you'd have to fetch from YouTube (ToS). |
| CALLHOME | LDC paywall for canonical version; HF mirror is CC-BY-NC-SA. Only 2 speakers, telephone-quality, weak domain match. |
| Libri2Mix / Libri3Mix | Requires generating from LibriSpeech (~430 GB) and WHAM! noise (NC license). Effort >> value — use LibriCSS instead. |

If a candidate fails (1) or (2), add it to this list with a one-line reason so we don't re-evaluate it.
