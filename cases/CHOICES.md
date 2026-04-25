# Audio case choices

What we surveyed when picking Day-1 cases, and why each candidate was kept,
deferred, or skipped. Re-read this before adding a new case so we don't
repeat dead-end research.

## Kept (Day-1 cases)

| Case | Speakers | License | Why |
| --- | --- | --- | --- |
| `librispeech-test-clean-mini` | 1 (per utterance, ~10 different speakers across the set) | Public domain (LibriVox audio) + CC-BY 4.0 metadata | Industry-standard ASR sanity baseline. Engine that can't hit <5% WER here is broken. Tiny — runs in seconds. |
| `scotus-glossip-v-oklahoma` | 9 (8 justices + counsel) | **Public domain** (US government work) | Domain-perfect for legal ASR + diarization. 50 min of clean studio audio. Official PDF transcripts at supremecourt.gov. |
| `deposition-greg-erwin` | 1 | Customer-provided, internal only | Bob Bakva's CSM reference file — the audio he uses to evaluate engines. No official transcript so latency + resources only until Phase 2 pseudo-truth. |

## Deferred (Phase 1.5 / 2)

| Case | Why later, not now |
| --- | --- |
| `ami-es2002a` | Real 4-speaker meeting, CC-BY 4.0. Worth adding once the harness is proven; needs HuggingFace mirror or AMI registration. |
| `libricss-session-1` | Synthetic far-field multi-speaker, CC-BY 4.0. Useful for stress-testing diarization on noisy audio. ~140 MB. |
| `scotus-second-case` | Adds variety to the legal corpus once Glossip is scored. |
| `same-speaker-pair` | Self-recorded audio of one person across two sessions, for the speaker-fingerprinting capability test. |

## Skipped (and why — don't re-research these)

| Source | Reason |
| --- | --- |
| NPR (Fresh Air etc.) | Transcripts copyrighted, non-commercial license. Cannot redistribute fetch script legally. |
| This American Life | Same — transcripts not licensed for redistribution; not diarized or time-aligned at speaker level. |
| Lex Fridman Podcast | Transcripts and audio licensing unclear; YouTube ToS issues for fetch scripts. |
| TED-LIUM v3 | CC-BY-NC-ND 3.0 — non-commercial, no derivatives. Disqualified. |
| ICSI Meeting Corpus | LDC paywall, restricted research-only license. |
| DIHARD II/III | NIST OpenSAT registration required, not openly redistributable. |
| VoxConverse | Audio files not hosted by the project; you'd have to fetch from YouTube (ToS). |
| CALLHOME | LDC paywall for the canonical version; HF mirror is CC-BY-NC-SA. Only 2 speakers, telephone-quality, weak domain match. |
| Libri2Mix / Libri3Mix | Requires generating from LibriSpeech (~430 GB) and WHAM! noise (NC license). Effort >> value — use LibriCSS instead. |

## Selection criteria (apply to new cases)

A case earns inclusion if it:

1. **Has a permissive license** for the audio (public domain, CC-BY, or
   customer-provided where redistribution isn't required).
2. **Has high-quality ground truth** (word-level or sentence-level reference;
   for multi-speaker, an RTTM file or convertible equivalent).
3. **Adds something the existing cases don't cover** — a new domain, a
   harder diarization scenario, a longer-form test.
4. **Is reproducible** — anyone should be able to re-fetch the audio from a
   stable URL using the included `fetch.sh`.

If a candidate fails (1) or (2), it goes in this file's "Skipped" section
with a one-line reason so we don't waste time evaluating it again.
