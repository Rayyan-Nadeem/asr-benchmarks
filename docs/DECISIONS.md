# Decisions

Architectural decisions that shaped this repo, with rationale. Update when
the answer changes.

## D-01: Python, not TypeScript

**Decision:** the benchmark harness is Python.

**Why:** every ASR-evaluation library lives in Python — `jiwer` (WER/CER
with word-level alignment), `pyannote.metrics` and `simpleder` (DER),
`meeteval` (cpWER for "who said what" composite scoring), `soundfile`
(audio I/O). Re-implementing them in TypeScript is weeks of work that
nobody else needs.

**What this doesn't preclude:** if Phase 3 ships a frontend, it can be
TS/Next/Tauri sitting on top of a FastAPI wrapper around this Python
harness. Keep them separated by HTTP, not language.

**Considered and rejected:** TypeScript-everywhere on the basis of
"the eventual product is a Windows app." The product surface and the
benchmark harness are different problems. Don't conflate them.

## D-02: Vendor-agnostic from line 1

**Decision:** every engine implements the same `Engine` protocol in
`engines/_base.py`. Result types are normalized; engine-native frames are
preserved on `Result.raw_dump`.

**Why:** the entire point of this repo is comparing engines. If the first
engine drove the protocol, every subsequent adapter would inherit
Speechmatics-specific assumptions. Generic from day one means Parakeet
(no WS), Whisper (offline batch), AssemblyAI (different protocol) all
slot in cleanly.

**Implementation note:** scorers consume the normalized `words: list[Word]`
view. Engine-native dumps stay around for debugging + auditability.

## D-03: 1× realtime is the canonical accuracy mode

**Decision:** WER, DER, entity-preservation runs are at 1× realtime.
`--fast` mode is opt-in for offline dumps where exact word boundaries
don't matter.

**Why (empirical, 2026-04-25):** the Speechmatics on-prem engine drops
the first ~13 s of audio when audio arrives faster than realtime — engine
buffer race during session warmup. LibriSpeech mini ran at WER 17.86 % in
`--fast` mode (28 deletions clustered at the start) vs. WER 2.98 % at
1× realtime, same audio, same config.

**Methodology consequence:** every accuracy number in the scoreboard is at
1× realtime. `--fast` is for resource-baseline runs and protocol dumps.
This is documented in `METHODOLOGY.md` § Audio handling.

## D-04: DepoDash production config baked in as the default

**Decision:** `engines/speechmatics_onprem.py` defaults `transcription_config`
to DepoDash's exact production cloud client struct (Ed's C++ source,
2026-04-22): `permitted_marks: [",",".","?"]`, `max_delay_mode: flexible`,
`speaker_diarization_config.max_speakers: 10`, `output_locale: en-US`,
`enable_entities: false`.

**Why:** benchmarking with non-production config is benchmarking a
fiction. If we score the engine with looser config than DepoDash actually
sends, we overstate; with stricter, we understate. Default-of-defaults is
a footgun (see `docs/SPEECHMATICS_NOTES.md` on the `permitted_marks` /
diarization issue).

**Override mechanism:** `--diarization`, `--max-delay`, etc. CLI flags pass
overlays to the adapter, but the *defaults* are production-shaped.

## D-05: simpleder for DER, not pyannote.metrics

**Decision:** `simpleder` is the DER scorer.

**Why:** zero deep-learning dependencies — `pyannote.metrics` pulls in
torch (~2 GB install) for what is, at the scoring layer, just set
arithmetic. Phase 2 brings torch in anyway when we add `pyannote.audio`
for diarization of engines that don't do it natively, but until then
simpleder keeps the install slim.

**Tuple shape gotcha:** simpleder expects `(speaker, start, end)` — speaker
**first**. Got bitten by this on first run; it's the order in
`metrics/diarization.py:words_to_rttm_segments` and `parse_rttm`. Don't
flip it back.

## D-06: jiwer + Whisper-style normalizer stack

**Decision:** WER/CER via `jiwer` 4.0+ with `ToLowerCase →
RemovePunctuation → ExpandCommonEnglishContractions → RemoveMultipleSpaces
→ Strip → ReduceToListOfListOfWords`.

**Why:** matches the de-facto industry baseline (OpenAI Whisper paper, all
recent ASR leaderboards). Without normalization, engines get penalized for
adding case + punctuation that the reference doesn't have (LibriSpeech is
uppercase, no punct; Speechmatics emits proper-cased + punctuated).

**S/D/I breakdown** is recorded alongside the WER number — it's the
diagnostic that explains *why* WER is what it is (substitution-heavy =
acoustic confusion; insertion-heavy = hallucination; deletion-heavy =
silence cuts / dropouts).

## D-07: Resource sampling via subprocess + nvidia-smi + docker stats

**Decision:** `samplers/resources.py` shells out to `nvidia-smi` and
`docker stats --no-stream --format json`, runs in a daemon thread, writes
JSONL.

**Why pynvml is not used:** pynvml fails inside Docker containers on
WSL2 — "GPU access blocked". Subprocess to the host nvidia-smi works
reliably from WSL. No clean Python path that works inside both host and
containers without conditional code.

**Per-container GPU memory attribution is not measurable on consumer GPUs.**
No MIG support on RTX 5080. We report aggregate GPU VRAM + util across all
containers; per-engine cost is inferred from delta vs. idle baseline.

## D-08: Audio committed to git (not gitignored)

**Decision:** `cases/<case>/audio.<ext>` is tracked in git.

**Why:** reproducibility wins. A clone of the repo is a complete benchmark
suite. Without committed audio, anyone re-running has to refetch (or use
broken hyperlinks for podcasts that vanish).

**Constraint accepted:** repo size grows with cases. Current size is ~110 MB
(SCOTUS 24 MB + LibriSpeech 2 MB + deposition 7 MB + generated wavs).
GitHub's recommended ceiling is 1 GB; we have headroom.

**Auto-generated `audio.wav`** sits next to the source `audio.mp3` etc. when
ffmpeg has converted it. We commit both — disk is cheap, and a fresh clone
shouldn't need ffmpeg before the first run.

## D-09: Quantitative degradation thresholds defined upfront

**Decision:** `METHODOLOGY.md` § Concurrency lists pass / degraded / broken
thresholds for WER drift, TTFT p95 multiplier, per-final p95 multiplier,
connection reject rate. Defined before runs, not after.

**Why:** a benchmark that defines "degraded" after seeing the numbers is
rationalized, not measured. Future Claude / engineers should add new
thresholds to that file *before* applying them to results.

## D-10: SCOTUS via Oyez API, not supremecourt.gov

**Decision:** SCOTUS audio + transcripts come from Oyez
(`api.oyez.org/cases/<year>/<docket>` and
`api.oyez.org/case_media/oral_argument_audio/<id>`).

**Why:** supremecourt.gov is bot-blocked by Akamai — generic curl returns
403 / error pages. Oyez serves the same content under a clean JSON API,
no rate-limiting issues, transcripts are speaker-labeled and time-aligned
at the text-block level. Audio is the same recording (S3-hosted MP3).

**License:** both are public domain (US government work).

## D-11: Repo on GitHub (private) + Azure DevOps (private)

**Decision:** primary remote is GitHub `Rayyan-Nadeem/asr-benchmarks` (private).
Azure DevOps mirror is the secondary remote (engagement-of-record location;
adds belt-and-suspenders for Bob's compliance comfort).

**Why GitHub primary:** Rayyan's professional portfolio + faster web UI.
Why Azure DevOps mirror: aligns with the existing `speechmaticsContainer`
location so all DepoDash engagement work is visible in one place to Bob's
team.

**Customer audio constraint:** the deposition audio is "internal use only"
per `cases/deposition-greg-erwin/case.yaml`. User explicitly authorized
inclusion in private GitHub on 2026-04-25. Re-confirm before any future
public push.
