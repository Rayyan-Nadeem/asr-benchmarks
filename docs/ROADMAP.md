# Roadmap

Phased plan, what's done, what's next, what's deferred.

## Phase 0 — single-engine baseline on borrowed hardware *(in progress, ends when the Alienware goes back)*

**Goal:** prove the harness works end-to-end and capture the first real
Speechmatics on-prem numbers while we still have access to the GPU box.
Concurrency + resource numbers can only be captured here.

- [x] Repo scaffold + `engines/_base.py` + `runner.py`
- [x] `engines/speechmatics_onprem.py` (WS adapter, DepoDash config baked in)
- [x] `metrics/{accuracy, diarization, latency}.py`
- [x] `samplers/resources.py` (nvidia-smi + docker stats + psutil, threaded)
- [x] `cases/librispeech-test-clean-mini` (1 speaker, public domain) → first scored WER
- [x] `cases/scotus-glossip-v-oklahoma` (5-min trim, real legal multi-speaker, public domain via Oyez) → first scored WER + DER
- [x] `cases/deposition-greg-erwin` (DepoDash CSM file, latency + resources only)
- [x] Concurrency ramp 1, 2, 4 (capped at 2 by current `SM_MAX_CONCURRENT_CONNECTIONS`)
- [x] `tools/render_scoreboard.py` + first `SCOREBOARD.md`
- [x] Push to GitHub (private)
- [ ] Push to Azure DevOps (waiting on the user to create the empty repo
  there; Azure PAT in this session can't create repos)
- [ ] Re-run concurrency ramp at higher N (4, 6, 8) after bumping
  `SM_MAX_CONCURRENT_CONNECTIONS` in the speechmaticsContainer `.env` and
  restarting the stack — gives the realistic per-stream cost curve before
  the laptop goes back

## Phase 1.5 — cloud engines *(no Alienware needed, can happen anywhere)*

Adds engines whose backends don't require local GPU: Speechmatics cloud,
AssemblyAI cloud (and Deepgram if interesting). Same `transcribe()`
interface, same scorecard, head-to-head against the on-prem column.

- [ ] `engines/speechmatics_cloud.py` against `wss://eu2.rt.speechmatics.com/v2`
  with API key. Same `transcription_config` defaults so on-prem vs cloud
  is a clean comparison.
- [ ] `engines/assemblyai.py` against AssemblyAI's streaming endpoint.
  DepoDash already has an integration so getting an API key should be
  straightforward.
- [ ] (Optional) `engines/deepgram.py` for completeness — Deepgram is the
  third major commercial real-time ASR.
- [ ] Re-run all three cases through each cloud engine. SCOREBOARD now has
  4-5 columns instead of 1.
- [ ] Hand-correct ~60 seconds of the Greg Erwin deposition for a first
  real WER number on Bob's reference file.

## Phase 2 — in-house alternatives *(unlocked when the i7 box arrives)*

The strategic phase. Adds the engines DepoDash could deploy themselves and
ask "is the cost / ops / accuracy trade worth replacing Speechmatics?"

- [ ] `engines/parakeet.py` — NVIDIA NeMo Parakeet (TDT 0.6B v3 or current
  SOTA), via NeMo Python or ONNX runtime. Apache license, top of Hugging
  Face Open ASR Leaderboard at time of evaluation.
- [ ] `engines/whisper.py` — `faster-whisper` (CTranslate2 backend) and/or
  `whisper.cpp`. Compare large-v3 + distil-large-v3 + maybe medium for the
  speed/accuracy frontier.
- [ ] (Optional) `engines/canary.py` — NVIDIA Canary if it competes with
  Parakeet on diarization-relevant metrics.
- [ ] Add `engines/<name>+pyannote.py` adapters for engines that don't
  natively diarize (Whisper, Parakeet) — runs `pyannote.audio` on the same
  audio, joins to engine words for cpWER.
- [ ] Re-run all cases through all engines. SCOREBOARD becomes the head-to-head
  Bob points at when making the buy/build call.
- [ ] Add `cases/ami-es2002a/` (real 4-speaker meeting, CC-BY 4.0) once the
  scoring methodology is stable, for a harder diarization stress test.
- [ ] Same-speaker cross-session fingerprinting test for engines that
  support it (NeMo ECAPA-TDNN; Speechmatics doesn't).

## Phase 3 — productization *(only if Phase 2 says "build in-house")*

These are placeholder ideas, NOT scoped or committed. They live here so the
benchmark architecture choices today don't paint Phase 3 into a corner.

- [ ] Wrap the runner in a FastAPI service so DepoDash can run it themselves
  on each engine release as a regression check.
- [ ] Cron-driven nightly re-runs across the full case set; alert on
  regressions vs. the previous baseline.
- [ ] `SCOREBOARD.md` rendered to a static site (GitHub Pages or similar)
  so non-engineers can read it.
- [ ] Optional: TS/Tauri thin client over the FastAPI for in-house use.
  (Not Electron — we've been burned by Electron's Windows GPU dep story
  in past projects.)

## Continuously deferred (skip-list)

These come up in conversation periodically; we don't do them on purpose.

- **Replacing the docker-compose Speechmatics stack with a Helm chart.**
  Out of scope for the benchmark. The engine being benchmarked is whatever
  the user has running locally.
- **Cost-per-minute as a benchmark column.** Cost is contractual + license-tier
  dependent; not directly comparable across vendors. Lives in business docs,
  not the technical scoreboard.
- **A "subjective quality" metric.** Explicitly out — we test what's
  measurable. UX-quality discussions belong in product, not in this repo.
- **Multi-language benchmarking.** Until DepoDash needs Spanish or Mandarin
  in the courtroom product, no value here.
