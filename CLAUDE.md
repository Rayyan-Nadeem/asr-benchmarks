# CLAUDE.md — asr-benchmarks (local, git-ignored)

> Local AI context file. **Not committed** — covered by `.gitignore`.
> Read this first, then read `docs/` for fuller project context before
> making non-trivial changes.

---

## 1. Project in one paragraph

`asr-benchmarks` is a vendor-agnostic harness for benchmarking real-time
speech-recognition (ASR) engines on a comparable scorecard: accuracy, latency,
resource cost, concurrency. Same audio + same `transcription_config` in →
canonical metrics out. Built for the **DepoDash** courtroom-transcription
engagement so Bob Bakva and team can answer the strategic question they're
already living with: *which engine becomes our long-term in-house ASR?* —
without re-asking Speechmatics, AssemblyAI, or anyone else for help.

The `engines/` interface is intentionally generic so adding Parakeet, Whisper,
or any cloud provider is a single new file. The `cases/` directory holds
audio + ground-truth pairs — public-domain (LibriSpeech, SCOTUS via Oyez) and
customer-provided (DepoDash deposition) — with a `case.yaml` per case.

For full strategic + technical context, read everything in `docs/`:

- `docs/VISION.md` — why this repo exists, what business problem it solves.
- `docs/ROADMAP.md` — phased execution plan (where we are, what's next).
- `docs/DECISIONS.md` — architectural decisions + their rationale.
- `docs/SPEECHMATICS_NOTES.md` — engine-specific gotchas (operating_point,
  permitted_marks, Blackwell quirks, license-gated concurrency).

`README.md` is the user-facing onboarding doc. `METHODOLOGY.md` is the
defensible "how we measured X" reference. Both are tracked in git.

---

## 2. Critical operating rules

1. **DepoDash production config is the default.** The Speechmatics engine
   adapter (`engines/speechmatics_onprem.py`) bakes in Ed's exact C++
   `TranscriptionConfig` struct from 2026-04-22: `permitted_marks`,
   `max_delay_mode: flexible`, `speaker_diarization_config.max_speakers: 10`,
   `output_locale: en-US`, `enable_entities: false`. Don't change these
   defaults without an explicit reason — benchmarking with non-production
   config is benchmarking a fiction. CLI overrides are fine for sensitivity
   analysis.

2. **1× realtime is the canonical accuracy mode.** `--fast` mode drops the
   first ~13 s of audio on Speechmatics (engine buffer race when audio
   arrives faster than realtime). Use `--fast` only for offline dumps where
   exact word boundaries don't matter. WER, DER, entity preservation must
   be measured at 1× realtime.

3. **Customer audio is not redistributable by default.** `cases/deposition-greg-erwin/`
   contains audio Bob provided for benchmarking. Its `case.yaml` flags it as
   "internal use only — do not redistribute." If pushing to a new remote or
   a public destination, the customer audio either stays out or the user
   explicitly authorizes its inclusion (see git history — this exact question
   came up on 2026-04-25 and the user answered "B: push everything to
   private GitHub").

4. **Quantitative thresholds before runs, not after.** `METHODOLOGY.md` §
   Concurrency defines pass/degraded/broken thresholds for WER drift, TTFT
   p95, per-final p95, connection reject rate. New thresholds are added to
   that file before they're applied to results.

5. **Audio formats: harness boundary is mono PCM S16LE @ 16 kHz.** Engines
   never see anything else. ffmpeg auto-converts at run time if a case
   ships only `audio.mp3` / `.flac` / `.m4a` / `.ogg`.

6. **Don't tell Bob about Mac dev environments or any non-target hardware.**
   This repo's validation is on Alienware RTX 5080 (Blackwell, sm_120, 16 GB
   VRAM); deliverables read as Windows-native. (Same operating principle
   carried over from the speechmaticsContainer engagement.)

---

## 3. State as of last commit (2026-04-27)

Phase 0 + bimodal hot-swap matrix complete on the Alienware RTX 5080 box.

**Engines wired:**
- `speechmatics_onprem` ← `ws://localhost:9000/v2`, ships with DepoDash prod config.
- `whisper` ← OpenAI whisper large-v3, batch (offline). Pair with any diarizer.
- (Phase 2: `parakeet`, `assemblyai`, `speechmatics_cloud`)

**Diarizers wired:**
- `native` — passthrough (engine's own labels)
- `none` — strip speakers
- `speechmatics_diar` — runs SM with diar=on, extracts RTTM only. Caches per audio so SM container doesn't have to be running for repeat runs.
- `pyannote` — pyannote.audio 3.1, MIT, gated on HF token.

**Cases shipped:**
- `librispeech-test-clean-mini` — 70 s, 1 speaker, public domain. Sanity baseline.
- `scotus-glossip-v-oklahoma` — 5 min trim from 67 min full audio, ~3 speakers
  in the trimmed window. Public domain via Oyez.
- `ami-es2004a-5min` — mid-meeting 5-min window of AMI ES2004a, 4 active
  speakers, 17% overlap, 94% speech density. The diarization stress test.
- `deposition-greg-erwin` — 8 min, 1 speaker, customer-provided. Latency +
  resources only (no ground-truth transcript).

**First scored numbers** (Speechmatics on-prem, 1× realtime, all defaults):
- LibriSpeech: WER 2.98% / CER 6.08% / TTFT 4019 ms / RTF 1.05 / GPU 7.6 GB
- SCOTUS 5min: WER 14.87% / DER 2.00% / TTFT 5131 ms / per-final p95 18.7 s /
  RTF 1.05 / GPU 7.7 GB / entities 6/16 (legal jargon dominated)
- Deposition (1× realtime): TTFT 5020 ms / RTF 1.05 / GPU 7.7 GB
- AMI ES2004a 5-min: DER 44.24-45.02% across baseline + 5 config variants
  (max_delay sweep + max_speakers sweep). Architectural ceiling, no knob
  fixes it.

**AMI bimodal hot-swap matrix (2026-04-27):**
- speechmatics + native: 45.02%
- speechmatics + pyannote: 40.12%
- whisper + speechmatics_diar: 46.70%
- whisper + pyannote: 42.75%
- whisper + none: 100% (no diarizer = no labels)
- pyannote alone (ceiling): **24.80%**

The pyannote-alone ceiling vs combined runs reveals a 15-20 pp tax from the
naive overlay approach (see SPEECHMATICS_NOTES.md). WhisperX-style integration
(use diarizer segments as source of truth, not word boundaries) would close
most of that gap. Not yet implemented — Phase 2 work.

**Concurrency ramp** (1, 2, 4 streams against SCOTUS 5min):
- N=1: clean baseline
- N=2: clean — same VRAM, same TTFT, same RTF as N=1
- N=4: half the streams rejected — cap is `SM_MAX_CONCURRENT_CONNECTIONS=2`
  in the .env. Need to bump and restart the stack to test higher.

---

## 4. Repo layout (where things live)

```
.
├── CLAUDE.md                # this file (gitignored)
├── README.md                # user-facing onboarding (tracked)
├── METHODOLOGY.md           # quantitative measurement choices (tracked)
├── requirements.txt         # Python deps (tracked)
├── .gitignore               # excludes .venv, __pycache__, CLAUDE.md, .claude, notes/
├── docs/                    # tracked project context — read these for fuller picture
│   ├── VISION.md
│   ├── ROADMAP.md
│   ├── DECISIONS.md
│   └── SPEECHMATICS_NOTES.md
├── engines/
│   ├── _base.py             # Engine protocol + Result/Word/RunMetadata
│   ├── speechmatics_onprem.py
│   └── whisper.py           # openai-whisper large-v3, batch
├── diarizers/
│   ├── _base.py             # Diarizer protocol + overlay_speakers helper
│   ├── native.py
│   ├── none.py
│   ├── speechmatics_diar.py # extracts RTTM from a SM run, caches per audio
│   └── pyannote.py          # pyannote.audio 3.1, gated HF token
├── cases/                   # one folder per audio case
│   ├── README.md            # convention + how to add a case
│   ├── CHOICES.md           # what was considered + why each kept/skipped
│   ├── librispeech-test-clean-mini/
│   ├── scotus-glossip-v-oklahoma/
│   ├── ami-es2004a-5min/    # the diarization stress test
│   └── deposition-greg-erwin/
├── metrics/
│   ├── accuracy.py          # WER/CER/SDI + entity preservation + confidence (jiwer)
│   ├── diarization.py       # DER (simpleder); RTTM read/write helpers
│   └── latency.py           # TTFT, per-final lag, RTF, partial stability
├── samplers/
│   └── resources.py         # nvidia-smi + docker stats + psutil; threaded JSONL
├── tools/
│   ├── oyez_to_reference.py # Oyez argument JSON → reference.txt + reference.rttm
│   ├── pyannote_only.py     # standalone pyannote DER (the diarizer ceiling)
│   ├── run_matrix.sh        # orchestrates the 5-row matrix for a case
│   └── render_scoreboard.py # results/runs/*.json → SCOREBOARD.md
├── runner.py                # CLI: --engine X --diarizer Y --case Z [--tag T]
└── results/
    ├── runs/                # one .json per run + matching __resources.jsonl
    ├── SCOREBOARD.md        # auto-rendered comparison table
    └── BOB_BRIEF.md         # talking points for Bob meeting
```

---

## 5. Working principles (Rayyan's operating rules — carried over)

1. Principal-engineer solution, not minimum-viable hack.
2. Phase strategically — land Phase 0 cleanly before pitching Phase 2.
3. Weight effort/pricing toward docs, methodology, and reproducibility — not
   the core measurement code.
4. Be the honest expert. Flag every methodology limitation in
   `METHODOLOGY.md` so a senior reviewer can't catch us hand-waving.
5. Protect scope ruthlessly. The harness benchmarks engines; it doesn't
   ship a UI, doesn't host a service, doesn't replace any DepoDash code.
6. Educate without condescending. Bob is making a buy/build decision; the
   `SCOREBOARD.md` is the artifact he'll point at.

---

## 6. Quick reference — commands

```bash
# install deps (use a venv)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# also (after pinning torch 2.7 + cu128 for Blackwell + legacy torchaudio):
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.7.*" "torchaudio==2.7.*"
.venv/bin/pip install openai-whisper "pyannote.audio>=3.1,<4" "huggingface_hub<0.23"

# single case
.venv/bin/python -m runner --engine speechmatics_onprem --case librispeech-test-clean-mini
.venv/bin/python -m runner --engine speechmatics_onprem --case scotus-glossip-v-oklahoma
.venv/bin/python -m runner --engine speechmatics_onprem --case ami-es2004a-5min
.venv/bin/python -m runner --engine speechmatics_onprem --case deposition-greg-erwin --fast

# bimodal: any engine × any diarizer
HF_TOKEN=hf_xxx .venv/bin/python -m runner --engine whisper --diarizer pyannote --case ami-es2004a-5min --tag whisper_pyannote
.venv/bin/python -m runner --engine whisper --diarizer speechmatics_diar --case ami-es2004a-5min
HF_TOKEN=hf_xxx .venv/bin/python tools/pyannote_only.py ami-es2004a-5min

# config sweep on diarization knobs
.venv/bin/python -m runner --engine speechmatics_onprem --case ami-es2004a-5min --max-delay 10 --max-speakers 4 --tag md10_ms4

# concurrency ramp
.venv/bin/python -m runner --engine speechmatics_onprem --case scotus-glossip-v-oklahoma --concurrency-ramp 1,2,4

# regenerate scoreboard
.venv/bin/python tools/render_scoreboard.py

# regenerate SCOTUS reference from Oyez JSON (after edits)
.venv/bin/python tools/oyez_to_reference.py cases/scotus-glossip-v-oklahoma/oyez_argument.json
```

---

## 7. Critical install gotchas (Alienware RTX 5080 Blackwell)

- **torch + torchaudio 2.7.* + cu128 wheels are the sweet spot.** Newer
  (2.10+) strips legacy torchaudio API that pyannote 3.4 needs. Older
  (≤2.6) lacks Blackwell sm_120 kernels.
- **`huggingface_hub<0.23`** required because pyannote.audio 3.4 internally
  calls `hf_hub_download(use_auth_token=...)` which was removed in newer hub.
- **`torch.load` weights_only=True default** in torch 2.6+ breaks pyannote's
  pickled checkpoints. `diarizers/pyannote.py` monkey-patches this.
- **VRAM ceiling on 16 GB card:** Whisper large-v3 ~10-15 GB + Speechmatics
  container ~7.7 GB > 16 GB. Stop SM container before Whisper runs.
  speechmatics_diar caches per-audio RTTM so it doesn't need re-extraction.
- **HF token:** required for pyannote, set as `HF_TOKEN` env var. Models
  also need terms-acceptance at:
  - https://huggingface.co/pyannote/speaker-diarization-3.1
  - https://huggingface.co/pyannote/segmentation-3.0

---

## 8. Reference links

- Project discussion log + decisions: see `docs/DECISIONS.md`.
- Companion repo (the engine being benchmarked):
  `/mnt/c/Users/Admin/speechmaticsContainer/` — read its
  `INTEGRATION_GUIDE.md` § 4 for the canonical Speechmatics
  `transcription_config`.
- GitHub: https://github.com/Rayyan-Nadeem/asr-benchmarks (private).
- Memory store with cross-conversation context:
  `~/.claude/projects/-home-dev/memory/MEMORY.md` and the per-topic files
  it indexes (DepoDash project, Speechmatics gotchas, WSL mount issues).
