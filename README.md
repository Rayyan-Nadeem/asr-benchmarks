# depodash-asr

A Docker container that exposes `ws://localhost:9000/v2` speaking the
Speechmatics realtime protocol verbatim, with an open-source ASR + diarizer
stack underneath. DepoDash's existing middleware swaps engines via a single
config change — no port required.

The goal: a drop-in replacement for the Speechmatics on-prem container that
beats it on the diarization metrics that matter for courtroom audio, runs on
hardware ~4× cheaper, and has no per-seat license to manage.

---

## Headline results

| Dimension | Result |
|---|---|
| Diarization confusion on clean turn-taking (SCOTUS) | **0.4 %** (Multitalker) vs SM's published 2.00 % — **5× better** |
| Diarization confusion on overlap-heavy meetings (AMI) | **3.3 %** (Multitalker) vs every published streaming-SM number landing in the 38–46 % DER band |
| Speaker count detected on real audio | Correct on every fixture (3/3 SCOTUS, 4/4 AMI), no hallucinated extras even when told to allow up to 12 |
| Hardware floor | **8 GB VRAM card** (Multitalker fits) vs SM's **16 GB hard minimum** |
| CPU-only path | ~2 GB RAM, no GPU at all (Parakeet-ONNX) — SM has no CPU path |
| Per-stream cost (self-hosted) | <$1000 laptop covers many concurrent sessions; no per-seat license fee |

Full measured numbers per case live in [`results/SCOREBOARD.md`](results/SCOREBOARD.md).

**The architectural moat:** Multitalker is the only engine in the production
stack that handles true simultaneous speech (cross-examination, objections).
Otter, Fireflies, AssemblyAI, Deepgram, Speechmatics — all of them collapse
overlapping voices to a single winner per word. We don't.

---

## The stack

### Layer 1 — ASR engines

| Engine | Best for | Trade-off |
|---|---|---|
| `auto-multispeaker` (default) | Production — routes per session | Two engines kept warm; negligible per-call overhead |
| `multitalker-parakeet` | Cross-examination, overlap, ≤4 speakers | Hard 4-spk cap; worse single-speaker WER |
| `nemotron-native` | >4 speakers (auto routes here) | Higher confusion than multitalker on its turf |
| `parakeet-nemo` | Post-session reference / batch cleanup | Finals at end of segment, not live |
| `fastconformer-hybrid` | 8 GB GPU / CPU mid-tier | No overlap handling |
| `parakeet-onnx` | CPU fallback (laptop, no GPU) | Slower; quantized accuracy hit |
| `nemotron-streaming` | Even smaller CPU fallback | Lower accuracy than ONNX variant |
| `whisper` | Short clips only | Drops ~22 % of words on long-form courtroom audio |
| `noop` | Protocol smoke test | Doesn't transcribe |

### Layer 2 — Diarizers

| Diarizer | Best for | Notes |
|---|---|---|
| `sortformer` (streaming v2.1) | Pipeline engines (Nemotron Native, Parakeet-NeMo) | GPU only; AOSC speaker cache |
| `pyannote` (3.1) | Arbitrary speaker counts, CPU paths | Slow on Mac; best on AMI overlap (38.94 % DER) |
| `passthrough` | Multitalker + auto-multispeaker | No-op; preserves engine-set labels |

### Layer 3 — Punctuation

| Punctuator | What it does |
|---|---|
| `distilbert` (default) | NVIDIA punctuation_en_distilbert post-process — adds periods, commas, question marks, capitalization (~10 ms GPU / 50 ms CPU per emission) |
| `passthrough` | Raw engine output, no post-process. Selectable for A/B testing only. |

### Always-on (built into engines)

- **SpeakerLabelSmoother** (inside Multitalker) — hysteresis + min-speaker-duration filter from the Streaming Sortformer paper §IV. Suppresses single-frame phantom-speaker flicker.
- **TitaNet speaker verification** (inside Multitalker) — after each multispk step, the dominant-channel chunk runs through NVIDIA `speakerverification_en_titanet_large` to produce a 192-dim embedding that feeds an EMA centroid per verified channel. Subsequent chunks map to the nearest centroid by cosine similarity (threshold 0.70). Replaces the previous heuristic sticky-lock for label stability under mic-distance drift. See `results/SCOREBOARD.md` for methodology + the AMI no-regression result.
- **Word-boundary commit** (inside Multitalker) — only emits text through the last word boundary; holds the trailing partial word until completion or a 1.5 s stale-flush.
- **Hybrid engine routing** (inside auto-multispeaker) — reads `transcription_config.speaker_diarization_config.max_speakers` and dispatches to Multitalker if ≤4, else Nemotron Native + Sortformer-streaming.
- **Arrival-order speaker remap** (in session.py) — translates engine-native labels to Speechmatics-compatible `S1`/`S2`/`S3`.

---

## Architecture — how a session flows

```
DepoDash middleware
        │
        │  ws://localhost:9000/v2
        │  StartRecognition { transcription_config }
        ▼
┌────────────────────────────────────────────────────────────┐
│  server/session.py — protocol shim + state machine         │
│  reads transcription_config, dispatches per-session        │
└────────────────────────────┬───────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        ▼                                         ▼
  auto-multispeaker meta-engine                   │
  ┌───────────────────────────────┐               │
  │ if max_speakers ≤ 4:          │               │
  │   → Multitalker Parakeet      │               │
  │     (4-channel speaker-kernel │               │
  │      ASR — true overlap)      │               │
  │ else:                         │               │
  │   → Nemotron Native +         │               │
  │     Sortformer-streaming      │               │
  │     (pipeline, arbitrary N)   │               │
  └───────────────────────────────┘               │
                                                  ▼
                            ┌──── diarizer.label() ────┐
                            │ passthrough — no-op       │
                            │   (auto + multitalker)    │
                            │ sortformer — GPU AOSC     │
                            │   (pipeline engines)      │
                            │ pyannote — CPU fallback   │
                            └────────────┬──────────────┘
                                         ▼
                            ┌──── punctuator.punctuate() ────┐
                            │ passthrough — raw output       │
                            │ distilbert — NVIDIA model      │
                            │   (110 MB; restores . , ? caps)│
                            └────────────┬───────────────────┘
                                         ▼
                            ┌──── arrival-order remap ────┐
                            │ engine labels → S1/S2/S3   │
                            └────────────┬────────────────┘
                                         ▼
                        AddPartialTranscript / AddTranscript
                                         ▼
                              DepoDash middleware
```

### Why Multitalker is different

Pipeline approach (Otter, Fireflies, SM, AssemblyAI, Deepgram):
```
audio → ASR (one transcript stream) → diarizer overlays speaker labels
        per word. When two voices overlap, ONE wins. The other is dropped.
```

Multitalker:
```
audio → Streaming Sortformer (per-frame speaker activity, 4 channels)
        → FastConformer encoder with learnable speaker kernels
        → 4 parallel ASR instances, one per detected speaker
        → 4 parallel transcripts. Overlap is preserved by construction.
```

### Why hybrid routing

Multitalker has a hard 4-speaker architectural cap (4 speaker-kernel
channels, 4 Sortformer output heads). DepoDash session shapes that exceed
the cap (multi-party depos, mediations, oral arguments) need a fallback.
The auto-multispeaker meta-engine routes per session — DepoDash's
middleware change is one new field in `StartRecognition`:
`transcription_config.speaker_diarization_config.max_speakers`.

| Session type | Typical speaker count | Routes to |
|---|---|---|
| Solo / standard deposition | 2–4 | Multitalker |
| Cross-examination heavy | 3–4 | Multitalker |
| Multi-party / class-action | 4–6 | Nemotron + Sortformer |
| Mediation | 4–8 | Nemotron + Sortformer |
| SCOTUS-style oral argument | 9–11 | Nemotron + Sortformer |

Both engines speak the same wire protocol. The swap is invisible to the
client.

---

## Hardware footprint

| Stack | Engine VRAM | Per-stream | Min GPU |
|---|---|---|---|
| Speechmatics + native | ~14–15 GB | ~14–15 GB | **16 GB hard floor** |
| Speechmatics + pyannote | ~14–15 GB | ~18–19 GB | **24 GB card** |
| **Multitalker + passthrough** | ~5 GB | ~5 GB | **8 GB card** |
| Nemotron Native + Sortformer-streaming | ~3 GB | ~4.5 GB | 8 GB card |
| Parakeet-NeMo + Sortformer-overlay | ~5 GB | ~6.5 GB | 8 GB card |
| Parakeet-ONNX + passthrough (Mac M-series) | ~2 GB RAM | ~2 GB RAM | **no GPU** |

`$4k Alienware spec (Speechmatics) → <$1000 laptop (us)`, plus a
per-courthouse shared-server option on any 12 GB+ consumer card.

---

## Running the container

Prerequisites: Docker 24+, NVIDIA driver 580+ (for the GPU shapes),
`nvidia-container-toolkit`, ~30 GB free disk, and a HuggingFace token with
read access to the gated NVIDIA + pyannote model cards (visit each card
while logged in and accept the terms once).

```bash
cd deploy
cp .env.example .env
$EDITOR .env                                  # set HF_TOKEN at minimum
docker compose -f deploy/docker-compose.yml up -d
curl http://localhost:9000/ready              # waits for first-boot model download
```

First boot pulls ~10 GB of model weights to a persistent volume; subsequent
restarts come up in ~30 s. The `auto-multispeaker` engine, `passthrough`
diarizer, and `distilbert` punctuator are the production defaults; override
per session at the StartRecognition layer (see Integration below) or globally
via the env vars documented in `deploy/.env.example`.

Day-to-day:

```bash
docker compose -f deploy/docker-compose.yml logs -f   # tail
docker compose -f deploy/docker-compose.yml restart    # restart
curl http://localhost:9000/ready                       # readiness probe
curl http://localhost:9100/current                     # which engine is loaded
```

---

## Integration — how DepoDash's middleware connects

Same WebSocket URL, same Speechmatics realtime v2 message frames, same
audio format (PCM S16LE 16 kHz mono, 80–100 ms per frame) as the legacy
on-prem container. The only middleware change is adding one field to
`StartRecognition`.

### StartRecognition payload

```json
{
  "message": "StartRecognition",
  "audio_format": {
    "type": "raw",
    "encoding": "pcm_s16le",
    "sample_rate": 16000
  },
  "transcription_config": {
    "language": "en",
    "operating_point": "enhanced",
    "diarization": "speaker",
    "enable_partials": true,
    "max_delay": 5.0,
    "max_delay_mode": "flexible",
    "output_locale": "en-US",
    "enable_entities": false,
    "punctuation_overrides": { "permitted_marks": [",", ".", "?"] },
    "speaker_diarization_config": {
      "max_speakers": 4
    }
  }
}
```

### `max_speakers` — the one new field that matters

Send the value that matches the session type. The container reads this to
pick the underlying engine; the field acts as an upper bound, not a forced
count.

| Session type | `max_speakers` to send |
|---|---|
| Solo deposition (witness + 1 attorney) | `4` |
| Standard deposition (witness + 2–3 attorneys) | `4` |
| Cross-examination heavy | `4` |
| Multi-party / class-action deposition | `8` |
| Mediation | `8` |
| SCOTUS-style oral argument | `12` |

Speechmatics ignores this field for routing; it's safe to send against
both containers.

### Server messages

Identical to SM: `RecognitionStarted`, `AudioAdded`, `AddPartialTranscript`,
`AddTranscript`, `EndOfTranscript`, `Error`, `Warning`. Per-word
`alternatives[0].speaker` uses the same arrival-order `S1`/`S2`/`S3`/...
scheme. Existing SM-protocol parsers work unmodified.

The `orchestrator_version` field in `RecognitionStarted` is the only
honest-distinction marker — it reads `2026.06.01+depodash-asr+0.1.0`
instead of SM's version string. Don't assert on it unless you specifically
want to detect "is this our container."

### Concurrency defaults

We ship `MAX_CONCURRENT_CONNECTIONS=4` (vs SM's default `1`) since the
underlying engines are lighter on VRAM. Tune via `deploy/.env`. Practical
per-shape concurrency: 1 multitalker session on 8 GB, 2–3 on 16 GB, 4–6 on
24 GB; fallback engine is roughly 2× cheaper per session.

---

## Running the benchmarks

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Stream a fixture through the live server, capture WER / RTF / TTFT
python tools/measure_via_ws.py \
    --case ami-es2004a-5min \
    --url ws://127.0.0.1:9000/v2 \
    --engine auto-multispeaker --diarizer passthrough \
    --max-speakers 4 --tag baseline

# Score diarization
python tools/score_der.py \
    results/auto-multispeaker+passthrough/ami-es2004a-5min.json \
    cases/ami-es2004a-5min/reference.rttm
```

Results land in `results/<engine>+<diarizer>/<case>.json` and the
canonical aggregate table is `results/SCOREBOARD.md`.

---

## Running the live demo

```bash
HF_TOKEN=... python scripts/demo/control.py
# open http://127.0.0.1:9100/
```

In production the demo runs at `https://depodash-lab.acmeplexus.com/`
behind basic auth.

---

## Repo layout

```
asr-benchmarks/
├── README.md                  this file
├── deploy/                    Docker container packaging
│   ├── Dockerfile             NVIDIA PyTorch base, CUDA 12.6
│   ├── docker-compose.yml     GPU passthrough + hf-cache volume
│   ├── .env.example           every config knob documented
│   ├── NOTICES                third-party model attributions
│   └── requirements.txt       pinned Python deps
├── engines/                   ASR adapters (one file per engine)
├── diarizers/                 speaker-labeling adapters
├── server/                    FastAPI + WebSocket layer + registries
│   └── punctuators/           punctuation post-process
├── tools/                     measure_via_ws.py, score_der.py, sweep_nemotron.py
├── scripts/demo/              live browser demo (depodash-lab.acmeplexus.com)
├── cases/                     benchmark fixtures (LibriSpeech, SCOTUS, AMI, depo)
└── results/
    └── SCOREBOARD.md          measured numbers — read this for the full data
```
