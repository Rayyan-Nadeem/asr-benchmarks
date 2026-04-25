# asr-benchmarks

Vendor-agnostic benchmark harness for real-time ASR (automatic speech
recognition) engines. Takes any audio case + any engine and produces a
comparable scorecard across **accuracy**, **latency**, **resource cost**, and
**concurrency**. Same audio in → same metrics out, regardless of which engine
is under test.

Built for the DepoDash courtroom-transcription engine selection. Day one runs
Speechmatics on-prem. Future engines (Parakeet, Whisper, AssemblyAI cloud,
Speechmatics cloud) drop in as new files in `engines/`; the rest of the harness
doesn't care.

## What it measures

| Category | Metric | Library |
| --- | --- | --- |
| Accuracy | WER, CER, S/D/I breakdown | `jiwer` |
| Accuracy | Entity preservation (proper nouns, dates, case numbers) | custom |
| Diarization | DER, JER | `simpleder` |
| Composite | cpWER, WDER ("who said what") | `meeteval` |
| Latency | TTFT, per-final lag, RTF | hand-rolled |
| Resources | GPU VRAM peak/steady, GPU util %, container CPU %, RAM | `nvidia-smi` + `docker stats` + `psutil` |
| Concurrency | Max stable streams, latency degradation curve | `asyncio.gather` ramp |

Quantitative degradation thresholds are defined upfront in `METHODOLOGY.md`,
not post-hoc — methodology is auditable.

## Quick start

```bash
# 1. install deps
pip install --user --break-system-packages -r requirements.txt

# 2. ensure target engine is reachable. For Speechmatics on-prem:
#    docker compose up -d in the speechmaticsContainer repo
curl -fsS http://localhost:8001/ready

# 3. run a single case
python -m runner --engine speechmatics_onprem --case librispeech-test-clean-mini

# 4. concurrency ramp
python -m runner --engine speechmatics_onprem --case scotus-glossip-v-oklahoma --concurrency-ramp 1,2,4

# 5. regenerate scoreboard
python tools/render_scoreboard.py
```

Results land in `results/runs/<timestamp>__<engine>__<case>.json`.
`SCOREBOARD.md` is regenerated from the JSONL run files.

## Repo layout

```
.
├── README.md                # this file
├── METHODOLOGY.md           # measurement choices + quantitative thresholds
├── requirements.txt         # Python deps
├── engines/                 # one file per engine, common interface
├── cases/                   # one folder per audio case (audio + reference + meta)
├── metrics/                 # accuracy, diarization, latency, composite
├── samplers/                # background resource sampling
├── tools/                   # PDF→reference parser, scoreboard renderer
├── runner.py                # CLI entry point
└── results/
    ├── runs/                # one JSON per (engine, case, run)
    └── SCOREBOARD.md        # auto-rendered comparison table
```

## Adding a new engine

Implement the protocol in `engines/_base.py`:

```python
class Engine(Protocol):
    name: str
    capabilities: Capabilities

    async def transcribe(self, audio: Path, config: dict) -> Result: ...
```

Save as `engines/<name>.py`. The runner picks it up via `--engine <name>`.

## Adding a new audio case

```
cases/<case-name>/
├── case.yaml        # metadata: speakers, duration, license, source URL
├── audio.<ext>      # the audio file (committed)
├── reference.txt    # ground-truth transcript (if available)
├── reference.rttm   # speaker timeline (if multi-speaker, optional)
└── fetch.sh         # script that produced audio.* (reproducibility)
```

See `cases/README.md` for the full convention and `cases/CHOICES.md` for what
was considered + why each candidate was kept or skipped.

## Status

Phase 0 — Speechmatics on-prem adapter only. Multi-engine comparisons land in
Phase 2 (when the i7 box arrives and Parakeet/Whisper adapters are added).
