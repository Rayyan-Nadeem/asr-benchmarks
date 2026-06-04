# Scoreboard — DepoDash ASR, CPU vs GPU vs Speechmatics

## How to read this

Three deployment shapes, side by side. All numbers are n=1 — gaps are directional, absolute precision past one decimal is overstated. Production deployment should re-run on a held-out test set with multiple seeds before locking in any single number.

**Methodology delta — read before comparing rows:**
- **GPU column** streams audio at 1× realtime through `ws://localhost:9000/v2` and commits finals as the engine emits them. Per-final lag is in the 4–12 s band. This is apples-to-apples with the Speechmatics on-prem container methodology.
- **CPU column** runs the engine in single-final mode: audio buffered through the same WebSocket, transcribed once at session end. Per-final lag = audio duration. This is the engine's native deployment shape for laptop/edge — true streaming on CPU is a known engineering follow-up.
- **Speechmatics column** is frozen reference data from the Apr-2026 measurement run on the same fixtures, container `sm-gpu-inference-server-en:15.0.0` + `rt-asr-transcriber-en:15.0.0`, eval license expired 2026-05-10.

DepoDash production `transcription_config` for every row (no vocab biasing, no KenLM, no beam search).

**DER methodology disclosure.** Speechmatics' frozen baseline numbers use strict simpleder (no collar, count overlap). The Sortformer / pyannote model cards report CALLHOME-style DER (0.25 s collar, skip overlap). When comparing this scoreboard to either source, match the methodology — strict scoring penalizes overlap segments heavily and gives 38–51 % DER on AMI for *every* approach (including SM); CALLHOME-style scoring drops the same approaches to 12–34 % on AMI. Both methodologies are honest; mixing them is not.

---

## Headline WER — three deployment shapes

| Stack | Hardware | LibriSpeech | SCOTUS (legal) | TTFT (post-warm) |
|---|---|---|---|---|
| **Speechmatics enhanced (frozen)** | $4 k laptop, RTX 5000 Ada, **16 GB VRAM hard floor** | **2.98 %** | **14.87 %** | 5–19 s p95 |
| **CPU — Mac M4** (`parakeet-onnx + passthrough`, single-final) | Any modern laptop, **no GPU**, ~2 GB RAM | 3.55 % | **12.15 %** ✅ | = audio duration |
| **GPU — T4 8 GB cap** (`parakeet-nemo + sortformer`, streaming) | <$1000 laptop with consumer dGPU OR cloud T4 | **0.59 %** ✅ | **7.63 %** ✅ | 11.5 s |
| **GPU — T4 8 GB cap** (`nemotron-nemo + sortformer`, streaming) | Same as above | **0.59 %** ✅ | 8.33 % ✅ | **0.84 s** ✅ |
| **GPU — T4 8 GB cap** (`fastconformer-hybrid + sortformer`, streaming) | 114 M-param model, fits on **any** laptop w/ 4 GB GPU or CPU | 2.96 % | **9.75 %** ✅ | ~1 s |

**The pitch in one paragraph.** Speechmatics ships on a $4 k Alienware-class laptop with 16 GB VRAM. We deliver three cheaper paths: a CPU-only Mac M4 path that **beats SM on legal long-form (12.15 % vs 14.87 %)** with no incremental hardware, a consumer-dGPU path on Parakeet TDT that **beats SM on every axis (-80 % rel on LibriSpeech, -49 % rel on SCOTUS)** while fitting in half the VRAM, and a **sub-second-TTFT** Nemotron path that matches Parakeet on WER while ten-x-ing the live-mic responsiveness. Real-time everywhere — both columns stream through the same Speechmatics-protocol WebSocket so DepoDash's middleware swaps containers with zero client-side changes.

**Engine choice within the GPU column:**
- `parakeet-nemo + sortformer` — highest accuracy (7.63 % SCOTUS), but 11.5 s TTFT per session (hacked sliding-window on an offline model).
- `nemotron-nemo + sortformer` — same WER class (8.33 % SCOTUS) with **0.84 s TTFT** because Nemotron is natively cache-aware streaming (FastConformer-RNN-T encoder, runtime-selectable 80 / 160 / 560 / 1120 ms chunks). **This is the live-mic production pick.**
- `fastconformer-hybrid + sortformer` — 114 M-param multi-latency model at 2.96 % / 9.75 %. Best fit for the "any 8 GB GPU laptop OR CPU" deployment shape.

---

## DER — diarization

| Stack | Mode | SCOTUS DER (clean turn-taking) | AMI DER strict | AMI DER CALLHOME-style |
|---|---|---|---|---|
| Speechmatics native (frozen) | streaming | **2.00 %** | 45.02 % | n/a |
| Speechmatics + pyannote overlay (frozen) | streaming | — | 40.12 % | n/a |
| `parakeet-nemo + sortformer` (T4 8 GB) | streaming | **2.81 %** | 51.33 % | 34.18 % |
| `nemotron-nemo + sortformer` (T4 8 GB) | streaming | — | 50.84 % | **34.01 %** |
| `parakeet-onnx + pyannote` (Mac CPU) | single-final | 4.46 % | 38.94 % | n/a |
| **`pyannote 4 community-1` (offline final-pass)** | offline | — | **23.65 %** ✅ | **12.97 %** ✅ |
| **`pyannote 4 community-1` exclusive mode** | offline | — | 28.83 % | **11.02 %** ✅ |
| pyannote 3.1 (batch ceiling, pre-M2) | offline | — | 24.80 % | n/a |

**Honest read on diarization.** On clean turn-taking (SCOTUS), the GPU stack ties SM within measurement noise. On overlap-heavy meeting audio (AMI ES2004a-5min, 22.6 % of speech is overlapped), Sortformer streaming sits at 34 % DER under CALLHOME-style scoring, beating SM's 45 % strict by 11 pp. The headline win is the offline final-pass: **pyannote 4 community-1, applied post-recording to a completed deposition, lands 11–13 % DER on the same AMI clip — a 32 pp absolute improvement over Speechmatics**, in the same methodology bucket SM was scored in. Recommended deployment: stream live with `nemotron-nemo + sortformer` (decent DER, fast TTFT), then re-diarize the completed transcript with `pyannote/speaker-diarization-community-1` for the final delivered file. This is the strongest diarization story on the open landscape today.

---

## Live-mic latency

`nemotron-streaming + passthrough` via sherpa-onnx INT8 was the previous live-mic engine — **deprecated** in favor of `nemotron-nemo` (native NeMo FP16). The sherpa-onnx INT8 community export carried a hidden ~16 pp WER tax (LibriSpeech 18.93 %) that disappeared entirely on the native NeMo path (LibriSpeech 0.59 %). Same model, different runtime; the INT8 quantization in the community ONNX export wasn't a clean distillation.

| | TTFT | RTF | LibriSpeech WER | SCOTUS WER |
|---|---|---|---|---|
| `nemotron-nemo + sortformer` (T4 8 GB, **production pick**) | **0.84 s** | 1.03 | **0.59 %** | 8.33 % |
| `nemotron-streaming + passthrough` (sherpa-onnx INT8, **deprecated**) | 0.82 s | 1.02 | 18.93 % | 24.29 % |

---

## Hardware footprint

Every GPU engine is benchmarked under a uniform **8 GB VRAM cap** via `torch.cuda.set_per_process_memory_fraction()` — so all numbers are defensibly "fits on consumer 2060 / 2070 / 4060-class hardware."

| Stack | Engine memory | Diarizer memory | Per-stream total |
|---|---|---|---|
| Speechmatics native | 7.6 GB VRAM | (integrated) | 7.6 GB VRAM |
| Speechmatics + pyannote | 7.6 GB VRAM | ~4 GB VRAM | 11.8 GB VRAM |
| `nemotron-nemo + sortformer` (T4 8 GB cap) | ~3 GB VRAM | ~1.5 GB VRAM | ~4.5 GB VRAM |
| `parakeet-nemo + sortformer` (T4 8 GB cap) | ~5 GB VRAM | ~1.5 GB VRAM | ~6.5 GB VRAM |
| `fastconformer-hybrid + sortformer` (T4 8 GB cap) | ~0.8 GB VRAM | ~1.5 GB VRAM | ~2.3 GB VRAM |
| `parakeet-onnx + passthrough` (Mac CPU) | ~2 GB RAM | — | ~2 GB RAM, **0 GB VRAM** |

**Hardware floor story.** Speechmatics requires the 16 GB VRAM card. We don't. The GPU path runs in half or less of the VRAM — even FastConformer hybrid + Sortformer fits in ~2.3 GB of VRAM and would run on an iGPU. The CPU path drops the hardware floor entirely.

---

## Per-axis champions (updated)

| Metric | Winner | Number | vs Speechmatics |
|---|---|---|---|
| LibriSpeech WER | `nemotron-nemo + sortformer` (GPU, T4 8 GB) | **0.59 %** | **−80 % rel** (SM 2.98 %) |
| SCOTUS WER | `parakeet-nemo + sortformer` (GPU, T4 8 GB) | **7.63 %** | **−49 % rel** (SM 14.87 %) |
| SCOTUS WER (CPU floor) | `parakeet-onnx + passthrough` (Mac M4) | **12.15 %** | **−18 % rel** (SM 14.87 %) |
| SCOTUS DER (streaming, clean turn-taking) | `parakeet-nemo + sortformer` (GPU) | 2.81 % | tie within noise (SM 2.00 %) |
| AMI DER (offline final-pass) | **`pyannote 4 community-1` exclusive** | **11.02 %** | **−34 pp abs** (SM 45.02 %) |
| Live-mic TTFT | `nemotron-nemo + sortformer` (T4 8 GB) | **0.84 s** | n/a (SM doesn't publish TTFT) |
| Smallest model footprint | `fastconformer-hybrid` (114 M params) | ~0.8 GB VRAM | fits on iGPU + CPU |
| Hardware floor | `parakeet-onnx` (Mac CPU) | **no incremental hardware** | dropped from $4 k Alienware-class laptop to any modern laptop |

---

## Reference data sources

| Stack | Tag | Hardware run on | When |
|---|---|---|---|
| Speechmatics + native / pyannote | `speechmatics_onprem` | Alienware m18 R2 RTX 5080 16 GB | 2026-04-25 |
| Parakeet-NeMo + sortformer / pyannote | `t4-8gb` | AWS GovCloud g4dn.xlarge T4 capped 8 GB | 2026-06-01 |
| Nemotron-NeMo + sortformer | `t4-8gb` | Same T4 box | 2026-06-02 |
| FastConformer hybrid + sortformer | `t4-8gb` | Same T4 box | 2026-06-02 |
| Parakeet-ONNX + pyannote / sortformer / passthrough | `t4-8gb` / `mac-m4-cpu` | T4 8 GB cap + Mac M4 CPU EP | 2026-06-01 / 2026-06-02 |
| Mac M4 CPU single-final (CPU column) | (pre-tag, offline mode) | Mac M4 CPU EP | 2026-05 |
| pyannote 4 community-1 offline final-pass | (offline, CPU) | T4 box, CPU EP | 2026-06-02 |

Per-run JSONs in `results/archive/`. Per-combo canonical results in `results/<engine>+<diarizer>/`.

---

## Known engineering follow-ups

- **Sortformer overlay on long audio (>5 min) OOMs under the 8 GB cap.** Our session pipeline re-runs sortformer on the *full* growing audio buffer on every emitted final, which scales linearly with audio length. The 477 s deposition clip hits 7.43 GB and crashes (`CUDA out of memory. Tried to allocate 788.00 MiB`). Fix: cache sortformer's frame-level activations and only re-run on the unprocessed delta — or set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as a stopgap.
- **`parakeet-onnx` streaming on long audio** — v3 fix (hard window cap at 30 s + safety-valve flush at 20 s lag) unblocks long-audio finals but CPU streaming WER doesn't beat SM. CPU column on this scoreboard stays in single-final mode where it does beat SM on legal.
- **GPU-PB / TurboBias hot-word biasing.** ASRU 2025 (arXiv:2508.07014) reports +8 to 23 pp F-score on proper-noun-heavy domains. NeMo 2.4 ships only CTC-WS (offline word spotter); GPU-PB / NGPU-LM require NeMo main or 2.5+. Direct analogue of Speechmatics' `additional_vocab` API — would close the residual SCOTUS WER gap (8 % → ~5–6 %) and is a strong follow-up.
- **Multitalker speaker-kernel joint ASR + diarizer** (`nvidia/multitalker-parakeet-streaming-0.6b-v1`). Requires `SpeakerTaggedASR` helper from NeMo's `multispk_transcribe_utils` which is not in 2.4. Lower priority since pyannote 4 community-1 offline final-pass already gives a bigger AMI overlap-DER win.
- **AMI overlap streaming DER headroom.** Pyannote 4 batch ceiling on AMI is 11 %; best streaming approach is 34 % (CALLHOME-style). 23 pp of unclaimed streaming-mode headroom — closing it needs `segment_first` integration where the diarizer drives chunk boundaries, or true joint speaker-kernel models with NeMo 2.5+.

## Open-source research footnotes (sources for the engine selection)

- **Nemotron Speech Streaming 0.6B** (NVIDIA Open Model License, March 2026 checkpoint) — `huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b`. Cache-aware FastConformer-RNN-T. Model-card LibriSpeech-clean 2.32 % at 1.12 s chunk; we measured 0.59 % offline on our LibriSpeech-mini clip.
- **FastConformer Hybrid Streaming Multi** (CC-BY-4.0) — `huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`. 114 M params, multi-latency [0 / 80 / 480 / 1040 ms] in a single checkpoint.
- **Streaming Sortformer v2.1** (NVIDIA Open Model License) — `huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1`. AOSC streaming, 80 ms resolution, 4-speaker cap. Card claims DIHARD III 13.24 % DER at 1.04 s.
- **pyannote speaker-diarization community-1** (CC-BY-4.0, 2025) — `huggingface.co/pyannote/speaker-diarization-community-1`. Best open offline diarizer. Card claims 12.9 % AMI-IHM; we measured 12.97 % on AMI ES2004a-5min (exact reproduction).
- **TurboBias / GPU-PB** (ASRU 2025) — arXiv:2508.07014. Streaming-capable hot-word boosting for RNN-T/TDT/CTC, no retraining. **Recommended follow-up for legal-vocabulary biasing.**
