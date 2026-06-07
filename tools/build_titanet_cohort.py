"""Build an impostor cohort of TitaNet embeddings for adaptive s-norm.

The TitaNet verifier in engines/multitalker_parakeet.py uses raw cosine
similarity as its same-speaker decision metric. Raw cosine is uncalibrated
across acoustic conditions — two distinct speakers in a shared room/mic
can sit at 0.70–0.78 cosine sim because shared acoustic context dominates
the embedding, not because their voices are similar. The fixed-threshold
approach (EAGER_THRESHOLD = 0.80) is a heuristic patch around this.

The principled fix is adaptive s-norm (Auckenthaler 2000; Cumani 2019;
NIST-SRE 2020 production standard): compute a z-score that normalizes
raw_sim against the distribution of "close-but-impostor" cohort scores.
A fixed z-score threshold is then session-invariant — it adjusts to the
acoustic context automatically.

This script builds the cohort. It walks a set of audio directories,
extracts random 3-second segments, runs TitaNet to produce 192-dim
L2-normalized embeddings, and writes them as a single tensor file
consumed by the verifier at session startup.

Usage:
    python tools/build_titanet_cohort.py \\
        --sources cases/ \\
        --output engines/cohort/titanet_cohort.pt \\
        --target 200
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch


SAMPLE_RATE = 16000
SEGMENT_LEN_S = 3.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_LEN_S)


def find_audio(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        for ext in ("*.wav", "*.flac", "*.mp3"):
            files.extend(Path(root).rglob(ext))
    return files


def load_segment(path: Path, rng: random.Random) -> np.ndarray | None:
    import soundfile as sf
    try:
        audio, sr = sf.read(str(path))
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != SAMPLE_RATE:
        try:
            from scipy.signal import resample_poly
            audio = resample_poly(audio, SAMPLE_RATE, sr).astype(np.float32)
        except Exception:
            return None
    if len(audio) < SEGMENT_SAMPLES + 1:
        return None
    max_start = len(audio) - SEGMENT_SAMPLES
    start = rng.randint(0, max_start)
    seg = audio[start:start + SEGMENT_SAMPLES]
    # Reject near-silence — would produce a degenerate embedding.
    if float(np.sqrt(np.mean(seg ** 2))) < 1e-3:
        return None
    return seg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--segments-per-file", type=int, default=8,
                        help="Max random segments to draw from each source file")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading TitaNet (speakerverification_en_titanet_large)...")
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    titanet = EncDecSpeakerLabelModel.from_pretrained(
        "nvidia/speakerverification_en_titanet_large"
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    titanet = titanet.to(device)

    print(f"Scanning {[str(p) for p in args.sources]}...")
    files = find_audio(args.sources)
    rng.shuffle(files)
    print(f"Found {len(files)} audio files")

    cohort: list[torch.Tensor] = []
    source_log: list[str] = []

    # Round-robin draw multiple segments per file so a single long file
    # doesn't dominate the cohort.
    target = args.target
    for file_idx, path in enumerate(files):
        if len(cohort) >= target:
            break
        for _ in range(args.segments_per_file):
            if len(cohort) >= target:
                break
            seg = load_segment(path, rng)
            if seg is None:
                break
            sig = torch.from_numpy(seg).unsqueeze(0).to(device)
            lens = torch.tensor([sig.shape[-1]], device=device)
            try:
                with torch.inference_mode():
                    _, emb = titanet.forward(
                        input_signal=sig, input_signal_length=lens
                    )
            except Exception as e:
                print(f"  skip {path.name}: {e}")
                break
            emb = emb.detach().cpu().flatten().to(torch.float32)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            cohort.append(emb)
            source_log.append(f"{path}")
        if (file_idx + 1) % 5 == 0:
            print(f"  ... {len(cohort)}/{target} (after {file_idx + 1} files)")

    if not cohort:
        raise RuntimeError(f"No valid audio segments extracted from {args.sources}")

    cohort_tensor = torch.stack(cohort)  # (N, 192)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": cohort_tensor,
            "sources": source_log,
            "model": "nvidia/speakerverification_en_titanet_large",
            "segment_len_s": SEGMENT_LEN_S,
            "sample_rate": SAMPLE_RATE,
        },
        args.output,
    )
    print(f"Saved cohort {tuple(cohort_tensor.shape)} -> {args.output}")
    print(f"  unique source files: {len(set(source_log))}")

    # Quick diagnostic: pairwise sim distribution
    sims = cohort_tensor @ cohort_tensor.T
    off_diag = sims[~torch.eye(sims.shape[0], dtype=torch.bool)]
    print(f"  off-diagonal pairwise sim: "
          f"mean={off_diag.mean().item():.3f} "
          f"std={off_diag.std().item():.3f} "
          f"p95={off_diag.quantile(0.95).item():.3f} "
          f"max={off_diag.max().item():.3f}")


if __name__ == "__main__":
    main()
