#!/usr/bin/env python3
"""Drive a Speechmatics-protocol WebSocket server with a benchmark case.

Streams `cases/<case>/audio.wav` at 1× realtime to a target URL with
DepoDash's production transcription_config, captures every `AddTranscript`
frame, and writes a `results/archive/<timestamp>__<tag>.json` shaped like
the existing run JSONs so the scoreboard renderer / metrics modules can
consume it.

Usage:
    python tools/measure_via_ws.py \\
        --case scotus-glossip-v-oklahoma \\
        --url ws://localhost:9000/v2 \\
        --engine parakeet-nemo --diarizer streaming-sortformer

Engine + diarizer in the JSON are labels — set them to match whatever the
target server has `ENGINE` / `DIARIZER` set to. They don't change behavior
here; they just get written into the output for the scoreboard.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import websockets


SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_BYTES = (SAMPLE_RATE * 2 * CHUNK_MS) // 1000


def load_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, SAMPLE_RATE):
            sys.exit(f"audio must be mono PCM S16LE @ {SAMPLE_RATE} Hz: {path}")
        return w.readframes(w.getnframes())


def build_config(diarization: str, max_speakers: int) -> dict:
    cfg = {
        "language": "en",
        "operating_point": "enhanced",
        "diarization": diarization,
        "enable_partials": False,
        "max_delay": 5.0 if diarization != "none" else 3.0,
        "max_delay_mode": "flexible",
        "output_locale": "en-US",
        "enable_entities": False,
        "punctuation_overrides": {"permitted_marks": [",", ".", "?"]},
    }
    # Always send max_speakers so engine-routing decisions (e.g., the
    # auto-multispeaker meta-engine that picks multitalker vs nemotron-native
    # based on this number) work even when the client requested diarization
    # to be off on the wire — they still need to declare the session shape.
    cfg["speaker_diarization_config"] = {"max_speakers": max_speakers}
    return cfg


async def run(url: str, audio_path: Path, config: dict) -> dict:
    pcm = load_pcm(audio_path)
    audio_duration = len(pcm) / (SAMPLE_RATE * 2)

    wall_start = time.monotonic()
    first_partial = None
    first_final = None
    frames = []
    partials = []
    finals = []

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({
            "message": "StartRecognition",
            "audio_format": {"type": "raw", "encoding": "pcm_s16le", "sample_rate": SAMPLE_RATE},
            "transcription_config": config,
        }))

        async def reader():
            nonlocal first_partial, first_final
            async for raw in ws:
                msg = json.loads(raw)
                kind = msg.get("message")
                t = time.monotonic() - wall_start
                if kind == "AddPartialTranscript":
                    if first_partial is None:
                        first_partial = t
                    partials.append({"t": t, "transcript": msg.get("metadata", {}).get("transcript", "")})
                elif kind == "AddTranscript":
                    if first_final is None:
                        first_final = t
                    finals.append({
                        "t": t,
                        "transcript": msg.get("metadata", {}).get("transcript", ""),
                        "results": msg.get("results", []),
                    })
                elif kind == "EndOfTranscript":
                    return
                elif kind == "Error":
                    print(f"server Error: {msg}", file=sys.stderr)
                    return

        reader_task = asyncio.create_task(reader())

        for i in range(0, len(pcm), CHUNK_BYTES):
            await ws.send(pcm[i : i + CHUNK_BYTES])
            await asyncio.sleep(CHUNK_MS / 1000)

        await ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": len(pcm) // CHUNK_BYTES}))
        await reader_task

    wall_end = time.monotonic()
    return {
        "audio_duration_s": audio_duration,
        "wall_clock_s": wall_end - wall_start,
        "ttft_s": first_partial if first_partial is not None else first_final,
        "first_final_s": first_final,
        "rtf": (wall_end - wall_start) / audio_duration if audio_duration else None,
        "partial_count": len(partials),
        "final_count": len(finals),
        "transcript": " ".join(f["transcript"] for f in finals if f["transcript"]),
        "finals": finals,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--url", default="ws://localhost:9000/v2")
    p.add_argument("--engine", default="unknown", help="label for the output JSON")
    p.add_argument("--diarizer", default="unknown", help="label for the output JSON")
    p.add_argument("--diarization-mode", choices=["none", "speaker", "channel"], default="speaker")
    p.add_argument("--max-speakers", type=int, default=10)
    p.add_argument("--tag", default="")
    p.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="repo root (defaults to two levels up from this script)",
    )
    args = p.parse_args()

    repo_root = Path(args.repo_root)
    case_dir = repo_root / "cases" / args.case
    audio_path = case_dir / "audio.wav"
    if not audio_path.exists():
        sys.exit(f"audio not found: {audio_path}")

    config = build_config(args.diarization_mode, args.max_speakers)
    print(f"streaming {audio_path} → {args.url}")
    print(f"config: {json.dumps(config)}")

    result = asyncio.run(run(args.url, audio_path, config))

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    diar_part = f"__diar-{args.diarizer}" if args.diarizer not in ("native", "unknown") else ""
    tag_part = f"__{args.tag}" if args.tag else ""
    fname = f"{timestamp}__{args.engine}__{args.case}{diar_part}{tag_part}.json"

    out_dir = repo_root / "results" / "archive"
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = case_dir / "reference.txt"
    wer = None
    if ref_path.exists() and result.get("transcript"):
        try:
            import jiwer
            from whisper.normalizers import EnglishTextNormalizer
            norm = EnglishTextNormalizer()
            ref_norm = norm(ref_path.read_text())
            hyp_norm = norm(result["transcript"])
            wer = jiwer.wer(ref_norm, hyp_norm) if ref_norm.strip() else None
        except Exception as e:
            print(f"(WER scoring unavailable: {e})")

    out_path = out_dir / fname
    out_path.write_text(json.dumps({
        "engine": args.engine,
        "diarizer": args.diarizer,
        "case": args.case,
        "tag": args.tag,
        "url": args.url,
        "config": config,
        "measured": result,
        "wer": wer,
    }, indent=2))

    combo_dir = repo_root / "results" / f"{args.engine}+{args.diarizer}"
    combo_dir.mkdir(parents=True, exist_ok=True)
    (combo_dir / f"{args.case}.json").write_text(out_path.read_text())

    print(f"\nwrote: {out_path}")
    print(f"       {combo_dir / f'{args.case}.json'}")
    print(f"audio: {result['audio_duration_s']:.2f}s  rtf: {result['rtf']:.3f}  partials: {result['partial_count']}  finals: {result['final_count']}")
    if result["ttft_s"] is not None:
        print(f"ttft:  {result['ttft_s']*1000:.0f} ms")
    if wer is not None:
        print(f"WER:   {wer*100:.2f}%")


if __name__ == "__main__":
    main()
