"""
Run a single (engine, case) combo end-to-end.

Pipeline: load case → start resource sampler → transcribe via engine →
stop sampler → score (WER, latency, DER if multi-speaker) → write run JSON.

Engines and cases are discovered by name from filesystem. New engines:
drop a file in `engines/`. New cases: drop a folder in `cases/`.

Usage:
    python -m runner --engine speechmatics_onprem --case scotus-glossip-v-oklahoma
    python -m runner --engine speechmatics_onprem --case deposition-greg-erwin --fast
    python -m runner --engine speechmatics_onprem --case scotus-glossip-v-oklahoma --concurrency-ramp 1,2,4
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from diarizers._base import Diarizer, overlay_speakers
from engines._base import Engine, Result
from metrics.accuracy import score_confidence, score_entity_preservation, score_wer
from metrics.diarization import score_der, words_to_rttm_segments
from metrics.latency import score_latency
from samplers.resources import ResourceSampler


REPO_ROOT = Path(__file__).resolve().parent
CASES_DIR = REPO_ROOT / "cases"
RESULTS_DIR = REPO_ROOT / "results" / "runs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_engine(name: str) -> Engine:
    mod = importlib.import_module(f"engines.{name}")
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and getattr(obj, "name", None) == name:
            return obj()
    raise ImportError(f"engines/{name}.py does not expose an Engine class with .name == '{name}'")


def load_diarizer(name: str) -> Diarizer:
    mod = importlib.import_module(f"diarizers.{name}")
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and getattr(obj, "name", None) == name:
            return obj()
    raise ImportError(f"diarizers/{name}.py does not expose a Diarizer class with .name == '{name}'")


def load_case(name: str) -> dict[str, Any]:
    case_dir = CASES_DIR / name
    if not case_dir.is_dir():
        raise FileNotFoundError(f"case {name!r} not found at {case_dir}")
    meta_path = case_dir / "case.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing")
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    meta["dir"] = case_dir
    # Audio: prefer audio.wav (ready to stream); fall back to anything else and convert lazily.
    wav = case_dir / "audio.wav"
    if not wav.exists():
        # Try to convert from any .mp3/.m4a/.flac/.ogg in the dir
        for ext in ("mp3", "m4a", "flac", "ogg"):
            src = case_dir / f"audio.{ext}"
            if src.exists():
                _ffmpeg_to_wav(src, wav)
                break
        else:
            raise FileNotFoundError(f"no audio.* found in {case_dir}")
    meta["audio_wav"] = wav
    return meta


def _ffmpeg_to_wav(src: Path, dst: Path) -> None:
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not on PATH — install with `sudo apt install ffmpeg`")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            str(dst),
        ],
        check=True,
    )


def _result_to_jsonable(result: Result) -> dict[str, Any]:
    """Strip non-JSON fields and shrink the dump for storage."""
    return {
        "engine": result.engine,
        "audio_path": str(result.audio_path),
        "config": result.config,
        "transcript": result.transcript,
        "words": [
            {
                "content": w.content,
                "start_time": w.start_time,
                "end_time": w.end_time,
                "speaker": w.speaker,
                "confidence": w.confidence,
                "is_punctuation": w.is_punctuation,
            }
            for w in result.words
        ],
        "metadata": {
            "audio_duration_s": result.metadata.audio_duration_s,
            "wall_clock_s": result.metadata.wall_clock_end - result.metadata.wall_clock_start,
            "first_audio_send": result.metadata.first_audio_send,
            "first_final_recv": result.metadata.first_final_recv,
            "first_partial_recv": result.metadata.first_partial_recv,
        },
        "raw_dump": result.raw_dump,
        "error": result.error,
    }


async def run_one(engine_name: str, case_name: str, fast: bool, transcription_overlay: dict | None, tag: str | None = None, diarizer_name: str = "native") -> dict[str, Any]:
    engine = load_engine(engine_name)
    case = load_case(case_name)
    diarizer = load_diarizer(diarizer_name)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    tag_suffix = f"__{tag}" if tag else ""
    diar_suffix = f"__diar-{diarizer_name}" if diarizer_name != "native" else ""
    run_id = f"{timestamp}__{engine_name}__{case_name}{diar_suffix}{tag_suffix}"
    run_path = RESULTS_DIR / f"{run_id}.json"
    resources_path = RESULTS_DIR / f"{run_id}__resources.jsonl"

    config: dict[str, Any] = {"fast": fast}
    if transcription_overlay:
        config["transcription_config"] = transcription_overlay

    print(f"[run] {run_id}")
    print(f"      audio: {case['audio_wav']}")
    print(f"      duration: {case.get('duration_s', '?')}s")

    # If a non-native diarizer is being used, force the engine to NOT diarize
    # (avoid wasting compute and avoid the engine's labels colliding with the
    # diarizer's labels in the overlay step).
    if diarizer_name not in ("native",) and "transcription_config" not in config:
        config["transcription_config"] = {}
    if diarizer_name not in ("native",):
        # Most engines won't have a transcription_config; this only matters for
        # speechmatics-style. Setting diarization=none is a hint, not an error.
        tc = config.setdefault("transcription_config", {})
        tc.setdefault("diarization", "none")

    # Start resource sampler in a thread alongside the async engine call
    containers = case.get("monitor_containers", ["sm-triton", "sm-rt-transcriber"])
    with ResourceSampler(resources_path, containers=containers, interval_s=1.0) as sampler:
        result = await engine.transcribe(case["audio_wav"], config=config)
        # Run the diarizer (sequentially — these tests don't need overlap).
        # NativeDiarizer / NoneDiarizer return [] and are no-ops.
        diar_segments = await diarizer.diarize(case["audio_wav"])
    res_summary = sampler.summary(skip_initial_seconds=10.0)

    # If the diarizer produced segments, overlay them onto the engine words.
    if diar_segments:
        overlay_speakers(result.words, diar_segments)

    # Score
    scores: dict[str, Any] = {}
    scores["diarizer"] = diarizer_name
    if result.error:
        scores["error"] = result.error

    reference_path = case["dir"] / "reference.txt"
    if reference_path.exists():
        reference = reference_path.read_text(encoding="utf-8")
        wer = score_wer(reference, result.transcript)
        scores["wer"] = asdict(wer)
        if case.get("key_terms"):
            scores["entity"] = asdict(score_entity_preservation(result.transcript, case["key_terms"]))

    rttm_path = case["dir"] / "reference.rttm"
    if rttm_path.exists():
        hyp_segs = words_to_rttm_segments(result.words)
        try:
            der = score_der(rttm_path, hyp_segs)
            scores["der"] = asdict(der)
        except Exception as e:
            scores["der_error"] = str(e)

    lat = score_latency(result.metadata, fast_mode=fast)
    scores["latency"] = asdict(lat)
    scores["resources"] = asdict(res_summary)
    scores["confidence"] = score_confidence(result.words)

    # Write the full run record
    record = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "engine": engine_name,
        "case": case_name,
        "diarizer": diarizer_name,
        "tag": tag,
        "case_meta": {k: v for k, v in case.items() if k not in ("dir", "audio_wav")},
        "scores": scores,
        "result": _result_to_jsonable(result),
        "resources_path": str(resources_path.relative_to(REPO_ROOT)),
    }
    run_path.write_text(json.dumps(record, indent=2, default=str))

    # Console summary
    print(f"      transcript: {len(result.words)} words")
    if "wer" in scores:
        print(f"      WER: {scores['wer']['wer']*100:.2f}%  CER: {scores['wer']['cer']*100:.2f}%")
        print(f"      S/D/I: {scores['wer']['substitutions']}/{scores['wer']['deletions']}/{scores['wer']['insertions']}")
    if "der" in scores:
        print(f"      DER: {scores['der']['der']*100:.2f}%  speakers ref/hyp: {scores['der']['speaker_count_ref']}/{scores['der']['speaker_count_hyp']}")
    if "entity" in scores:
        e = scores["entity"]
        print(f"      entities: {e['preserved']}/{e['total']} preserved" + (f"  missing: {e['missing']}" if e['missing'] else ""))
    p95 = f"{lat.final_lag_p95_ms:.0f}ms" if lat.final_lag_p95_ms is not None else "n/a (fast mode)"
    print(f"      latency: TTFT={lat.ttft_ms:.0f}ms  per-final p95={p95}  RTF={lat.rtf:.3f}")
    if res_summary.gpu_mem_used_mb_peak:
        print(f"      GPU: peak {res_summary.gpu_mem_used_mb_peak} MiB  util p50 {res_summary.gpu_util_percent_p50}%  temp peak {res_summary.gpu_temp_c_peak}°C")
    print(f"      → {run_path}")

    return record


async def run_concurrency_ramp(engine_name: str, case_name: str, levels: list[int], hold_s: float) -> None:
    print(f"[ramp] engine={engine_name} case={case_name} levels={levels} hold={hold_s}s")
    for n in levels:
        print(f"\n[ramp] === N={n} ===")
        engine_factory = lambda: load_engine(engine_name)
        case = load_case(case_name)

        # Launch N concurrent transcribes of the same audio
        tasks = []
        for i in range(n):
            engine = engine_factory()
            tasks.append(engine.transcribe(case["audio_wav"], config={"fast": False}))

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        run_id = f"{timestamp}__{engine_name}__{case_name}__ramp-N{n}"
        resources_path = RESULTS_DIR / f"{run_id}__resources.jsonl"

        with ResourceSampler(resources_path, containers=case.get("monitor_containers", ["sm-triton", "sm-rt-transcriber"]), interval_s=1.0) as sampler:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        res_summary = sampler.summary(skip_initial_seconds=10.0)

        success = [r for r in results if isinstance(r, Result) and not r.error]
        failed = [r for r in results if not (isinstance(r, Result) and not r.error)]

        ttfts = [score_latency(r.metadata).ttft_ms for r in success if score_latency(r.metadata).ttft_ms is not None]
        rtfs = [score_latency(r.metadata).rtf for r in success]

        record = {
            "run_id": run_id,
            "engine": engine_name,
            "case": case_name,
            "concurrency": n,
            "successes": len(success),
            "failures": len(failed),
            "failure_reasons": [str(f) if not isinstance(f, Result) else f.error for f in failed],
            "ttft_ms_p50": _stat(ttfts, 50),
            "ttft_ms_p95": _stat(ttfts, 95),
            "rtf_p50": _stat(rtfs, 50),
            "rtf_p95": _stat(rtfs, 95),
            "resources": asdict(res_summary),
        }
        (RESULTS_DIR / f"{run_id}.json").write_text(json.dumps(record, indent=2, default=str))
        print(f"[ramp] N={n} success={len(success)} fail={len(failed)} TTFT p95={record['ttft_ms_p95']:.0f}ms RTF p95={record['rtf_p95']:.3f} GPU peak={res_summary.gpu_mem_used_mb_peak} MiB")


def _stat(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k); hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, help="engine module name in engines/")
    p.add_argument("--case", required=True, help="case folder name in cases/")
    p.add_argument("--fast", action="store_true", help="send audio as fast as WS allows (offline accuracy runs)")
    p.add_argument("--tag", help="label this run; appears in SCOREBOARD as a separate row (e.g. 'max_delay_10')")
    p.add_argument("--diarizer", default="native", help="diarizers/ module name (native|none|pyannote|speechmatics_diar). default: native (engine's own labels)")
    p.add_argument("--diarization", choices=["none", "speaker"], help="override transcription_config.diarization (engine-internal flag, separate from --diarizer)")
    p.add_argument("--max-delay", type=float, help="override max_delay (seconds, range 2-20)")
    p.add_argument("--max-delay-mode", choices=["fixed", "flexible"], help="override max_delay_mode")
    p.add_argument("--max-speakers", type=int, help="override speaker_diarization_config.max_speakers (range 2-100)")
    p.add_argument("--enable-partials", action="store_true", help="set enable_partials=true")
    p.add_argument("--operating-point", choices=["enhanced", "standard"], help="override operating_point")
    p.add_argument("--concurrency-ramp", help="comma-separated stream counts, e.g. '1,2,4'")
    p.add_argument("--ramp-hold-s", type=float, default=180.0)
    args = p.parse_args()

    overlay: dict[str, Any] = {}
    if args.diarization:
        overlay["diarization"] = args.diarization
    if args.max_delay is not None:
        overlay["max_delay"] = args.max_delay
    if args.max_delay_mode:
        overlay["max_delay_mode"] = args.max_delay_mode
    if args.max_speakers is not None:
        overlay["speaker_diarization_config"] = {"max_speakers": args.max_speakers}
    if args.enable_partials:
        overlay["enable_partials"] = True
    if args.operating_point:
        overlay["operating_point"] = args.operating_point

    if args.concurrency_ramp:
        levels = [int(x) for x in args.concurrency_ramp.split(",")]
        asyncio.run(run_concurrency_ramp(args.engine, args.case, levels, args.ramp_hold_s))
    else:
        asyncio.run(run_one(args.engine, args.case, args.fast, overlay or None, tag=args.tag, diarizer_name=args.diarizer))


if __name__ == "__main__":
    main()
