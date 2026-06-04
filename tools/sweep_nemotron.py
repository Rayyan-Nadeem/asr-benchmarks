"""Tune nemotron-native streaming WER toward the SM baseline.

Runs measure_via_ws.py across a grid of:
    - left attention context (NEMOTRON_LEFT_CTX)
    - right attention context (NEMOTRON_RIGHT_CTX)
    - beam_size (NEMOTRON_BEAM_SIZE)
    - optional KenLM legal-vocab path (NEMOTRON_KENLM_PATH)

For each cell, runs the LS-clean + SCOTUS + deposition fixtures, captures
WER + RTF + TTFT, and writes a summary CSV. The orchestrator gets
restarted with new env between cells so the engine reloads with the
swept config. Each cell takes ~10-20 minutes wall-clock to run all three
fixtures; budget hours not minutes for a full sweep.

Usage:
    # quick triage sweep (8 cells, ~2 h)
    python3 tools/sweep_nemotron.py --preset quick

    # full grid (~24 cells, ~6 h)
    python3 tools/sweep_nemotron.py --preset full

    # focused — beam-size only, fixed context
    python3 tools/sweep_nemotron.py --beam 1 2 4 8 --left 70 --right 13

    # KenLM legal vocab (requires --kenlm pointing at a .arpa/.bin file)
    python3 tools/sweep_nemotron.py --kenlm /opt/kenlm/legal.bin \
                                   --beam 4 --left 70 --right 13

Must run on the box (orchestrator restart depends on `sudo systemctl
restart orchestrator`). The sweep harness writes its CSV to
results/sweep/nemotron_native_<timestamp>.csv.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "sweep"
FIXTURES = ["librispeech-test-clean-mini", "scotus-glossip-v-oklahoma", "deposition-greg-erwin"]

PRESETS = {
    "quick": dict(left=[70], right=[13, 6], beam=[1, 4], kenlm=[None]),
    "full":  dict(left=[35, 70, 105], right=[1, 6, 13], beam=[1, 4, 8], kenlm=[None]),
    "att":   dict(left=[35, 70, 105], right=[1, 6, 13, 26], beam=[1], kenlm=[None]),
    "beam":  dict(left=[70], right=[13], beam=[1, 2, 4, 8, 16], kenlm=[None]),
}


def _systemd_restart_orchestrator() -> None:
    """Restart the orchestrator so it picks up new NEMOTRON_* env vars."""
    subprocess.run(
        ["sudo", "systemctl", "restart", "orchestrator"], check=True
    )
    # Wait for ready (model load is slow on first boot but fast after).
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            import urllib.request
            urllib.request.urlopen("http://127.0.0.1:9000/ready", timeout=2)
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError("orchestrator did not come up after 180s")


def _set_env_for_orchestrator(env_overrides: dict[str, str]) -> None:
    """Write NEMOTRON_* env to the systemd override drop-in so the
    orchestrator inherits them on restart. Idempotent."""
    override_dir = Path("/etc/systemd/system/orchestrator.service.d")
    override_dir.mkdir(parents=True, exist_ok=True)
    override_file = override_dir / "nemotron-sweep.conf"
    body = ["[Service]"]
    body.extend(f'Environment="{k}={v}"' for k, v in env_overrides.items())
    override_file.write_text("\n".join(body) + "\n")
    subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)


def _switch_to_engine(engine: str, diarizer: str) -> None:
    """Tell the orchestrator to bind a specific engine + diarizer."""
    import urllib.request
    url = f"http://127.0.0.1:9100/switch?engine={engine}&diarizer={diarizer}"
    req = urllib.request.Request(url, method="POST")
    urllib.request.urlopen(req, timeout=15).read()
    # Wait for /ready after switch.
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://127.0.0.1:9000/ready", timeout=2)
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"server didn't come ready after switch to {engine}")


def _run_one_fixture(case: str, tag: str) -> dict:
    """Stream one fixture through the currently-loaded engine."""
    out = subprocess.run(
        [
            "python3", str(REPO / "tools" / "measure_via_ws.py"),
            "--case", case,
            "--url", "ws://127.0.0.1:9000/v2",
            "--engine", "nemotron-native",
            "--diarizer", "sortformer",
            "--diarization-mode", "speaker",
            "--max-speakers", "4",
            "--tag", tag,
        ],
        capture_output=True, text=True, timeout=900,
    )
    # Parse the last 3 lines of stdout for "audio: ... rtf: ..." style
    metrics = {"stdout_tail": out.stdout[-500:], "stderr_tail": out.stderr[-300:]}
    for line in out.stdout.splitlines():
        if "WER:" in line:
            metrics["WER"] = line.split("WER:")[1].strip().rstrip("%")
        if "audio:" in line and "rtf:" in line:
            # Quick parse: "audio: 70.00s  rtf: 1.011  partials: ..."
            try:
                parts = dict(zip(line.replace(":", "").split()[::2],
                                 line.replace(":", "").split()[1::2]))
                metrics["audio_s"] = parts.get("audio", "").rstrip("s")
                metrics["rtf"] = parts.get("rtf", "")
                metrics["finals"] = parts.get("finals", "")
            except Exception:
                pass
        if "ttft:" in line:
            metrics["ttft_ms"] = line.split("ttft:")[1].strip().rstrip(" ms")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="quick")
    ap.add_argument("--left", type=int, nargs="+", help="Override left ctx")
    ap.add_argument("--right", type=int, nargs="+", help="Override right ctx")
    ap.add_argument("--beam", type=int, nargs="+", help="Override beam sizes")
    ap.add_argument("--kenlm", nargs="+", help="Override KenLM paths (or 'none')")
    ap.add_argument("--fixtures", nargs="+", default=FIXTURES)
    args = ap.parse_args()

    grid = dict(PRESETS[args.preset])
    if args.left:  grid["left"]  = args.left
    if args.right: grid["right"] = args.right
    if args.beam:  grid["beam"]  = args.beam
    if args.kenlm: grid["kenlm"] = [None if k == "none" else k for k in args.kenlm]

    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    csv_path = RESULTS / f"nemotron_native_{ts}.csv"

    cells = list(itertools.product(grid["left"], grid["right"], grid["beam"], grid["kenlm"]))
    print(f"sweep: {len(cells)} cells × {len(args.fixtures)} fixtures "
          f"= {len(cells) * len(args.fixtures)} runs", file=sys.stderr)
    print(f"output: {csv_path}", file=sys.stderr)

    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["left_ctx", "right_ctx", "beam_size", "kenlm",
                    "case", "WER", "RTF", "TTFT_ms", "finals", "tag"])

        for left, right, beam, kenlm in cells:
            tag_base = f"sweep_L{left}_R{right}_B{beam}"
            if kenlm:
                tag_base += "_kenlm"
            print(f"\n=== {tag_base} ===", file=sys.stderr)
            env = {
                "NEMOTRON_LEFT_CTX": str(left),
                "NEMOTRON_RIGHT_CTX": str(right),
                "NEMOTRON_BEAM_SIZE": str(beam),
            }
            if kenlm:
                env["NEMOTRON_KENLM_PATH"] = kenlm
            _set_env_for_orchestrator(env)
            _systemd_restart_orchestrator()
            _switch_to_engine("nemotron-native", "sortformer")

            for case in args.fixtures:
                tag = f"{tag_base}_{case}"
                print(f"  {case}...", file=sys.stderr)
                try:
                    m = _run_one_fixture(case, tag)
                except subprocess.TimeoutExpired:
                    m = {"WER": "TIMEOUT"}
                w.writerow([
                    left, right, beam, kenlm or "",
                    case, m.get("WER", ""), m.get("rtf", ""),
                    m.get("ttft_ms", ""), m.get("finals", ""), tag,
                ])
                f.flush()

    print(f"\ndone — {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
