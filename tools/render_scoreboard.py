"""
Render SCOREBOARD.md from results/runs/*.json.

For each (engine, case) pair, picks the most recent non-ramp run and emits
a row with WER, CER, DER (if scored), entity preservation, latency, RTF, GPU
peak. Concurrency ramp runs go in a separate section that compares N>1 to
N=1 against the upfront thresholds in METHODOLOGY.md.

Deterministic over input files. Re-run any time `results/runs/` changes.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "results" / "runs"
OUT_PATH = REPO_ROOT / "results" / "SCOREBOARD.md"


@dataclass
class Run:
    path: Path
    timestamp: str
    engine: str
    case: str
    diarizer: str                   # "native" if not specified
    tag: str | None                 # config variant label
    concurrency: int | None         # None for single-stream baseline
    data: dict


def load_runs() -> list[Run]:
    runs: list[Run] = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        if path.name.endswith("__resources.jsonl"):
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        runs.append(Run(
            path=path,
            timestamp=data.get("timestamp_utc", ""),
            engine=data.get("engine", ""),
            case=data.get("case", ""),
            diarizer=data.get("diarizer", "native"),
            tag=data.get("tag"),
            concurrency=data.get("concurrency"),
            data=data,
        ))
    return runs


def fmt_pct(x: float | None, places: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x*100:.{places}f}%"


def fmt_ms(x: float | None) -> str:
    if x is None:
        return "—"
    if x < 0:
        # Negative per-final lag = fast-mode artifact (audio sent before finals
        # arrive). The number isn't meaningful in that mode.
        return "n/a (fast)"
    return f"{x:.0f} ms"


def fmt_int(x) -> str:
    return "—" if x is None else f"{x:,}"


def render() -> str:
    runs = load_runs()
    if not runs:
        return "# Scoreboard\n\nNo runs yet. Run `python -m runner --engine ... --case ...`.\n"

    # Group: latest single-stream run per (engine, diarizer, case, tag).
    # diarizer = "native" is the engine's own labels (default).
    # tag=None is the "default" config; named tags are config variants and
    # get their own row.
    latest_baseline: dict[tuple[str, str, str, str | None], Run] = {}
    ramp_runs: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for r in runs:
        if r.concurrency is None:
            key = (r.engine, r.diarizer, r.case, r.tag)
            cur = latest_baseline.get(key)
            if cur is None or r.timestamp > cur.timestamp:
                latest_baseline[key] = r
        else:
            ramp_runs[(r.engine, r.case)].append(r)

    lines: list[str] = ["# Scoreboard", ""]
    lines.append("Auto-generated from `results/runs/*.json`. Re-render with `python tools/render_scoreboard.py`.")
    lines.append("")
    lines.append("Methodology + thresholds: see [METHODOLOGY.md](../METHODOLOGY.md).")
    lines.append("")

    # ---- Accuracy + latency table (single-stream baseline) ----
    lines.append("## Single-stream baseline")
    lines.append("")
    lines.append("| Case | Engine | Diarizer | Config | WER | CER | DER | Entities | Mean conf | TTFT | per-final p95 | RTF | GPU peak |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    # Sort by case → engine → diarizer → tag so each case's stack is contiguous
    for (engine, diarizer, case, tag), r in sorted(latest_baseline.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1], kv[0][3] or "")):
        s = r.data.get("scores", {})
        wer = s.get("wer") or {}
        der = s.get("der") or {}
        ent = s.get("entity") or {}
        lat = s.get("latency") or {}
        res = s.get("resources") or {}
        conf = s.get("confidence") or {}

        ent_str = "—"
        if ent and ent.get("total"):
            ent_str = f"{ent['preserved']}/{ent['total']}"

        gpu_str = "—"
        if res.get("gpu_mem_used_mb_peak"):
            gpu_str = f"{res['gpu_mem_used_mb_peak']:,} MiB"

        conf_str = "—"
        if conf.get("mean") is not None:
            conf_str = f"{conf['mean']:.3f}"

        config_label = f"`{tag}`" if tag else "_default_"
        lines.append(
            f"| `{case}` | `{engine}` | `{diarizer}` | {config_label} "
            f"| {fmt_pct(wer.get('wer'))} "
            f"| {fmt_pct(wer.get('cer'))} "
            f"| {fmt_pct(der.get('der'))} "
            f"| {ent_str} "
            f"| {conf_str} "
            f"| {fmt_ms(lat.get('ttft_ms'))} "
            f"| {fmt_ms(lat.get('final_lag_p95_ms'))} "
            f"| {lat.get('rtf', 0):.3f} "
            f"| {gpu_str} |"
        )
    lines.append("")

    # ---- S/D/I detail ----
    lines.append("## Accuracy detail (S/D/I + entity preservation)")
    lines.append("")
    lines.append("| Case | Engine | Diarizer | Config | Subs | Dels | Ins | Ref words | Hyp words | Missing entities |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for (engine, diarizer, case, tag), r in sorted(latest_baseline.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1], kv[0][3] or "")):
        s = r.data.get("scores", {})
        wer = s.get("wer") or {}
        ent = s.get("entity") or {}
        miss = ", ".join(ent.get("missing", [])) if ent.get("missing") else "—"
        if not wer:
            continue
        config_label = f"`{tag}`" if tag else "_default_"
        lines.append(
            f"| `{case}` | `{engine}` | `{diarizer}` | {config_label} "
            f"| {fmt_int(wer.get('substitutions'))} "
            f"| {fmt_int(wer.get('deletions'))} "
            f"| {fmt_int(wer.get('insertions'))} "
            f"| {fmt_int(wer.get('ref_word_count'))} "
            f"| {fmt_int(wer.get('hyp_word_count'))} "
            f"| {miss} |"
        )
    lines.append("")

    # ---- Concurrency ramp ----
    if ramp_runs:
        lines.append("## Concurrency ramp")
        lines.append("")
        for (engine, case), rs in sorted(ramp_runs.items()):
            rs_sorted = sorted(rs, key=lambda r: (r.concurrency or 0, r.timestamp))
            lines.append(f"### `{engine}` × `{case}`")
            lines.append("")
            lines.append("| N | Successes | Failures | TTFT p50 | TTFT p95 | RTF p50 | RTF p95 | GPU peak |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for r in rs_sorted:
                d = r.data
                res = (d.get("resources") or {})
                gpu = f"{res['gpu_mem_used_mb_peak']:,} MiB" if res.get("gpu_mem_used_mb_peak") else "—"
                lines.append(
                    f"| {d.get('concurrency')} "
                    f"| {d.get('successes', 0)} "
                    f"| {d.get('failures', 0)} "
                    f"| {fmt_ms(d.get('ttft_ms_p50'))} "
                    f"| {fmt_ms(d.get('ttft_ms_p95'))} "
                    f"| {d.get('rtf_p50', 0):.3f} "
                    f"| {d.get('rtf_p95', 0):.3f} "
                    f"| {gpu} |"
                )
            lines.append("")

    # ---- Engines + cases inventory ----
    lines.append("## Inventory")
    lines.append("")
    engines = sorted({r.engine for r in runs})
    cases = sorted({r.case for r in runs})
    lines.append(f"- **Engines:** {', '.join(f'`{e}`' for e in engines)}")
    lines.append(f"- **Cases:** {', '.join(f'`{c}`' for c in cases)}")
    lines.append(f"- **Total runs:** {len(runs)}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    out = render()
    OUT_PATH.write_text(out)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
