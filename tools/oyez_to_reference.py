"""
Parse an Oyez `oral_argument_audio` JSON into reference.txt + reference.rttm.

Oyez structure: argument → sections → turns (one speaker per turn) → text_blocks
(timed sentence/clause spans). We collapse turns into RTTM speaker segments and
concatenate all text_blocks for the reference transcript.

Usage:
    python tools/oyez_to_reference.py cases/scotus-glossip-v-oklahoma/oyez_argument.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def speaker_label(turn: dict) -> str:
    """Make a stable, RTTM-safe speaker label from the Oyez speaker block."""
    sp = turn.get("speaker") or {}
    name = sp.get("name") or sp.get("last_name") or f"unk_{sp.get('ID', 'X')}"
    # RTTM speaker tokens shouldn't contain spaces — use last name only when possible
    last = sp.get("last_name") or name.split()[-1]
    return re.sub(r"[^A-Za-z0-9_]", "_", last)


def parse(arg_json_path: Path) -> tuple[str, list[tuple[float, float, str]]]:
    """Return (reference_text, rttm_segments)."""
    data = json.loads(arg_json_path.read_text(encoding="utf-8"))
    transcript = data["transcript"]

    text_parts: list[str] = []
    rttm: list[tuple[float, float, str]] = []

    for section in transcript.get("sections", []):
        for turn in section.get("turns", []):
            label = speaker_label(turn)
            t_start = float(turn.get("start", 0))
            t_stop = float(turn.get("stop", t_start))
            if t_stop > t_start:
                rttm.append((t_start, t_stop, label))
            for tb in turn.get("text_blocks", []):
                text = (tb.get("text") or "").strip()
                if text:
                    text_parts.append(text)
    return "\n".join(text_parts), rttm


def write_outputs(case_dir: Path, ref_text: str, rttm: list[tuple[float, float, str]], file_id: str) -> None:
    (case_dir / "reference.txt").write_text(ref_text + "\n", encoding="utf-8")
    with open(case_dir / "reference.rttm", "w") as f:
        for start, stop, speaker in rttm:
            duration = stop - start
            f.write(f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("argument_json", type=Path)
    p.add_argument("--file-id", default=None, help="RTTM file_id (default: case dir name)")
    args = p.parse_args()

    case_dir = args.argument_json.parent
    file_id = args.file_id or case_dir.name

    text, rttm = parse(args.argument_json)
    write_outputs(case_dir, text, rttm, file_id)

    speakers = sorted({s for _, _, s in rttm})
    print(f"wrote {case_dir / 'reference.txt'} ({len(text)} chars, {len(text.split())} words)")
    print(f"wrote {case_dir / 'reference.rttm'} ({len(rttm)} turns, {len(speakers)} speakers)")
    print(f"speakers: {', '.join(speakers)}")


if __name__ == "__main__":
    main()
