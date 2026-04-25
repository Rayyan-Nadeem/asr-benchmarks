# Cases

Each subdirectory is one audio case. The harness picks the case by folder
name (`--case <folder>`), reads `case.yaml` for metadata, and runs every
configured engine against `audio.<ext>`.

## Layout

```
cases/<case-name>/
├── case.yaml            # required — see schema below
├── audio.<ext>          # required — mp3/wav/flac/m4a/ogg, will be converted to wav at run time
├── audio.wav            # optional — auto-generated from audio.<ext> on first run
├── reference.txt        # optional — ground-truth transcript for WER scoring
├── reference.rttm       # optional — speaker timeline for DER scoring
└── fetch.sh             # optional — script that produced audio.* (reproducibility)
```

## case.yaml schema

```yaml
name: my-case
description: |
  What the audio is, where it came from, why it's useful.
source: "URL or attribution"
license: "Public domain / CC-BY 4.0 / Customer-provided / etc."
duration_s: 300                 # approximate seconds, for sizing
speakers: 1                     # speaker count (1 if mono)
domain: legal | meeting | interview | broadcast | read
language: en

monitor_containers:             # containers to attribute resources to
  - sm-triton
  - sm-rt-transcriber

key_terms:                      # for entity preservation scoring
  - "Proper Noun"
  - "Case Number"

ground_truth: null              # populated by the maintainer when a transcript is added
```

## Adding a new case

1. `mkdir cases/<case-name>/`
2. Write `case.yaml` from the schema above.
3. Drop the audio in as `audio.mp3` (or any other ffmpeg-readable format).
4. If you have a transcript, write it as `reference.txt` (plain text, one sentence per line is fine).
5. If multi-speaker and you want DER, write speaker turns to `reference.rttm` (RTTM format).
6. Optional: write a `fetch.sh` so the next person can reproduce.
7. Run the case: `python -m runner --engine speechmatics_onprem --case <case-name>`.

See `CHOICES.md` for the audio sources we considered when building Day-1 cases.
