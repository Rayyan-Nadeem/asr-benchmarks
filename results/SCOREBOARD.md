# Scoreboard

Auto-generated from `results/runs/*.json`. Re-render with `python tools/render_scoreboard.py`.

Methodology + thresholds: see [METHODOLOGY.md](../METHODOLOGY.md).

## Single-stream baseline

| Case | Engine | Diarizer | Config | WER | CER | DER | Entities | Mean conf | TTFT | per-final p95 | RTF | GPU peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ami-es2004a-5min` | `speechmatics_onprem` | `native` | _default_ | — | — | 44.24% | — | 0.985 | 8354 ms | 12685 ms | 1.031 | 7,684 MiB |
| `ami-es2004a-5min` | `speechmatics_onprem` | `native` | `matrix_sm_native` | — | — | 45.02% | — | 0.986 | 5201 ms | 7055 ms | 1.010 | 7,684 MiB |
| `ami-es2004a-5min` | `speechmatics_onprem` | `native` | `max_delay_10` | — | — | 45.02% | — | 0.986 | 5368 ms | 17988 ms | 1.048 | 7,684 MiB |
| `ami-es2004a-5min` | `speechmatics_onprem` | `native` | `max_delay_15` | — | — | 45.02% | — | 0.986 | 5377 ms | 18286 ms | 1.048 | 7,684 MiB |
| `ami-es2004a-5min` | `speechmatics_onprem` | `native` | `max_speakers_4` | — | — | 45.02% | — | 0.986 | 5383 ms | 17496 ms | 1.046 | 7,684 MiB |
| `ami-es2004a-5min` | `speechmatics_onprem` | `native` | `md10_ms4` | — | — | 45.02% | — | 0.986 | 5394 ms | 18097 ms | 1.048 | 7,684 MiB |
| `ami-es2004a-5min` | `speechmatics_onprem` | `native` | `md15_ms4_fixed` | — | — | 45.02% | — | 0.986 | 5389 ms | 17255 ms | 1.045 | 7,684 MiB |
| `ami-es2004a-5min` | `whisper` | `none` | `matrix_whisper_only` | — | — | 100.00% | — | 0.867 | 57420 ms | n/a (fast) | 0.191 | 10,044 MiB |
| `ami-es2004a-5min` | `whisper` | `speechmatics_diar` | `matrix_whisper_smdiar` | — | — | 46.70% | — | 0.869 | 60609 ms | n/a (fast) | 0.202 | 10,032 MiB |
| `deposition-greg-erwin` | `speechmatics_onprem` | `native` | _default_ | — | — | — | — | 0.990 | 5020 ms | 26215 ms | 1.052 | 7,782 MiB |
| `librispeech-test-clean-mini` | `speechmatics_onprem` | `native` | _default_ | 2.98% | 6.08% | — | 3/3 | 0.991 | 4019 ms | 6348 ms | 1.053 | 7,634 MiB |
| `librispeech-test-clean-mini` | `whisper` | `none` | `whisper_smoke` | 3.57% | 5.66% | — | 2/3 | 0.983 | 142969 ms | 73849 ms | 2.042 | 15,719 MiB |
| `scotus-glossip-v-oklahoma` | `speechmatics_onprem` | `native` | _default_ | 14.87% | 7.78% | 2.00% | 6/16 | 0.995 | 5131 ms | 18702 ms | 1.051 | 7,684 MiB |

## Accuracy detail (S/D/I + entity preservation)

| Case | Engine | Diarizer | Config | Subs | Dels | Ins | Ref words | Hyp words | Missing entities |
|---|---|---|---|---|---|---|---|---|---|
| `librispeech-test-clean-mini` | `speechmatics_onprem` | `native` | _default_ | 4 | 0 | 1 | 168 | 169 | — |
| `librispeech-test-clean-mini` | `whisper` | `none` | `whisper_smoke` | 4 | 1 | 1 | 168 | 168 | Nelly |
| `scotus-glossip-v-oklahoma` | `speechmatics_onprem` | `native` | _default_ | 42 | 20 | 36 | 659 | 675 | Roberts, Sotomayor, Kagan, Kavanaugh, Barrett, Jackson, Alito, Thomas, habeas, 22-7466 |

## Concurrency ramp

### `speechmatics_onprem` × `scotus-glossip-v-oklahoma`

| N | Successes | Failures | TTFT p50 | TTFT p95 | RTF p50 | RTF p95 | GPU peak |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0 | 5602 ms | 5602 ms | 1.053 | 1.053 | 7,684 MiB |
| 2 | 2 | 0 | 5096 ms | 5232 ms | 1.046 | 1.046 | 7,684 MiB |
| 4 | 2 | 2 | 5069 ms | 5070 ms | 1.049 | 1.049 | 7,782 MiB |

## Inventory

- **Engines:** `speechmatics_onprem`, `whisper`
- **Cases:** `ami-es2004a-5min`, `deposition-greg-erwin`, `librispeech-test-clean-mini`, `scotus-glossip-v-oklahoma`
- **Total runs:** 19
