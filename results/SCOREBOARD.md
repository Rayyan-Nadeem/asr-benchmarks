# Scoreboard

Auto-generated from `results/runs/*.json`. Re-render with `python tools/render_scoreboard.py`.

Methodology + thresholds: see [METHODOLOGY.md](../METHODOLOGY.md).

## Single-stream baseline

| Engine | Case | WER | CER | DER | Entities | TTFT | per-final p95 | RTF | GPU peak |
|---|---|---|---|---|---|---|---|---|---|
| `speechmatics_onprem` | `deposition-greg-erwin` | — | — | — | — | 852 ms | n/a (fast) | 0.137 | 7,634 MiB |
| `speechmatics_onprem` | `librispeech-test-clean-mini` | 2.98% | 6.08% | — | 3/3 | 4019 ms | 6348 ms | 1.053 | 7,634 MiB |
| `speechmatics_onprem` | `scotus-glossip-v-oklahoma` | 14.87% | 7.78% | 2.00% | 6/16 | 5131 ms | 18702 ms | 1.051 | 7,684 MiB |

## Accuracy detail (S/D/I + entity preservation)

| Engine | Case | Subs | Dels | Ins | Ref words | Hyp words | Missing entities |
|---|---|---|---|---|---|---|---|
| `speechmatics_onprem` | `librispeech-test-clean-mini` | 4 | 0 | 1 | 168 | 169 | — |
| `speechmatics_onprem` | `scotus-glossip-v-oklahoma` | 42 | 20 | 36 | 659 | 675 | Roberts, Sotomayor, Kagan, Kavanaugh, Barrett, Jackson, Alito, Thomas, habeas, 22-7466 |

## Concurrency ramp

### `speechmatics_onprem` × `scotus-glossip-v-oklahoma`

| N | Successes | Failures | TTFT p50 | TTFT p95 | RTF p50 | RTF p95 | GPU peak |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0 | 5602 ms | 5602 ms | 1.053 | 1.053 | 7,684 MiB |
| 2 | 2 | 0 | 5096 ms | 5232 ms | 1.046 | 1.046 | 7,684 MiB |
| 4 | 2 | 2 | 5069 ms | 5070 ms | 1.049 | 1.049 | 7,782 MiB |

## Inventory

- **Engines:** `speechmatics_onprem`
- **Cases:** `deposition-greg-erwin`, `librispeech-test-clean-mini`, `scotus-glossip-v-oklahoma`
- **Total runs:** 8
