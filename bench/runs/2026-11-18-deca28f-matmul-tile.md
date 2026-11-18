# Q4_0 and Q8_0 matmul op times with a tile kernel for 4 to 16 rows

Recorded 2026-11-18 on an Apple M4 Pro (24 GB), macOS 27.2.1. The timer
and its rules are the ones of `bench/runs/2026-11-17-6996c8f-matmul.md`
(`bench/README.md`, "The matmul timers"). Only `cold` rows, layout
`batch_axis` (`[M, 1, K]`, the input of a Fornace decode step).

## Code

- Fornace `deca28f`, with `Cargo.toml` patched to a local Candle (not
  committed).
- Candle: the fork `b4090fe1`, with upstream PR #3898 (`cf5b24f7`) applied
  (`0d65fd9e`), and with the change of Candle commit `9ab1f412` ("metal:
  tile kernel for the q4_0 and q8_0 matmul with 4 to 16 rows", on upstream
  `main` `cfcbd718` plus #3898) applied on top, not committed. The kernel is
  `kernel_mul_mv_tile` in `candle-metal-kernels/src/metal_src/quantized.metal`.
  From 4 rows, Q4_0 and Q8_0 go to it. With 2 or 3 rows they stay on the
  one-launch mat-vec path of #3898.

## Command

From `crates/fornace-engine` of that worktree:

```sh
FORNACE_OP_COST_OUT=$PWD/../../bench/runs/2026-11-18-deca28f-matmul-tile-candle.csv \
FORNACE_OP_COST_FORMATS=q8_0,q4_0 FORNACE_OP_COST_LAYOUTS=batch_axis FORNACE_OP_COST_MODES=cold \
cargo test --release --features metal -p fornace-engine --test matmul_op_cost -- --ignored --nocapture --test-threads=1
```

"Today", "#3898", and "ggml" are from `bench/runs/2026-11-17-6996c8f-matmul-*.csv`
(the day before, same machine). "Today" is the layout that Fornace uses:
a launch for each row up to B = 4 (`q8`) or B = 6 (`q4`), then the matmul
kernel. The step time is the sum of op time × calls in one step.

## All the matmuls of one decode step, ms

| Model | Format | B | Today | #3898 | Tile kernel | ggml | Today / tile | Tile / ggml |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5B | Q8_0 | 1 | 3.17 | 3.01 | 3.06 | 2.66 | 1.04 | 1.15 |
| 0.5B | Q8_0 | 2 | 5.46 | 3.15 | 3.21 | 2.71 | 1.70 | 1.18 |
| 0.5B | Q8_0 | 4 | 10.00 | 4.91 | 4.43 | 2.74 | 2.26 | 1.62 |
| 0.5B | Q8_0 | 8 | 11.85 | 8.85 | 4.52 | 2.80 | 2.62 | 1.61 |
| 0.5B | Q8_0 | 16 | 11.83 | 16.64 | 6.63 | 3.99 | 1.78 | 1.66 |
| 0.5B | Q4_0 | 1 | 2.14 | 2.09 | 2.16 | 1.33 | 0.99 | 1.62 |
| 0.5B | Q4_0 | 2 | 3.74 | 2.58 | 2.63 | 2.25 | 1.42 | 1.17 |
| 0.5B | Q4_0 | 4 | 7.05 | 4.16 | 3.61 | 2.29 | 1.95 | 1.58 |
| 0.5B | Q4_0 | 8 | 11.39 | 7.52 | 3.67 | 2.33 | 3.10 | 1.57 |
| 0.5B | Q4_0 | 16 | 11.47 | 13.98 | 5.68 | 3.85 | 2.02 | 1.48 |
| 1.5B | Q8_0 | 1 | 7.90 | 7.86 | 7.94 | 7.37 | 1.00 | 1.08 |
| 1.5B | Q8_0 | 2 | 13.39 | 8.03 | 8.20 | 8.18 | 1.63 | 1.00 |
| 1.5B | Q8_0 | 4 | 24.50 | 12.32 | 11.94 | 8.29 | 2.05 | 1.44 |
| 1.5B | Q8_0 | 8 | 28.07 | 23.05 | 12.20 | 8.47 | 2.30 | 1.44 |
| 1.5B | Q8_0 | 16 | 28.27 | 44.84 | 19.26 | 12.66 | 1.47 | 1.52 |
| 1.5B | Q4_0 | 1 | 4.59 | 4.55 | 4.64 | 3.85 | 0.99 | 1.21 |
| 1.5B | Q4_0 | 2 | 7.98 | 5.37 | 5.70 | 6.31 | 1.40 | 0.90 |
| 1.5B | Q4_0 | 4 | 14.76 | 9.59 | 8.70 | 6.35 | 1.70 | 1.37 |
| 1.5B | Q4_0 | 8 | 27.86 | 17.92 | 9.10 | 6.53 | 3.06 | 1.39 |
| 1.5B | Q4_0 | 16 | 28.10 | 35.29 | 16.29 | 11.23 | 1.73 | 1.45 |

## One op at M = 8, µs

| Model | Format | Op | Today | #3898 | Tile kernel | ggml |
|---|---|---|---:|---:|---:|---:|
| 0.5B | Q8_0 | q | 35.7 | 19.8 | 11.9 | 5.1 |
| 0.5B | Q8_0 | k_v | 34.3 | 6.3 | 6.2 | 1.9 |
| 0.5B | Q8_0 | o | 35.8 | 19.7 | 11.9 | 5.1 |
| 0.5B | Q8_0 | gate_up | 66.6 | 78.5 | 34.9 | 23.2 |
| 0.5B | Q8_0 | down | 159.5 | 64.8 | 46.6 | 26.9 |
| 0.5B | Q8_0 | lm_head | 1457.9 | 2277.5 | 852.0 | 712.0 |
| 0.5B | Q4_0 | q | 34.3 | 18.1 | 10.1 | 4.8 |
| 0.5B | Q4_0 | k_v | 32.9 | 6.0 | 6.2 | 2.3 |
| 0.5B | Q4_0 | o | 34.3 | 17.8 | 9.9 | 4.8 |
| 0.5B | Q4_0 | gate_up | 63.5 | 66.0 | 28.1 | 18.7 |
| 0.5B | Q4_0 | down | 152.1 | 53.0 | 34.1 | 22.5 |
| 0.5B | Q4_0 | lm_head | 1466.9 | 1924.6 | 728.6 | 551.1 |
| 1.5B | Q8_0 | q | 61.4 | 41.2 | 24.4 | 12.9 |
| 1.5B | Q8_0 | k_v | 52.3 | 10.2 | 10.5 | 2.8 |
| 1.5B | Q8_0 | o | 61.3 | 41.0 | 24.7 | 12.8 |
| 1.5B | Q8_0 | gate_up | 181.1 | 205.3 | 99.2 | 71.8 |
| 1.5B | Q8_0 | down | 324.1 | 186.4 | 107.1 | 83.4 |
| 1.5B | Q8_0 | lm_head | 2482.4 | 3463.3 | 1677.7 | 1231.3 |
| 1.5B | Q4_0 | q | 62.3 | 32.7 | 19.1 | 11.3 |
| 1.5B | Q4_0 | k_v | 50.1 | 8.9 | 8.9 | 2.8 |
| 1.5B | Q4_0 | o | 61.3 | 32.4 | 19.1 | 11.4 |
| 1.5B | Q4_0 | gate_up | 173.7 | 157.3 | 73.8 | 49.0 |
| 1.5B | Q4_0 | down | 336.8 | 146.6 | 74.7 | 71.2 |
| 1.5B | Q4_0 | lm_head | 2433.4 | 2685.1 | 1303.8 | 999.3 |

## Notes

- The tile kernel dequantizes the weights to half, as `kernel_mul_mm`
  does. Its results differ from the mat-vec kernels by about 5e-4 of the
  largest value (Candle test `qmatmul_mv_rows_match_single_row_calls_*`).
- "Today" and "ggml" were recorded on 2026-11-17, the tile kernel on
  2026-11-18. Between two runs, speed can drift by about 7 %
  (`bench/README.md`). At B = 1, where the code is the same, the two runs differ
  by 0.5 to 3.5 %.
