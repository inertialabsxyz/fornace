# A tile kernel makes Candle's Q4_0 matmuls for 8 rows 2× faster on Metal, and lifts Fornace's agent load at Q4 from 318 to 436 tok/s

## Why it matters

When an agent sends several requests to one local server, the server
batches them. Each decode step then runs every matmul with 2 to 16 rows
of input, one row for each request. The scheduler result
([2026-11-13](2026-11-13-scheduler-qwen2-metal.md)) found that Candle's
matmuls on Metal cost about 12 ms a step for any batch of 2 to 8 rows
with Qwen2.5-0.5B on an M4 Pro, and that llama.cpp gave 1.91× the
tokens per second of Fornace at Q4 with 8 requests. This result measures
each matmul against MLX and llama.cpp, finds the cause in Candle's
quantized matmul kernels, and adds a kernel that reads each weight block
once for up to 8 rows. The Q4 decode step of 8 rows falls from 13.5 to
6.0 ms, and the agent load at 8 requests rises from 318 to 436 tok/s.
The kernel is a pull request to Candle (#4156).

## What was measured and how

| Item | Value |
|---|---|
| Machine | Apple M4 Pro, 20 GPU cores, 24 GB unified memory, macOS 27.2.1 |
| Fornace | `6996c8f` (before), `c0d8372` (after). Candle 0.11.0 with changes (fork `inertialabsxyz/candle`): `b4090fe1` before, `e8540dbb` after (#3898 and the new kernel added) |
| Candle PR | Branch `metal/qmatmul-mv-rows` (`7c1c5cdf`): upstream `main` `cfcbd718`, plus #3898, plus the kernel and a bench |
| References | MLX 0.32.4 (`bf16`), llama.cpp b12110 (`cd959ba`, Q8_0 and Q4_0), Metal |
| Model | Qwen2.5-0.5B-Instruct: `bf16`, Q8_0, Q4_0. The 1.5B shapes as a check |
| Op timer | One matmul at the decode shapes, M = 1, 2, 4, 8, 16. The weights come from a ring of copies of 256 MiB or more, so each op reads them from memory, as in a decode step. Each sample is 32 to 256 ops. The median of 100 samples after 10 warm-up samples. The same rules in all three engines (`bench/README.md`, "The matmul timers") |
| Step time | The sum of op time × calls for one decode step: q, k, v, o, gate, up, and down for 24 layers, then lm_head |
| Fornace runs | The step table and the agent load of the scheduler result, at C = 8, Fornace and llama.cpp in one run, in both orders |

Data:

- Op times before: `bench/runs/2026-11-17-6996c8f-matmul.md` (and the
  CSV files with the same prefix).
- Op times with the new kernel: `bench/runs/2026-11-18-deca28f-matmul-tile.md`.
- Metal traces: `bench/runs/2026-11-18-28db357-matmul-trace.md`.
- Fornace after: `bench/runs/2026-11-18-c0d8372-batch-step.txt`,
  `bench/runs/2026-11-18-c0d8372-agent-load-c8-ab.csv` and `-ba.csv`,
  `bench/runs/2026-11-18-c0d8372-batch-tokens.txt`.
- The Candle bench, before and after in one session:
  `bench/runs/2026-11-19-7c1c5cdf-candle-qmatmul-rows-*.txt`.

### The gap before: all the matmuls of one 0.5B decode step (`6996c8f`)

Candle as Fornace ran it: one launch for each row up to B = 4 (Q8_0) or
B = 6 (Q4_0), then the matmul kernel.

| Format | B | Candle ms | Reference ms | Reference |
|---|---:|---:|---:|---|
| `bf16` | 1 | 4.87 | 4.10 | MLX |
| `bf16` | 8 | 12.76 | 4.80 | MLX |
| Q8_0 | 1 | 3.17 | 2.66 | llama.cpp |
| Q8_0 | 8 | 11.85 | 2.80 | llama.cpp |
| Q4_0 | 1 | 2.14 | 1.33 | llama.cpp |
| Q4_0 | 8 | 11.39 | 2.33 | llama.cpp |

- At one row the gap is small and spread over all ops.
- At 8 rows it is 4× to 5× in the quantized formats. The largest single
  loss is `down` (N = 896, K = 4,864): 3.1 to 3.4 ms a step in all three
  formats. At Q4_0, 8 rows, `down` takes 152.1 µs in Candle and 22.5 µs
  in llama.cpp, 43 % of the roofline (bytes / 273 GB/s).
- With the open Candle PR #3898 (one launch for up to 8 rows), the Q4_0
  step at 8 rows is 7.52 ms.

### The new kernel: all the matmuls of one decode step, ms

`bench/runs/2026-11-18-deca28f-matmul-tile.md`:

| Model | Format | B | Before | #3898 | New kernel | llama.cpp |
|---|---|---:|---:|---:|---:|---:|
| 0.5B | Q4_0 | 4 | 7.05 | 4.16 | 3.61 | 2.29 |
| 0.5B | Q4_0 | 8 | 11.39 | 7.52 | 3.67 | 2.33 |
| 0.5B | Q4_0 | 16 | 11.47 | 13.98 | 5.68 | 3.85 |
| 0.5B | Q8_0 | 8 | 11.85 | 8.85 | 4.52 | 2.80 |
| 1.5B | Q4_0 | 8 | 27.86 | 17.92 | 9.10 | 6.53 |
| 1.5B | Q8_0 | 8 | 28.07 | 23.05 | 12.20 | 8.47 |

The Candle bench of the PR, before (#3898) and after on the same day, in
the order before, after, after, before, agrees: 0.5B Q4_0 at 8 rows
7.42 → 3.59 ms, Q8_0 8.81 → 4.53 ms. At 1 and 2 rows the code does not
change, and the two differ by up to 5 %.

### Fornace with the new kernel (`c0d8372`)

One decode step at 513 positions, fused attention:

| Format | B = 1 ms | B = 8 before (`aa23bc8`) | B = 8 after |
|---|---:|---:|---:|
| `q4` | 3.33 | 13.49 | 5.97 |
| `q8` | 4.35 | 13.69 | 7.04 |
| `bf16` | 6.21 | 13.72 | 13.63 |

The agent load at C = 8, total tok/s (Fornace first / llama.cpp first):

| Engine | Format | Scheduler result (`38171e7`) | This result |
|---|---|---:|---:|
| Fornace | `q4` | 317.6 | 435.5 / 435.8 |
| Fornace | `bf16` | 303.6 | 297.4 / 301.3 |
| llama.cpp | `q4` | 607.5 | 616.7 / 655.8 |
| llama.cpp | `bf16` | 489.8 | 489.9 / 541.2 |

At Q4 the gap to llama.cpp falls from 1.91× to 1.42× to 1.50×. `bf16`
does not change: it does not use the new kernel.

## Cause and workaround

**Candle had no quantized matmul kernel that reads a weight once for
several rows.** With 2 to 16 rows it had two paths, each slow for a
different reason:

1. *The mat-vec kernel* (`kernel_mul_mv_q4_0_f32`,
   `candle-metal-kernels/src/metal_src/quantized.metal`) takes one input
   row for each row of threadgroups. So it reads all the weights once
   for each row: 8 times at 8 rows. Candle launched it once for each row
   (`candle-core/src/quantized/metal.rs`). PR #3898 makes it one launch,
   but each row still reads the weights.
2. *The matmul kernel* (`kernel_mul_mm`) has a tile of 32 input rows and
   64 weight rows. At 8 rows that is one row of tiles, so `down` gets 14
   threadgroups, and each walks all of K = 4,864 in series. The time is
   about 0.04 µs × K for the measured shapes, and 24 of the 32 rows of
   each tile are empty.

A Metal System Trace shows the GPU busy 95 to 97 % of the run in all
Candle paths, with a GPU time for each op equal to the op time. So the
cost is GPU work, not the CPU cost of a launch.

llama.cpp runs these shapes with `kernel_mul_mv_mma`: up to 8 input rows
in one 8 × 8 simdgroup-matrix tile, with K split over 8 or 16 SIMD
groups. The new Candle kernel (`kernel_mul_mv_tile`) has that design,
built from Candle's own dequantize and simdgroup code. Q4_0 and Q8_0
with 4 or more rows go to it. 2 and 3 rows stay on the one-launch path
of #3898, which is faster there. It dequantizes the weights to half, as
`kernel_mul_mm` does, so its results differ from the mat-vec kernel by
about 5e-4 of the largest value.

**The workaround** is the Candle fork at `e8540dbb`, or PR #4156 on top
of #3898. Fornace uses the fork from `c0d8372`. Fornace also removes its
own layout switch, because `[B, 1, K]` and `[B, K]` now take the same
kernel.

**What is left.**

- The new kernel is 1.4× to 1.6× slower than llama.cpp for the matmuls
  of one step. Most of that is in the small matmuls: at 0.5B Q4_0, 8
  rows, `q` takes 10.1 µs against 4.8, and `k_v` 6.2 against 2.3.
- At 4 rows, some single ops are slower than with #3898 (`k_v` Q4_0 5.2
  against 4.5 µs, `down` Q8_0 43.4 against 35.6 µs). The whole step at 4
  rows is still faster (1.09× to 1.14×).
- The `bf16` matmul for 2 to 16 rows has the same fault as the
  quantized matmul kernel: a 32 × 32 tile for any M below 16
  (`candle-metal-kernels/src/kernels/mlx_gemm.rs`). It loses 8.0 ms a
  step at 8 rows against MLX. This result does not change it.
- The step table gives 961 tok/s for a `q4` step of 8 rows with a cache
  for each row, and the server gives 436 tok/s. The rest is not
  measured. Possible causes: prompts that run between the steps, batches
  with fewer than 8 rows, and the HTTP and sampling cost.

## What it means if you run models on your own hardware

If you run a quantized model with Candle on a Mac and send several
requests at once, the matmuls were the largest cost of each decode step.
With the new kernel, a Q4 step of 8 requests with Qwen2.5-0.5B on an M4
Pro takes 6.0 ms instead of 13.5 ms, and the tokens per second of an
agent load with 8 requests rise 37 %. llama.cpp is still faster: 1.4×
to 1.5× at Q4. With one request nothing changes. With `bf16` weights
nothing changes yet.

## Correctness

- Candle tests `qmatmul_mv_rows_match_single_row_calls_q4_0` and
  `_q8_0`: 2 to 8 and 12 rows against one call for each row, at five
  shapes, within 2e-3 of the largest value. All quantized tests and the
  tests of `candle-metal-kernels` pass.
- 8 agent requests in one batch against each alone
  (`bench/runs/2026-11-18-c0d8372-batch-tokens.txt`): the file is
  identical to that of the scheduler result. `q8` and `q4` give the same
  tokens in 8 of 8 requests, `bf16` in 5 of 8 (the same near ties).

## The predictions against the measurement

Written before the first run (15 November).

| Prediction | Measured | |
|---|---|---|
| At M = 1, Candle is within 15 % of the reference for each shape | The small matmuls (q, o, k_v): 1.3× to 2.9×. gate_up and down: 1.07× to 1.19× for `bf16` and Q8_0, but 1.64× for Q4_0. `lm_head`: Candle is 6 to 11 % faster than llama.cpp at Q8_0 and Q4_0 | Failed, except for the large `bf16` and Q8_0 matmuls |
| At M = 2 to 8, Candle is 2× or more slower than the reference | `down` 4.5× to 5.2×, `k_v` 11× to 18×; `bf16` gate_up 1.7× to 1.9× | Held for most shapes |
| The largest single-kernel gap at B = 8 is the matmul of the MLP | `down`, 3.1 to 3.4 ms a step in all three formats | Held |
| A Q4_0 kernel for 2 to 8 rows makes a `q4` step of 8 rows about 1.5× a step of one row | 1.80× (5.97 against 3.33 ms) | Failed |
| With it, the agent load at C = 8, `q4`, gives 450 to 550 total tok/s | 435.5 and 435.8 | Failed, 3 % below |

## The pull request

huggingface/candle#4156, opened on 20 November 2026. It is stacked on
#3898, which has had no review since August. On the date of this result
it is open, not merged.

## What this does not prove

- One machine (M4 Pro), one model for the Fornace runs (Qwen2.5-0.5B),
  and the 1.5B shapes only in the op timer.
- Only Q4_0 and Q8_0. The K-quants (Q4_K_M and others), which most
  people download, use other Candle kernels. They probably have the same
  gap, but it is not measured.
- The tile sizes and the 4-row gate are tuned on this machine. Another
  Apple GPU can have other best values.
- The op times before and with the new kernel in the Fornace timer were
  recorded on two days. The Candle bench repeats the comparison in one
  session and agrees within 3 %.
- The cause of the gap between the step table and the server is not
  measured.
- `bf16` is not changed. Its gap to MLX at 8 rows is still 2.7× for the
  matmuls.
- Each reference engine ran at one version, with settings that were not
  tuned. llama.cpp `bf16` varies by up to 10 % between the two orders.

## Reproduce

Commit `18a032a`. On macOS, with Homebrew and Xcode with its Metal
Toolchain.

```sh
git clone https://github.com/inertialabsxyz/fornace && cd fornace
git checkout 18a032a
make model && make quantize
scripts/get-reference-engines.sh --install
cargo build --release --features metal -p fornace-server -p fornace-bench
# Op times with the new kernel (the fork at e8540dbb is in Cargo.toml)
(cd crates/fornace-engine && FORNACE_OP_COST_OUT=$PWD/../../bench/runs/repro-matmul-candle.csv \
  FORNACE_OP_COST_FORMATS=q8_0,q4_0 FORNACE_OP_COST_LAYOUTS=batch_axis FORNACE_OP_COST_MODES=cold \
  cargo test --release --features metal -p fornace-engine --test matmul_op_cost -- --ignored --nocapture --test-threads=1)
# The references
bench/ref/run.sh $PWD/bench/runs/repro-matmul
# The step table and the batch tokens check
cargo test --release --features metal -p fornace-engine \
  --test batch_step_cost -- --ignored --nocapture --test-threads=1
cargo test --release --features metal -p fornace-bench \
  --test batch_tokens -- --ignored --nocapture
# The agent load at C = 8, both orders
fb() { target/release/fornace-bench --model ./models/qwen2.5-0.5b-instruct \
         --transport http --attention fused --repeat-penalty 1.0 --load agent \
         --concurrency 8 --parallel 8 --dtypes f16,q4 "$@"; }
fb --engines fornace,llamacpp --out bench/runs/repro-metal-gap-ab.csv
fb --engines llamacpp,fornace --out bench/runs/repro-metal-gap-ba.csv
```

The numbers before the new kernel: the same op timer at `6996c8f`. The
Candle bench, on the PR branch and on #3898:

```sh
cargo bench -p candle-core --features metal --bench bench_main -- qmatmul_rows_
```

Expected: the op times near the tables above; a `q4` step of 8 rows near
6.0 ms; the batch tokens file identical to
`bench/runs/2026-11-13-71cae6e-batch-tokens.txt`; at C = 8, Fornace
`q4` near 436 and `bf16` near 300 total tok/s, llama.cpp near 600 to 660
and 490 to 540. Speed drifts between commands, so compare within one
run.
