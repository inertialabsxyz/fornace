# A scheduler cuts the slowest first token of 8 agent requests to Qwen2 on Metal from 8.3 s to 0.9 s, and doubles the tokens per second at BF16

## Why it matters

One agent on one machine sends several requests at once: tool calls,
sub-agents, a summary and a check. The comparison result
([2026-11-13](2026-11-13-comparison-qwen2-metal.md)) measured such a
load on Fornace, llama.cpp, and mistral.rs. Fornace ran one request at a
time, so with 8 requests at once the last one waited 8.3 s for its first
token at BF16, and llama.cpp gave 3.3× its tokens per second. This result
adds a scheduler to `fornace-server`: requests join and leave one batch,
each with its own length, position, and KV cache. The wait goes from
seconds to under one second, and the tokens per second double at BF16. At
Q4 they rise only 1.33×. The rest of the gap to llama.cpp is in the
batched decode step: Candle's matmul kernels cost about 12 ms a step for
any batch of 2 to 8 rows on an M4 Pro.

## What was measured and how

| Item | Value |
|---|---|
| Machine | Apple M4 Pro, 20 GPU cores, 24 GB unified memory, macOS 27.2 |
| Fornace | `fornace-server` at `105e894` (against its queue) and `38171e7` (against the other engines). Candle 0.11.0 with three changes (fork `inertialabsxyz/candle`, commit `b4090fe1`). Fused attention, `--max-batch 8` |
| The queue | The same server with `--max-batch 1`: one request at a time, as before this result |
| llama.cpp | `llama-server` 0.9.0, build 12110 (`cd959bae9`), Homebrew. `-ngl 99 -fa on -lm none -np 8 -c 32768`, prompt caches off |
| mistral.rs | `mistralrs serve` v0.9.5 (`465191d`), built with Xcode 27.2 and its Metal Toolchain, so with precompiled Metal kernels. Its default paged attention, `--max-seqs 8 --kv-cache 32768tokens` (the cache of llama.cpp), `--prefix-cache-n 0`. `--dtype f32` for GGUF |
| Model | Qwen2.5-0.5B-Instruct, `bf16` and `q4` (Q4_0, the same tensor bytes in all engines) |
| Load | The agent load of the comparison result: a shared system prompt of 256 tokens, then an own part of 64, 256, or 1,024 tokens of code or German text, and 32, 128, or 256 new tokens. C clients each send their next request when the last one is done. A level has 16 requests (C = 1, 2, 4) or 32 (C = 8) |
| Requests | `/v1/completions`, streamed, greedy, repeat penalty 1.0, end-of-sequence token ignored. One warm-up request before each level, not recorded |
| Timing | The client times each request. It never uses the timings of a server |
| Order | Each run twice, in one order and then the reverse. The tables give the mean of the two |

Data: `bench/runs/2026-11-13-105e894-scheduler.csv` (the scheduler
against the queue and llama.cpp), and
`bench/runs/2026-11-13-38171e7-three-engines-paged.csv` (the three
engines). The step tables: `bench/runs/2026-11-13-aa23bc8-batch-step.txt`.

### The scheduler against the queue (`105e894`)

Total tok/s (all new tokens / the wall time of the level):

| Server | Format | C = 1 | C = 2 | C = 4 | C = 8 |
|---|---|---:|---:|---:|---:|
| Queue | `bf16` | 141.1 | 141.5 | 141.9 | 142.1 |
| Scheduler | `bf16` | 141.0 | 129.7 | 194.2 | 299.6 |
| Queue | `q4` | 231.4 | 231.7 | 234.3 | 235.4 |
| Scheduler | `q4` | 232.4 | 258.8 | 280.5 | 312.3 |

The time to the first token at C = 8 (ms):

| Server | Format | p50 | Largest |
|---|---|---:|---:|
| Queue | `bf16` | 6,316 | 8,312 |
| Scheduler | `bf16` | 221 | 911 |
| Queue | `q4` | 3,927 | 5,000 |
| Scheduler | `q4` | 239 | 991 |

With one request (C = 1, `alice-512.txt`, 128 tokens, ABBA), the
scheduler is −0.8 to +0.1 % of the queue in decode, with the same first
token (`bf16` 159.4 against 160.1 tok/s, `q8` 225.0 against 226.9, `q4`
297.3 against 297.0).

### The three engines (`38171e7`)

| Engine | Format | C = 1 | C = 2 | C = 4 | C = 8 |
|---|---|---:|---:|---:|---:|
| Fornace | `bf16` | 139.1 | 129.3 | 194.2 | 303.6 |
| mistral.rs | `bf16` | 124.4 | 193.9 | 269.9 | 356.2 |
| llama.cpp | `bf16` | 154.9 | 245.1 | 335.5 | 489.8 |
| Fornace | `q4` | 226.6 | 257.5 | 281.4 | 317.6 |
| llama.cpp | `q4` | 262.2 | 351.6 | 446.0 | 607.5 |

At C = 8:

| Engine | Format | First token p50 ms | Largest ms | Decode of one request, tok/s | Peak MiB |
|---|---|---:|---:|---:|---:|
| Fornace | `bf16` | 180 | 912 | 44.0 | 1,464 |
| mistral.rs | `bf16` | 528 | 1,510 | 59.8 | 2,148 |
| llama.cpp | `bf16` | 243 | 728 | 80.0 | 2,040 |
| Fornace | `q4` | 225 | 957 | 43.7 | 1,079 |
| llama.cpp | `q4` | 246 | 752 | 102.1 | 984 |

- At C = 8, llama.cpp gives 1.61× (`bf16`) and 1.91× (`q4`) the Fornace
  total, and mistral.rs 1.17× (`bf16`). The llama.cpp `bf16` levels with
  a batch vary by 9 % between the two orders (467.9 and 511.7 at C = 8),
  so 1.54 to 1.69×.
- mistral.rs with Q4_0 (GGUF) failed at C = 4: "input tensor is not
  contiguous", shape `[2, 1, 896]`, when two prompts run in one step
  (`bench/runs/2026-11-13-38171e7-mistralrs-q4-error.txt`). It finished
  C = 1 (208.2 tok/s) and C = 2 (242.4).
- In its default mode mistral.rs gains 2.9× from C = 1 to C = 8. With
  paged attention off (the mode of the comparison result), it lost 13 %
  (`bench/runs/2026-11-13-14dd90f-three-engines.csv`).

### What one step of B rows costs (`aa23bc8`)

A test in Fornace (`tests/batch_step_cost.rs`): one decode step at 513
positions, fused attention, with B copies of `alice-512.txt`. The
quantized matmuls in two layouts: one matrix-vector launch for each row
(`[B, 1, K]`, as Candle does), and one matmul launch for all rows
(`[B, K]`). Dense `bf16` weights always take the matmul launch.

| B | `bf16` ms | `q8`, one launch for each row | `q8`, one matmul | `q4`, one launch for each row | `q4`, one matmul |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.24 | 4.45 | 4.44 | 3.35 | 3.38 |
| 2 | 13.33 | **6.62** | 12.58 | **5.15** | 12.38 |
| 4 | 13.40 | **11.07** | 13.10 | **8.36** | 12.78 |
| 6 | 13.66 | 15.41 | **13.38** | **11.77** | 13.09 |
| 8 | 13.72 | 19.74 | **13.69** | 15.01 | **13.49** |

The scheduler takes the faster layout: one matmul from B = 5 (`q8`) and
from B = 7 (`q4`). The rows of a batch keep their own caches, so the KV
append and the attention run once for each row. That adds 2.4 ms at
B = 8 (`tests/batch_step_cost.rs`, `row_caches_step_cost`).

## Cause and workaround

**The queue was the larger part of the wait.** With one request at a
time, the 8th request of a level waits for the 7 before it, about 0.6 s
each at `q4` and 1 s at `bf16`. The scheduler moves at most one waiting
request into the batch before each step. Its prompt runs alone, then it
takes part in each step. So a request waits for the prompts of the
requests before it and a few steps, not for whole requests. At C = 8 the
p50 first token falls from 6.3 s to 0.22 s (`bf16`) and from 3.9 s to
0.24 s (`q4`).

**The batched step is the rest of the gap.**

1. *The floor of Candle's matmul kernels for 2 to 8 rows.* A `bf16` step
   of 2 rows costs 2.14× a step of one row, and a step of 8 rows only
   2.20×. The quantized matmul for more than one row on the M axis has
   the same floor, about 12 ms. One row runs the matrix-vector kernel
   (GEMV); two rows or more run the tiled matmul kernel. The floor is
   almost the same in all three formats and grows only 0.1 to 0.2 ms for
   each row, so it is not the read of the weights. Its cause is not
   measured. A Metal System Trace of one step at B = 2 would show which
   kernels take the time. This is a Candle limit on Metal. llama.cpp runs
   a step of 4 rows in 1.33× the GPU time of one row (comparison result,
   Metal System Trace). This gap is the candidate for the next part of
   this work. This result does not promise a fix.
2. *One matrix-vector launch for each row with quantized weights.* For
   an input `[B, 1, K]`, Candle's quantized Metal matmul launches its
   kernel once for each of the B rows
   (`candle-core/src/quantized/metal.rs`, lines 303 to 336), so each row
   reads all weights again. At `q4` each row adds 1.7 ms. Fornace avoids
   it from B = 7 (`q4`) and B = 5 (`q8`) with one `[B, K]` matmul, which
   meets the floor of point 1.
3. *One KV append and one attention launch for each row.* Fornace keeps
   a cache for each request, because the fused decode attention kernel
   of Candle takes no mask or row lengths. With 8 rows this is about 800
   more kernel launches in a step, 2.4 ms. A fused kernel with a length
   for each row would remove it.
4. *The sampling.* The first scheduler sampled each row from its own
   slice of the logits on the GPU, so each row needed one copy to the
   CPU. One copy for all rows (`14dd90f`) gave 1 to 2 % at C = 8.

One Metal System Trace of a Fornace `q4` step with 8 rows
(`bench/runs/2026-11-13-14dd90f-trace-q4-batch8.txt`, traced, for
explanation only): 14.7 ms of GPU time and one copy to the CPU. The
trace before the sampling fix
(`bench/runs/2026-11-13-105e894-trace-q4-batch8.txt`) counts 25 compute
command buffers in such a step.

**`bf16` with 2 requests is slower than one at a time** (129.7 against
141.0 tok/s at C = 2), because of point 1. A rule that runs 2 rows as
two steps of one row would avoid it. It is not in this result.

**There is no workaround in Fornace today** for the floor of point 1.
Use llama.cpp if an agent sends many requests at once and the tokens
per second matter most.

## What it means if you run models on your own hardware

If an agent or several tasks use one local server, the server must batch
requests. One request at a time makes the last of 8 requests wait 5 to 8
seconds for its first word on an M4 Pro with a 0.5B model. With a
scheduler, all three engines give the typical request its first token in
0.2 to 0.5 s, and the slowest in 0.7 to 1.5 s. The tokens per second
differ more: at 8 requests, llama.cpp gives 1.6 to 1.9× Fornace, and
mistral.rs 1.17× Fornace at BF16. mistral.rs needs a build with Xcode's
Metal compiler for its batching to work, and its GGUF path failed with
several requests at once. Memory stays low in all three: 1.0 to 2.1 GiB
for the server with 8 requests.

## Correctness

- **The same tokens in a batch as alone.** On the CPU in F32, requests
  that join and leave a batch give exactly the tokens and the text of
  each request alone (`tests/batch_api.rs`, `tests/rows.rs`). On Metal,
  8 agent requests in one batch against each alone
  (`bench/runs/2026-11-13-71cae6e-batch-tokens.txt`): `q8` and `q4` give
  the same tokens in 8 of 8 requests. `bf16` gives the same tokens in 5
  of 8. The 3 others differ at tokens 10, 11, and 13, where the two
  largest logits differ by 0.375, 0.062, and 0.125: one to three BF16
  steps. A batch runs the matmul kernel, not the matrix-vector kernel, so
  the sums come in another order and a near tie can change.
- **The same tokens as the other engines.** One request to each server
  gives the same 64 greedy tokens of `alice-512.txt` in all three
  engines and all three formats
  (`bench/runs/2026-11-13-71cae6e-same-tokens.csv`).
- **Each server counted the Fornace prompt tokens** of each request (for
  mistral.rs, the request without a stream before each level, because it
  sends no usage in a stream).

## The predictions against the measurement

Written before the runs, after the step tables.

| Prediction | Measured | |
|---|---|---|
| `q4` at C = 8: 270 to 340 tok/s, 1.15 to 1.45× the queue | 312.3, 1.33× | Held |
| `bf16` at C = 8: 270 to 340 tok/s, 1.9 to 2.3× the queue | 299.6, 2.11× | Held |
| `bf16` at C = 2 gives less than at C = 1 | 129.7 against 141.0 | Held |
| At C = 8, `bf16` and `q4` give about the same total | 300 and 312 | Held |
| The largest first token at C = 8: 0.8 to 1.5 s | 0.91 and 0.99 s | Held |
| The first token p50 at C = 8: 150 to 400 ms | 221 and 239 ms | Held |
| The decode of one request at C = 8: 40 to 60 tok/s | 42.9 and 42.7 | Held |
| At C = 1, within 2 % of the queue | −0.8 to +0.1 % | Held |
| llama.cpp gives 1.6 to 2.2× the Fornace total at C = 8 | 1.61× (`bf16`), 1.91× (`q4`) | Held |
| (step 1) With one matmul, a `q4` step of 8 rows costs less than 2× one row | 4.0× | Failed. The floor |
| (step 2) A cache for each row adds less than 0.3 ms at B = 8 | 2.4 ms | Failed |
| (step 6) One copy of the logits saves about 12 % of an 8-row step | 1 to 2 % | Failed. The traced copies included the cost of the tracer |

## What this does not prove

- One machine (M4 Pro), one model (Qwen2.5-0.5B), `bf16` and Q4_0 only.
- One synthetic agent load, not the trace of a real agent. Concurrency 1
  to 8 only.
- Greedy decoding, no repeat penalty, the end-of-sequence token ignored.
- The prompt caches are off in all engines. The requests share a system
  prompt of 256 tokens, and a cache of it would help each engine.
- The scheduler of Fornace is first come, first served, with the prompt
  of a new request run alone between two steps. llama.cpp puts the
  prompt tokens into the decode step. A different policy can change the
  first-token numbers.
- The step tables and the thresholds of the matmul layout are for this
  machine and this model. Another GPU or model can have other crossover
  points.
- The cause of the floor of Candle's matmul kernels is not measured.
- Each reference engine ran at one version, with settings that were not
  tuned. mistral.rs needs Xcode's Metal compiler for its default mode.
  llama.cpp `bf16` with a batch varies by up to 15 % from run to run.

## Reproduce

Commit `38171e7`. On macOS, with Homebrew and Xcode with its Metal
Toolchain (`xcodebuild -downloadComponent MetalToolchain`).

```sh
git clone https://github.com/inertialabsxyz/fornace && cd fornace
git checkout 38171e7
make model && make quantize
scripts/get-reference-engines.sh --install     # llama.cpp, and mistral.rs twice
scripts/make-reference-weights.sh ./models/qwen2.5-0.5b-instruct
cargo build --release --features metal -p fornace-server -p fornace-bench
fb() { target/release/fornace-bench --model ./models/qwen2.5-0.5b-instruct \
         --transport http --attention fused --repeat-penalty 1.0 --load agent \
         --concurrency 1,2,4,8 --parallel 8 --out bench/runs/repro-scheduler.csv "$@"; }
fb --engines fornace --dtypes f16,q4 --max-batch 1      # the queue
fb --engines fornace --dtypes f16,q4                    # the scheduler
fb --engines llamacpp --dtypes f16,q4
MISTRALRS_TAG=v0.9.5-xcode fb --engines mistralrs --mistralrs-paged --dtypes f16
```

`--dtypes f16` gives `bf16` on Metal. The summary at the end of each
command gives, for each level: the p50 and the largest first token, the
p50 decode tok/s, the total tok/s, and the peak memory.

Expected at C = 8: the queue near 142 (`bf16`) and 235 (`q4`) total
tok/s, the scheduler near 300 and 315, llama.cpp near 470 to 510 and 600,
mistral.rs near 355 (`bf16`). The largest first token of the queue near
8.3 and 5.0 s, of the scheduler near 0.9 and 1.0 s. Speed drifts between
commands, so compare within one run.

The step table and the token checks:

```sh
cargo test --release --features metal -p fornace-engine \
  --test batch_step_cost -- --ignored --nocapture --test-threads=1
cargo test --release --features metal -p fornace-bench \
  --test batch_tokens -- --ignored --nocapture
FORNACE_MODEL_DIR=$PWD/models/qwen2.5-0.5b-instruct \
  cargo test --release -p fornace-engine --test batch_api --test rows
```

Expected: the step table near the one above; "none" for all `q8` and
`q4` requests and differences at tokens 10, 11, and 13 for three `bf16`
requests; 5 tests pass.
