# Batch 4 gives 1.6× the total decode speed of batch 1 for Qwen2 on Metal, but each sequence decodes at 0.40×

## Why it matters

One user on a laptop can have several sequences to run at the same time:
agent subtasks, best-of-n answers, or an evaluation over a dataset. A
batch runs them in one forward pass, and the weights are read once for
all rows. Decode is limited by memory bandwidth, so a simple estimate says
that a batch of 4 costs almost the same as one sequence. On this Mac, a
batch of 4 gives 1.6× the total tokens per second, not 4×. Each sequence
gets its tokens at 0.40× the speed. The cause is a change of matmul
kernel between one row and two rows.

## What was measured and how

| Item | Value |
|---|---|
| Machine | Apple M4 Pro, 20 GPU cores, 24 GB unified memory, macOS 27.2 |
| Engine | fornace-engine at `dae67a1`, Candle 0.11.0 with the reuse cap of PR 3851 (fork `inertialabsxyz/candle`, commit `79fe58df`) |
| Model | Qwen2.5-0.5B-Instruct, BF16 on Metal |
| Batch | Static. B copies of one prompt, all of equal token count, no padding (`Engine::generate_batch`) |
| Prompts | `bench/prompts/alice-128.txt`, `alice-512.txt`, `alice-2048.txt` (129, 513, and 2,049 tokens) |
| KV cache | `prealloc` |
| Attention | `fused` (`candle_nn::ops::sdpa`), except case D: `naive` |
| Generation | 128 tokens for each sequence, end-of-sequence ignored |
| Sampler | Greedy, repeat penalty 1.1 over the last 64 tokens of each row, on the CPU |
| Method | `fornace-bench`, untraced: one process for each batch size, 1 warm-up, 3 recorded runs. One command for each case. All runs on 2026-11-05 |
| Data | `bench/runs/2026-11-05-dae67a1-batch.csv` (the case is in the `tag` column), `bench/runs/2026-11-05-dae67a1-batch-rerun.csv`, committed in `bd334e2` |

For B sequences, T prompt tokens each, and N new tokens each:

- Total decode tok/s is B × (N − 1) / decode time.
- Decode tok/s per sequence is the total / B.
- Total prefill tok/s is B × T / prefill time.
- Time to first token is the time to the first token of all sequences.

Peak memory is the largest footprint of the process since it started, in
MiB (2²⁰ bytes). All values are the mean of 3 runs.

**Case A: 512 tokens, fused attention.**

| | Batch 1 | Batch 4 | Ratio |
|---|---:|---:|---:|
| **Total decode, tok/s** | 159.1 | **256.8** | **1.61×** |
| Decode per sequence, tok/s | 159.1 | 64.2 | 0.40× |
| Decode step, ms | 6.26 | 15.59 | 2.49× |
| Total prefill, tok/s | 5,994 | 6,479 | 1.08× |
| Time to first token, ms | 85.6 | 316.7 | 3.7× |
| Peak memory, MiB | 1,153.5 | 1,289.7 | +136 |

**Case B: a short and a long prompt, fused attention.**

| Prompt tokens | Total prefill, B = 1 → 4 | Total decode, B = 1 → 4 | Per sequence at B = 4 | Time to first token, B = 1 → 4 | Peak MiB, B = 1 → 4 |
|---:|---|---|---:|---|---|
| 129 | 4,164 → 5,996 (1.44×) | 164.9 → 260.1 (1.58×) | 65.0 | 31.0 → 86.1 ms | 1,126 → 1,182 |
| 2,049 | 6,173 → 5,341 (0.87×) | 151.8 → 224.5 (1.48×) | 56.1 | 332 → 1,534 ms | 1,292 → 1,844 |

**Case C: batch size, 512 tokens, fused attention.**

| B | Total decode, tok/s | Per sequence, tok/s | Decode step, ms | Time to first token, ms | Peak MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 147.5 | 147.5 | 6.79 | 93 | 1,153 |
| 2 | 139.5 | 69.7 | 14.50 | 168 | 1,204 |
| 4 | 257.0 | 64.3 | 15.58 | 315 | 1,290 |
| 8 | **394.0** | 49.3 | 20.46 | 638 | 1,469 |
| 16 | 419.8 | 26.3 | 38.61 | 1,375 | 1,818 |

**Case D: 2,048 tokens, naive attention.**

| | Batch 1 | Batch 4 |
|---|---:|---:|
| Total decode, tok/s | 100.0 | 144.4 |
| Time to first token, ms | 1,080 | 4,862 |
| Peak memory, MiB | 1,666 | 3,113 |

- The total stops rising fast at B = 8: 2.7× batch 1 in case C. From
  B = 8 to B = 16 it rises only 7 %.
- Batch 2 gives *less* total than batch 1, in all 3 runs.
- Batch 1 in case C is 7 % slower than in case A, with the same
  settings. See "Speed drifts between runs" below. The ratio inside one
  command is stable: a rerun of case A gives 148.7 and 239.8 tok/s, also
  1.61×.

**Predictions.** Before the runs, an estimate from memory bandwidth
predicted these values. A decode step at batch 1 reads about 996 MB:
the weights and the KV cache. At 273 GB/s that takes 3.6 ms (*calc*).
The measured step is 6.3 ms.

| Prediction | Measured | Holds? |
|---|---|---|
| Total decode at batch 4 is 2.5 to 4× batch 1 | 1.61× | No |
| Decode per sequence stays the same or falls a little | 0.40× | No |
| Total prefill at 512 tokens changes little | 1.08× | Yes |
| Time to first token is about 4× higher | 3.7× | Yes |
| Total prefill at 128 tokens rises | 1.44× | Yes |
| Total decode rises, then flattens below B = 33 | Flattens at B = 8, with a dip at B = 2 | Yes, with the dip |
| Peak memory rises much more than the KV cache size | +1,447 MiB against about 80 MB (case D) | Yes |

## Cause and workaround

**Cause: a different matmul kernel for one row.** At decode, each linear
layer gets a `[B, 1, K]` input. `candle-nn` reshapes a contiguous input
to `[B, K]` before the matmul (`candle-nn/src/linear.rs:58-63`), so the
matmul has M = B rows. Candle's Metal matmul uses a matrix-vector kernel
(GEMV) when M = 1, and the tiled GEMM otherwise
(`candle-metal-kernels/src/kernels/mlx_gemm.rs:520`).

Thus batch 1 runs GEMV, and every batch of 2 or more runs GEMM. The step
time follows this:

| B | Decode step, ms (case C) | MLP time for each layer, traced, ms |
|---:|---:|---:|
| 1 | 6.79 | 0.30 |
| 2 | 14.50 | 0.53 |
| 4 | 15.58 | 0.82 |
| 8 | 20.46 | 0.81 |
| 16 | 38.61 | 0.84 |

The step more than doubles from 1 row to 2 rows. Then it rises little up
to 8 rows. A probable reason is that a GEMM tile holds more rows than 2
to 16, so a few rows pay for a full tile. The tile sizes were not checked.

**Cause after B = 8: sampling on the CPU.** Each row copies 151,936
logits to the CPU and applies the repeat penalty. In a debug trace, the
`sample` span is 1.5 % of a decode step at B = 1, 11 % at B = 8, and
27 % at B = 16. From B = 8 to B = 16, the untraced step grows by 18 ms,
and the traced `sample` span grows by 15 ms. Traces add GPU waits, so
these times explain the knee only. The traces are in `bench/runs/`
(`2026-11-05-dae67a1-trace-batch{1,2,4,8,16}.txt`).

**Not the cause: the cache read and attention.** At 2,048 tokens, batch 4
decodes at 224.5 tok/s, against 256.8 at 512 tokens. Thus the longer
cache costs about 13 %, from two commands that ran at different times.
The attention kernel stays below 0.25 ms for each layer at every batch
size in the traces.

**Workaround.** There is no workaround in Fornace today. A GEMM path for
a small M (2 to 16 rows), or a batched GEMV, could bring batch 4 nearer
to the estimate. Sampling on the GPU would move the knee after B = 8.
Neither is built.

## What it means if you run models on your own hardware

If you run several sequences at once on a Mac with Candle, measure batch
2 before you use it: on this machine it gives less total than batch 1.
Batch 4 gives 1.6× the total, and batch 8 gives 2.7×. But each sequence
then decodes at 0.40× and 0.33× the speed, and the first token comes 3.7×
and 6.9× later. Thus batching helps when you need all the answers, as in
an evaluation or best-of-n. It does not help when you wait for one
answer. Keep the prompts the same length, because this engine does not
pad.

## Correctness

A test compares each row of a batch of 4 with the same prompt at batch 1
(`crates/fornace-engine/tests/batch.rs`). It forces the batch-1 tokens
into the batch and compares the logits at each of 32 steps. It uses two
batches: 4 copies of `alice-512`, and 4 different 512-token windows of
`alice-2048`.

| Device, attention | Max diff after prefill | Max diff in 31 decode steps | First top token change |
|---|---:|---:|---|
| CPU, F32, naive | 0.0000 | 0.0001 | none |
| Metal, BF16, naive | 0.0000 | 0.83 | steps 4, 9, 18 (different prompts) |
| Metal, BF16, fused | 0.0000 | 0.52 | steps 9, 10 (different prompts) |

- Prefill is bit-equal to batch 1 in every case. There is no row mixup,
  offset error, or mask error.
- On Metal, decode logits differ. A probable cause is the kernel change
  above: different kernels add in a different order.
- The top token changes only where the batch-1 margin between the two
  best logits (0.000 to 0.19) is smaller than the difference. Thus the
  text changes only at near-ties.

## Speed drifts between runs

Batch 1 at 512 tokens gave 159.1 tok/s in case A. In case C and in 3
later runs, it gave 147.5 to 149.2. Batch 4 also fell, from 256.8 to
239.8. Inside one process, the spread of 3 runs is below 1 %. Between
commands at different times it is about 7 %. The order of the workers is
not the cause: batch 1 has the same speed alone, first, or after batch
4. macOS recorded no thermal or performance warning. The cause is not
known. Compare rows from one command.

## What this does not prove

- All runs are from one day, on one machine, with one model and one
  dtype (BF16).
- Prompts in one batch have equal token counts. The engine has no
  padding. Batches of prompts with different lengths were not measured.
- The fused decode kernel in Candle takes no attention mask. Thus a
  padded batch cannot use fused decode. This gap limits batches of mixed
  length.
- The batching is static. All sequences start together and run to the
  same length. Continuous batching was not measured.
- The cause of the step at B = 2 comes from the kernel choice in the
  source and from the step times. The GEMM tile sizes were not checked.
  The kernels were not timed apart from the traced spans.
- The memory above the KV cache is activations and pooled buffers. This
  result does not separate them.
- The cause of the 7 % drift between runs is not known.
- A batch of copies does not test sequences that end at different times.
  End-of-sequence was ignored.
- Other models and other Macs were not measured.

## Reproduce

Commit `dae67a1`.

```sh
git clone https://github.com/inertialabsxyz/fornace && cd fornace
make model   # downloads Qwen2.5-0.5B-Instruct to ./models/qwen2.5-0.5b-instruct
git checkout dae67a1
cargo build --release --features fornace-engine/metal -p fornace-bench
fb() {
  ./target/release/fornace-bench --model ./models/qwen2.5-0.5b-instruct \
    --kv-cache prealloc --max-new-tokens 128 --out bench/runs/repro-batch.csv "$@"
}
fb --batch 1,4 --prompts bench/prompts/alice-512.txt --attention fused --tag "case A"
fb --batch 1,4 --prompts bench/prompts/alice-128.txt,bench/prompts/alice-2048.txt --attention fused --tag "case B"
fb --batch 1,2,4,8,16 --prompts bench/prompts/alice-512.txt --attention fused --tag "case C"
fb --batch 1,4 --prompts bench/prompts/alice-2048.txt --attention naive --tag "case D"
```

The defaults give 1 warm-up, 3 recorded runs, and `--ignore-eos true`.

Expected: for case A, the summary shows about 150 to 160 tok/s at batch 1
and about 240 to 257 tok/s at batch 4, a ratio of about 1.6. In case C,
batch 2 is below batch 1, and batch 8 is near 2.7× batch 1.

The rerun in "Speed drifts between runs" (batch 1 alone in 3 processes,
then batch 4 before batch 1):

```sh
for i in 1 2 3; do
  fb --batch 1 --prompts bench/prompts/alice-512.txt --attention fused --tag "case A rerun; B=1 alone $i"
done
fb --batch 4,1 --prompts bench/prompts/alice-512.txt --attention fused --tag "case A rerun; order 4,1"
```

Expected: batch 1 is the same in all 4 commands, within 1 %. The ratio of
batch 4 to batch 1 in the last command is about 1.6.

The correctness test:

```sh
FORNACE_MODEL_DIR=./models/qwen2.5-0.5b-instruct \
  cargo test --release --features metal -p fornace-engine \
  --test batch -- --nocapture --test-threads=1
```

Expected: 3 tests pass, and the prefill difference is 0.0000 for every
row.
