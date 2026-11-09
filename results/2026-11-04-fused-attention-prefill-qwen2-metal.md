# Fused attention makes a 4,096-token Qwen2 prefill on Metal 4.5× faster and lowers its peak from 3,403 MiB to 1,470 MiB

## Why it matters

Candle has a fused attention kernel for Metal (`candle_nn::ops::sdpa`).
Candle's Qwen2 model does not use it. It computes attention in steps,
and each step writes a [14, T, T] score matrix to memory: 470 MB at
T = 4,096. The cost of these steps grows with the square of the prompt
length. Fornace adds the fused kernel as an option, `--attention fused`,
to measure what it gives. With a 4,096-token prompt and a preallocated
KV cache, prefill is 4.5× faster, the peak memory is less than half, and
decode is 75 % faster. Candle does not change. The kernel is already in
`candle-nn`.

## What was measured and how

| Item | Value |
|---|---|
| Machine | Mac, Apple silicon, 24 GB unified memory, macOS 27.2 |
| Engine | fornace-engine at `c3693e3`, Candle 0.11.0 with the reuse cap of PR 3851 (fork `inertialabsxyz/candle`, commit `79fe58df`) |
| Model | Qwen2.5-0.5B-Instruct, BF16 on Metal |
| Prompts | `bench/prompts/alice-512.txt`, `alice-2048.txt`, `alice-4096.txt` (513, 2,049, and 4,097 tokens) |
| Attention | `naive`: matmul, causal mask, softmax, matmul, after `repeat_kv`. `fused`: `candle_nn::ops::sdpa` on the grouped KV heads, causal in the kernel |
| KV caches | `cat` (`Tensor::cat`, as in Candle's Qwen2), `cat-contiguous`, `prealloc` (`candle_nn::kv_cache::KvCache`, the Fornace default) |
| Generation | 128 tokens for each run, end-of-sequence ignored |
| Sampler | Greedy, repeat penalty 1.1 over the last 64 tokens |
| Method | `fornace-bench`, untraced: one process for each prompt, KV cache, and attention, 1 warm-up, 3 recorded runs. All runs on 2026-11-04 |
| Data | `bench/runs/2026-11-04-c3693e3-attention.csv` (a fused row has `attention=fused` in the `tag` column) |

Peak memory is the largest footprint of the process since it started, in
MiB (2²⁰ bytes). The table gives the largest of the 3 runs. The other
columns are the mean of the 3 runs.

| Prompt | KV cache | Peak MiB, naive → fused | Prefill tok/s | Time to first token, ms | Decode tok/s |
|---|---|---|---|---|---|
| 513 | `cat` | 1,198.3 → 1,181.6 | 4,012 → 5,639 | 127.9 → 91.0 | 85.2 → 131.9 |
| 513 | `cat-contiguous` | 1,188.9 → 1,168.3 | 4,222 → 6,073 | 121.5 → 84.5 | 136.1 → 162.6 |
| 513 | `prealloc` | 1,164.7 → 1,153.9 | 4,203 → 5,951 | 122.1 → 86.2 | 137.7 → 157.0 |
| 2,049 | `cat` | 1,773.7 → 1,422.1 | 2,056 → 6,063 | 996.5 → 338.2 | 39.7 → 111.3 |
| 2,049 | `cat-contiguous` | 1,761.8 → 1,419.8 | 2,076 → 5,982 | 986.8 → 342.6 | 99.9 → 141.9 |
| 2,049 | `prealloc` | 1,683.3 → 1,287.8 | 2,083 → 5,949 | 983.8 → 344.4 | 106.2 → 147.6 |
| 4,097 | `cat` | 3,487.5 → 1,741.5 | 1,216 → 5,506 | 3,369.6 → 744.1 | 22.6 → 86.5 |
| 4,097 | `cat-contiguous` | 3,476.0 → 1,736.4 | 1,227 → 5,504 | 3,339.6 → 744.4 | 74.2 → 131.3 |
| 4,097 | `prealloc` | **3,402.8 → 1,470.4** | **1,265 → 5,770** | 3,238.5 → 710.0 | 81.3 → 142.5 |

- Fused prefill stays at 5,500 to 6,100 tok/s from 513 to 4,097 tokens.
  Naive prefill falls from about 4,200 to about 1,250 tok/s.
- With `prealloc`, decode is 14 % faster at 513 tokens, 39 % at 2,049,
  and 75 % at 4,097.
- `prealloc` with fused attention has the lowest peak and the fastest
  decode at 2,049 and 4,097 tokens.
- One group spread more than 5 %: 513 tokens, `cat`, fused decode
  (8.2 %).

## Cause and workaround

**Cause.** The naive path runs attention as separate ops
(`crates/fornace-engine/src/model.rs`, the `AttentionKind::Naive` arm).
First, `repeat_kv` copies the 2 KV heads to the 14 query heads. Then a
matmul writes the [14, T, T] scores. The scale, the mask, and the
softmax each write them again, and a last matmul reads them. Each op is a separate Metal kernel,
and each one reads and writes the full score matrix. Both the time and
the memory grow with T².

The fused kernel does all of this in one pass. It reads the grouped KV
heads directly, it applies the causal mask in the kernel, and it never
stores the score matrix (`candle-nn-0.11.0/src/ops.rs:1032-1330`,
`candle-metal-kernels-0.11.0/src/kernels/sdpa.rs`).

**Workaround.** Call `candle_nn::ops::sdpa` in place of the naive steps.
In Fornace it is `fused_attention` (`model.rs:87`), selected with
`--attention fused`. Three details matter:

1. Give K and V before `repeat_kv`. The kernel takes grouped KV heads
   when the query heads are a multiple of the KV heads.
2. Set `do_causal` only when more than one query token is present
   (`q_len > 1`, `model.rs:321`). With `do_causal`, the kernel aligns the
   queries to the end of the keys, so a chunk at an offset sees all
   earlier keys.
3. For one query token, the kernel reads only the head stride of K and V
   (`sdpa.rs:285-286`). It assumes that the rows of each head are next to
   each other. The `cat` cache stores V as a transposed view, so Fornace
   copies it first (`dense_rows`, `model.rs:103`).

The kernel has no CPU version. On the CPU, `Engine::load` returns
`FusedAttentionNeedsMetal` (`crates/fornace-engine/src/lib.rs:190`).

## What it means if you run models on your own hardware

If you run a Candle model on Metal, check whether its attention uses
`candle_nn::ops::sdpa`. If it does not, long prompts pay a cost that
grows with the square of their length, in time and in memory. On this
Mac, the fused kernel makes a 4,096-token prompt ready in 0.71 s in
place of 3.24 s. Use it with a preallocated KV cache for the best decode
speed and the lowest peak.

## Correctness

BF16 rounds differently in the two kernels. A test gives the same tokens
to both paths and compares the logits at each step
(`crates/fornace-engine/tests/fused_attention.rs`). It fails if the top
token changes at a step where the naive margin between the two best
logits is larger than the difference between the two paths. It passes at
all three lengths:

| Prompt | Max diff after prefill | Max diff in 31 decode steps | Steps where the top token changes (naive margin) |
|---|---|---|---|
| 512 | 0.875 | 2.06 | 8 (0.125), 14 (0.625) |
| 2,048 | 0.563 | 0.81 | 7 (0.125), 27 (0.000) |
| 4,096 | 0.438 | 0.88 | 1 (0.000), 2 (0.125), 10 (0.188) |

Output: `bench/runs/2026-11-04-c3693e3-fused-logits.txt`. The largest
logits are about 15. So the generated text changes, but only where two
tokens are almost equal.

## What this does not prove

- All runs are from one day, on one machine, with one model and one dtype
  (BF16).
- The BF16 logit differences in "Correctness" include an error in the
  RoPE tables of this commit: positions above 256 were not exact in BF16
  (fixed in `5ea3a39`, see the quantization result). The speed and memory
  numbers do not change.
- The generated text changes at near-ties. This result does not measure
  the quality of the text on a task. Do that before fused attention
  becomes the default.
- This result does not compare the two paths in F32.
- The decode gain has two possible causes: no `repeat_kv` copy, and one
  kernel in place of five (matmul, scale, mask, softmax, matmul). This
  result does not separate them.
- With the `cat` caches, the fused peak at 4,097 tokens is about 270 MiB
  higher with the 4× pool cap than it was with an earlier 2× cap
  (1,741.5 against 1,467.2 MiB, `bench/runs/2026-11-02-ce22896-attention.csv`).
  `prealloc` does not change. The cause is probably the larger reuse
  limit. No allocation log confirms it.
- The kernel does not apply a sliding window. For Qwen2.5 the window is
  32,768, the same as `max_position_embeddings`, so it never masks a
  position. Other models can differ.
- Other models in Candle were not measured.

## Reproduce

Commit `c3693e3`.

```sh
git clone https://github.com/inertialabsxyz/fornace && cd fornace
make model   # downloads Qwen2.5-0.5B-Instruct to ./models/qwen2.5-0.5b-instruct
git checkout c3693e3
for T in 512 2048 4096; do
  for KV in cat cat-contiguous prealloc; do
    for A in naive fused; do
      cargo run --release --features metal -p fornace-bench -- \
        --model ./models/qwen2.5-0.5b-instruct \
        --kv-cache $KV --attention $A \
        --prompts bench/prompts/alice-$T.txt \
        --out bench/runs/repro-attention.csv
    done
  done
done
```

Expected: at `alice-4096.txt` with `prealloc`, the summary shows a peak
of about 3,400 MiB with `naive` and about 1,470 MiB with `fused`. Prefill
is about 1,250 tok/s with `naive` and about 5,800 tok/s with `fused`.

The correctness test:

```sh
FORNACE_MODEL_DIR=./models/qwen2.5-0.5b-instruct \
  cargo test --release --features metal -p fornace-engine \
  --test fused_attention -- --nocapture
```

Expected: `test fused_logits_are_close_to_naive ... ok`, and the same
steps and margins as the table in "Correctness".
