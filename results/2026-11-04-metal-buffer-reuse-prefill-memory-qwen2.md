# A 1 MiB KV cache tensor holds a 512 MiB buffer in each layer, and makes a 4,096-token Qwen2 prefill on Metal peak at 14.5 GiB instead of 3.4 GiB

## Why it matters

On a Mac, the GPU and the CPU share one memory. A 0.5B model with about
1 GB of weights should fit easily in 24 GB. A `Tensor::cat` KV cache, as
in Candle's Qwen2, makes the peak much larger than the model. With it, a
4,096-token prompt takes the process to 14.5 GiB. At 8,192 tokens it
would not fit in 24 GB (this is an estimate from the T² growth, not a
measurement). The cause is a reuse rule in the Metal buffer pool. A cap
on that rule, as in the open Candle PR 3851, removes the extra memory.
Speed does not change.

## What was measured and how

| Item | Value |
|---|---|
| Machine | Mac, Apple silicon, 24 GB unified memory, macOS 27.2 |
| Engine, before | fornace-engine at `bd42949`, Candle 0.11.0 from crates.io |
| Engine, after | fornace-engine at `243a9a2`, Candle 0.11.0 with the reuse cap of PR 3851 (fork `inertialabsxyz/candle`, commit `79fe58df`) |
| Model | Qwen2.5-0.5B-Instruct, BF16 on Metal |
| Prompts | `bench/prompts/alice-512.txt`, `alice-2048.txt`, `alice-4096.txt` (513, 2,049, and 4,097 tokens) |
| KV caches | `cat` (`Tensor::cat`, as in Candle's Qwen2), `cat-contiguous` (`cat` after a contiguous copy of K and V), `prealloc` (`candle_nn::kv_cache::KvCache`) |
| Generation | 128 tokens for each run, end-of-sequence ignored |
| Sampler | Greedy, repeat penalty 1.1 over the last 64 tokens |
| Method | `fornace-bench`, untraced: one process for each prompt and KV cache, 1 warm-up, 3 recorded runs. All runs on 2026-11-04 |
| Data, before | `bench/runs/2026-11-04-bd42949-before.csv` |
| Data, after | `bench/runs/2026-11-04-243a9a2-after.csv` |

Peak memory is the largest footprint of the process since it started, in
MiB (2²⁰ bytes). The table gives the largest of the 3 runs. The other
columns are the mean of the 3 runs.

| Prompt | KV cache | Peak MiB, before → after | Prefill tok/s | Time to first token, ms | Decode tok/s |
|---|---|---|---|---|---|
| 513 | `cat` | 1,375.9 → 1,198.3 | 4,027 → 4,087 | 127.4 → 125.5 | 85.9 → 88.3 |
| 513 | `cat-contiguous` | 1,338.2 → 1,187.1 | 4,065 → 4,215 | 126.2 → 121.7 | 131.3 → 134.2 |
| 513 | `prealloc` | 1,165.5 → 1,165.7 | 4,102 → 4,131 | 125.1 → 124.2 | 136.9 → 135.9 |
| 2,049 | `cat` | 4,627.7 → 1,773.5 | 2,135 → 2,147 | 960.6 → 954.3 | 40.7 → 40.8 |
| 2,049 | `cat-contiguous` | 4,487.8 → 1,761.8 | 2,194 → 2,168 | 933.8 → 945.3 | 103.9 → 103.3 |
| 2,049 | `prealloc` | 1,683.2 → 1,683.7 | 2,192 → 2,172 | 934.8 → 943.4 | 109.2 → 107.6 |
| 4,097 | `cat` | **14,872.1 → 3,487.3** | 1,226 → 1,260 | 3,341.4 → 3,255.2 | 23.0 → 23.5 |
| 4,097 | `cat-contiguous` | **14,366.9 → 3,469.7** | 1,227 → 1,286 | 3,339.3 → 3,185.8 | 74.9 → 76.9 |
| 4,097 | `prealloc` | 3,272.9 → 3,403.2 | 1,289 → 1,286 | 3,178.5 → 3,185.1 | 82.7 → 81.6 |

- With the cap, the three caches have almost the same peak at each
  length.
- Without the cap, `cat` and `cat-contiguous` have almost the same peak.
  The contiguous copy is not the cause.
- The speed differences are within 5 %, in both directions. This result
  does not claim a speed gain.
- The low decode speed of `cat` is a different effect. Its cache is not
  contiguous, so each decode step copies the full cache with strided
  kernels. The cap does not change it.

## Cause and workaround

**Cause.** The pool gives a request the smallest free buffer that is
large enough. It has no upper limit on the size
(`candle-core-0.11.0/src/metal_backend/device.rs:472-486`). A buffer is
free when no tensor holds it. Attention builds a score matrix of
[14, T, T] in BF16 in each layer: 470 MB at T = 4,096, in a 512 MiB
buffer.

A `Tensor::cat` cache keeps the K tensor of each layer after RoPE
(`crates/fornace-engine/src/cache.rs:109`). This tensor is 1 MiB. RoPE
gives a new contiguous K, so `cat` and `cat-contiguous` keep the same K.
From layer 3, this K gets a free 512 MiB score buffer of the layer
before, and it holds that buffer until decode replaces it. So each later
layer must allocate a new score buffer. At the end of prefill, the pool
has 25 buffers of 512 MiB: 13.4 GB.

An allocation log in Candle 0.11.0, with the `cat-contiguous` cache,
shows each step:

| Layer | New 512 MiB buffers | Held 512 MiB buffers at the softmax |
|---|---|---|
| 0 | 3 | 2 |
| 1 | 0 | — |
| 2 | 1 | 3 |
| 3 to 23 | 1 each | 4, 5, … 24 |

Logs: `bench/runs/2026-11-02-5514568-alloc-*-4096.txt`. These come from
a build with log lines added to Candle. The first line of each file says
so. The logs are for `cat-contiguous` and `prealloc` only.

The `prealloc` cache does not have the problem. It copies K and V into
one buffer that `Tensor::zeros` allocates, and `Tensor::zeros` uses a
separate pool on Metal (`device.rs:269,294`). The K tensor after RoPE is
then a temporary, and it frees its small buffer at once.

A GPU wait hides the problem. After a wait, Candle gives all free
buffers back to Metal (`drop_unused_buffers`, `device.rs:131`). Then no
large free buffer is left for a small tensor to take. With a wait around
each layer and op (a debug trace), the `cat-contiguous` peak at 4,096
tokens is 2,981.4 MiB. This number comes from a traced run at commit
`af15bae` (`bench/runs/2026-11-02-af15bae-trace-cat-contiguous-debug-4096.txt`).

**Workaround.** Three ways, best first:

1. **Cap the reuse size in the pool.** Reuse a buffer only if it is at
   most two power-of-two size classes above the rounded request, so at
   most 4× its size. This is the cap of the open Candle PR 3851:

   ```diff
   +const REUSE_SLACK: u32 = 2;
   +
   +fn reuse_cap(size: usize) -> usize {
   +    buf_size(size).saturating_mul(1usize << REUSE_SLACK)
   +}
   ...
    fn find_available_buffer(size: usize, buffers: &BufferMap) -> Option<Arc<Buffer>> {
   +    let cap = reuse_cap(size);
        ...
   -        if buffer_size >= &size && buffer_size < &best_buffer_size {
   +        if buffer_size >= &size && buffer_size <= &cap && buffer_size < &best_buffer_size {
   ```

   This is the "after" column. Fornace uses it from `243a9a2` through a
   `[patch.crates-io]` entry. The patch must cover `candle-core`,
   `candle-nn`, and `candle-metal-kernels`, so that they come from one
   source.
2. **Use a preallocated KV cache** (`candle_nn::kv_cache::KvCache`). The
   `prealloc` rows show the peak without a cap.
3. **Use the fused attention kernel** (`candle_nn::ops::sdpa`). It does
   not build the score matrix, so there is no large buffer to take. This
   is a separate result.

## What it means if you run models on your own hardware

If you run a Candle model on Metal with a long prompt, look at the
memory of the process, not at the size of the model. A small tensor that
lives long can hold a large buffer. Until the pool has a cap, use a
preallocated KV cache, or a fused attention kernel. The extra memory
grows with the square of the prompt length, so a prompt that fits today
can fail at twice the length.

## What this does not prove

- All runs are from one day, on one machine, with one model and one dtype
  (BF16).
- The factor 4 in the cap comes from PR 3851. This result did not tune
  it. With the cap, the `prealloc` peak is 4 % higher at 4,097 tokens.
- These runs did not compare the generated text with and without the cap.
- The allocation log covers only `cat-contiguous` and `prealloc`, and
  only buffers of 64 MiB or more. About 8 MiB in each layer is not
  explained.
- Other models in Candle were not measured. The same pool serves all
  models, so the same effect is possible where a small tensor lives long.
- The 8,192-token number in "Why it matters" is an estimate.

## Reproduce

Before (crates.io Candle), commit `bd42949`. After (with the cap of
PR 3851), commit `243a9a2`.

```sh
git clone https://github.com/inertialabsxyz/fornace && cd fornace
make model   # downloads Qwen2.5-0.5B-Instruct to ./models/qwen2.5-0.5b-instruct
for commit in bd42949 243a9a2; do
  git checkout $commit
  for T in 512 2048 4096; do
    for KV in cat cat-contiguous prealloc; do
      cargo run --release --features metal -p fornace-bench -- \
        --model ./models/qwen2.5-0.5b-instruct \
        --kv-cache $KV \
        --prompts bench/prompts/alice-$T.txt \
        --out bench/runs/repro-$commit.csv
    done
  done
done
git checkout main
```

Expected: at `alice-4096.txt`, the summary shows a `cat` peak of about
14,900 MiB and a `cat-contiguous` peak of about 14,400 MiB at `bd42949`.
At `243a9a2`, both are about 3,500 MiB. The `prealloc` peak is about
3,200–3,400 MiB at both commits.

To check that the build uses the fork at `243a9a2`:

```sh
cargo tree -p fornace-engine --features metal | grep "candle-core v"
```

Expected: `candle-core v0.11.0 (https://github.com/inertialabsxyz/candle?rev=79fe58df…)`.
