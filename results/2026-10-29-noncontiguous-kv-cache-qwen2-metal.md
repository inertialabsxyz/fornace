# A non-contiguous KV cache cuts Qwen2 decode speed by 59 % at 2k context in Candle on Metal

## Why it matters

Decode speed is the rate at which text appears. In Candle's Qwen2, V is
not contiguous after prefill. So each `Tensor::cat` that appends to the
KV cache copies the full cache with strided kernels, in all 24 layers,
for each token. The cost increases with the context: `cat` is 33 %
slower at 513 tokens and 59 % slower at 2,049 tokens. One `.contiguous()`
call before the append removes the cost. Candle PR #4077 has the fix. A
related bug is open for gemma3 (PR #3902): V is also not contiguous, but
the cache uses `slice_set`, so the model fails. Other models in Candle
can have the same problem.

## What was measured and how

| Item | Value |
|---|---|
| Machine | Mac, Apple silicon, 24 GB unified memory |
| Engine | fornace-engine at commit `e76e1ba`, Candle 0.11.0 |
| Model | Qwen2.5-0.5B-Instruct, BF16 on Metal |
| Prompts | `bench/prompts/alice-512.txt` (513 tokens), `bench/prompts/alice-2048.txt` (2,049 tokens) |
| Generation | 128 tokens for each run, end-of-sequence ignored |
| Sampler | Greedy, repeat penalty 1.1 over the last 64 tokens |
| Method | `fornace-bench`: one process for each KV cache, 1 warm-up, 3 recorded runs |
| Data | `bench/runs/2026-11-02-e76e1ba.csv` |

At 2,049 tokens:

| KV cache | Decode tok/s (runs) | Mean | Mean ms, last 64 steps | Peak memory |
|---|---|---|---|---|
| `cat` (Candle original) | 41.8, 41.9, 42.0 | 41.9 | 24.1–24.2 | 3,373 MiB |
| `cat-contiguous` (the fix) | 102.2, 101.0, 102.0 | 101.7 | 9.8–10.0 | 4,626 MiB |
| `prealloc` (`slice_set` into a fixed buffer) | 105.3, 104.9, 104.8 | 105.0 | 9.5–9.6 | 1,666–1,683 MiB |

At 513 tokens:

| KV cache | Decode tok/s (runs) | Mean | Mean ms, last 64 steps | Peak memory |
|---|---|---|---|---|
| `cat` | 89.2, 88.8, 88.3 | 88.7 | 11.4–11.5 | 1,362 MiB |
| `cat-contiguous` | 130.5, 131.8, 133.8 | 132.0 | 7.5–8.0 | 1,340 MiB |
| `prealloc` | 135.4, 136.1, 132.9 | 134.8 | 7.4–7.6 | 1,164–1,165 MiB |

Peak memory is the largest footprint of the process since it started.
It includes prefill. Each process ran the 513-token prompt first.

## Cause and workaround

**Cause.** After prefill, V is a transposed view (`model.rs:203-205`).
Candle's Qwen2 keeps this view in the cache (`qwen2.rs:181-189`). When an
input is not contiguous, `Tensor::cat` transposes, joins with strided
copies, and transposes back (`candle-core/src/tensor_cat.rs:61-76`). The
result is also not contiguous, so the next step pays again. K is
contiguous, because RoPE writes a new tensor.

**Workaround.** Call `.contiguous()` on K and V before the `cat`
(`cache.rs:97-99`). That is the full change. `prealloc` is fast for the
same reason: `slice_set` accepts only a contiguous source, so the cache
calls `.contiguous()` first (`cache.rs:114`).

**Peak memory.** At 2,049 tokens, the peak of `prealloc` is
1,666–1,683 MiB, and the peak of `cat-contiguous` is 4,626 MiB. The KV
cache does not cause this difference: K and V for all 24 layers are only
about 25 MB. A `--trace summary` run at commit `0e81d96` shows that the
difference is in prefill. With `cat-contiguous`, prefill adds 2,925 MB
of Metal memory, and decode releases 2,871 MB of it. With `prealloc`,
prefill adds 71 MB. The cause is not known yet.

**Upstream numbers.** Candle PR #3143 found `cat` 2–5× faster than
`slice_set` on GPU. This agrees with a comparison where both inputs were
not contiguous. With contiguous inputs, the two methods are within 4 %.

## What it means if you run models on your own hardware

If you use Candle's Qwen2, or a model with the same cache code, each
token copies the full cache. Until the fix is merged, add `.contiguous()`
before the cache append, or use a preallocated cache. Decode is then
about 1.5× faster at 513 tokens and 2.4× faster at 2,049 tokens.

## What this does not prove

- All runs are from one day.
- The data is from one machine, one model, and one dtype (BF16).
- The cause of the lower peak memory of `prealloc` is not known.

## Reproduce

Commit `e76e1ba`.

```sh
git clone https://github.com/inertialabsxyz/fornace && cd fornace
git checkout e76e1ba
make model   # downloads Qwen2.5-0.5B-Instruct to ./models/qwen2.5-0.5b-instruct
cargo run --release --features metal -p fornace-bench -- \
  --model ./models/qwen2.5-0.5b-instruct \
  --kv-cache cat,cat-contiguous,prealloc \
  --prompts bench/prompts/alice-512.txt,bench/prompts/alice-2048.txt
```

Expected: at `alice-2048.txt`, the summary table shows about 42 tok/s
for `cat` and about 100 tok/s for the other two caches. The CSV goes to
`bench/runs/<date>-e76e1ba.csv`.

One run with the engine:

```sh
cargo run --release -p fornace-engine --features metal --bin fornace -- \
  --model ./models/qwen2.5-0.5b-instruct \
  --prompt "$(cat bench/prompts/alice-2048.txt)" \
  --max-new-tokens 128 --ignore-eos --kv-cache cat
```

Expected: about 40 tok/s with `--kv-cache cat`, and about 100 tok/s with
`--kv-cache cat-contiguous`. The unit test
`all_kinds_return_the_same_tensors` (`cache.rs:140-171`) asserts two
things: the `cat` cache is not contiguous at each step, and all three
caches return the same values.
