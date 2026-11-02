# A non-contiguous KV cache cuts Qwen2 decode speed by 59 % at 2k context in Candle on Metal

> **Correction, 2026-11-02.** The first version of this result gave
> 58, 123, and 115 tok/s at 2,048 tokens, and a peak of about 1.4 GB for
> each cache. These numbers do not reproduce. The same commands at the
> same commit (`f630e02`) now give about 39, 99, and 105 tok/s, and a
> peak of up to 4.6 GB. The Candle version and the macOS version are the
> same. We do not know the cause. The claim did not change: the
> contiguous caches are about 2.5× faster than `cat` at 2,048 tokens.
> The numbers below are from the benchmark harness. The explanation of
> the lower peak memory of `prealloc` was wrong, and it is replaced.

## Why it matters

Decode speed is what a user feels: it is the rate at which text appears.
Candle's Qwen2 implementation appends to the KV cache with `Tensor::cat`.
After prefill, V is a transposed view and is not contiguous, so every
later `cat` copies the full cache with strided kernels, in all 24 layers,
on every token. The cost grows with the context length: `cat` is 33 %
slower at 512 tokens, and 59 % slower at 2,048. One `.contiguous()` call
before the append removes it. Candle PR #4077 has the fix. The same class
of bug is open for gemma3 (PR #3902), so other models in Candle may have
it.

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
| `cat` (Candle original) | 41.8, 41.9, 42.0 | 41.9 | 24.1 | 3,373 MiB |
| `cat-contiguous` (the fix) | 102.2, 101.0, 102.0 | 101.7 | 9.8–10.0 | 4,626 MiB |
| `prealloc` (`slice_set` into a fixed buffer) | 105.3, 104.9, 104.8 | 105.0 | 9.5–9.6 | 1,666–1,683 MiB |

At 513 tokens:

| KV cache | Decode tok/s (runs) | Mean | Mean ms, last 64 steps | Peak memory |
|---|---|---|---|---|
| `cat` | 89.2, 88.8, 88.3 | 88.7 | 11.4–11.5 | 1,362 MiB |
| `cat-contiguous` | 130.5, 131.8, 133.8 | 132.0 | 7.5–8.0 | 1,340 MiB |
| `prealloc` | 135.4, 136.1, 132.9 | 134.8 | 7.4–7.6 | 1,164–1,165 MiB |

Peak memory is the largest footprint of the process since it started, so
it includes prefill. In each process, the 513-token runs came before the
2,049-token runs.

The fresh-process check at the original commit `f630e02` agrees with the
harness. One process, one generation, 2,048 tokens: `cat` 39.3 and 39.1
tok/s, `cat-contiguous` 98.9 and 99.3, `prealloc` 104.3 and 105.3.

On 2026-10-29, at `f630e02`, an info-level trace showed that over the
decode, the `cat` cache allocates a net +183 MB of Metal memory, and the
contiguous caches allocate +0.3 MB. We did not measure this again.

## Cause and workaround

**Cause.** After prefill, V is a transposed view (`model.rs:203-205`).
Candle's Qwen2 stores it as it is (`qwen2.rs:181-189`). For an input
that is not contiguous, `Tensor::cat` transposes, joins with strided
copies, and transposes back (`candle-core/src/tensor_cat.rs:61-76`). The
result is also not contiguous, so the next step pays again. K is not
affected, because RoPE writes a new contiguous tensor.

**Workaround.** Call `.contiguous()` on K and V before the `cat`
(`cache.rs:97-99`). This is the whole change. A preallocated cache with
`slice_set` gets the same speed for the same reason: `slice_set` needs a
contiguous source and calls `.contiguous()` itself.

**Peak memory.** At 2,048 tokens, `prealloc` has a much lower peak than
`cat-contiguous`: about 1.7 GB against 4.6 GB. The first version of this
result said that the cause is that no step holds two copies of the
cache. That is not correct. The K and V of all 24 layers at 2,048 tokens
are only about 25 MB. A `--trace summary` run at commit `0e81d96` shows
that the difference is in prefill. With `cat-contiguous`, prefill adds
2,925 MB of Metal memory, and the decode steps release 2,871 MB of it.
With `prealloc`, prefill adds 71 MB. We do not know the cause yet.

**On the upstream numbers.** Candle PR #3143 found `cat` 2–5× faster
than `slice_set` on GPU. The data here is consistent with both inputs
being non-contiguous in that comparison. With a contiguous input, the two
methods are within a few percent. PR #3419 reports fragmentation and a
leak with the cat cache. The 2026-10-29 trace showed that the net memory
growth over the decode is linear and stops when the input is contiguous.

## What it means if you run models on your own hardware

If you use Candle's Qwen2, or a model with the same cache code, at a
context above a few hundred tokens, you pay for a full cache copy on
every token. Until the fix is merged, add `.contiguous()` before the
cache append in your copy of the model, or use a preallocated cache. The
fix makes decode about 1.5× faster at 512 tokens and about 2.4× faster at
2,048 tokens.

## What this does not prove

- The absolute speeds depend on conditions that we cannot explain. On
  each day the runs agree within a few percent. Between 2026-10-29 and
  2026-11-02, the same commands gave speeds 2 % to 28 % lower. The ratio
  between the caches held on both days.
- The data is from one machine, one model, and one dtype (BF16).
- The cause of the lower peak memory of `prealloc` is not known.
- The +183 MB decode figure is from 2026-10-29 only.

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

Expected: the summary table gives about 42 tok/s for `cat` at
`alice-2048.txt`, and about 100 tok/s for `cat-contiguous` and
`prealloc`. The CSV goes to `bench/runs/<date>-e76e1ba.csv`.

For one run with the engine directly:

```sh
cargo run --release -p fornace-engine --features metal --bin fornace -- \
  --model ./models/qwen2.5-0.5b-instruct \
  --prompt "$(cat bench/prompts/alice-2048.txt)" \
  --max-new-tokens 128 --kv-cache cat
```

Expected: decode at about 40 tok/s with `--kv-cache cat`, and about
100 tok/s with `--kv-cache cat-contiguous`. The unit test
`all_kinds_return_the_same_tensors` (`cache.rs:140-171`) asserts that the
`cat` cache is not contiguous at every step and that all three kinds
return the same values.
