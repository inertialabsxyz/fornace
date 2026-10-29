# A non-contiguous KV cache halves Qwen2 decode speed at 2k context in Candle on Metal

## Why it matters

Decode speed is what a user feels: it is the rate at which text appears.
Candle's Qwen2 implementation appends to the KV cache with `Tensor::cat`.
After prefill, V is a transposed view and is not contiguous, so every
later `cat` copies the full cache with strided kernels, in all 24 layers,
on every token. The cost grows with the context length: about 20 % at
512 tokens, and half the speed at 2,048. One `.contiguous()` call before
the append removes it. Candle PR #NNNN has the fix. The same class of bug
is open for gemma3 (PR #3902), so other models in Candle may have it.

## What was measured and how

| Item | Value |
|---|---|
| Machine | Mac, Apple silicon, 24 GB unified memory |
| Engine | fornace-engine at commit `f630e02`, Candle 0.11.0 |
| Model | Qwen2.5-0.5B-Instruct, BF16 on Metal |
| Prompt | 2,048 tokens (`bench/prompts/alice-2048.txt`) |
| Sampler | Greedy, repeat penalty 1.1 over the last 64 tokens |
| Method | Untraced runs, 2 to 3 repeats, no formal warm-up |

| KV cache | Decode tok/s (runs) | Mean | Mean ms, last 64 steps | Peak memory |
|---|---|---|---|---|
| `cat` (Candle original) | 60.9, 54.4 | ≈ 58 | 24.9 | 1,470 MB |
| `cat-contiguous` (the fix) | 124.5, 121.7, 122.3 | ≈ 123 | 9.4–9.6 | ≈ 1,395 MB |
| `prealloc` (`slice_set` into a fixed buffer) | 108.0, 122.2 | ≈ 115 | 9.3 | 1,253 MB |

At 512 tokens: `cat` 107 tok/s, `prealloc` about 137 tok/s.

An info-level trace shows the same thing from the memory side: over the
decode, the `cat` cache allocates a net +183 MB of Metal memory; the
contiguous caches allocate +0.3 MB.

## Cause and workaround

**Cause.** After prefill, V is a transposed view (`model.rs:201-203`).
Candle's Qwen2 stores it as it is (`qwen2.rs:181-189`). For an input
that is not contiguous, `Tensor::cat` transposes, joins with strided
copies, and transposes back (`candle-core/src/tensor_cat.rs:61-76`). The
result is also not contiguous, so the next step pays again. K is not
affected, because RoPE writes a new contiguous tensor.

**Workaround.** Call `.contiguous()` on K and V before the `cat`
(`cache.rs:97-101`). This is the whole change. A preallocated cache with
`slice_set` gets the same speed for the same reason: `slice_set` needs a
contiguous source and calls `.contiguous()` itself. Its real benefit is
lower peak memory, because no step holds two copies of the cache.

**On the upstream numbers.** Candle PR #3143 found `cat` 2–5× faster
than `slice_set` on GPU. The data here is consistent with both inputs
being non-contiguous in that comparison. With a contiguous input, the two
methods are within a few percent. PR #3419 reports fragmentation and a
leak with the cat cache; here the net memory growth is linear and stops
when the input is contiguous.

## What it means if you run models on your own hardware

If you use Candle's Qwen2, or a model with the same cache code, at a
context above a few hundred tokens, you pay for a full cache copy on
every token. Until the fix is merged, add `.contiguous()` before the
cache append in your copy of the model, or use a preallocated cache. The
fix costs nothing at short contexts and doubles decode speed at long ones.

## Reproduce

Commit `f630e02`.

```sh
git clone https://github.com/inertialabsxyz/fornace && cd fornace
git checkout f630e02
make model   # downloads Qwen2.5-0.5B-Instruct to ./models/qwen2.5-0.5b-instruct
cargo run --release -p fornace-engine --features metal --bin fornace -- \
  --model ./models/qwen2.5-0.5b-instruct \
  --prompt "$(cat bench/prompts/alice-2048.txt)" \
  --max-new-tokens 128 --kv-cache cat
cargo run --release -p fornace-engine --features metal --bin fornace -- \
  --model ./models/qwen2.5-0.5b-instruct \
  --prompt "$(cat bench/prompts/alice-2048.txt)" \
  --max-new-tokens 128 --kv-cache cat-contiguous
```

Expected: the first run reports decode at about 55–60 tok/s; the second
at about 120 tok/s. The unit test `all_kinds_return_the_same_tensors`
(`cache.rs:139-170`) asserts that the `cat` cache is not contiguous at
every step and that all three kinds return the same values.
