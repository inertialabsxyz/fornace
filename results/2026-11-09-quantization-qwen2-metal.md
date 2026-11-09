# Q4 weights make Qwen2 decode on Metal 1.85× faster, at a 23 % perplexity cost

## Why it matters

A small model on a laptop is limited by memory bandwidth at decode: each
new token reads all the weights. Weights of 8 or 4 bits are fewer bytes to
read, so decode should be faster, and the model should take less memory.
The cost is quality. This result measures both sides on one Mac, with
Candle's Q8_0 and Q4_0 formats. Q8 gives 1.40× the decode speed and 22 %
less memory for almost no quality loss. Q4 gives 1.85× and 48 % less
memory, but its perplexity is 23 % higher. A batch of several sequences
gains little from quantized weights, because Candle reads the weights again
for each row. The work also found two bugs. BF16 RoPE tables put tokens
after position 256 at the wrong angle, which made BF16 perplexity 3.2×
worse at 2,048 tokens. And Candle's quantized Metal matmul ignores the
start offset of its input.

## What was measured and how

| Item | Value |
|---|---|
| Machine | Apple M4 Pro, 20 GPU cores, 24 GB unified memory, macOS 27.2 |
| Engine | fornace-engine at `43e280a`, Candle 0.11.0 with the reuse cap of PR 3851 (fork `inertialabsxyz/candle`, commit `79fe58df`) |
| Model | Qwen2.5-0.5B-Instruct |
| Formats | `bf16`: dense BF16. `f32`: dense F32. `q8`: Q8_0. `q4`: Q4_0. Q8_0 and Q4_0 quantize blocks of 32 weights by rounding to the nearest value, with one scale for each block |
| Quantized weights | `make quantize` writes `model-q8_0.gguf` (525.1 MB) and `model-q4_0.gguf` (278.2 MB) from `model.safetensors`. Every matmul weight and the tied embedding and `lm_head` are quantized. Biases and norms stay F32 |
| Compute dtype | BF16 for `bf16`. F32 for `f32`, `q8`, and `q4`: Candle's quantized Metal matmuls take F32 input. The KV cache has the compute dtype |
| Prompts | `bench/prompts/alice-128.txt`, `alice-512.txt`, `alice-2048.txt` (129, 513, and 2,049 tokens) |
| KV cache and attention | `prealloc`, `fused` (`candle_nn::ops::sdpa`) |
| Generation | 128 tokens for each sequence, end-of-sequence ignored |
| Sampler | Greedy, repeat penalty 1.1 over the last 64 tokens, on the CPU |
| Quality | Perplexity on `bench/prompts/alice-2048.txt` (2,049 tokens): exp of the mean of −log p(next token), with no repeat penalty. Lower is better. Also the top-1 agreement with dense F32 at every position of `alice-512.txt` (`tests/quantized.rs`) |
| Method | `fornace-bench`, untraced: one process for each format and batch size, 1 warm-up, 3 recorded runs. The rows that are compared ran in one command. All runs on 2026-11-09 |
| Data | `bench/runs/2026-11-09-43e280a-quant.csv` (the case is in the `tag` column), committed in `8ab7d7c` |

Peak memory is the largest footprint of the process since it started, in
MiB (2²⁰ bytes), the largest of the 3 runs. The other values are the mean
of the 3 runs.

**Case A: 512 tokens, batch 1.**

| Format | Decode tok/s | Decode step, ms | Prefill tok/s | Time to first token, ms | Peak memory, MiB | Perplexity | Top-1 agreement with F32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bf16` | 157.6 | 6.24 | 6,055 | 84.7 | 1,157.5 | 12.473 | 98.2 % |
| `q8` | **220.5 (1.40×)** | 4.53 | 5,881 | 87.2 | 899.9 (−258) | 12.501 (+0.2 %) | 98.6 % |
| `q4` | **292.1 (1.85×)** | 3.43 | 5,832 | 88.0 | 601.5 (−556) | 15.361 (+23.2 %) | 69.3 % |
| `f32` | 96.1 | 10.34 | 5,315 | 96.5 | 4,717.2 | 12.456 | — |

- The perplexity change is against `bf16`. Against `f32`, it is +0.4 %
  for `q8` and +23.3 % for `q4`.
- The decode spread of the 3 runs is 0.6 to 2.9 %.
- `f32` is a control: it has the F32 activations of the quantized formats,
  with dense weights. Its peak memory includes the load, which converts
  the BF16 file to F32. Do not use it as a memory row.

**Case B: a short and a long prompt, batch 1.**

| Prompt tokens | Format | Decode tok/s | Prefill tok/s | Time to first token, ms | Peak memory, MiB |
|---:|---|---:|---:|---:|---:|
| 129 | `bf16` | 159.0 | 4,207 | 30.7 | 1,129.0 |
| 129 | `q8` | 232.7 (1.46×) | 4,461 | 28.9 | 839.3 |
| 129 | `q4` | 314.2 (1.98×) | 4,477 | 28.8 | 542.5 |
| 2,049 | `bf16` | 150.8 | 6,217 | 329.6 | 1,294.4 |
| 2,049 | `q8` | 208.6 (1.38×) | 5,747 | 356.5 | 1,175.3 |
| 2,049 | `q4` | 260.1 (1.72×) | 5,665 | 361.7 | 878.5 |

The `bf16` group at 129 tokens spread 8.5 %. The other groups spread
5.2 % or less.

**Case C: batch size, 512 tokens.**

| B | `bf16` total tok/s | `bf16` per sequence | `q4` total tok/s | `q4` per sequence | `q4` step, ms | `q4` peak MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 155.7 | 155.7 | 290.1 | 290.1 | 3.45 | 601.2 |
| 2 | 144.3 | 72.1 | 346.1 | 173.0 | 5.79 | 691.1 |
| 4 | 256.5 | 64.1 | 377.5 | 94.4 | 10.64 | 874.1 |
| 8 | **394.9** | 49.4 | 377.4 | 47.2 | 21.30 | 1,240.3 |

**Predictions.** Before the runs, an estimate from memory bandwidth
predicted these values. It assumed that the part of each step that is not
the weight read (about 2.6 ms) stays the same, and that the quantized
kernels reach the same share of the bandwidth as the BF16 kernels.

| Prediction (*calc*) | Measured | Holds? |
|---|---|---|
| `q8` decode about 4.6 ms, 220 tok/s (1.4×) | 4.53 ms, 220.5 tok/s (1.40×) | Yes |
| `q4` decode about 3.7 ms, 270 tok/s (1.7×) | 3.43 ms, 292.1 tok/s (1.85×) | Yes, faster |
| Prefill lower than `bf16` | 3 to 4 % lower at 512 tokens, 8 to 9 % at 2,048, 6 % higher at 129 | Mostly, and small |
| Peak about 463 MB (`q8`) and 710 MB (`q4`) lower, minus the F32 activations | 270 MB and 583 MB lower at 512 tokens | Yes |
| `q8` perplexity within 1 % | +0.2 % | Yes |
| `q4` perplexity clearly higher | +23.2 % | Yes |
| `q4` has no drop at batch 2 | 290.1 → 346.1 tok/s | Yes, but see "Cause" |

## Cause and workaround

**Decode: the weight read falls, the fixed cost does not.** The weights
are 988 MB in BF16, 525 MB in Q8_0, and 278 MB in Q4_0 (*calc*, and the
file sizes). With the KV cache at 640 positions (8 MB in BF16, 16 MB in
F32), a step reads about 996, 541, and 294 MB. At 273 GB/s (Apple
specification) that takes 3.65, 1.98, and 1.08 ms (*calc*). The measured
steps are 6.24, 4.53, and 3.43 ms. So the part that is not the read stays
near 2.4 to 2.6 ms in each format. At Q4, that fixed cost is more than two
thirds of a step: kernel launches and small ops. A smaller format cannot
remove it.

Debug traces of case A agree. They add GPU waits, so they explain only
(`bench/runs/2026-11-09-43e280a-trace-quant-{f16,f32,q8,q4}.txt`).
`lm_head`, the largest single matmul, falls with its bytes: 1.30, 0.76,
and 0.46 ms for `bf16`, `q8`, and `q4`. The MLP of a layer falls less:
0.29, 0.23, and 0.21 ms. The dequantization in the kernel does not show as
a cost: Q4 is faster than Q8 in every case.

**Memory: less saving than the weights, because of F32.** Q8 saves 463 MB
of weights and Q4 710 MB (*calc*). At 512 tokens the peak falls by 270 MB
and 583 MB. The rest goes to F32 activations and an F32 KV cache, which is
twice the BF16 size. At 2,049 tokens, the saving falls to 125 MB and
436 MB (119 and 416 MiB).

**Long prompts: the F32 KV cache.** From 129 to 2,049 tokens, `q4` decode
falls 17 % (314.2 → 260.1 tok/s) and `bf16` falls 5 % (159.0 →
150.8 tok/s). The quantized step is shorter, and its F32 cache is twice
the size, so the cache read is a larger share of it.

**Batches: Candle reads the weights again for each row.** In Candle's
quantized Metal matmul, a decode input of B rows goes to `fwd_mv`, which
launches the matrix-vector kernel once for each row
(`candle-core/src/quantized/metal.rs:324`). A comment there gives the
reason: "the implementation in candle-metal-kernels doesn't handle batch
properly". So no weight read is shared, and the `q4` step grows by about
2.6 ms for each row after the first. At batch 8, `bf16` gives more total
tokens than `q4` (394.9 against 377.4 tok/s).

**Workarounds in Fornace.**

- Candle's quantized Metal matmul takes only contiguous input, and for
  more than one row it ignores the start offset of the input. The
  quantized projection in `model.rs` copies such an input to a new buffer
  first (`Proj`, `43e280a`). A test prints whether Candle still ignores
  the offset (`tests/quantized.rs`,
  `metal_quantized_matmul_needs_a_chunk_at_offset_zero`).
- No workaround for the batch read. A kernel launch that takes all rows
  would let a batch share the weight read. It is not built.

## What it means if you run models on your own hardware

If you run a small model on a Mac with Candle, use Q8. It decodes 1.4×
faster, uses 22 % less memory, and its perplexity is 0.4 % from F32. Use
Q4 only if you need the speed or the memory: it decodes 1.85× faster and
uses 48 % less memory, but on 3 positions in 10 its top token is not the
F32 top token. Long prompts keep less of the gain, because the cache stays
F32. If you run several sequences at once, a BF16 batch of 8 gives more
total tokens than Q4 at any batch size.

## Correctness

`tests/quantized.rs` compares each format with dense F32 on Metal. It
decodes `alice-512.txt` greedily with F32, gives the same tokens to the
other model, and compares the logits at each of 32 steps. It also compares
the top token at each of the 512 prompt positions.

| Against dense F32 | Max logit diff after prefill | Max diff in 31 decode steps | Top token changes in 32 steps | Top-1 agreement |
|---|---:|---:|---:|---:|
| Metal, `bf16` | 0.25 | 0.64 | 1 | 503/512 = 98.2 % |
| Metal, `q8` | 0.25 | 0.37 | 0 | 505/512 = 98.6 % |
| Metal, `q4` | 3.98 | 6.44 | 5 | 355/512 = 69.3 % |
| CPU, `q8` | 0.50 | 0.55 | 1 | 500/512 = 97.7 % |

- A batch of 4 prompts in `q8` and `q4` gives the same text as each
  prompt alone (`a_quantized_batch_gives_the_text_of_batch_one`).
- The `bf16` row uses RoPE tables computed in F32 (`5ea3a39`). See the
  next section.

## Two bugs found during the work

**BF16 RoPE tables put late tokens at the wrong angle.** RoPE rotates
each token by an angle that depends on its position. Fornace's model code
is a copy of Candle's `qwen2.rs`, which converts the positions to the
compute dtype before it computes the angles. BF16 has an 8-bit mantissa,
so a position above 256 is not exact: 257 becomes 256, and above 1,024
the positions round in steps of 8. Thus late tokens get the angle of a
nearby position. Commit `5ea3a39` computes the angles in F32 and converts
only sin and cos to the compute dtype.

| Text | `f32` | `bf16` before (`819de0a`) | `bf16` after (`43e280a`) |
|---|---:|---:|---:|
| `alice-128.txt` (129 tokens) | 8.991 | 9.280 | 9.067 |
| `alice-512.txt` (513 tokens) | 9.667 | 12.486 | 9.742 |
| `alice-2048.txt` (2,049 tokens) | 12.456 | **39.686** | 12.473 |

Perplexity, fused attention, Metal. Data:
`bench/runs/2026-11-09-819de0a-43e280a-rope.csv`, committed in
`b436a65`. The error grows with the
length of the text, as rounded positions predict. After the fix, `bf16`
is within 0.9 % of `f32` at each length. Speed and memory do not change:
the work is the same. A unit test checks that each BF16 table value is the
F32 value rounded once (`bf16_rope_tables_are_the_f32_tables_rounded_once`
in `crates/fornace-engine/src/model.rs`). Candle's `qwen2.rs` has the same
code. Candle PR #3520 (open) makes the same change, and these
measurements are now on it. A text search finds the same pattern in about
20 other Candle model files. Only Qwen2 was measured.

**Candle's quantized Metal matmul ignores the start offset of its
input.** For an input of more than one row, `fwd` in
`candle-core/src/quantized/metal.rs` builds a new layout from the shape
and passes its start offset, which is always 0. So a view of rows 4 to 7
gives the result for rows 0 to 3. `Tensor::copy` does not help, because it
keeps the offset. The quantized matmul also takes only contiguous input,
and the last position of each row in a batch is a strided view. Fornace
copies such an input to a new buffer at offset 0 (`Proj` in
`crates/fornace-engine/src/model.rs`, `43e280a`). The model with one row
was not affected: its inputs are new tensors, and the one-row path uses
the offset. The test `metal_quantized_matmul_needs_a_chunk_at_offset_zero`
prints whether Candle still ignores the offset. The fix in Candle is one
line, with a test: Candle PR #4105.

## What this does not prove

- All runs are from one day, on one machine, with one model.
- Only Q8_0 and Q4_0. K-quants need rows that are a multiple of 256, and
  most tensors of this model have 896. No imatrix, no mixed formats.
- The perplexity is on one text, `alice-2048.txt`. That text is public
  domain and is probably in the training data, so the absolute values are
  low. The order of the formats is valid. The size of the gaps on other
  text is not measured.
- There is no task score. Perplexity and top-1 agreement do not show if a
  Q4 answer is still correct.
- The quantized formats compute in F32. A Q8 or Q4 path with BF16
  activations, or a quantized KV cache, was not measured.
- The cause of the fixed cost of about 2.5 ms for each step is not
  separated into kernel launches and small ops.
- Whether Candle's matrix-vector kernel can take several rows in one
  launch was not checked.
- Batches are copies of one prompt, static, and of equal length.

## Reproduce

Commit `43e280a`.

```sh
git clone https://github.com/inertialabsxyz/fornace && cd fornace
make model      # downloads Qwen2.5-0.5B-Instruct to ./models/qwen2.5-0.5b-instruct
git checkout 43e280a
make quantize   # writes model-q8_0.gguf and model-q4_0.gguf
cargo build --release --features fornace-engine/metal -p fornace-bench
fb() {
  ./target/release/fornace-bench --model ./models/qwen2.5-0.5b-instruct \
    --kv-cache prealloc --attention fused --max-new-tokens 128 \
    --out bench/runs/repro-quant.csv "$@"
}
fb --dtypes f16,f32,q8,q4 --batch 1 --prompts bench/prompts/alice-512.txt \
   --quality bench/prompts/alice-2048.txt --tag "case A"
fb --dtypes f16,q8,q4 --batch 1 --prompts bench/prompts/alice-128.txt,bench/prompts/alice-2048.txt --tag "case B"
fb --dtypes f16,q4 --batch 1,2,4,8 --prompts bench/prompts/alice-512.txt --tag "case C"
```

The defaults give 1 warm-up, 3 recorded runs, and `--ignore-eos true`.
`--dtypes f16` gives `bf16` on Metal.

Expected: in case A, decode near 158, 220, and 292 tok/s for `bf16`,
`q8`, and `q4`, and perplexity 12.473, 12.456, 12.501, and 15.361 for
`bf16`, `f32`, `q8`, and `q4`. The perplexity is deterministic. In case C,
`q4` is flat from batch 4 to batch 8, and `bf16` passes it at batch 8.

The correctness test:

```sh
FORNACE_MODEL_DIR=./models/qwen2.5-0.5b-instruct \
  cargo test --release --features metal -p fornace-engine \
  --test quantized -- --nocapture --test-threads=1
```

Expected: 4 tests pass, with the agreements in "Correctness". The output
also says "candle still ignores the input offset".

The RoPE fix, before and after. Run each command at its commit:

```sh
for q in alice-128 alice-512 alice-2048; do
  ./target/release/fornace-bench --model ./models/qwen2.5-0.5b-instruct \
    --dtypes f16,f32 --prompts bench/prompts/alice-128.txt --attention fused \
    --max-new-tokens 2 --repeats 1 --warmup 0 \
    --quality bench/prompts/$q.txt --out bench/runs/repro-rope.csv
done
```

Expected: the perplexities in the table of "Two bugs found during the
work". At `819de0a`, `bf16` gives 39.686 on `alice-2048.txt`. At
`43e280a`, it gives 12.473.
