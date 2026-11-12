# Prefilling a shared system prompt once cuts the time to first token of 4 Qwen2 sequences 2.7× on Metal, and a paged KV cache is not needed for it

## Why it matters

Servers use a paged KV cache. It keeps keys and values in fixed-size
blocks, so more users fit in one GPU, and a system prompt that many users
share is stored once. This result asks what a paged cache gives one user
on a laptop: several sequences with one system prompt, or one long
session. On a Mac with Candle, the shared system prompt is a real gain:
it is prefilled once, and the first token comes 2.7× sooner. But a
contiguous cache gets the same gain when it copies the shared prefix into
each sequence once. The paged cache adds a gather in every layer at every
step, and with Candle's Metal kernels that makes a full answer 47 to 61 %
slower. At batch 1 it saves no memory.

## What was measured and how

| Item | Value |
|---|---|
| Machine | Apple M4 Pro, 20 GPU cores, 24 GB unified memory, macOS 27.2 |
| Engine | fornace-engine at `bee2196`, Candle 0.11.0 with the reuse cap of PR 3851 (fork `inertialabsxyz/candle`, commit `79fe58df`) |
| Model | Qwen2.5-0.5B-Instruct, `bf16` (dense BF16) and `q4` (Q4_0 weights, F32 activations and F32 cache) |
| `prealloc` cache | One contiguous buffer for each sequence, with room for the prompt and the new tokens |
| `paged` cache | A pool of blocks of 16 positions, and a block table for each sequence. Each step writes the new positions into their blocks (`scatter_set`), and then gathers all positions of each sequence into a contiguous K and V for attention (`index_select`). Written for Fornace: Candle has no paged cache |
| Shared prefix | `--share-prefix`: the tokens that all sequences have in common at the start are prefilled once, as one sequence. Then the rest of each sequence is prefilled as a batch. `paged` shares the full blocks of the prefix, and each sequence gets a copy of the last block if it is not full. `prealloc` copies the prefix into each sequence once |
| Prompts | `bench/prompts/alice-512.txt` and `alice-2048.txt` (513 and 2,049 tokens). The system-prompt case: each sequence is `alice-512.txt` and its own 64 tokens of `alice-2048.txt` (577 tokens) |
| Attention | `fused` (`candle_nn::ops::sdpa`) |
| Generation | 128 tokens for each sequence, end-of-sequence ignored. Greedy, repeat penalty 1.1 |
| Method | `fornace-bench`, untraced: one process for each setting, 1 warm-up, 3 recorded runs. The rows that are compared ran in one command. All runs on 2026-11-12 |
| Data | `bench/runs/2026-11-12-bee2196-paged.csv` (the case is in the `tag` column). Block counts: `bench/runs/2026-11-12-bee2196-paged-stderr.txt` |

Each position of this model holds 12 KiB of cache in BF16 and 24 KiB in
F32, in all 24 layers (*calc*). The values are the mean of the 3 runs.
Peak memory is the largest of the 3, in MiB (2²⁰ bytes). "Total" is the
time to the first token plus the decode time of all 4 sequences (*calc*
from the CSV: ttft + 4 × 127 / decode tok/s).

**A system prompt and 4 different user messages** (case G).

| Format | Cache | Shared | Time to first token, ms | Decode step, ms | Total, ms | Peak, MiB |
|---|---|---|---:|---:|---:|---:|
| `bf16` | prealloc | no | 374.5 | 16.36 | 2,449 | 1,294.8 |
| `bf16` | prealloc | **yes** | **137.5 (2.7× lower)** | 16.38 | **2,216 (−10 %)** | 1,226.5 |
| `bf16` | paged | no | 436.7 | 27.68 | 3,878 | 1,305.9 |
| `bf16` | paged | yes | 156.2 | 27.68 | 3,598 (+47 %) | 1,194.8 |
| `q4` | prealloc | no | 418.1 | 11.11 | 1,829 | 871.2 |
| `q4` | prealloc | **yes** | **145.0 (2.9× lower)** | 11.17 | **1,559 (−15 %)** | 709.2 |
| `q4` | paged | no | 484.1 | 22.49 | 3,265 | 890.2 |
| `q4` | paged | yes | 164.3 | 22.57 | 2,953 (+61 %) | 642.2 |

The percentages compare with `prealloc` without sharing. With sharing,
the paged cache holds 80 blocks, against 176 without (the 32 full blocks
of the system prompt once).

**4 copies of one prompt** (case C, best-of-n). The time to first token
falls 3.6× (`bf16`, 327.5 → 90.1 ms) and 3.8× (`q4`, 355.4 → 93.9 ms)
with `prealloc` sharing, and the total falls by 10 and 14 %. With paged
sharing, the first token comes at 96.0 and 103.0 ms, and the total is 43
and 59 % longer than `prealloc` without sharing.

**One sequence** (cases C and B).

| Prompt | Format | Cache | Decode tok/s | Decode step, ms | Time to first token, ms | Peak, MiB |
|---:|---|---|---:|---:|---:|---:|
| 513 | `bf16` | prealloc | 159.2 | 6.29 | 84.9 | 1,162.3 |
| 513 | `bf16` | paged | 139.5 (−12 %) | 7.24 | 93.8 | 1,162.8 |
| 513 | `q4` | prealloc | 284.0 | 3.53 | 88.7 | 596.8 |
| 513 | `q4` | paged | 229.0 (−19 %) | 4.43 | 97.0 | 599.4 |
| 2,049 | `bf16` | prealloc | 150.9 | 6.61 | 338.7 | 1,285.9 |
| 2,049 | `bf16` | paged | 99.6 (−34 %) | 10.15 | 377.5 | 1,289.2 |
| 2,049 | `q4` | prealloc | 259.0 | 3.86 | 371.0 | 865.6 |
| 2,049 | `q4` | paged | 138.7 (−46 %) | 7.28 | 411.3 | 874.5 |

The 513-token rows come from case C. Case A gave the same rows, but its
`q4` `prealloc` group had one slow run (214.6 tok/s against 285.7 and
278.8), so its spread is 33 %. All other groups spread 4.5 % or less.

**A cache that reserves 32,768 positions** (case D, `--kv-capacity
32768`, 513 tokens). The peak rises against case A by 366 MiB (`bf16`
`prealloc`), 372 MiB (`bf16` paged), 747 MiB (`q4` `prealloc`), and 748
MiB (`q4` paged). The reservation is 384 MiB in BF16 and 768 MiB in F32
(*calc*). The paged pool reserves the same as `prealloc`.

**Block size** (case E, `bf16`, 2,049 tokens, paged): 8, 16, and 64
positions give 96.9, 96.7, and 96.6 tok/s.

**Predictions.** Before the runs, an estimate from the bytes that the
gather moves predicted these values.

| Prediction (*calc*) | Measured | Holds? |
|---|---|---|
| Paged at batch 1, 513 tokens: 1 to 5 % slower (`bf16`), 2 to 8 % (`q4`) | 12 % and 19 % | No |
| Paged at batch 1, 2,049 tokens: +0.2 ms, 3 to 8 % slower (`bf16`); +0.4 ms, 8 to 15 % (`q4`) | +3.54 ms, 34 %; +3.42 ms, 46 % | No |
| Peak at batch 1 within 1 MiB | +0.5 to +8.9 MiB | Nearly |
| Prefill within the drift | Time to first token +8 to +9 ms (513), +39 to +40 ms (2,049) | No |
| Shared prompt: time to first token near the batch-1 value | 90.1 ms (`prealloc`) and 96.0 ms (paged), against 84.9 and 93.8 ms | Yes |
| Shared prompt: the paged cache gives the gain | `prealloc` with one copy gives the same gain, and is faster | No |
| Reservation of 32,768: the paged peak rises by the blocks in use only | It rises by the full reservation | No |
| Block size changes nothing | 96.6 to 96.9 tok/s | Yes |

## Cause and workaround

**The shared prefix is the gain.** With 4 sequences, the prefill of the
shared tokens runs once, not 4 times. That does not need blocks. A
contiguous cache that copies the prefix into each sequence once gets the
same time to first token, and it keeps its decode speed.

**The gather costs more than its bytes.** At 2,049 tokens, the gather
moves 53 MB in BF16 and 107 MB in F32 in each step (*calc*). At 273 GB/s
that is 0.2 and 0.4 ms. But the step grows by 3.5 ms in BF16 and 3.4 ms
in F32: about the same in both. Thus the bytes do not set the cost. The
number of elements does.

A timing of the kernels alone, 48 calls (one decode step), at the shape
of one layer of the pool:

| Format | Positions | `index_select` | `index_select` by block of 16 | Strided copy | Element-wise kernel, same data |
|---|---:|---:|---:|---:|---:|
| BF16 | 640 | 1.39 ms | 1.06 ms | 1.39 ms | 0.32 ms |
| BF16 | 2,176 | 3.88 ms | 3.01 ms | 4.17 ms | 0.39 ms |
| F32 | 640 | 1.35 ms | 1.07 ms | 1.43 ms | 0.31 ms |
| F32 | 2,176 | 3.92 ms | 3.05 ms | 4.26 ms | 0.45 ms |

Data: `bench/runs/2026-11-12-bee2196-gather-kernels.txt`. At 2,176
positions, `index_select` takes 3.5 ms more than an element-wise kernel
over the same data, the same as the measured growth of the step. Its
time does not change with the dtype. Candle's Metal `index_select`
computes the source index of each element with 64-bit division and
modulo (`candle-metal-kernels/src/metal_src/indexing.metal`, `index`). A
gather by block is 22 % faster, and is still 7 to 8 times slower than
the element-wise kernel. So the block size does not change the speed. A
strided copy, the kernel type behind the non-contiguous cache of the
first Fornace finding, has the same cost.

Debug traces agree. They add GPU waits, so they explain only
(`bench/runs/2026-11-12-bee2196-trace-paged-*.txt`). At 2,048 tokens,
`attn.gather` is 0.32 ms in each layer (`bf16`) and 0.36 ms (`q4`),
against 0.11 ms for `attn.kv_append` with `prealloc`.

**Prefill.** Candle's Metal `scatter` kernel starts one thread for each
head and element of a row, 128 threads for this model. Each thread loops
over the new positions in series (`indexing.metal`, `scatter`). At decode
there is one new position. At prefill there is the full prompt, so the
time to first token rises by up to 40 ms at 2,049 tokens.

**Batches.** The one-token kernel of `sdpa` reads K and V with only the
head stride. A gathered batch of several sequences does not have that
layout, so it is copied again. At batch 4 the paged decode step is 63 to
102 % longer than with `prealloc`.

**Memory.** The paged pool is allocated once, at its full size. For one
sequence, it is as large as a `prealloc` cache of the same capacity. With
a shared system prompt, the paged cache has a 32 MiB (`bf16`) and 67 MiB
(`q4`) lower peak than `prealloc` sharing, because the prefix is stored
once.

**Workarounds in Fornace.** Share the prefix with the contiguous cache
(`--kv-cache prealloc --share-prefix true`). No workaround for the
gather. A paged attention kernel that reads the blocks in place, or a
gather kernel at the speed of the element-wise kernel, would remove most
of the cost. Neither is built.

## What it means if you run models on your own hardware

If you run several sequences with one system prompt on a Mac with
Candle, prefill the system prompt once and copy it into each sequence.
With 4 sequences, the first token comes 2.7 to 2.9 times sooner, and the
full answers are 10 to 15 % faster. You do not need a paged cache for
this. With Candle today, a paged cache makes each answer 47 to 61 %
slower, and it saves only tens of MiB with this model. If you run one
sequence at a time, use a contiguous cache sized for the prompt and the
answer. A paged pool must still be as large as the longest session, so
it does not save memory for one user.

## Correctness

`tests/paged.rs` decodes greedily with `prealloc`, gives the same tokens
to the paged cache, and compares the logits at each of 16 steps.

| Device, format | Attention | Batch | Block sizes | Max logit difference |
|---|---|---|---|---:|
| Metal, `bf16` | naive, fused | 1, 4 | 1, 7, 16 | 0 |
| Metal, `q4` | fused | 1, 4 | 7, 16 | 0 |
| CPU, F32 | naive | 1, 4 | 7, 16 | 0 |

A shared prefix gives the same logits as 4 full prefills (difference 0
on the CPU in F32 and on Metal in BF16), for the paged cache (block 7 and
16) and for `prealloc`. This holds for 4 copies of one prompt and for a
448-token system prompt with 4 different 64-token user messages. Each
sequence of a shared batch gives the text of a single generation.

## What this does not prove

- All runs are from one day, on one machine, with one model.
- This model has only 2 KV heads, so its cache is small for its size. A
  model with more KV heads, or a longer context, makes the cache, and the
  memory saving of shared blocks, larger.
- One user, 4 sequences, no scheduler, and no pool shared between
  requests. A server with many sequences in one pool was not measured.
- Gather only. No paged attention kernel was built. The gain of a faster
  kernel is an estimate from a timing of the kernels alone.
- Sequences of equal length only, and greedy sampling only. The user
  messages are 64 tokens each.
- No prefix reuse across calls, for example a second chat turn.
- The cause of the extra 0.5 to 8.9 MiB at batch 1 is not measured.
- The peak fall with a shared prefix is not separated into the cache and
  the prefill activations.

## Reproduce

Commit `bee2196`.

```sh
git clone https://github.com/inertialabsxyz/fornace && cd fornace
make model      # downloads Qwen2.5-0.5B-Instruct to ./models/qwen2.5-0.5b-instruct
git checkout bee2196
make quantize   # writes model-q4_0.gguf
cargo build --release --features fornace-engine/metal -p fornace-bench
fb() {
  ./target/release/fornace-bench --model ./models/qwen2.5-0.5b-instruct \
    --dtypes f16,q4 --attention fused --max-new-tokens 128 \
    --out bench/runs/repro-paged.csv "$@"
}
P=bench/prompts
fb --prompts $P/alice-512.txt  --kv-cache prealloc,paged --share-prefix false,true \
   --batch 1,4 --tag "case C"
fb --prompts $P/alice-2048.txt --kv-cache prealloc,paged --batch 1 --tag "case B"
fb --prompts $P/alice-512.txt  --kv-cache prealloc,paged --kv-capacity 32768 --tag "case D"
fb --prompts $P/alice-2048.txt --dtypes f16 --kv-cache paged --block-size 8,16,64 --tag "case E"
fb --prompts $P/alice-512.txt  --kv-cache prealloc,paged --share-prefix false,true \
   --batch 4 --suffix $P/alice-2048.txt --suffix-tokens 64 --tag "case G"
```

The defaults give 1 warm-up, 3 recorded runs, and `--ignore-eos true`.
`--dtypes f16` gives `bf16` on Metal. Case A is case C at batch 1 only.

Expected: in case G, the shared `prealloc` rows give a time to first
token near 140 ms (`bf16`) and 145 ms (`q4`), against about 375 and 420
ms without sharing, with the same decode step. The paged rows have a
decode step near 28 ms (`bf16`) and 23 ms (`q4`), against 16 and 11 ms
for `prealloc`. Speed drifts between commands, so compare rows from one
command.

The kernel timing:

```sh
cargo test --release --features metal -p fornace-engine \
  --test metal_gather_kernels -- --ignored --nocapture
```

Expected: at 2,176 positions, `index_select` near 3.9 ms and the
element-wise kernel near 0.4 ms, in BF16 and F32.

The correctness test:

```sh
FORNACE_MODEL_DIR=./models/qwen2.5-0.5b-instruct \
  cargo test --release --features metal -p fornace-engine \
  --test paged -- --nocapture --test-threads=1
```

Expected: 6 tests pass. Every "max |prealloc - paged|" and every
"max |4 prefills - 1 shared prefill|" line is 0.
