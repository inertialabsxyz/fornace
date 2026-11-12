
# A 0.5B draft makes Qwen2.5-1.5B decode 1.35× faster on code and 38 % slower on prose on Metal

## Why it matters

Speculative decoding is the one method that gives one user more tokens
for each step without a change to the model. A small draft model guesses
the next K tokens, and the large model checks them in one forward pass.
With greedy sampling, the text is the text of the large model alone. On a
server GPU it often gives 2 to 3×. On a laptop, both models share one GPU
and one memory bandwidth. This result asks what it gives on a Mac with
Candle: Qwen2.5-1.5B-Instruct as the target, and Qwen2.5-0.5B-Instruct as
the draft. On free prose it is a loss at every setting. On code that
copies from its own context it is a gain of 1.35×. The cost that decides
both is the check pass: checking K + 1 tokens in one pass costs 2.2 to
2.6 decode steps, not one.

## What was measured and how

| Item | Value |
|---|---|
| Machine | Apple M4 Pro, 20 GPU cores, 24 GB unified memory, macOS 27.2 |
| Engine | fornace-engine at `9673634`. Candle 0.11.0 with three changes (fork `inertialabsxyz/candle`, commit `b4090fe1`): the reuse cap of PR 3851, `KvCache::truncate`, and the causal mask fix of PR 4062 |
| Target | Qwen2.5-1.5B-Instruct, `bf16` |
| Draft | Qwen2.5-0.5B-Instruct, `bf16`, `q8` (Q8_0), or `q4` (Q4_0). The two models share one tokenizer |
| Speculative decoding | Greedy. In each round, the draft guesses K tokens, one step each. The target checks them in one forward pass of K + 1 tokens, and accepts the guesses while they equal its own choice. Then it gives one token of its own. Both caches drop the rejected positions. Written for Fornace: Candle has no speculative decoding |
| Prompts | `bench/prompts/alice-512.txt` (English prose, *Alice's Adventures in Wonderland*), `alice-de-512.txt` (the same chapter in German), `code-512.txt` (Rust source from Fornace). 512 tokens each |
| Settings | `--attention fused --kv-cache prealloc`, 128 new tokens, end-of-sequence ignored, repeat penalty 1.1 |
| Method | `fornace-bench`, untraced: one process for each setting, 1 warm-up, 3 recorded runs. The rows that are compared ran in one command. All runs on 2026-11-12 |
| Data | `bench/runs/2026-11-12-9673634-spec.csv` (the case is in the `tag` column) |

α is the share of the checked guesses that the target accepts. After the
first reject in a round, the later guesses are not checked, so they do
not count. The values are the mean of the 3 runs. The spread of the 3
runs is 2.4 % or less in each group, except the target alone in case A
(6.6 %).

**Prose** (case A, `alice-512.txt`). The target alone: 65.4 tok/s.

| Draft | K = 2 | K = 4 | K = 8 | α |
|---|---:|---:|---:|---:|
| `bf16` | 36.0 (0.55×) | 31.0 (0.47×) | 20.7 (0.32×) | 0.46 to 0.47 |
| `q8` | 39.4 (0.60×) | 34.2 (0.52×) | 24.6 (0.38×) | 0.46 to 0.48 |
| `q4` | **40.3 (0.62×)** | 36.4 (0.56×) | 26.4 (0.40×) | 0.44 |

In German (case C, `q4` draft, K = 2): 42.4 tok/s against 65.9 (0.64×),
α = 0.49.

**Code** (case C, `code-512.txt`). The target alone: 65.6 tok/s.

| Draft | K = 2 | K = 4 | K = 8 | α |
|---|---:|---:|---:|---:|
| `q4` | 65.1 (0.99×) | 80.4 (1.23×) | **88.4 (1.35×)** | 0.91 to 0.93 |
| `q8` | 61.3 (0.94×) | 73.8 (1.13×) | 82.9 (1.26×) | 0.92 to 0.93 |

**Memory and first token.** With a draft, the time to first token rises
from 265 ms to 345 to 349 ms: the draft also prefills the prompt. The
peak rises from 3,231 MiB by 1,030 MiB (`bf16` draft), 729 MiB (`q8`),
and 429 MiB (`q4`).

**The same model as its own draft** (case B, 0.5B `bf16` target, the
same `bf16` weights as the draft, K = 4): α = 0.981, 101.6 tok/s against
161.1 (0.63×).

## Cause and workaround

**The check pass costs 2.2 to 2.6 decode steps.** A decode step of one
token runs each matmul as a matrix-vector product (GEMV). A pass of 2 or
more tokens runs Candle's tiled matrix-matrix kernel (GEMM). A timing of
one forward pass of M tokens after the 513-token prompt (median of 10):

| Target | M = 1 | M = 2 | M = 5 | M = 9 |
|---|---:|---:|---:|---:|
| 1.5B `bf16` | 16.6 ms | 32.8 ms | 33.7 ms | 34.2 ms |
| 1.5B `q4` | 6.8 ms | 33.2 ms | 33.5 ms | 33.3 ms |
| 0.5B `bf16` | 6.4 ms | 14.2 ms | 14.9 ms | 15.1 ms |

Data: `bench/runs/2026-11-12-b6d6d66-verify-cost.txt`. From M = 2 to 9
the pass is nearly flat, so extra guesses cost only draft steps. A
quantized target is no help: at M ≥ 2 it costs as much as `bf16`.

**A draft step is not cheap.** For each guess, the 0.5B draft costs 6.7
ms (`bf16`), 4.9 ms (`q8`), and 3.6 ms (`q4`), against 15.28 ms for a
1.5B step. A step has a fixed cost of about 2.5 ms (kernel launches and
the choice of the token on the CPU) that does not fall with the model
size. So the draft has a third of the parameters but costs 24 to 44 % of
a target step.

**α depends on the text.** On prose, the two models agree on fewer than
half of the guesses. On code, the answer completes the current `match`
arm and then copies the module comment from the start of the prompt, so
most of it is already in the context. The text does not loop, and the
target does not reach its end-of-sequence token in the 128 tokens.

**The formula holds.** With E(α, K) = (1 − α^(K+1)) / (1 − α) tokens for
each round, and the measured α and times, the predicted speed is within
1 % of the measurement on every prose row. On code at K = 8 it is 8 to
12 % too high. The accepts on code come in long runs, not independently,
and the run stops at 128 tokens.

**Break even on prose.** For the best prose row (`q4`, K = 2), the check
pass would have to fall from 33.2 ms to 17.6 ms (1.15 decode steps). With
today's costs, α would have to rise from 0.44 to 0.88.

**Two bugs that only a check pass reaches.** Both are fixed at
`9673634`.

- Candle's fused Metal attention with a causal mask and fewer queries
  than keys: when the key length is a multiple of the key tile (16 for
  head dim 128, 32 for head dim 64), the last key block is not masked.
  The queries then see later positions. In speculative decoding this hit
  about 1 check pass in 16 on the 1.5B, and the wrong keys and values
  stayed in the cache after the rollback. Logits differed by up to 19.
  Upstream PR 4062 fixes it. The Fornace pin carries that change.
- Fornace's causal mask for a pass of more than one token at an offset
  mixed F32 and BF16 values, so naive attention in BF16 failed. One-token
  decode never reached it.

**Workarounds in Fornace.** Use a draft only for text that copies from
the context, such as code edits, and use a `q4` draft at K = 8. For free
prose, decode without a draft. A matrix-vector path for 2 to 9 rows would
make the check pass near one decode step. Then, by the formula, prose
would break even and code would reach about 1.8× (*calc*). It is not built. It is
a candidate for a later step, the same kernel that the batching result
asks for.

## What it means if you run models on your own hardware

On a Mac with Candle, a draft model does not make a chat answer faster.
For Qwen2.5-1.5B with a 0.5B draft, an answer in free prose comes 38 to
68 % slower, and the draft costs 0.4 to 1.0 GiB more memory and 80 ms
more before the first token. Use speculation where the answer repeats
text that the model can see: code edits, a document that the answer
quotes, a template. There, a `q4` draft at K = 8 gave 1.35×, with the
same text as the large model alone. Before you choose a K, measure what a
pass of K + 1 tokens costs on your engine. On this one it costs 2.2 decode
steps, and that sets the gain more than the draft does.

## Correctness

With greedy sampling, the text with a draft must be the text of the
target alone, token for token. `tests/speculative.rs`, 128 tokens of
`alice-512.txt`:

| Test | Device | Pairs and K | Result |
|---|---|---|---|
| The 0.5B as its own draft | CPU, F32 | K = 1, 4 | Same text, α = 1.000 |
| 0.5B with a `q8` draft; 1.5B with a 0.5B draft | CPU, F32 | K = 1, 3, 8 | Same text in all 6 runs |
| The same pairs | Metal, F32 | K = 1, 3, 8 | Same text in all 6 runs |
| The same pairs | Metal, BF16 | K = 1, 3, 8 | The text leaves the target's text at token 15 (0.5B) and token 16 (1.5B), at a top-2 margin of 0.000 and 0.125 logit |

In BF16 the check pass (GEMM, full attention kernel) and a decode step
(GEMV, one-token attention kernel) round in a different order. At a near
tie the two can choose different tokens. Each token is still the target's
own choice, under the rounding of the check pass. The test allows a
difference only at a margin of at most 2 BF16 steps at the top logit.
Case B shows the size of the effect: with the same weights as the draft,
1.9 % of the guesses are rejected.

A second test truncates the cache to each of 35 offsets and runs a check
pass of 5 tokens there. In F32, its logits equal the logits of one-token
steps within 0.0002, on the CPU and on Metal, with fused and naive
attention, for both models.

## What this does not prove

- All runs are from one day, on one machine, with one model family: a
  1.5B target and a 0.5B draft. A larger target (3B, 7B) has a longer
  step for the same draft, so the draft is relatively cheaper.
- Greedy sampling only. The rejection sampler for temperature sampling is
  not built, and α is different with sampling.
- One sequence (batch 1) and the contiguous cache only.
- A model draft only. A draft from the prompt itself (n-gram lookup), or
  trained heads (Medusa, EAGLE), were not measured. For code, a lookup in
  the prompt could guess as well at almost no cost.
- K is fixed for a run. An adaptive K was not tried.
- 128 tokens with the end-of-sequence token ignored, and three 512-token
  prompts. One code prompt is not all code. Its high α comes from text
  that the answer copies from the prompt.
- The time of the check pass for each M comes from one test at one
  context length (513).
- The share of the fixed cost of a step (about 2.5 ms) between the
  kernel launches and the token choice on the CPU was not measured.

## Reproduce

Commit `9673634`.

```sh
git clone https://github.com/inertialabsxyz/fornace && cd fornace
make model      # downloads Qwen2.5-0.5B-Instruct to ./models/qwen2.5-0.5b-instruct
scripts/get-model.sh Qwen/Qwen2.5-1.5B-Instruct ./models/qwen2.5-1.5b-instruct
git checkout 9673634
make quantize   # writes the 0.5B model-q8_0.gguf and model-q4_0.gguf
cargo build --release --features fornace-engine/metal -p fornace-bench
fb() {
  ./target/release/fornace-bench --dtypes f16 --attention fused --max-new-tokens 128 \
    --draft-model ./models/qwen2.5-0.5b-instruct --out bench/runs/repro-spec.csv "$@"
}
T=./models/qwen2.5-1.5b-instruct
P=bench/prompts
fb --model $T --prompts $P/alice-512.txt --draft-dtypes f16,q8,q4 --spec-k 0,2,4,8 --tag "case=A"
fb --model ./models/qwen2.5-0.5b-instruct --prompts $P/alice-512.txt \
   --draft-dtypes f16 --spec-k 0,4 --tag "case=B"
fb --model $T --prompts $P/alice-512.txt,$P/code-512.txt,$P/alice-de-512.txt \
   --draft-dtypes q4 --spec-k 0,2 --tag "case=C"
fb --model $T --prompts $P/code-512.txt --draft-dtypes q4,q8 --spec-k 0,2,4,8 --tag "case=C-code-k"
```

The defaults give 1 warm-up, 3 recorded runs, `--kv-cache prealloc`, and
`--ignore-eos true`. `--dtypes f16` gives `bf16` on Metal. The summary
table at the end of each command gives tok/s and α for each row.

Expected: in case A, the target alone near 65 tok/s, and the `q4` draft
at K = 2 near 40 tok/s with α near 0.44. In the code command, the `q4`
draft at K = 8 near 88 tok/s with α near 0.93, against about 66 for the
target alone. Speed drifts between commands, so compare rows from one
command.

The cost of the check pass:

```sh
cargo test --release --features metal -p fornace-engine \
  --test verify_cost -- --ignored --nocapture
```

Expected: for the 1.5B in `bf16`, M = 1 near 16 ms and M = 2 to 9 near
33 to 34 ms.

The correctness tests (about 6 minutes with both models):

```sh
FORNACE_MODEL_DIR=$PWD/models/qwen2.5-0.5b-instruct \
  cargo test --release --features metal -p fornace-engine \
  --test speculative -- --nocapture --test-threads=1
```

Expected: 5 tests pass. Every F32 line says "first difference: None",
"top token changes at (one-token margin): none", or "offsets with a top
token change: none".
