# With 8 requests at once from one agent, llama.cpp gives 2.5× the tokens per second of Fornace on Metal; with one request, it is only 1.1× faster

## Why it matters

People who run a model on their own machine choose an engine by its speed.
This result compares Fornace with the two engines that people use on a
Mac, llama.cpp and mistral.rs, on the same machine, the same model, the
same weight bytes, and the same HTTP client. With one request at a time,
llama.cpp decodes 1.08 to 1.14× faster than Fornace, and mistral.rs is
0.93 to 0.94× of Fornace. That gap is small. But one machine does not
send one request at a time. An agent sends tool calls, sub-agents, and
checks at the same time. With 8 such requests, llama.cpp gives 2.5× the
total tokens per second of Fornace at Q4, and its slowest first token
comes after 0.7 s, not 5.0 s. Fornace runs one request at a time. That is
the largest gap, and it is in the server and in the batched step, not in
the single step.

## What was measured and how

| Item | Value |
|---|---|
| Machine | Apple M4 Pro, 20 GPU cores, 24 GB unified memory, macOS 27.2 |
| Fornace | `fornace-server` at `a1fffdc` (cases A to C) and `160e5c2` (cases D, E). Candle 0.11.0 with three changes (fork `inertialabsxyz/candle`, commit `b4090fe1`). Fused attention, preallocated KV cache, one request at a time (a queue) |
| llama.cpp | `llama-server` 0.9.0, build 12110 (`cd959bae9`), Homebrew. `-ngl 99 -fa on -lm none`, both prompt caches off (`--no-cache-prompt -cram 0`). `-np 1` (cases A, B, E) or `-np 8` (case D) |
| mistral.rs | `mistralrs serve` v0.9.5 (`465191d`, upstream Candle `66a8cf18`), built with `--features metal`. `--paged-attn off`, `--prefix-cache-n 0`, and `--dtype f32` for GGUF. `--max-seqs 1` or `8` |
| Models | Qwen2.5-0.5B-Instruct (cases A to D), Qwen2.5-1.5B-Instruct (case E) |
| Formats | `bf16`, `q8` (Q8_0), `q4` (Q4_0). The Q8_0 and Q4_0 tensors have the same bytes in all engines: Fornace reads the files of `make quantize`, the others the files of `llama-quantize --pure` (SHA-256 below) |
| Requests | `/v1/completions`, the raw prompt, streamed, greedy, repeat penalty 1.0 (off), end-of-sequence token ignored. 128 new tokens (cases A to C, E). One warm-up request, not recorded |
| Timing | The client times each request: the first line of text, the last line, and `usage.completion_tokens`. It never uses the timings of a server |
| Order | Each case ran twice: Fornace, mistral.rs, llama.cpp, and then in the reverse order. The tables give the mean of the two |

Data: `bench/runs/2026-11-13-a1fffdc-compare.csv` (cases A to C),
`bench/runs/2026-11-13-160e5c2-compare.csv` (cases D and E),
`bench/runs/2026-11-13-9b1a8bf-compare.csv` (the first run, all cases,
with `-lm` at its default). The agent load:
`bench/runs/2026-11-13-160e5c2-compare-agent-load-32.csv`.

### One request at a time (case A, `alice-512.txt`, 513 prompt tokens)

| Engine | Format | Decode tok/s | Prefill tok/s | First token ms | Peak MiB | Decode / Fornace | Prefill / Fornace |
|---|---|---:|---:|---:|---:|---:|---:|
| Fornace | `bf16` | 160.0 | 5,755 | 89.1 | 1,211 | 1.00× | 1.00× |
| llama.cpp | `bf16` | 173.5 | 7,001 | 73.3 | 1,683 | 1.08× | 1.22× |
| mistral.rs | `bf16` | 150.3 | 4,450 | 115.3 | 2,192 | 0.94× | 0.77× |
| Fornace | `q8` | 224.9 | 5,561 | 92.2 | 956 | 1.00× | 1.00× |
| llama.cpp | `q8` | 244.6 | 6,937 | 74.0 | 994 | 1.09× | 1.25× |
| mistral.rs | `q8` | 209.8 | 5,481 | 93.6 | 1,086 | 0.93× | 0.99× |
| Fornace | `q4` | 297.4 | 5,552 | 92.4 | 650 | 1.00× | 1.00× |
| llama.cpp | `q4` | 324.6 | 7,078 | 72.5 | 626 | 1.09× | 1.27× |
| mistral.rs | `q4` | 275.3 | 5,448 | 94.2 | 858 | 0.93× | 0.98× |

Prefill tok/s is the prompt tokens / the time to the first token, so it
includes the HTTP request and the tokenizer. Peak memory is the peak
physical footprint of the server process. `-lm none` makes llama.cpp read
the weights into memory. By default it maps the file, and a mapped file
is not in the footprint (212 MiB in every format). `-lm none` does not
change its speed: 1.008× at `q4` and 0.999× at `bf16`, in an ABBA check
(`bench/runs/2026-11-13-a1fffdc-llama-load-mode.txt`).

The first run (`9b1a8bf`) gave llama.cpp 1.12, 1.14, and 1.12× in
decode. Between the runs, Fornace moved 1 to 2 % and llama.cpp 4 to 7 %.
So the decode gap is 1.08 to 1.14×, and the prefill gap 1.22 to 1.34×.

### Prompt length (case B)

| Prompt | Format | Fornace decode | llama.cpp / Fornace, decode | Fornace first token ms | llama.cpp / Fornace, prefill |
|---|---|---:|---:|---:|---:|
| `alice-128.txt` | `bf16` | 168.4 | 1.03× | 31.9 | 1.26× |
| `alice-128.txt` | `q4` | 318.5 | 1.01× | 30.0 | 1.22× |
| `alice-2048.txt` | `bf16` | 155.8 | 1.11× | 338.3 | 1.16× |
| `alice-2048.txt` | `q4` | 272.6 | 1.11× | 369.6 | 1.22× |

The decode gap grows with the context: 1.01 to 1.03× at 129 tokens, 1.11×
at 2,049. mistral.rs: 0.95 to 0.96× of Fornace in decode.

### An agent load (case D)

The load of one agent on one machine. Each request has the same system
prompt (the first 256 tokens of `alice-2048.txt`), then its own part of
64, 256, or 1,024 tokens of code or German text, and asks for 32, 128,
or 256 new tokens. Every 9 requests have all 9 pairs of length and
output. C clients each send their next request when the last one is
done. A level has at least 4 × C and at least 16 requests. mistral.rs
runs at `bf16` only (see "Correctness").

Total tok/s (all new tokens / the wall time of the level):

| Engine | Format | C = 1 | C = 2 | C = 4 | C = 8 |
|---|---|---:|---:|---:|---:|
| Fornace | `bf16` | 144.6 | 144.9 | 144.6 | 145.1 |
| llama.cpp | `bf16` | 156.8 | 247.3 | 324.3 | 471.5 |
| mistral.rs | `bf16` | 127.6 | 110.1 | 109.2 | 110.8 |
| Fornace | `q4` | 235.1 | 239.2 | 237.8 | 237.4 |
| llama.cpp | `q4` | 266.1 | 359.5 | 472.3 | 595.9 |

Time to the first token (ms, p50 and the largest, over both orders), and
the decode tok/s of one request (p50):

| Engine | Format | C | First token p50 | First token, largest | Decode of one request |
|---|---|---:|---:|---:|---:|
| Fornace | `q4` | 4 | 1,355 | 3,015 | 302.5 |
| Fornace | `q4` | 8 | 3,919 | 4,984 | 301.7 |
| llama.cpp | `q4` | 4 | 184 | 335 | 197.2 |
| llama.cpp | `q4` | 8 | 236 | 718 | 102.6 |
| Fornace | `bf16` | 8 | 6,166 | 8,191 | 164.2 |
| llama.cpp | `bf16` | 8 | 227 | 739 | 75.9 |
| mistral.rs | `bf16` | 8 | 369 | 1,368 | 13.6 |

- Fornace gives the same total at every C. Its queue runs one request at
  a time, so a request waits for the requests before it.
- llama.cpp at C = 8 gives 2.2× (`q4`) and 3.0× (`bf16`) its total at
  C = 1, and 2.5× and 3.2× the total of Fornace.
- The llama.cpp levels with a batch vary from run to run. Over six runs
  at `bf16`: 312 to 366 tok/s at C = 4, and 451 to 528 at C = 8. At
  `q4`, C = 4: 452 to 493. The Fornace levels stay within 1 to 4 %.
- mistral.rs with `--paged-attn off` gives less in total at C ≥ 2 than at
  C = 1. It ran with paged attention off, because its paged attention
  kernels did not compile without the Metal compiler. In its default mode
  it batches: see [the scheduler result](2026-11-13-scheduler-qwen2-metal.md).
- llama.cpp with its prompt caches on, at C = 4: +14 % and +17 % (`q4`),
  and +9 % and +23 % (`bf16`) of total tok/s, in two runs. The system
  prompt is the same in every request, and the cache reuses it.

### The 1.5B (case E)

| Format | Fornace decode | llama.cpp / Fornace | mistral.rs / Fornace | Fornace first token ms | llama.cpp / Fornace, prefill |
|---|---:|---:|---:|---:|---:|
| `bf16` | 66.9 | 1.09× | 0.99× | 266.7 | 1.20× |
| `q4` | 158.7 | 1.17× | 0.98× | 278.9 | 1.27× |

The `q4` decode gap is larger with the 1.5B (1.17×) than with the 0.5B
(1.09×).

### The server against the engine in the process (case C, Fornace only)

| Format | Penalty | Process decode | HTTP decode | Process first token ms | HTTP first token ms |
|---|---:|---:|---:|---:|---:|
| `bf16` | 1.0 | 163.2 | 162.2 | 85.8 | 87.6 |
| `bf16` | 1.1 | 156.6 | 159.8 | 86.0 | 86.2 |
| `q4` | 1.0 | 298.7 | 297.9 | 89.4 | 90.8 |
| `q4` | 1.1 | 281.6 | 285.7 | 88.9 | 88.4 |

- HTTP adds 1.4 to 1.8 ms to the first token at penalty 1.0, and 0.02 to
  0.03 ms to a step. Over the three runs: 1.4 to 2.9 ms, and less than
  0.1 ms.
- The repeat penalty (1.1 over the last 64 tokens, on the CPU) adds
  about 0.21 ms to a step: 4.0 % at `bf16` and 5.7 % at `q4` in the
  process.

## Cause and workaround

### The step model

```
step time ≈ weight bytes / bandwidth + fixed cost
```

Two points for each engine, `bf16` and `q4` (*calc*, from
`decode_ms_mean_last64`). The `bf16` rows of Fornace have BF16
activations and the `q4` rows F32, so this fit is approximate.

| Model | Engine | Step `bf16` ms | Step `q4` ms | Fit GB/s | Fixed ms |
|---|---|---:|---:|---:|---:|
| 0.5B | Fornace | 6.25 | 3.38 | 247 | 2.25 |
| 0.5B | llama.cpp | 5.76 | 3.08 | 264 | 2.02 |
| 0.5B | mistral.rs | 6.65 | 3.60 | 233 | 2.41 |
| 1.5B | Fornace | 14.97 | 6.33 | 257 | 2.94 |
| 1.5B | llama.cpp | 13.69 | 5.38 | 267 | 2.13 |
| 1.5B | mistral.rs | 15.05 | 6.40 | 257 | 3.01 |

The M4 Pro reads about 273 GB/s. All three engines read near it. The
fixed cost is 2 ms or more in every engine, so for a 0.5B model the fixed
cost is most of a `q4` step everywhere. The fixed cost of llama.cpp is
the same for both models (2.0 and 2.1 ms). That of Fornace grows from
2.25 to 2.94 ms. So the extra cost of Fornace grows with the model.

### What a step does on the GPU

One Metal System Trace for each row, over 2 s of steady requests
(`bench/runs/2026-11-13-a1fffdc-metal-traces.txt`, traced, so for
explanation only). Medians for one decode step:

| Trace | Command buffers | GPU busy ms | GPU idle ms |
|---|---:|---:|---:|
| Fornace `q4` | 11 | 2.80 | 0.48 |
| llama.cpp `q4` | 3 | 2.38 | 0.80 |
| Fornace `bf16` | 11 | 5.50 | 0.55 |
| llama.cpp `bf16` | 3 | 5.04 | 0.69 |
| llama.cpp `q4`, 4 requests in one step | 3 | 3.16 | 1.11 |
| Prefill, 513 tokens, Fornace `q4` | 14 | 87.6 | — |
| Prefill, 513 tokens, llama.cpp `q4` | 5 | 66.7 | — |

The command buffers include the copy of the logits to the CPU.

### The gap list

**1. The agent load: 2.5× (`q4`) and 3.2× (`bf16`) at C = 8.** At C = 4:
2.0× and 2.2×. The largest first token at C = 8: 4,984 against 718 ms
(`q4`). Two causes:

- *The queue.* Fornace runs one request at a time. llama.cpp puts the
  tokens of all slots in one batch and runs it with one decode call
  (`tools/server/server-context.cpp`, lines 3584 and 3631 to 3639 at build
  12110). A trace of llama.cpp with 4 requests in one step: 3.16 ms of
  GPU time, 1.33× a step of one request, for 4 tokens. It reads the
  weights once for all rows. This is a limit of the Fornace server, not
  of Candle. The fix is a scheduler with continuous batching, the next
  step of this work.
- *The batched step.* A scheduler is not enough with the Candle kernels
  of today. The batching result
  ([2026-11-05](2026-11-05-batching-qwen2-metal.md)) measured that Candle
  runs its dense matrix-vector kernel only at one row, so 2 rows at
  `bf16` cost 2.1× one row. With Q4_0, each row is one quantized
  matrix-vector launch, so 4 rows give only about 1.2× the total. This is
  a Candle limit. A candidate fix: a quantized matmul for 2 to 8 rows
  that reads each weight block once for all rows.

**2. The prefill: 1.22 to 1.34× (`q4` largest).** At `q4`, the prefill of
513 tokens has 87.6 ms of GPU time in Fornace against 66.7 ms in
llama.cpp. The Fornace prefill at `q4` is slower than at `bf16` (92.4
against 89.1 ms to the first token), so the quantized matmul of the
prefill is the probable cause. One Fornace debug trace
(`bench/runs/2026-11-13-9b1a8bf-trace-q4-prefill.txt`): the matmuls are
76 % of the prefill. Candle runs the quantized matmul with F32
activations. llama.cpp has its own Metal kernel for a matmul with
quantized weights, on SIMD-group matrices (`kernel_mul_mm` in
`ggml/src/ggml-metal/kernels/mul_mm.metal`, line 155, and the
`simdgroup_multiply_accumulate` at line 314). This is a Candle limit. A
candidate fix: a quantized matmul kernel on SIMD-group matrices.

**3. The decode step: 1.08 to 1.14× at the 0.5B, 1.17× at the 1.5B `q4`.**
The trace shows that the gap is GPU time: 0.42 to 0.46 ms more in each
step, in 11 command buffers against 3. The GPU of Fornace is idle for a
shorter time than that of llama.cpp (0.48 against 0.80 ms at `q4`). So
the extra cost of Fornace is not between the steps. It is in its
kernels. Candle encodes each operation in its own compute encoder and
starts a new command buffer after 50 of them
(`candle-metal-kernels/src/metal/commands.rs`, line 18). Eleven command
buffers in a step are about 500 kernel launches (*calc*). llama.cpp
encodes the graph of a step in 1 or 2 command buffers with concurrent
dispatch, and fuses chains of operations into one kernel
(`ggml/src/ggml-metal/ggml-metal-context.m`, lines 488 to 500, and
`ggml/src/ggml-metal/ggml-metal-ops.cpp`, lines 235 to 237). The trace
does not say which Fornace kernels take the extra time: it names only
labelled kernels, and the matmul kernels of Candle have no label. So the
exact cause is not known. A GPU capture in Xcode, or a label on each
Candle kernel, would find it. This is a Candle limit. A candidate fix:
fused kernels (the normalisation and the matmul, the rotary embedding of
Q and K) and fewer encoders.

**The largest Metal gap.** On the Metal side, the largest gap is the
batched step for a few rows (gap 1, second part). The scheduler result
([2026-11-13](2026-11-13-scheduler-qwen2-metal.md)) measured it: Candle's
matmul kernels cost about 12 ms a step for any batch of 2 to 8 rows on
this Mac. That floor is the candidate for the next part of this work.
This result does not promise a fix.

**Earlier results.** The batching result and the paged-cache result say
that one user does not need a batch. They measured copies of one prompt.
An agent on one machine sends several different requests at once. With
that load, the batch is the largest gap of all.

### Not a gap: memory and quality

- Memory: Fornace uses less memory than mistral.rs in every format, and
  less than llama.cpp at `bf16` (1,211 against 1,683 MiB). At `q4`,
  Fornace and llama.cpp are within 4 % (650 and 626 MiB).
- Quality: see "Correctness".

## What it means if you run models on your own hardware

For one request at a time, the three engines are near each other on a
Mac: llama.cpp is about 10 % faster than Fornace in decode, and about
25 % faster to the first token, and mistral.rs is about 6 % slower. A
choice between them on single-request speed changes little. Prefill is
the larger difference, and it grows with long prompts. If an agent, or
several tasks, send requests to the same local server at the same time,
the difference is large. Use an engine that batches requests together.
With 8 requests at once, llama.cpp gave 2.5 to 3.2× the total tokens per
second of Fornace, and the slowest request got its first token after
0.7 s, not 5 to 8 s. Turn the prompt cache on when the requests share a
system prompt: it gave llama.cpp another 9 to 23 %.

## Correctness

- **The same tokens.** For each format, all three engines give the same
  64 greedy tokens of `alice-512.txt` as Fornace in the process, with no
  repeat penalty (`bench/runs/2026-11-13-25323e3-dirty-same-tokens.csv`).
  The check compares text. At `q4` the text is a loop (`[1] [2] [3] ...`),
  so the check at `q4` is weaker. Fornace over HTTP gives the same text
  as Fornace in the process.
- **The same prompt tokens.** Every server counted the Fornace token
  count of every prompt: 513 for `alice-512.txt`, and each request of the
  agent load. mistral.rs sends no usage in a stream, so for mistral.rs
  the check covers the request without a stream before each level, and
  the token count of a streamed request is `max_tokens` (the stream ends
  with the finish reason `length`). Qwen2.5 adds no BOS token.
- **The same quality.** With the scoring method of `llama-perplexity`
  (chunks of ctx tokens, only the second half of each chunk scored),
  Fornace and llama.cpp give the same perplexity to the third decimal at
  `q8` and `q4`, and within 0.3 % at `bf16`
  (`bench/runs/2026-11-13-a1fffdc-quality.csv`,
  `bench/runs/2026-11-13-25323e3-dirty-llama-perplexity.txt`):

  | Text, ctx | Fornace `bf16`, `q8`, `q4` | llama.cpp `bf16`, `q8`, `q4` |
  |---|---|---|
  | `alice-2048.txt`, 1024 | 14.760, 14.852, 17.327 | 14.794, 14.852, 17.326 |
  | `alice-4096.txt`, 2048 | 14.249, 14.288, 17.063 | 14.239, 14.289, 17.062 |

  The quantization result ([2026-11-09](2026-11-09-quantization-qwen2-metal.md))
  gave +23 % for `q4`. That number scores every position, including those
  with a short context. With the llama.cpp method, the `q4` cost is +17 to
  +20 %. Both are correct for their method. Compare a perplexity only with
  one of the same method.
- **mistral.rs with GGUF and more than one request.** At `q8` and `q4`,
  with 4 requests at once, 3 of 4 requests failed with `input tensor is
  not contiguous` in a batched prompt step. So case D has mistral.rs at
  `bf16` only.

## The predictions against the measurement

Written before the runs, after the first numbers without HTTP.

| Prediction | Measured | |
|---|---|---|
| llama.cpp decodes 1.15 to 1.25× faster | 1.08 to 1.14× | Failed, a little lower |
| The format changes the decode gap by less than 0.1× | 1.08, 1.09, 1.09× | Held |
| mistral.rs decodes 1.05 to 1.15× faster than Fornace | 0.93 to 0.94× | Failed |
| The prefill gap is 1.2 to 1.4×, larger than the decode gap | 1.22 to 1.34× | Held |
| mistral.rs uses the most memory at `q4` | 858 MiB, against 650 and 626 | Held |
| HTTP adds less than 1 ms to the first token | 1.4 to 2.9 ms | Failed |
| HTTP adds less than 0.1 ms to each token | 0.01 to 0.07 ms | Held |
| The repeat penalty costs 1 to 2 % | 1.5 to 6.7 % over the runs; 4.0 and 5.7 % in the process (run 3) | Failed |
| The Fornace total is the same at every C, near 230 (`q4`) and 135 (`bf16`) | 235 to 239, and 145 | Held |
| The largest Fornace first token at C = 8 is near 4.2 s (`q4`) and 7 s (`bf16`) | 5.0 and 8.2 s | About 17 % higher |
| llama.cpp at C = 4 gives 2 to 3.5× its total at C = 1 | 1.78× (`q4`), 2.07× (`bf16`) | Failed at `q4` |
| mistral.rs gains less than llama.cpp from C | It loses 13 % | Held |
| The `q4` decode gap is smaller with the 1.5B | 1.17×, against 1.09× | Failed |
| (before any measurement) The largest decode gap is at `q4`, near 2×, because of the fixed cost | 1.09× | Failed. Both engines have a fixed cost near 2 ms |

## What this does not prove

- One machine (M4 Pro), one model family (Qwen2.5, 0.5B and 1.5B), on one
  day.
- Q8_0 and Q4_0 only, with every tensor in the named format. Most users
  run K-quants (for example Q4_K_M), which were not measured.
- Greedy sampling with no repeat penalty, and 128 new tokens with the
  end-of-sequence token ignored (except case D).
- The default build of each reference engine, at one version: llama.cpp
  from Homebrew, mistral.rs built from source without the Metal compiler
  (so with `--paged-attn off`). Each engine has settings that were not
  tuned. A different build or setting can change its numbers.
- Concurrency 1 to 8 only. One synthetic agent load, not the trace of a
  real agent.
- The prompt caches were off, except one llama.cpp level. A real agent
  server would keep them on.
- The llama.cpp levels with a batch vary by up to 16 % between runs. The
  tables give one run. The range is in the text.
- The traces count command buffers and GPU time, not kernels. Which
  Fornace kernels take the extra decode time is not known.
- The prefill tok/s over HTTP includes the request and the tokenizer.

## Reproduce

Commit `a1fffdc`. On macOS, with Homebrew.

```sh
git clone https://github.com/inertialabsxyz/fornace && cd fornace
git checkout a1fffdc
make model && make quantize
scripts/get-model.sh Qwen/Qwen2.5-1.5B-Instruct ./models/qwen2.5-1.5b-instruct
make quantize MODEL=./models/qwen2.5-1.5b-instruct
scripts/get-reference-engines.sh --install     # llama.cpp and mistral.rs
scripts/make-reference-weights.sh ./models/qwen2.5-0.5b-instruct
scripts/make-reference-weights.sh ./models/qwen2.5-1.5b-instruct
cargo build --release --features metal -p fornace-server -p fornace-bench
fb() { target/release/fornace-bench --transport http --attention fused \
         --repeat-penalty 1.0 --out bench/runs/repro-compare.csv "$@"; }
M=./models/qwen2.5-0.5b-instruct; P=bench/prompts
# Case A
fb --model $M --engines fornace,mistralrs,llamacpp --dtypes f16,q8,q4 \
   --prompts $P/alice-512.txt --requests 3
# Case D (mistral.rs at f16 only)
fb --model $M --load agent --concurrency 1,2,4,8 --parallel 8 \
   --engines fornace,llamacpp --dtypes f16,q4
fb --model $M --load agent --concurrency 1,2,4,8 --parallel 8 \
   --engines mistralrs --dtypes f16
# Case E
fb --model ./models/qwen2.5-1.5b-instruct --engines fornace,mistralrs,llamacpp \
   --dtypes f16,q4 --prompts $P/alice-512.txt --requests 3
```

`--dtypes f16` gives `bf16` on Metal. The summary at the end of each
command gives, for each level: the p50 and the largest first token, the
p50 decode tok/s, the total tok/s, and the peak memory.

Expected: in case A at `q4`, Fornace near 297 decode tok/s, llama.cpp
near 325 to 338, and mistral.rs near 275 to 285. In case D at C = 8,
Fornace near 237 total tok/s at `q4`, and llama.cpp near 595. Speed
drifts between commands, so compare the engines within one command.

The same tokens and the perplexity:

```sh
target/release/fornace-bench --model $M --same-tokens --engines fornace,llamacpp,mistralrs \
  --dtypes f16,q8,q4 --attention fused --prompts $P/alice-512.txt \
  --max-new-tokens 64 --repeat-penalty 1.0
target/release/fornace-bench --model $M --dtypes f16,q8,q4 --attention fused \
  --prompts $P/alice-128.txt --max-new-tokens 2 --repeats 1 --warmup 0 \
  --quality $P/alice-2048.txt --quality-ctx 1024
llama-perplexity -m $M/model-q4_0.llama.gguf -f $P/alice-2048.txt -c 1024 -ngl 99 -fa on
```

Expected: "none in 64" for every engine and format. Fornace near 14.760,
14.852, 17.327, and `llama-perplexity` near 17.326 for `q4`.

The SHA-256 of the weight files:

| File | SHA-256 |
|---|---|
| 0.5B `model.safetensors` | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |
| 0.5B `model-q8_0.gguf` | `c6ea40baee71fb0d0008246057bbcfed0b9221dc5de73a306e8c5e40d83c0e94` |
| 0.5B `model-q4_0.gguf` | `a2c10bd8495e524ca16a5be9491d373878084bb21137e48358e2a3b5c30f1bfd` |
| 0.5B `model-bf16.llama.gguf` | `19d95e2afa1ab58fc1b3a8130e787b9c2aea4cb15dc432f303ff3d6a472d4f2f` |
| 0.5B `model-q8_0.llama.gguf` | `bce18b02e687a837bc240212d261ed8f4b80f0607f5b465885da7c1d145e40ae` |
| 0.5B `model-q4_0.llama.gguf` | `8872d61bc9215c4bcece098ca412e4bc1c2e88d1d478e693b2f8b7dd3eb5a666` |
| 1.5B `model.safetensors` | `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` |
| 1.5B `model-q4_0.gguf` | `ed4ac94e013862ad2e7dc77f590ea720919b9f1b1d584b0774b7b1ead14d7248` |
| 1.5B `model-bf16.llama.gguf` | `123254df7aac5b3bc8a896c4adc467c19c60535992b29bc5a4b3ab9cca96edd0` |
| 1.5B `model-q4_0.llama.gguf` | `e6dc487a269784468e8800f5bb9e52882949086c7759a066a23a27bdbfa9f829` |

The Q8_0 and Q4_0 files of Fornace and of llama.cpp have different
SHA-256, because the tensor names and the metadata differ. Their tensor
bytes are the same (checked tensor by tensor).
