# Benchmarks

`fornace-bench` runs each engine on the same model and the same prompts. It
writes one CSV row for each run.

## How to run

Download the model first with `scripts/get-model.sh`. Speculative decoding
also needs Qwen2.5-1.5B-Instruct as the target:

```sh
scripts/get-model.sh Qwen/Qwen2.5-1.5B-Instruct ./models/qwen2.5-1.5b-instruct
make quantize MODEL=./models/qwen2.5-1.5b-instruct
```

Then run:

```sh
cargo run --release -p fornace-bench -- \
    --model ./models/qwen2.5-0.5b-instruct \
    --engines fornace,mistralrs,llamacpp \
    --out bench/runs/latest.csv
```

- `--model <DIR>`: the directory that contains the model files.
- `--prompts <FILES>`: the prompt files, separated by commas. The default
  is `alice-512.txt`, `alice-2048.txt`, and `alice-4096.txt` in
  `bench/prompts/`.
- `--dtypes <LIST>`: the data types of the weights, separated by commas:
  `f16` (dense, BF16 on Metal and F32 on the CPU), `f32` (dense F32), `q8`
  (Q8_0), or `q4` (Q4_0). The default is `f16`. `q8` and `q4` read
  `model-q8_0.gguf` and `model-q4_0.gguf` in the model directory. Make them
  with `make quantize`. See the `dtype` column.
- `--kv-cache <LIST>`: the KV caches, separated by commas: `cat`,
  `cat-contiguous`, `prealloc`, or `paged`. The default is `prealloc`.
- `--block-size <LIST>`: the block sizes of the `paged` cache, separated
  by commas. The default is `16`. Only `paged` rows use them.
- `--kv-capacity <N>`: the positions that the cache reserves for each
  sequence. The default is the prompt and `--max-new-tokens`, which the
  run knows in advance. `prealloc` reserves this many positions. The
  `paged` pool holds this many positions for each sequence, in blocks.
- `--share-prefix <LIST>`: `false`, `true`, or `false,true`. The default
  is `false`. With `true`, a row with a batch above 1 prefills the tokens
  that all sequences have in common at the start (the shared prefix)
  once, as one sequence. Then it prefills the rest of each sequence as a
  batch. `paged` shares the full blocks of the prefix, and each sequence
  gets a copy of the last block if it is not full. `prealloc` copies the
  prefix into each sequence once. Other caches use only `false`.
- `--suffix <FILE>` and `--suffix-tokens <N>` (default 64): give each
  sequence its own end. Sequence b is the prompt, then tokens b × N to
  (b + 1) × N of the file. So all sequences share the prompt, as a system
  prompt, and each has its own user message.
- `--batch <LIST>`: the batch sizes, separated by commas. A batch of B runs
  B copies of the prompt at the same time. The default is `1`.
- `--attention <KIND>`: `naive` or `fused`. The default is `naive`. Fused
  needs `--device metal`.
- `--device <DEVICE>`: `cpu` or `metal`. The default is `metal` on macOS
  and `cpu` on other systems.
- `--max-new-tokens <N>`: the number of tokens to generate for each
  sequence. The default is 128.
- `--repeats <N>`: the number of recorded runs for each prompt. The default
  is 3.
- `--warmup <N>`: the number of runs before the recorded runs. They are not
  recorded. The default is 1.
- `--ignore-eos <BOOL>`: do not stop at the end-of-sequence token. The
  default is `true`, so each run generates exactly `--max-new-tokens`
  tokens.
- `--draft-model <DIR>`: a draft model for speculative decoding. It must
  have the tokenizer of `--model`. It needs `--batch 1` and `--kv-cache
  prealloc`.
- `--draft-dtypes <LIST>`: the data types of the draft weights, separated
  by commas, as for `--dtypes`. The default is `f16`.
- `--spec-k <LIST>`: the guesses of the draft in each round, separated by
  commas. The default is `0`. A K of `0` runs `--model` without a draft,
  so `--spec-k 0,2,4` puts the target alone and the speculative rows in one
  command. Each K above 0 runs once with each draft dtype. A K above 0
  needs `--draft-model`. See "Metrics for speculative decoding".
- `--engines <LIST>`: the engines to measure, separated by commas. The
  default is `fornace`.
- `--out <CSV>`: the CSV file for the results. If it exists, the rows are
  appended. The default is `bench/runs/<date>-<commit>.csv`.
- `--tag <TEXT>`: a note for the `tag` column.
- `--quality <FILE>`: a text file for the `quality` column. Each process
  computes the perplexity of the model on it once, after its recorded
  runs, and writes it to each of its rows.
- `--repeat-penalty <P>`: the repeat penalty, over the last 64 generated
  tokens. The default is 1.1. 1.0 turns it off.
- `--transport <LIST>`: `process`, `http`, or both, separated by commas.
  The default is `process`. See "Over HTTP".

On macOS, add `--features metal` to use the GPU.

Each (dtype, KV cache, block size, shared prefix, batch, draft dtype, K)
set runs in its own process, so that each set has its own peak memory.

A `paged` worker also prints the blocks of the cache at the end of each
run to stderr: the blocks in use, the blocks in the pool, the positions
stored, and the share of the positions in the blocks in use that holds
nothing. These numbers are not in the CSV file.

## Reference engines

The comparison runs llama.cpp and mistral.rs on the same model. Two
scripts prepare them:

```sh
scripts/get-reference-engines.sh --install
scripts/make-reference-weights.sh ./models/qwen2.5-0.5b-instruct
```

- `get-reference-engines.sh` installs llama.cpp with Homebrew, and builds
  mistral.rs at a fixed tag with Metal in `~/.cache/fornace/ref` (set
  `FORNACE_REF_DIR` to change it). Without `--install`, it installs
  nothing. It writes the version and the commit of each engine to
  `bench/engines.txt`.
- `make-reference-weights.sh` writes `model-bf16.llama.gguf`,
  `model-q8_0.llama.gguf`, and `model-q4_0.llama.gguf` from
  `model.safetensors`, with the llama.cpp converter and
  `llama-quantize --pure`. It prints the SHA-256 of each file. The Q8_0
  and Q4_0 tensors are the same bytes as the files of `make quantize`.
  Only the tensor names and the metadata differ.

## Over HTTP

With `--transport http`, the bench starts the server of each engine, sends
the requests, and times them as a client. One server runs for each
(engine, dtype), and all levels use it. Build the Fornace server first,
with the profile and the features of the bench:

```sh
cargo build --release --features metal -p fornace-server -p fornace-bench
target/release/fornace-bench --model ./models/qwen2.5-0.5b-instruct \
    --engines fornace,llamacpp,mistralrs --transport http \
    --dtypes f16,q8,q4 --attention fused --repeat-penalty 1.0 \
    --prompts bench/prompts/alice-512.txt
```

- `--concurrency <LIST>`: the requests in flight at the same time,
  separated by commas. The default is `1`. Each value is one level. At
  concurrency C, C threads each send their next request when the last one
  is done.
- `--requests <N>`: the recorded requests of each level. The default is 3.
- `--load <KIND>`: `plain` (every request is the prompt file) or `agent`.
  The default is `plain`. See "The agent load".
- `--port <N>`: the port of the server. The default is 8090. If the port
  is in use, the bench stops.
- `--server-context <N>`: the context positions of each server slot. The
  default is 4096.
- `--parallel <N>`: the slots of llama.cpp (`-np`) and mistral.rs
  (`--max-seqs`). The default is the largest `--concurrency`. Fornace has
  one queue, and runs one request at a time.
- `--cache-prompt`: llama.cpp only. Keep its prompt caches on.

Only `fornace` runs with `--transport process`. HTTP runs batch 1,
`prealloc`, and no draft.

The command of each server is in `crates/fornace-bench/src/servers.rs`.
The bench writes it to stderr before each start, and the server output
to a log file in the temporary directory. The server is ready when its
`/health` route gives 200, or the bench stops after 120 s.

- llama.cpp: `llama-server` from `PATH`, with `-ngl 99 -fa on`,
  `-c` (the slots × `--server-context`), `-np`, and both prompt caches
  off (`--no-cache-prompt -cram 0`).
- mistral.rs: `mistralrs serve` from `scripts/get-reference-engines.sh`,
  with `--prefix-cache-n 0 --max-seqs <N> --paged-attn off`. GGUF also
  gets `--dtype f32`. The model name of each request is `default`.

Each request is a `/v1/completions` request with the raw text of the
prompt, `stream: true`, `temperature: 0`, `ignore_eos`, and the repeat
penalty in the field of each engine (`repetition_penalty` for
mistral.rs). Before each level, one request without a stream warms the
server up. It is not recorded. Its `usage` must give the Fornace prompt
token count, and with EOS ignored, `max_tokens` new tokens.

The client records the time before it sends a request, and the arrival
of each SSE line with text. It never uses the times of the server.

- `ttft_ms`: the first line with text, from the send. It includes the
  HTTP request, the time in the queue, the tokenizer, and the prefill.
- `decode_tok_s`: (n − 1) / (the last line with text − the first).
  n is `usage.completion_tokens`. A line can hold more than one token,
  so the bench does not count lines. mistral.rs sends no usage in a
  stream. With EOS ignored and the finish reason `length`, n is then
  `max_tokens`, and the row has `n=max_tokens` in the tag.
- `prefill_tok_s`: the prompt tokens / `ttft_ms`. It includes the request
  time, so it is a little lower than in a process.
- `decode_ms_mean_last64`: the mean gap between the last 64 lines with
  text, if each line holds one token. Otherwise empty.
- `peak_mem_mb`: the peak physical footprint of the server process, read
  after the level. The levels before it are in the peak.

The summary has one line for each level: the p50 and the largest
`ttft_ms`, the p50 `decode_tok_s`, the total tok/s (all generated tokens
/ the wall time of the level), and the peak memory. With fewer than 100
requests, p99 is the largest value, so the table gives the maximum.

### The agent load

`--load agent` sends the requests of one agent harness on one machine.
The list is the same in each run and in each engine.

- Each request starts with the first 256 tokens of `alice-2048.txt`.
- Then it has its own part, from `code-512.txt` and `alice-de-512.txt` in
  turn, with 64, 256, and 1,024 tokens in turn. The own parts of one
  source start 31 tokens apart, and wrap to the start of the source.
- `max_tokens` is 32, 128, and 256. It moves one step after each full
  cycle of own lengths, so requests 0 to 2 get 32, requests 3 to 5 get
  128, and requests 6 to 8 get 256. Every 9 requests have all 9 pairs of
  own length and `max_tokens`. A level of 16 or 32 requests is not a
  multiple of 9, so some pairs come more often. The list file gives each
  pair.
- A level at concurrency C sends at least 4 × C requests, and at least
  16.

The bench builds each prompt from token ids, and decodes it to text. An
own part that starts inside a character starts at the next token. The
bench writes the list to `<CSV name>-agent-load.csv`: the index, the
source, the first token, the token counts, and `max_tokens`. The
`prompt_tokens` column is the count of the decoded text, which can differ
from the sum of the parts by a token.

## Metrics for a batch

A row with `batch` B has B sequences. Each sequence has T prompt tokens
and N generated tokens. All sequences start together, and every decode
step makes one token for each sequence.

- `prompt_tokens` and `generated_tokens` are for one sequence: T and N.
- `prefill_tok_s` is the total for all sequences: B × T / prefill time.
- `decode_tok_s` is the total for all sequences: B × (N − 1) / decode time.
- The decode tok/s for one sequence is `decode_tok_s` / `batch`.
- `ttft_ms` is the time to the first token of all sequences.
- `decode_ms_mean_last64` is the time of one decode step for all
  sequences.

At batch 1, these are the same as the numbers for one sequence.

With a shared prefix (`shared` in the tag), the prefill computes the
shared tokens once, not B times. `prefill_tok_s` keeps its definition, B × T /
prefill time, so it counts the shared tokens B times. Compare
`ttft_ms` instead.

## Metrics for speculative decoding

A row with a draft has one sequence (batch 1). In each round, the draft
guesses K tokens, one step each, and `--model` (the target) checks them in
one forward pass of K + 1 tokens. Greedy sampling only.

- `ttft_ms` includes the prefill of the draft, because the first round
  needs the draft cache. So does `prefill_tok_s`.
- `decode_tok_s` keeps its definition: (N − 1) / decode time.
- `decode_ms_mean_last64` is a mean over tokens, not over rounds. Each
  token of a round gets the round time divided by the tokens of the round.
- `alpha` is `accepted` / `tested`. After the first reject in a round, the
  later guesses are not checked, so they are not in `tested`. Do not use
  `accepted` / (`rounds` × K).

## Runs

Commit one CSV file for each run to `bench/runs/`. Each result gives the
name of its CSV file and the commit.

## Columns

The columns are in this order. Each row is one recorded generation.

- `commit`: the short git commit hash at run time, with `-dirty` if tracked files have changes, or `unknown` if git fails.
- `date`: the date of the run in UTC, in ISO 8601 format (`YYYY-MM-DD`).
- `engine`: the engine name: `fornace`, `mistralrs`, or `llamacpp`.
- `model`: the directory name of `--model`.
- `device`: the device: `cpu` or `metal`.
- `dtype`: the data type of the weights that the engine used: `bf16`, `f32`, `q8`, or `q4`. This can differ from `--dtypes`: `f16` gives `bf16` on Metal and `f32` on the CPU. With `q8` and `q4`, the model computes in F32.
- `kv_cache`: the KV cache: `cat`, `cat-contiguous`, `prealloc`, or `paged`.
- `batch`: the number of sequences that run at the same time. See "Metrics for a batch".
- `prompt_file`: the file name of the prompt.
- `prompt_tokens`: the number of tokens in the prompt of one sequence.
- `generated_tokens`: the number of tokens that the engine generated for one sequence.
- `run`: the number of the recorded generation, from 1 to `--repeats`.
- `load_ms`: the time to load the engine, in milliseconds. The engine loads once for each `dtype`, `kv_cache`, and `batch`, so all rows of one load have the same value.
- `prefill_tok_s`: the prompt tokens processed each second, for all sequences together.
- `decode_tok_s`: the generated tokens each second, after the first token, for all sequences together.
- `ttft_ms`: the time from the start to the first generated token of all sequences, in milliseconds.
- `decode_ms_mean_last64`: the mean time of the last 64 decode steps, in milliseconds. One step makes one token for each sequence. If there are fewer than 64 steps, the mean of all steps.
- `peak_mem_mb`: the peak memory of the process, in MiB, with one decimal. On macOS this is the physical footprint, which includes Metal buffers. On Linux it is the peak RSS. Each `dtype`, `kv_cache`, and `batch` runs in its own process, so the peak is for that set only. In one set, the peak of a prompt includes the prompts that ran before it.
- `quality`: the perplexity of the model on the `--quality` text, with 3 decimals, or empty without `--quality`. Perplexity is exp of the mean of −log p(next token) over all tokens after the first, with no repeat penalty. Lower is better.
- `tag`: the text of `--tag`, then `attention=fused` for fused attention, then `block=<N>` for a `paged` row, then `shared` for a shared prefix (`paged` or `prealloc`), then `capacity=<N>` for `--kv-capacity`, then `suffix=<file>:<N>` for `--suffix`, then `quality=<file name>` for the `--quality` text, separated by `; `. Empty if none applies.
- `draft`: the directory name of the draft model and the format of its weights, for example `qwen2.5-0.5b-instruct:q8`. Empty without a draft. The columns after it are empty without a draft too.
- `spec_k`: K, the guesses of the draft in each round.
- `rounds`: the speculative rounds in the run.
- `accepted`: the guesses that the target accepted, in all rounds.
- `tested`: the guesses that the target checked, in all rounds: j + 1 in a round with a reject after j accepts, and K in a round with no reject.
- `alpha`: `accepted` / `tested`, with 3 decimals.
- `draft_ms`: the total time of the draft steps, in milliseconds, untraced. It includes the extra draft step after a round with no reject.
- `verify_ms`: the total time of the verify passes, in milliseconds, untraced. It includes the logits to the CPU and the choice of the target's token at each position.
- `engine_version`: `fornace@<commit>`, `llama.cpp@b<build>`, or `mistral.rs@v<version>`.
- `transport`: `process` (the engine in a worker) or `http` (the server of the engine).
- `concurrency`: the requests in flight at the same time. 1 for process rows.
- `request`: the index of the request in its level, from 0. Empty for process rows. For HTTP rows, `run` is `request` + 1, and `batch` is 1.
- `repeat_penalty`: the repeat penalty of the row. 1.0 is off.

The eight columns from `draft` to `verify_ms` are new on 2026-11-12. The last five are new on 2026-11-13. Older CSV files do not have them. Do not append new rows to an older file.

For HTTP rows, `kv_cache` is `prealloc` for Fornace and empty for the other engines. `load_ms` is the time from the start of the server to its first 200 from `/health`. The tag has `np=<N>` for llama.cpp and mistral.rs, `load=agent` for the agent load, and `cache_prompt` for `--cache-prompt`.
