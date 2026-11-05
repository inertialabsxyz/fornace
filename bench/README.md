# Benchmarks

`fornace-bench` runs each engine on the same model and the same prompts. It
writes one CSV row for each run.

## How to run

Download the model first with `scripts/get-model.sh`. Then run:

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
  `f16`, `q8`, or `q4`. The default is `f16`. See the `dtype` column.
- `--kv-cache <LIST>`: the KV caches, separated by commas: `cat`,
  `cat-contiguous`, or `prealloc`. The default is `prealloc`.
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
- `--engines <LIST>`: the engines to measure, separated by commas. The
  default is `fornace`.
- `--out <CSV>`: the CSV file for the results. If it exists, the rows are
  appended. The default is `bench/runs/<date>-<commit>.csv`.
- `--tag <TEXT>`: a note for the `tag` column.

On macOS, add `--features metal` to use the GPU.

Each (dtype, KV cache, batch) set runs in its own process, so that each
set has its own peak memory.

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
- `dtype`: the data type of the weights that the engine used. This can differ from `--dtypes`. Nothing is quantized today, so it is `bf16` on Metal and `f32` on the CPU.
- `kv_cache`: the KV cache: `cat`, `cat-contiguous`, or `prealloc`.
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
- `quality`: a quality score. This column is empty until quantization is added.
- `tag`: the text of `--tag`, or empty.
