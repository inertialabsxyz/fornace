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
- `--engines <LIST>`: the engines to measure, separated by commas. The
  default is `fornace`.
- `--out <CSV>`: the CSV file for the results. The default is
  `bench/runs/latest.csv`.

On macOS, add `--features metal` to use the GPU.

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
- `batch`: the number of sequences that run at the same time.
- `prompt_file`: the file name of the prompt.
- `prompt_tokens`: the number of tokens in the prompt.
- `generated_tokens`: the number of tokens that the engine generated.
- `run`: the number of the recorded generation, from 1 to `--repeats`.
- `load_ms`: the time to load the engine, in milliseconds. The engine loads once for each `dtype` and `kv_cache`, so all rows of one load have the same value.
- `prefill_tok_s`: the prompt tokens processed each second.
- `decode_tok_s`: the generated tokens each second, after the first token.
- `ttft_ms`: the time from the start to the first generated token, in milliseconds.
- `decode_ms_mean_last64`: the mean time of the last 64 decode steps, in milliseconds. If there are fewer than 64 steps, the mean of all steps.
- `peak_mem_mb`: the peak memory of the process, in MiB, with one decimal. On macOS this is the physical footprint, which includes Metal buffers. On Linux it is the peak RSS. Each `dtype` and `kv_cache` runs in its own process, so the peak is for that pair only. In one pair, the peak of a prompt includes the prompts that ran before it.
- `quality`: a quality score. This column is empty until quantization is added.
- `tag`: the text of `--tag`, or empty.
