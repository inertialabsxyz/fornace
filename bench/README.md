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

- `commit`: the git commit hash of the engine.
- `date`: the date of the run, in ISO 8601 format.
- `engine`: the engine name: `fornace`, `mistralrs`, or `llamacpp`.
- `model`: the model name.
- `dtype`: the data type of the weights: `f16`, `q8`, or `q4`.
- `batch`: the number of sequences that run at the same time.
- `prompt_tokens`: the number of tokens in the prompt.
- `generated_tokens`: the number of tokens that the engine generated.
- `prefill_tokens_per_s`: the prompt tokens processed each second.
- `decode_tokens_per_s`: the generated tokens each second.
- `time_to_first_token_ms`: the time from the start to the first generated token, in milliseconds.
- `peak_memory_mb`: the peak memory of the process, in megabytes. On macOS
  this is the physical footprint, which includes Metal buffers. On Linux it
  is the peak RSS.
- `quality`: a quality score. This column is empty until quantization is added.
