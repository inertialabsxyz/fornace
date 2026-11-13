# Fornace

Fornace is an ongoing project on local AI inference in Rust. It builds small
inference engines, down to the kernels, and measures each change on real
hardware. It also looks at how to prove where a model ran. The first engine
runs Qwen2.5-0.5B on [Candle](https://github.com/huggingface/candle).

## Results

Each result is a short note. Its title gives the result with a number, for
example: "A KV cache makes decode 6x faster on an M4 and costs 1.2 GB at 8k
context". The full list is in [results/README.md](results/README.md). The
three newest results are below.

No results yet.

## Crates

| Crate | What it is | Status |
|---|---|---|
| [`fornace-engine`](crates/fornace-engine) | The inference engine on Candle: a library and the `fornace` command | in progress |
| [`fornace-bench`](crates/fornace-bench) | The benchmark harness: it writes one CSV row per run | in progress |
| [`fornace-server`](crates/fornace-server) | An OpenAI-compatible HTTP server for one engine: the `fornace-server` command | in progress |
| [`fornace`](crates/fornace) | The root crate: it exports the engine | root |

## How to run

Download the model to `./models/qwen2.5-0.5b-instruct`:

```sh
make model
```

Generate text from a prompt:

```sh
make run PROMPT="The capital of Spain is"
```

Run the formatter check, clippy, and the tests:

```sh
make check
```

The model test runs only when `FORNACE_MODEL_DIR` is set:

```sh
FORNACE_MODEL_DIR=$PWD/models/qwen2.5-0.5b-instruct make test
```

Run the benchmark:

```sh
make bench
```

On macOS, `make run`, `make build`, and `make bench` use the GPU (Metal).

## The server

Start the server on port 8080:

```sh
cargo run --release --features metal -p fornace-server -- \
    --model ./models/qwen2.5-0.5b-instruct
```

- Endpoints: `POST /v1/completions`, `POST /v1/chat/completions`,
  `GET /v1/models`, and `GET /health`. `/health` gives 503 until the engine
  is loaded.
- Flags: `--dtype` (default `f16`), `--attention` (default `fused`),
  `--device`, `--host` (default `127.0.0.1`), `--port` (default 8080),
  `--max-context` (default 4096), and `--repeat-penalty` (default 1.1).
- The engine runs one request at a time. Other requests wait in a queue,
  in the order they arrive.
- Decoding is greedy. A `temperature` above 0, a `top_p` below 1, an `n`
  above 1, `logprobs`, or a prompt that is a list gives 400. A prompt and
  `max_tokens` above `--max-context` also gives 400. Without `max_tokens`,
  the server generates up to the end of the context.
- `stop` (a string or a list) is supported. Two fields are extensions:
  `ignore_eos` (as in llama.cpp) and `repeat_penalty` (1.0 turns it off).
- With `"stream": true`, the server sends one SSE chunk for each token in
  the OpenAI shape. The last chunk has the `finish_reason` and the `usage`.
  Then it sends `data: [DONE]`. A chat stream starts with a chunk that has
  the role and no text. If the client closes the stream, the engine stops
  at the next token.
- The chat endpoint uses a fixed Qwen2.5 chat template, not the template
  in `tokenizer_config.json`. A chat with no system message first gets the
  default Qwen system message. Tools are not supported.

## Licence

MIT. See [LICENSE](LICENSE).

More from Inertia Labs: [inertialabs.xyz](https://inertialabs.xyz).
