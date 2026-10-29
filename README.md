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
| [`fornace`](crates/fornace) | The root crate: it exports the engine | root |

## How to run

Download the model to `./models/qwen2.5-0.5b`:

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
FORNACE_MODEL_DIR=./models/qwen2.5-0.5b make test
```

Run the benchmark:

```sh
make bench
```

On macOS, `make run`, `make build`, and `make bench` use the GPU (Metal).

## Licence

MIT. See [LICENSE](LICENSE).

More from Inertia Labs: [inertialabs.xyz](https://inertialabs.xyz).
