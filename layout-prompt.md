Create the initial layout of a public Rust workspace named `fornace`, at
github.com/inertialabsxyz/fornace. Write all text in ASD-STE100 Simplified
Technical English. Create the files, run `cargo build` and `cargo test` to
check that everything compiles, then make one git commit with the message
`chore: workspace layout`.

Do not implement any inference code. This commit is the layout only.

## What Fornace is

Fornace is a series of findings from Inertia Labs on AI inference that you
run yourself: Rust inference engines down to the kernels, and how to prove
where a model ran. Each piece of work produces one or more crates in this
workspace and one or more findings in `findings/`. A finding is a short
note named by its result with a number, for example "A KV cache makes
decode 6x faster on an M4 and costs 1.2 GB at 8k context". The first piece
of work builds a small inference engine on Candle, one commit per idea,
each commit adding one row to one benchmark table; each row that says
something becomes a finding.

Never use the word "lab" in any file.

## Layout
fornace/
├── Cargo.toml [workspace], members = [“crates/*”], resolver = “2”,
│ [workspace.package] with version 0.0.1, edition 2024,
│ license “MIT OR Apache-2.0”,
│ repository “https://github.com/inertialabsxyz/fornace”
├── rust-toolchain.toml channel = “stable”
├── LICENSE-MIT
├── LICENSE-APACHE
├── README.md
├── CONTRIBUTING.md
├── .gitignore target/, models/, *.safetensors, *.gguf, .DS_Store
├── .github/workflows/ci.yml
├── scripts/
│ └── get-model.sh downloads Qwen/Qwen2.5-0.5B (safetensors) with
│ huggingface-cli into ./models/qwen2.5-0.5b
├── findings/
│ ├── README.md the list of findings, newest first: date, title
│ │ (the result with its number), commit, link
│ └── TEMPLATE.md the fixed order of a finding (see below)
├── bench/
│ ├── README.md how to run the harness and what each column means
│ └── results/
│ └── .gitkeep one CSV per run is committed here; findings cite
│ the file and the commit
└── crates/
├── fornace/ root crate
├── fornace-engine/ the engine, library plus CLI
└── fornace-bench/ the benchmark harness


## findings/TEMPLATE.md

A markdown template with these sections, in this order, each with one
line that says what goes there:

1. Title: the result in one sentence with a number.
2. Why it matters: two or three sentences. Who hits this limit and what
   it costs them.
3. What was measured and how: the table (or a link to the CSV in
   bench/results), the hardware, the model, the dtype, the commit hash.
4. Cause and workaround: the cause if found; the workaround if there is
   one; otherwise say that there is none.
5. What it means if you run models on your own hardware: one paragraph.
6. Reproduce: the commit, the command, and the expected output.

## Crates

### crates/fornace
A library crate. `lib.rs` has the crate doc that explains the series and
re-exports `fornace_engine` under `pub use fornace_engine as engine;`.
Description: "Fornace: findings on small models that run where your data lives."

### crates/fornace-engine
A library crate with one binary.

`src/lib.rs` defines the public API as stubs that compile but return
`Err(Error::NotImplemented)`:

- `pub struct Config { pub model_dir: PathBuf, pub dtype: Dtype,
  pub device: DeviceKind, pub max_new_tokens: usize, pub batch: usize }`
- `pub enum Dtype { F16, Q8, Q4 }` and `pub enum DeviceKind { Cpu, Metal }`,
  both with `FromStr` and `Display`
- `pub struct Engine` with `pub fn load(config: &Config) -> Result<Engine>`
- `pub struct GenerationReport { pub prompt_tokens: usize,
  pub generated_tokens: usize, pub prefill_tokens_per_s: f64,
  pub decode_tokens_per_s: f64, pub time_to_first_token_ms: f64,
  pub peak_rss_bytes: u64, pub text: String }`, deriving
  `serde::Serialize` and `serde::Deserialize`
- `impl Engine { pub fn generate(&mut self, prompt: &str)
  -> Result<GenerationReport> }`
- `pub enum Error` with thiserror: `NotImplemented`, `Io`, `Model(String)`
- `pub type Result<T> = std::result::Result<T, Error>;`

Modules as empty files with a one-line doc comment each: `model.rs`,
`cache.rs`, `sampler.rs`, `metrics.rs`. Add `mod` lines in `lib.rs`.

`src/bin/fornace.rs` is the CLI, with clap derive:
`--model <DIR>` (required), `--prompt <TEXT>` (required),
`--max-new-tokens <N>` default 128, `--dtype f16|q8|q4` default f16,
`--device cpu|metal` default metal on macOS and cpu elsewhere,
`--batch <N>` default 1, `--json` to print the report as JSON instead of
text. It builds a `Config`, calls `Engine::load` and `generate`, and
prints the result or the error. No logic beyond that.

`tests/generate.rs` has one integration test that reads the model path
from the environment variable `FORNACE_MODEL_DIR`, skips with a message
when it is not set, and otherwise loads the engine and asserts that
greedy decoding of the prompt "The capital of Spain is" starts with
"Madrid". Mark the body so that it compiles now and fails with
`NotImplemented` when the variable is set.

The library has no `println!`, no argument parsing, and no `std::process::exit`.

Dependencies: candle-core, candle-nn, candle-transformers (latest
versions, with a `metal` feature on the crate that enables Candle's
metal feature), tokenizers, safetensors, serde, serde_json, thiserror,
anyhow (binary only), clap (binary only, derive feature).

### crates/fornace-bench
A binary crate. `src/main.rs` with clap: `--model <DIR>`,
`--engines fornace,mistralrs,llamacpp` default `fornace`,
`--out <CSV>` default `bench/results/latest.csv`. For now it prints a
one-line message that the harness lands in the next commit, and exits 0.
Depends on fornace-engine, clap, serde, csv.

## README.md

Sections, in this order: the one-paragraph description of the series;
the two questions it asks; a "Findings" section that says the list is in
findings/README.md and shows the three newest (empty for now); a "Crates"
table with columns Crate, What it is, Status (fornace-engine "in
progress", fornace-bench "in progress", fornace "root"); how to run
(`scripts/get-model.sh`, the three cargo commands for run, test, and
bench); the publication rule (one commit per idea, one row per commit,
each row that says something becomes a finding, at least one finding
every two weeks); licence; and one line that points to inertialabs.xyz.

## bench/README.md

How to run `fornace-bench`, and one line per CSV column: commit, date,
engine, model, dtype, batch, prompt_tokens, generated_tokens,
prefill_tokens_per_s, decode_tokens_per_s, time_to_first_token_ms,
peak_rss_mb, quality (empty until quantization lands).

## CI

`.github/workflows/ci.yml`: on push and pull request; one job on
`ubuntu-latest` with `cargo fmt --check`, `cargo clippy --workspace
--all-targets -- -D warnings`, and `cargo test --workspace`; one job on
`macos-latest` with `cargo build --workspace --features metal` only. No
model download in CI.

## Checks before the commit

- `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and `cargo test --workspace` all pass.
- `cargo run -p fornace-engine --bin fornace -- --model ./models/x
  --prompt hi` prints the NotImplemented error and exits non-zero.
- `grep -ri "kiln\|\blab\b" .` outside `target/` finds nothing.
