//! The scheduler brief, step 5: on Metal, each request of the agent load
//! gives the same tokens in one batch as alone, or the check gives the
//! first token that differs and the logit margin there.
//!
//! This needs the GPU and the model, so it is ignored by default. Run it
//! from the repository root with:
//!
//! ```sh
//! cargo test --release --features metal -p fornace-bench \
//!     --test batch_tokens -- --ignored --nocapture
//! ```
//!
//! For each format (`bf16`, `q8`, `q4`): requests 0 to 7 of the agent load
//! (the list of `--load agent`), EOS ignored, no repeat penalty. Alone: one
//! request in a batch of one. Together: the requests join one before each
//! step, as in the server, so the batch grows to 8 rows and then empties.
//! The margin is the difference between the two largest logits of the
//! request alone at that token, from one prefill of the prompt and the
//! tokens before it (*approximate*: a prefill runs other kernels than a
//! decode step).

#![cfg(feature = "metal")]

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use fornace_bench::agent;
use fornace_engine::batch::{Batch, Done};
use fornace_engine::{
    AttentionKind, DeviceKind, Dtype, Engine, GenerateOptions, KvCacheKind, LoadConfig,
};
use tokenizers::Tokenizer;

const REQUESTS: usize = 8;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn ids(tokenizer: &Tokenizer, file: &str) -> Vec<u32> {
    let text = std::fs::read_to_string(repo().join("bench/prompts").join(file)).unwrap();
    tokenizer
        .encode(text.as_str(), true)
        .unwrap()
        .get_ids()
        .to_vec()
}

fn engine(dir: &Path, dtype: Dtype) -> Engine {
    Engine::load(&LoadConfig {
        model_dir: dir.to_path_buf(),
        dtype,
        device: DeviceKind::Metal,
        max_new_tokens: 1,
        batch: 1,
        kv_cache: KvCacheKind::Prealloc,
        block_size: fornace_engine::DEFAULT_BLOCK_SIZE,
        kv_capacity: None,
        share_prefix: false,
        attention: AttentionKind::Fused,
        sync_spans: false,
        ignore_eos: true,
        repeat_penalty: 1.0,
        draft: None,
    })
    .expect("load the engine")
}

fn options(max_tokens: usize) -> GenerateOptions {
    GenerateOptions {
        max_new_tokens: max_tokens,
        ignore_eos: true,
        repeat_penalty: 1.0,
    }
}

fn quiet() -> fornace_engine::batch::OnToken {
    Box::new(|_| ControlFlow::Continue(()))
}

fn drain(batch: &mut Batch, engine: &mut Engine, done: &mut Vec<Done>) {
    while !batch.is_empty() {
        done.extend(batch.step(engine).unwrap());
    }
}

/// The two largest logits of the next token after `prompt` and `tokens`,
/// alone: their difference.
fn margin(engine: &mut Engine, prompt: &str, tokenizer: &Tokenizer, tokens: &[u32]) -> f32 {
    // The prompt and the tokens before the difference, as one text, gives
    // the same ids only if the decode round-trips. Use the ids instead.
    let mut ids = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();
    ids.extend_from_slice(tokens);
    engine.next_token_margin(&ids).unwrap()
}

#[test]
#[ignore = "a check on Metal with the model"]
fn agent_requests_give_the_same_tokens_in_a_batch() {
    let dir = repo().join("models/qwen2.5-0.5b-instruct");
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipped: {} is missing", dir.display());
        return;
    }
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let system = ids(&tokenizer, "alice-2048.txt")[..agent::SYSTEM_TOKENS].to_vec();
    let sources = [
        ids(&tokenizer, "code-512.txt"),
        ids(&tokenizer, "alice-de-512.txt"),
    ];
    let lens: Vec<usize> = sources.iter().map(Vec::len).collect();
    let list = agent::build(&agent::plan(REQUESTS, &lens), &system, &sources, &tokenizer).unwrap();

    println!("| Format | Request | Prompt tokens | max_tokens | First difference | Margin there |");
    println!("|---|---:|---:|---:|---|---:|");
    for dtype in [Dtype::F16, Dtype::Q8, Dtype::Q4] {
        let mut engine = engine(&dir, dtype);
        let name = engine.effective_dtype().to_string();
        let alone: Vec<Vec<u32>> = list
            .iter()
            .map(|r| {
                let mut batch = Batch::new();
                let mut done = Vec::new();
                let (_, left) = batch
                    .join(&mut engine, &r.prompt, &options(r.plan.max_tokens), quiet())
                    .unwrap();
                done.extend(left);
                drain(&mut batch, &mut engine, &mut done);
                done.remove(0).tokens
            })
            .collect();

        let mut batch = Batch::new();
        let mut done = Vec::new();
        for r in &list {
            let (_, left) = batch
                .join(&mut engine, &r.prompt, &options(r.plan.max_tokens), quiet())
                .unwrap();
            done.extend(left);
            done.extend(batch.step(&mut engine).unwrap());
        }
        drain(&mut batch, &mut engine, &mut done);
        done.sort_by_key(|d| d.id);

        for (i, (row, want)) in done.iter().zip(&alone).enumerate() {
            assert_eq!(row.tokens.len(), want.len(), "{name} request {i}");
            let first = row.tokens.iter().zip(want).position(|(a, b)| a != b);
            let (at, m) = match first {
                None => (format!("none in {}", want.len()), String::from("—")),
                Some(k) => {
                    let m = margin(&mut engine, &list[i].prompt, &tokenizer, &want[..k]);
                    (k.to_string(), format!("{m:.3}"))
                }
            };
            println!(
                "| {name} | {i} | {} | {} | {at} | {m} |",
                list[i].prompt_tokens, list[i].plan.max_tokens
            );
        }
    }
}
