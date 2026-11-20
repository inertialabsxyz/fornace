//! The test of the static mask (brief, step 1): the model gives the name of
//! the first command for each request, with the mask off, with a scan of
//! the vocabulary, or with a bitmask for each trie node. Writes two files:
//! `<date>-<commit>-names-<mask>-<model>[-<tag>]-outcomes.jsonl` and
//! `...-timings.jsonl`.

use std::fs;
use std::io::Write;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use fornace_constrain::bench::user_message;
use fornace_constrain::chat::chat_prompt;
use fornace_constrain::runs::{commit, parse_ids, today, writer};
use fornace_constrain::tasks::Task;
use fornace_constrain::trie::{COMMAND_NAMES, Mode, TrieMask};
use fornace_engine::{DeviceKind, Dtype, Engine, GenerateOptions, LoadConfig};
use serde::Serialize;

/// The temperature of the bench (brief, "Fixed conditions").
const TEMPERATURE: f64 = 0.7;
/// The repeat penalty of the bench (brief, "Fixed conditions").
const REPEAT_PENALTY: f32 = 1.0;
/// The longest name, `ASK_OPERATOR`, is 12 bytes. With the mask off, the
/// model can write more.
const MAX_NAME_TOKENS: usize = 16;

/// The system message: the names only, not the JSON form.
const SYSTEM: &str = "You are an operations assistant for a payments ledger. \
An operator sends a request. Reply with the name of the first command to send to the ledger, \
and nothing else. The names: AUTHORIZE, CAPTURE, VOID, REFUND, TRANSFER, ASK_OPERATOR, DONE.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum MaskFlag {
    /// No hook: `generate_stream`, the path of `main`.
    Off,
    Scan,
    Bitmask,
}

#[derive(Parser)]
struct Args {
    #[arg(long, value_enum)]
    mask: MaskFlag,
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "f16")]
    dtype: Dtype,
    #[arg(long, default_value = "metal")]
    device: DeviceKind,
    #[arg(long, default_value = "data/constrained/tasks.jsonl")]
    tasks: PathBuf,
    /// The task IDs. The brief asks for 200 requests.
    #[arg(long, default_value = "0-199")]
    ids: String,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long, default_value = "bench/runs/constrained")]
    out_dir: PathBuf,
    #[arg(long)]
    tag: Option<String>,
}

/// One line of the outcomes file.
#[derive(Serialize)]
struct Outcome {
    task: usize,
    seed: u64,
    mask: MaskFlag,
    model: String,
    tokens: Vec<u32>,
    text: String,
    /// The text is one of the 7 names, byte for byte.
    permitted: bool,
}

/// One line of the timings file.
#[derive(Serialize)]
struct Timing {
    task: usize,
    seed: u64,
    prompt_tokens: usize,
    generated_tokens: usize,
    prefill_ms: f64,
    /// The decode time of all steps after the first. With a mask, it
    /// includes the mask time of those steps.
    decode_ms: f64,
    /// The mean time of the mask for each token. `None` with the mask off.
    mask_ms_per_token: Option<f64>,
    /// The time to build the mask, one time for the run. `None` with the
    /// mask off.
    build_ms: Option<f64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let text = fs::read_to_string(&args.tasks).with_context(|| format!("read {:?}", args.tasks))?;
    let all: Vec<Task> = text
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let ids = parse_ids(&args.ids)?;
    let model_name = args
        .model
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model")
        .to_string();
    let mask_name = format!("{:?}", args.mask).to_lowercase();
    let mut stem = format!("{}-{}-names-{mask_name}-{model_name}", today(), commit());
    if let Some(tag) = &args.tag {
        stem.push('-');
        stem.push_str(tag);
    }
    fs::create_dir_all(&args.out_dir)?;
    let mut outcomes = writer(&args.out_dir, &format!("{stem}-outcomes.jsonl"))?;
    let mut timings = writer(&args.out_dir, &format!("{stem}-timings.jsonl"))?;

    let config = LoadConfig {
        model_dir: args.model.clone(),
        dtype: args.dtype,
        device: args.device,
        max_new_tokens: MAX_NAME_TOKENS,
        batch: 1,
        kv_cache: Default::default(),
        block_size: fornace_engine::DEFAULT_BLOCK_SIZE,
        kv_capacity: None,
        share_prefix: false,
        attention: Default::default(),
        sync_spans: false,
        ignore_eos: false,
        repeat_penalty: REPEAT_PENALTY,
        draft: None,
    };
    let mut engine = Engine::load(&config).context("load the model")?;
    let mut mask = match args.mask {
        MaskFlag::Off => None,
        MaskFlag::Scan | MaskFlag::Bitmask => {
            let mode = if args.mask == MaskFlag::Scan {
                Mode::Scan
            } else {
                Mode::Bitmask
            };
            let vocab = engine.token_bytes();
            // The engine stops on these two tokens (`generate_rows`).
            let eos = ["<|im_end|>", "<|endoftext|>"]
                .iter()
                .map(|t| {
                    vocab
                        .iter()
                        .position(|b| b == t.as_bytes())
                        .map(|i| i as u32)
                })
                .collect::<Option<Vec<u32>>>()
                .context("the vocabulary has no ChatML end tokens")?;
            let len = logits_len(&args.model)?;
            let start = Instant::now();
            let mask = TrieMask::command_names(vocab, eos, len, mode);
            let build_ms = start.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "mask: {} trie nodes, built in {build_ms:.1} ms",
                mask.trie().len()
            );
            Some((mask, build_ms))
        }
    };

    let options = GenerateOptions {
        max_new_tokens: MAX_NAME_TOKENS,
        ignore_eos: false,
        repeat_penalty: REPEAT_PENALTY,
        temperature: Some(TEMPERATURE),
        seed: args.seed,
    };
    let (mut runs, mut permitted) = (0, 0);
    for &id in &ids {
        let task = all.get(id).with_context(|| format!("no task {id}"))?;
        let request = task.request.as_deref().context("a task with no request")?;
        let prompt = chat_prompt(SYSTEM, &user_message(request, &task.start, &[]));
        let mut tokens = Vec::new();
        let mut on_token = |event: fornace_engine::TokenEvent| {
            tokens.push(event.id);
            ControlFlow::Continue(())
        };
        let (report, mask_ms, build_ms) = match &mut mask {
            None => (
                engine.generate_stream(&prompt, &options, &mut on_token)?,
                None,
                None,
            ),
            Some((mask, build_ms)) => {
                mask.reset();
                mask.clear_times();
                let report =
                    engine.generate_stream_with_hook(&prompt, &options, mask, &mut on_token)?;
                (report, Some(mask.mask_ms_per_token().0), Some(*build_ms))
            }
        };
        let text = report.texts[0].clone();
        let ok = COMMAND_NAMES.contains(&text.as_str());
        let outcome = Outcome {
            task: id,
            seed: args.seed,
            mask: args.mask,
            model: model_name.clone(),
            tokens,
            text,
            permitted: ok,
        };
        writeln!(outcomes, "{}", serde_json::to_string(&outcome)?)?;
        let timing = Timing {
            task: id,
            seed: args.seed,
            prompt_tokens: report.prompt_tokens,
            generated_tokens: report.generated_tokens,
            prefill_ms: report.time_to_first_token_ms,
            decode_ms: report.decode_ms_per_step * report.generated_tokens.saturating_sub(1) as f64,
            mask_ms_per_token: mask_ms,
            build_ms,
        };
        writeln!(timings, "{}", serde_json::to_string(&timing)?)?;
        runs += 1;
        permitted += usize::from(ok);
        eprintln!("task {id:3}: {:?} ({permitted}/{runs})", outcome.text);
    }
    outcomes.flush()?;
    timings.flush()?;
    println!("{stem}: permitted names {permitted} of {runs}");
    Ok(())
}

/// The length of the logits: `vocab_size` in `config.json`. Qwen2.5 pads
/// the vocabulary, so this is longer than the tokenizer.
fn logits_len(model: &std::path::Path) -> Result<usize> {
    let path = model.join("config.json");
    let config: serde_json::Value = serde_json::from_str(&fs::read_to_string(&path)?)?;
    match config["vocab_size"].as_u64() {
        Some(len) => Ok(len as usize),
        None => bail!("{path:?} has no vocab_size"),
    }
}
