//! Runs one configuration of the constrained sampling bench on a list of
//! tasks and seeds, and writes two files to the output directory:
//! `<date>-<commit>-<config>-<model>[-<tag>]-outcomes.jsonl` and
//! `...-timings.jsonl`.

use std::fs;
use std::io::Write;
use std::ops::ControlFlow;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use fornace_constrain::bench::{Config, Generation, MAX_COMMAND_TOKENS, Model, json_end, run_task};
use fornace_constrain::grammar::{GRAMMAR_B, Grammars};
use fornace_constrain::runs::{commit, parse_ids, today, writer};
use fornace_constrain::tasks::Task;
use fornace_engine::{DeviceKind, Dtype, Engine, GenerateOptions, LoadConfig};

/// The temperature of the bench (brief, "Fixed conditions").
const TEMPERATURE: f64 = 0.7;
/// The repeat penalty of the bench (brief, "Fixed conditions").
const REPEAT_PENALTY: f32 = 1.0;

#[derive(Parser)]
struct Args {
    #[arg(long, value_enum)]
    config: Config,
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "f16")]
    dtype: Dtype,
    #[arg(long, default_value = "metal")]
    device: DeviceKind,
    #[arg(long, default_value = "data/constrained/tasks.jsonl")]
    tasks: PathBuf,
    /// The task IDs, for example `0-29,180-189`. All tasks if not given.
    #[arg(long)]
    ids: Option<String>,
    /// The run seeds, for example `1,2,3`.
    #[arg(long, default_value = "1,2,3", value_delimiter = ',')]
    seeds: Vec<u64>,
    #[arg(long, default_value = "bench/runs/constrained")]
    out_dir: PathBuf,
    /// A suffix for the file names, so that a second run does not write
    /// over the first.
    #[arg(long)]
    tag: Option<String>,
}

struct EngineModel {
    engine: Engine,
    /// The grammar of the configuration. `None` for A.
    grammar: Option<(Grammars, &'static str)>,
}

impl Model for EngineModel {
    fn command(&mut self, prompt: &str, seed: u64) -> Result<Generation> {
        let options = GenerateOptions {
            max_new_tokens: MAX_COMMAND_TOKENS,
            ignore_eos: false,
            repeat_penalty: REPEAT_PENALTY,
            temperature: Some(TEMPERATURE),
            seed,
        };
        let mut text = String::new();
        let mut on_token = |event: fornace_engine::TokenEvent| {
            text.push_str(&event.text);
            if json_end(&text).is_some() {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        let (report, mask_ms_per_token, compile_ms) = match &self.grammar {
            None => (
                self.engine
                    .generate_stream(prompt, &options, &mut on_token)?,
                None,
                None,
            ),
            Some((grammars, lark)) => {
                let mut mask = grammars.mask(lark)?;
                let report = self.engine.generate_stream_with_hook(
                    prompt,
                    &options,
                    &mut mask,
                    &mut on_token,
                )?;
                if let Some(error) = mask.error() {
                    bail!("the grammar engine failed: {error}");
                }
                (
                    report,
                    Some(mask.mask_ms_per_token()),
                    Some(mask.compile_ms()),
                )
            }
        };
        let decode_steps = report.generated_tokens.saturating_sub(1) as f64;
        Ok(Generation {
            text: report.texts[0].clone(),
            prompt_tokens: report.prompt_tokens,
            generated_tokens: report.generated_tokens,
            prefill_ms: report.time_to_first_token_ms,
            decode_ms: report.decode_ms_per_step * decode_steps,
            mask_ms_per_token,
            compile_ms,
        })
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let text = fs::read_to_string(&args.tasks).with_context(|| format!("read {:?}", args.tasks))?;
    let all: Vec<Task> = text
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let ids = match &args.ids {
        Some(ids) => parse_ids(ids)?,
        None => (0..all.len()).collect(),
    };
    let model_name = args
        .model
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model")
        .to_string();
    let config_name = format!("{:?}", args.config).to_lowercase();
    let mut stem = format!("{}-{}-{config_name}-{model_name}", today(), commit());
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
        max_new_tokens: MAX_COMMAND_TOKENS,
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
    let engine = Engine::load(&config).context("load the model")?;
    let grammar = match args.config {
        Config::A => None,
        Config::B => Some((grammars(&args.model)?, GRAMMAR_B)),
    };
    let mut model = EngineModel { engine, grammar };
    let (mut runs, mut successes) = (0, 0);
    for &id in &ids {
        let task = all.get(id).with_context(|| format!("no task {id}"))?;
        for &seed in &args.seeds {
            let (outcome, task_timings) =
                run_task(&mut model, &model_name, args.config, task, seed)?;
            writeln!(outcomes, "{}", serde_json::to_string(&outcome)?)?;
            for timing in &task_timings {
                writeln!(timings, "{}", serde_json::to_string(timing)?)?;
            }
            runs += 1;
            successes += usize::from(outcome.score.success);
            eprintln!(
                "task {id:3} seed {seed}: {} in {} outputs, success {} ({successes}/{runs})",
                outcome.ended_by,
                outcome.outputs.len(),
                outcome.score.success
            );
        }
    }
    outcomes.flush()?;
    timings.flush()?;
    println!("{stem}: task success {successes} of {runs}");
    Ok(())
}

/// The parser factory for the model. The logits length is `vocab_size` in
/// `config.json`, because Qwen2.5 pads the vocabulary.
fn grammars(model: &std::path::Path) -> Result<Grammars> {
    let path = model.join("config.json");
    let config: serde_json::Value = serde_json::from_str(&fs::read_to_string(&path)?)?;
    let len = config["vocab_size"]
        .as_u64()
        .with_context(|| format!("{path:?} has no vocab_size"))? as usize;
    let start = std::time::Instant::now();
    // The two tokens that stop the engine (`generate_rows`).
    let (grammars, eos) = Grammars::from_model_dir(model, &["<|im_end|>", "<|endoftext|>"], len)?;
    eprintln!(
        "grammar factory: EOS {eos:?}, built in {:.0} ms",
        start.elapsed().as_secs_f64() * 1e3
    );
    Ok(grammars)
}
