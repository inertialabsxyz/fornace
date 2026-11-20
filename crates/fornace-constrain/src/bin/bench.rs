//! Runs one configuration of the constrained sampling bench on a list of
//! tasks and seeds, and writes two files to the output directory:
//! `<date>-<commit>-<config>-<model>[-<tag>]-outcomes.jsonl` and
//! `...-timings.jsonl`.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::Command as Process;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::Parser;
use fornace_constrain::bench::{Config, Generation, MAX_COMMAND_TOKENS, Model, json_end, run_task};
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

struct EngineModel(Engine);

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
        let report = self.0.generate_stream(prompt, &options, &mut |event| {
            text.push_str(&event.text);
            if json_end(&text).is_some() {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })?;
        let decode_steps = report.generated_tokens.saturating_sub(1) as f64;
        Ok(Generation {
            text: report.texts[0].clone(),
            prompt_tokens: report.prompt_tokens,
            generated_tokens: report.generated_tokens,
            prefill_ms: report.time_to_first_token_ms,
            decode_ms: report.decode_ms_per_step * decode_steps,
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
    let mut model = EngineModel(Engine::load(&config).context("load the model")?);
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

fn writer(dir: &Path, name: &str) -> Result<BufWriter<File>> {
    let path = dir.join(name);
    if path.exists() {
        bail!("{path:?} exists. Give another --tag");
    }
    eprintln!("writing {path:?}");
    Ok(BufWriter::new(File::create(&path)?))
}

/// `0-29,180,185-189` as a list of IDs.
fn parse_ids(text: &str) -> Result<Vec<usize>> {
    let mut ids = Vec::new();
    for part in text.split(',') {
        match part.split_once('-') {
            Some((a, b)) => ids.extend(a.trim().parse::<usize>()?..=b.trim().parse::<usize>()?),
            None => ids.push(part.trim().parse()?),
        }
    }
    Ok(ids)
}

/// The short commit hash, with `-dirty` if the tree has changes.
fn commit() -> String {
    let run = |args: &[&str]| {
        Process::new("git")
            .args(args)
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let hash = run(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "nogit".to_string());
    let dirty =
        run(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    if dirty { format!("{hash}-dirty") } else { hash }
}

/// The UTC date as `YYYY-MM-DD`.
fn today() -> String {
    let days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() / 86_400)
        .unwrap_or(0) as i64;
    // The civil date from the day count (H. Hinnant, "chrono-compatible
    // low-level date algorithms").
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}
