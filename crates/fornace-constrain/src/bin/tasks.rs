//! Makes and checks `data/constrained/tasks.jsonl`.
//!
//! - `tasks generate` makes the 300 tasks from a seed. With `--model`, the
//!   model writes the request text of each task.
//! - `tasks check` checks a tasks file: the class counts, a replay of each
//!   shortest sequence, the tasks again from their seeds, and the requests.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use fornace_constrain::chat::chat_prompt;
use fornace_constrain::ledger::{Step, shortest_full};
use fornace_constrain::tasks::{
    Class, OVER_LIMIT, PREFIX, REQUEST_SYSTEM, SIZES, Task, USUAL, WRONG_STATUS, check_request,
    class_of, generate, generate_one, request_prompt,
};
use fornace_engine::{DeviceKind, Dtype, Engine, GenerateOptions, LoadConfig};

/// The tries of the model for one request (the prompt of step 3).
const REQUEST_TRIES: u64 = 5;
/// The temperature of the model that writes the requests (decision 3).
const REQUEST_TEMPERATURE: f64 = 0.7;
/// A request longer than this is not a request.
const REQUEST_MAX_TOKENS: usize = 200;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Make the tasks, and the request texts with `--model`.
    Generate {
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
        /// The model that writes the requests. With no model, the tasks
        /// have no request text.
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, default_value = "q8")]
        dtype: Dtype,
        #[arg(long, default_value = "metal")]
        device: DeviceKind,
    },
    /// Check a tasks file.
    Check { file: PathBuf },
}

fn main() -> Result<()> {
    match Args::parse().command {
        Cmd::Generate {
            seed,
            out,
            model,
            dtype,
            device,
        } => {
            let mut tasks = generate(seed);
            eprintln!("{} tasks from seed {seed}", tasks.len());
            let mut failed = Vec::new();
            if let Some(model) = model {
                failed = write_requests(&mut tasks, model, dtype, device)?;
            }
            let mut file = fs::File::create(&out).with_context(|| format!("create {out:?}"))?;
            for task in &tasks {
                writeln!(file, "{}", serde_json::to_string(task)?)?;
            }
            eprintln!("wrote {out:?}");
            if !failed.is_empty() {
                bail!(
                    "{} tasks have no request after {REQUEST_TRIES} tries: {failed:?}",
                    failed.len()
                );
            }
            Ok(())
        }
        Cmd::Check { file } => check(&file),
    }
}

/// The model writes the request of each task. A text that fails
/// [`check_request`] gets another try with the next seed. The result is
/// the IDs of the tasks with no request after [`REQUEST_TRIES`] tries.
fn write_requests(
    tasks: &mut [Task],
    model_dir: PathBuf,
    dtype: Dtype,
    device: DeviceKind,
) -> Result<Vec<usize>> {
    let config = LoadConfig {
        model_dir,
        dtype,
        device,
        max_new_tokens: REQUEST_MAX_TOKENS,
        batch: 1,
        kv_cache: Default::default(),
        block_size: fornace_engine::DEFAULT_BLOCK_SIZE,
        kv_capacity: None,
        share_prefix: false,
        attention: Default::default(),
        sync_spans: false,
        ignore_eos: false,
        repeat_penalty: 1.0,
        draft: None,
    };
    let mut engine = Engine::load(&config).context("load the model")?;
    let start = Instant::now();
    let mut failed = Vec::new();
    for task in tasks.iter_mut() {
        let prompt = chat_prompt(REQUEST_SYSTEM, &request_prompt(task));
        for attempt in 0..REQUEST_TRIES {
            let seed = task.seed.wrapping_add(attempt);
            let options = GenerateOptions {
                max_new_tokens: REQUEST_MAX_TOKENS,
                ignore_eos: false,
                repeat_penalty: 1.0,
                temperature: Some(REQUEST_TEMPERATURE),
                seed,
            };
            let report =
                engine.generate_stream(&prompt, &options, &mut |_| ControlFlow::Continue(()))?;
            let text = report.texts[0].trim().to_string();
            let result = if report.generated_tokens >= REQUEST_MAX_TOKENS {
                Err(format!("no end in {REQUEST_MAX_TOKENS} tokens"))
            } else {
                check_request(task, &text)
            };
            match &result {
                Ok(()) => eprintln!("task {:3} seed {seed}: {text}", task.id),
                Err(reason) => eprintln!("task {:3} seed {seed}: REJECT {reason}: {text}", task.id),
            }
            match result {
                Ok(()) => {
                    task.request = Some(text);
                    task.request_seed = Some(seed);
                    break;
                }
                Err(reason) => task.request_rejects.push(format!("{reason}: {text}")),
            }
        }
        if task.request.is_none() {
            failed.push(task.id);
        }
        eprintln!(
            "task {:3}: {} tries, {:.0} s in all",
            task.id,
            task.request_rejects.len() + usize::from(task.request.is_some()),
            start.elapsed().as_secs_f64()
        );
    }
    Ok(failed)
}

fn check(file: &PathBuf) -> Result<()> {
    let text = fs::read_to_string(file).with_context(|| format!("read {file:?}"))?;
    let tasks: Vec<Task> = text
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let mut errors = Vec::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut lengths: BTreeMap<usize, usize> = BTreeMap::new();
    let mut exact = 0;
    let mut rejects = 0;
    for (i, task) in tasks.iter().enumerate() {
        let id = task.id;
        if id != i {
            errors.push(format!("task {i} has the ID {id}"));
        }
        if class_of(id) != (task.class, task.prefix_id) {
            errors.push(format!("task {id}: the class does not agree with the ID"));
        }
        let key = match (task.class, task.prefix_id) {
            (Class::WrongStatus, true) => "wrong_status, prefix ID".to_string(),
            (class, _) => format!("{class:?}"),
        };
        *counts.entry(key).or_default() += 1;
        // The task again from its seed: the same state, goal, and sequence.
        let mut again = generate_one(id, task.seed, SIZES);
        if let (Some(again), Some(_)) = (&mut again, &task.request) {
            again.request = task.request.clone();
            again.request_seed = task.request_seed;
            again.request_rejects = task.request_rejects.clone();
        }
        if again.as_ref() != Some(task) {
            errors.push(format!("task {id}: the seed does not give the same task"));
        }
        if let Some(sequence) = &task.shortest {
            let mut state = task.start.clone();
            for command in sequence {
                if state.execute(command) != Ok(Step::Continue) {
                    errors.push(format!("task {id}: the ledger refuses {command:?}"));
                }
            }
            if !task.goal.is_met(&task.start, &state) {
                errors.push(format!("task {id}: the replay does not get to the goal"));
            }
            *lengths.entry(sequence.len()).or_default() += 1;
            // The full search checks each shorter length up to 2.
            if sequence.len() <= 3 {
                if shortest_full(&task.start, &task.goal, sequence.len() - 1).is_some() {
                    errors.push(format!("task {id}: the full search is shorter"));
                }
                exact += 1;
            }
        } else if task.class == Class::Usual {
            errors.push(format!("task {id}: a usual task with no sequence"));
        }
        match &task.request {
            Some(request) => {
                if let Err(reason) = check_request(task, request) {
                    errors.push(format!("task {id}: request: {reason}"));
                }
            }
            None => errors.push(format!("task {id}: no request")),
        }
        rejects += task.request_rejects.len();
    }
    let expected = [
        ("Usual".to_string(), USUAL),
        ("OverLimit".to_string(), OVER_LIMIT),
        ("WrongStatus".to_string(), WRONG_STATUS - PREFIX),
        ("wrong_status, prefix ID".to_string(), PREFIX),
    ];
    for (key, count) in expected {
        let found = counts.get(&key).copied().unwrap_or(0);
        if found != count {
            errors.push(format!("{key}: {found} tasks, expected {count}"));
        }
    }
    println!("tasks: {}", tasks.len());
    println!("classes: {counts:?}");
    println!("shortest lengths of the usual tasks: {lengths:?}");
    println!("lengths that the full search proves (up to 3): {exact}");
    println!("rejected request texts: {rejects}");
    if errors.is_empty() {
        println!("check: pass");
        Ok(())
    } else {
        for error in &errors {
            println!("{error}");
        }
        bail!("check: {} errors", errors.len())
    }
}
