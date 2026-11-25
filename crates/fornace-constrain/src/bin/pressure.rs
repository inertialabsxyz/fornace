//! E1 of the changes brief: the mask pressure of a record run of C
//! (`bench --config c --mask-pressure`). Writes markdown to standard
//! output:
//!
//! - the distribution of the signal for each task group;
//! - for each threshold of the grid, the result of the stop, from a replay
//!   of the record run ([`fornace_constrain::pressure::replay`]);
//! - the selected, low, and high thresholds of the rule.
//!
//! The rule (written before the data, 25 November 2026): the selected
//! threshold has the largest value of (trap successes − false alarms) on
//! seed 1. A tie selects the higher threshold. The low and high thresholds
//! are two grid positions below and above it.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use fornace_constrain::bench::{Outcome, TrapClass, score};
use fornace_constrain::pressure::{PressureLine, first_stop, replay};
use fornace_constrain::tasks::{Class, Task};

/// A task group of the distribution table: a label and a test.
type GroupTest = (&'static str, fn(&Outcome) -> bool);

/// The thresholds of the rule.
const GRID: [f32; 11] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];

#[derive(Parser)]
struct Args {
    /// The outcomes file of the record run: `...-outcomes.jsonl`. The mass
    /// file `...-mass.jsonl` is next to it.
    outcomes: PathBuf,
    #[arg(long, default_value = "data/constrained/tasks.jsonl")]
    tasks: PathBuf,
}

/// The counts of one threshold, for a set of runs.
#[derive(Default, Clone, Copy)]
struct Counts {
    runs: usize,
    usual: usize,
    usual_success: usize,
    false_alarms: usize,
    /// False alarms that the bench sent.
    bench_alarms: usize,
    traps: usize,
    trap_success: usize,
    /// Trap successes from an `ASK_OPERATOR` of the bench.
    bench_trap_success: usize,
    shortened: usize,
    wrong_ids: usize,
    unit_errors: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tasks: Vec<Task> = fs::read_to_string(&args.tasks)
        .with_context(|| format!("read {:?}", args.tasks))?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let outcomes: Vec<Outcome> = fs::read_to_string(&args.outcomes)
        .with_context(|| format!("read {:?}", args.outcomes))?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let name = args
        .outcomes
        .to_str()
        .context("a UTF-8 path")?
        .strip_suffix("-outcomes.jsonl")
        .context("an outcomes file")?;
    let mass_path = format!("{name}-mass.jsonl");
    let mut signals: HashMap<(usize, u64), Vec<f32>> = HashMap::new();
    for line in fs::read_to_string(&mass_path)
        .with_context(|| format!("read {mass_path}"))?
        .lines()
    {
        let line: PressureLine = serde_json::from_str(line)?;
        let list = signals.entry((line.task, line.seed)).or_default();
        anyhow::ensure!(
            list.len() == line.generation,
            "the mass file is not in order"
        );
        list.push(line.max_value_mass());
    }
    let file = |p: &str| {
        std::path::Path::new(p)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(p)
            .to_string()
    };
    println!("# E1: mask pressure of C\n");
    println!(
        "Made by `pressure` from `{}` and `{}`.\n",
        file(args.outcomes.to_str().unwrap_or_default()),
        file(&mass_path)
    );
    println!(
        "The signal of a generation is its largest blocked mass in an ID or an amount field. \
         The stop results come from a replay of the record run.\n"
    );

    distribution(&outcomes, &signals);

    let count = |seeds: &[u64], threshold: Option<f32>| -> Result<Counts> {
        let mut c = Counts::default();
        for o in outcomes.iter().filter(|o| seeds.contains(&o.seed)) {
            let task = tasks.get(o.task).context("a task of the outcomes")?;
            let s = &signals[&(o.task, o.seed)];
            anyhow::ensure!(
                s.len() == o.outputs.len(),
                "task {} seed {}",
                o.task,
                o.seed
            );
            let (outputs, end) = match threshold {
                Some(t) => replay(&task.start, &o.outputs, s, t),
                None => (o.outputs.clone(), end_state(task, &o.outputs)),
            };
            let sc = score(task, &end, &outputs);
            anyhow::ensure!(
                threshold.is_some() || sc == o.score,
                "the replay with no stop does not give the score of task {} seed {}",
                o.task,
                o.seed
            );
            let bench = threshold.is_some_and(|t| first_stop(s, t).is_some());
            c.runs += 1;
            if task.class == Class::Usual {
                c.usual += 1;
                c.usual_success += usize::from(sc.success);
                c.false_alarms += usize::from(sc.false_alarm);
                c.bench_alarms += usize::from(sc.false_alarm && bench);
            } else {
                c.traps += 1;
                c.trap_success += usize::from(sc.success);
                c.bench_trap_success += usize::from(sc.success && bench);
                c.shortened += usize::from(sc.trap_class == Some(TrapClass::ShortenedValue));
                c.wrong_ids += usize::from(sc.trap_class == Some(TrapClass::WrongId));
            }
            c.unit_errors += sc.unit_errors;
        }
        Ok(c)
    };

    // The rule, on seed 1 only.
    let mut best: Option<(i64, usize)> = None;
    for (i, &t) in GRID.iter().enumerate() {
        let c = count(&[1], Some(t))?;
        let value = c.trap_success as i64 - c.false_alarms as i64;
        if best.is_none_or(|(v, _)| value >= v) {
            best = Some((value, i));
        }
    }
    let (_, selected) = best.context("an empty grid")?;
    let low = selected.saturating_sub(2);
    let high = (selected + 2).min(GRID.len() - 1);
    println!("## The thresholds of the rule (seed 1)\n");
    println!(
        "Selected {}, low {}, high {}.\n",
        GRID[selected], GRID[low], GRID[high]
    );

    for (title, seeds) in [
        ("Seed 1 (the threshold comes from these runs)", &[1u64][..]),
        ("Seeds 2 and 3 (not used for the threshold)", &[2, 3][..]),
        ("All seeds", &[1, 2, 3][..]),
    ] {
        println!("## {title}\n");
        println!(
            "| Threshold | Runs | Usual success | False alarms | From the bench | Trap success | From the bench | Shortened values | Wrong IDs | Unit errors |"
        );
        println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
        let mut rows = vec![("none (C)".to_string(), None)];
        for (i, &t) in GRID.iter().enumerate() {
            let mark = if i == selected {
                " (selected)"
            } else if i == low && low != selected {
                " (low)"
            } else if i == high && high != selected {
                " (high)"
            } else {
                ""
            };
            rows.push((format!("{t}{mark}"), Some(t)));
        }
        for (label, t) in rows {
            let c = count(seeds, t)?;
            println!(
                "| {label} | {} | {}/{} | {} | {} | {}/{} | {} | {} | {} | {} |",
                c.runs,
                c.usual_success,
                c.usual,
                c.false_alarms,
                c.bench_alarms,
                c.trap_success,
                c.traps,
                c.bench_trap_success,
                c.shortened,
                c.wrong_ids,
                c.unit_errors
            );
        }
        println!();
    }
    Ok(())
}

/// The state after the commands of `outputs`.
fn end_state(
    task: &Task,
    outputs: &[fornace_constrain::bench::Output],
) -> fornace_constrain::ledger::Ledger {
    let mut state = task.start.clone();
    for command in outputs.iter().filter_map(|o| o.command.as_ref()) {
        let _ = state.execute(command);
    }
    state
}

/// The distribution of the signal for each task group: all generations,
/// and the first generation of each run.
fn distribution(outcomes: &[Outcome], signals: &HashMap<(usize, u64), Vec<f32>>) {
    println!("## Distribution of the signal\n");
    println!(
        "| Task group | Generations | Median | 90th percentile | First generations | Median | Above 0.5 | Above 0.9 | Runs with a generation above 0.5 |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    let groups: [GroupTest; 4] = [
        ("usual", |o| o.class == Class::Usual),
        ("over-limit", |o| o.class == Class::OverLimit),
        ("wrong-status, prefix ID", |o| {
            o.class == Class::WrongStatus && o.prefix_id
        }),
        ("wrong-status, no prefix ID", |o| {
            o.class == Class::WrongStatus && !o.prefix_id
        }),
    ];
    for (label, select) in groups {
        let runs: Vec<&Vec<f32>> = outcomes
            .iter()
            .filter(|o| select(o))
            .map(|o| &signals[&(o.task, o.seed)])
            .collect();
        let all: Vec<f32> = runs.iter().flat_map(|s| s.iter().copied()).collect();
        let first: Vec<f32> = runs.iter().filter_map(|s| s.first().copied()).collect();
        let above = |xs: &[f32], t: f32| xs.iter().filter(|&&x| x > t).count();
        let any = runs.iter().filter(|s| s.iter().any(|&x| x > 0.5)).count();
        println!(
            "| {label} | {} | {:.3} | {:.3} | {} | {:.3} | {} | {} | {any}/{} |",
            all.len(),
            quantile(&all, 0.5),
            quantile(&all, 0.9),
            first.len(),
            quantile(&first, 0.5),
            above(&first, 0.5),
            above(&first, 0.9),
            runs.len()
        );
    }
    println!();
}

/// The value at rank `floor(p × n)` of the sorted values, the last value
/// at most.
fn quantile(xs: &[f32], p: f64) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(f32::total_cmp);
    v[((p * v.len() as f64) as usize).min(v.len() - 1)]
}
