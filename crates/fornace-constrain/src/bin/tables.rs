//! The tables of the brief (step 4 and step 5), made from the run files.
//! Give the outcomes files of one matrix. The timings file of each run is
//! next to it. Writes markdown to standard output.
//!
//! Each rate has a 95 % Wilson interval. The paired counts compare two
//! configurations on the same task and seed.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use fornace_constrain::bench::{Answer, Outcome, Timing, TrapClass};
use fornace_constrain::ledger::Command;
use fornace_constrain::tasks::{Class, Task, Unit};

#[derive(Parser)]
struct Args {
    /// The outcomes files: `...-outcomes.jsonl`.
    #[arg(required = true)]
    outcomes: Vec<PathBuf>,
    #[arg(long, default_value = "data/constrained/tasks.jsonl")]
    tasks: PathBuf,
}

/// One run: the outcomes and the timings of one configuration.
struct Run {
    label: String,
    file: String,
    outcomes: Vec<Outcome>,
    timings: Vec<Timing>,
}

/// The task groups of the trap table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Group {
    Usual,
    OverLimit,
    WrongStatusPrefix,
    WrongStatusPlain,
}

impl Group {
    fn of(o: &Outcome) -> Group {
        match (o.class, o.prefix_id) {
            (Class::Usual, _) => Group::Usual,
            (Class::OverLimit, _) => Group::OverLimit,
            (Class::WrongStatus, true) => Group::WrongStatusPrefix,
            (Class::WrongStatus, false) => Group::WrongStatusPlain,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Group::Usual => "usual",
            Group::OverLimit => "over-limit",
            Group::WrongStatusPrefix => "wrong-status, prefix ID",
            Group::WrongStatusPlain => "wrong-status, no prefix ID",
        }
    }
}

/// The 95 % Wilson interval of `k` in `n`, in percent.
fn wilson(k: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }
    let (k, n, z) = (k as f64, n as f64, 1.96f64);
    let p = k / n;
    let centre = p + z * z / (2.0 * n);
    let half = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt();
    let scale = 1.0 + z * z / n;
    (
        100.0 * (centre - half) / scale,
        100.0 * (centre + half) / scale,
    )
}

/// `k/n = p % [lo, hi]`.
fn rate(k: usize, n: usize) -> String {
    if n == 0 {
        return "—".to_string();
    }
    let (lo, hi) = wilson(k, n);
    format!(
        "{k}/{n} = {:.1} % [{lo:.1}, {hi:.1}]",
        100.0 * k as f64 / n as f64
    )
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}

fn ms(value: Option<f64>) -> String {
    value.map_or("—".to_string(), |v| format!("{v:.3}"))
}

/// The `ASK_OPERATOR` that ends a successful trap task came from the bench,
/// not from the model: the fallback of D, or the change of workaround 3 or
/// 4.
fn ask_by_bench(o: &Outcome) -> bool {
    o.outputs
        .iter()
        .find(|out| matches!(out.command, Some(Command::AskOperator { .. })))
        .is_some_and(|out| out.fallback || out.changed_by.is_some())
}

/// The table of the brief, step 5: one row for each run.
fn main_table(runs: &[Run]) -> String {
    let mut text = String::from(
        "| Configuration | Runs | Parse failures | Valid-command rate | Usual success | Over-limit success | Wrong-status success | False alarms | Commands for each task | Tokens for each task | Mask for each token, median (ms) | Compile, median (ms) |\n\
         | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
    );
    for run in runs {
        let o = &run.outcomes;
        let success = |class: Class| {
            let all: Vec<_> = o.iter().filter(|x| x.class == class).collect();
            rate(all.iter().filter(|x| x.score.success).count(), all.len())
        };
        let parse: usize = o.iter().map(|x| x.score.parse_failures).sum();
        let rejected_parse: usize = o
            .iter()
            .flat_map(|x| &x.outputs)
            .flat_map(|out| &out.rejected)
            .filter(|r| matches!(r.reason, Answer::ParseFailure(_)))
            .count();
        let valid: usize = o.iter().map(|x| x.score.valid_commands).sum();
        let commands: usize = o.iter().map(|x| x.score.commands).sum();
        let usual = o.iter().filter(|x| x.class == Class::Usual).count();
        let alarms = o.iter().filter(|x| x.score.false_alarm).count();
        let outputs: usize = o.iter().map(|x| x.outputs.len()).sum();
        let tokens: usize = o.iter().map(|x| x.tokens).sum();
        let n = o.len().max(1) as f64;
        let mask = median(
            run.timings
                .iter()
                .filter_map(|t| t.mask_ms_per_token)
                .collect(),
        );
        let compile = median(run.timings.iter().filter_map(|t| t.compile_ms).collect());
        let parse_text = if rejected_parse > 0 {
            format!("{parse} (+{rejected_parse} rejected samples)")
        } else {
            parse.to_string()
        };
        text.push_str(&format!(
            "| {} | {} | {parse_text} | {} | {} | {} | {} | {} | {:.2} | {:.1} | {} | {} |\n",
            run.label,
            o.len(),
            rate(valid, commands),
            success(Class::Usual),
            success(Class::OverLimit),
            success(Class::WrongStatus),
            if usual > 0 {
                format!("{alarms}/{usual}")
            } else {
                "—".to_string()
            },
            outputs as f64 / n,
            tokens as f64 / n,
            ms(mask),
            ms(compile),
        ));
    }
    text
}

/// The trap successes with the `ASK_OPERATOR` of the model only, for the
/// runs where the bench can send one.
fn bench_ask_table(runs: &[Run]) -> String {
    let mut text = String::from(
        "| Configuration | Task class | Task success | `ASK_OPERATOR` from the bench | Task success without it |\n\
         | --- | --- | --- | --- | --- |\n",
    );
    for run in runs {
        let any = run
            .outcomes
            .iter()
            .flat_map(|o| &o.outputs)
            .any(|out| out.fallback || out.changed_by.is_some());
        if !any {
            continue;
        }
        for group in [
            Group::Usual,
            Group::OverLimit,
            Group::WrongStatusPrefix,
            Group::WrongStatusPlain,
        ] {
            let all: Vec<_> = run
                .outcomes
                .iter()
                .filter(|o| Group::of(o) == group)
                .collect();
            if all.is_empty() {
                continue;
            }
            let wins = all.iter().filter(|o| o.score.success).count();
            let by_bench = all
                .iter()
                .filter(|o| o.score.success && group != Group::Usual && ask_by_bench(o))
                .count();
            text.push_str(&format!(
                "| {} | {} | {} | {by_bench} | {} |\n",
                run.label,
                group.name(),
                rate(wins, all.len()),
                rate(wins - by_bench, all.len()),
            ));
        }
    }
    text
}

const TRAP_COLUMNS: [(TrapClass, &str); 10] = [
    (TrapClass::AskOperator, "`ASK_OPERATOR`"),
    (TrapClass::ShortenedValue, "Shortened value"),
    (TrapClass::LimitValue, "Limit value"),
    (TrapClass::DifferentValue, "Different valid value"),
    (TrapClass::RequestedValue, "Requested value"),
    (TrapClass::WrongId, "Wrong ID"),
    (TrapClass::RequestedId, "Requested ID"),
    (TrapClass::DifferentCommand, "Different command"),
    (TrapClass::Done, "`DONE`"),
    (TrapClass::NoCommand, "No command"),
];

/// The table of the brief, step 4: the class of the first command of each
/// trap task, and the unit errors in the same runs.
fn trap_table(runs: &[Run], units: &HashMap<usize, Unit>) -> String {
    let mut text = String::from("| Configuration | Task class | Runs |");
    for (_, name) in TRAP_COLUMNS {
        text.push_str(&format!(" {name} |"));
    }
    text.push_str(" Unit errors, cents | Unit errors, dollars |\n|");
    text.push_str(&" --- |".repeat(TRAP_COLUMNS.len() + 5));
    text.push('\n');
    for run in runs {
        for group in [
            Group::OverLimit,
            Group::WrongStatusPrefix,
            Group::WrongStatusPlain,
        ] {
            let all: Vec<_> = run
                .outcomes
                .iter()
                .filter(|o| Group::of(o) == group)
                .collect();
            if all.is_empty() {
                continue;
            }
            let mut counts: BTreeMap<String, usize> = BTreeMap::new();
            for o in &all {
                if let Some(class) = o.score.trap_class {
                    *counts.entry(format!("{class:?}")).or_default() += 1;
                }
            }
            let units_of = |unit: Unit| -> usize {
                all.iter()
                    .filter(|o| units.get(&o.task) == Some(&unit))
                    .map(|o| o.score.unit_errors)
                    .sum()
            };
            text.push_str(&format!(
                "| {} | {} | {} |",
                run.label,
                group.name(),
                all.len()
            ));
            for (class, _) in TRAP_COLUMNS {
                let k = counts.get(&format!("{class:?}")).copied().unwrap_or(0);
                text.push_str(&format!(" {k} |"));
            }
            text.push_str(&format!(
                " {} | {} |\n",
                units_of(Unit::Cents),
                units_of(Unit::Dollars)
            ));
        }
    }
    text
}

/// The paired counts of two runs on the same tasks and seeds: only the
/// first succeeds, only the second, both, and neither.
fn paired(a: &Run, b: &Run, trap: bool) -> Option<(usize, usize, usize, usize, usize)> {
    let second: HashMap<(usize, u64), bool> = b
        .outcomes
        .iter()
        .map(|o| ((o.task, o.seed), o.score.success))
        .collect();
    let (mut a_only, mut b_only, mut both, mut neither, mut n) = (0, 0, 0, 0, 0);
    for o in &a.outcomes {
        if (o.class != Class::Usual) != trap {
            continue;
        }
        let Some(&other) = second.get(&(o.task, o.seed)) else {
            continue;
        };
        n += 1;
        match (o.score.success, other) {
            (true, false) => a_only += 1,
            (false, true) => b_only += 1,
            (true, true) => both += 1,
            (false, false) => neither += 1,
        }
    }
    (n > 0).then_some((a_only, b_only, both, neither, n))
}

fn paired_table(runs: &[Run], pairs: &[(&str, &str)]) -> String {
    let mut text = String::from(
        "| First | Second | Tasks | Runs | First only | Second only | Both | Neither |\n\
         | --- | --- | --- | --- | --- | --- | --- | --- |\n",
    );
    let find = |label: &str| runs.iter().find(|r| r.label == label);
    for &(first, second) in pairs {
        let (Some(a), Some(b)) = (find(first), find(second)) else {
            continue;
        };
        for trap in [false, true] {
            if let Some((a_only, b_only, both, neither, n)) = paired(a, b, trap) {
                text.push_str(&format!(
                    "| {first} | {second} | {} | {n} | {a_only} | {b_only} | {both} | {neither} |\n",
                    if trap { "trap" } else { "usual" }
                ));
            }
        }
    }
    text
}

fn label(outcome: &Outcome) -> String {
    let config = format!("{:?}", outcome.config);
    match outcome.workaround {
        Some(w) => format!("{config} + workaround {}", w.number()),
        None => config,
    }
}

fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {path:?}"))?;
    text.lines()
        .map(|line| serde_json::from_str(line).with_context(|| format!("a line of {path:?}")))
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tasks: Vec<Task> = read_jsonl(&args.tasks)?;
    let units: HashMap<usize, Unit> = tasks.iter().map(|t| (t.id, t.unit)).collect();
    let mut runs = Vec::new();
    for path in &args.outcomes {
        let name = path.to_string_lossy().to_string();
        let timings_path = PathBuf::from(name.replace("-outcomes.jsonl", "-timings.jsonl"));
        let outcomes: Vec<Outcome> = read_jsonl(path)?;
        let first = outcomes
            .first()
            .with_context(|| format!("{path:?} is empty"))?;
        runs.push(Run {
            label: label(first),
            file: path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
            timings: read_jsonl(&timings_path)?,
            outcomes,
        });
    }
    runs.sort_by(|a, b| a.label.cmp(&b.label));

    println!("# Constrained sampling: the tables\n");
    println!("Made by `tables` from these files in `bench/runs/constrained/`:\n");
    for run in &runs {
        println!("- {}: `{}`", run.label, run.file);
    }
    println!("\nEach rate: k/n = rate [95 % Wilson interval].");
    println!("For D, \"Tokens for each task\" includes the rejected samples.");
    println!("The times are medians over the generations of the run.\n");
    println!("## Configurations (brief, step 5)\n");
    println!("{}", main_table(&runs));
    println!("## `ASK_OPERATOR` from the bench\n");
    println!(
        "The fallback of D, and the change of workarounds 3 and 4. For the usual tasks, the \
         column is 0: an `ASK_OPERATOR` there is a false alarm, not a success.\n"
    );
    println!("{}", bench_ask_table(&runs));
    println!("## The first command of the trap tasks (brief, step 4)\n");
    println!("{}", trap_table(&runs, &units));
    println!("## Paired counts\n");
    println!(
        "{}",
        paired_table(
            &runs,
            &[
                ("A", "B"),
                ("B", "C"),
                ("C", "D"),
                ("C", "C + workaround 1"),
                ("C", "C + workaround 3"),
                ("C", "C + workaround 4"),
                ("C", "C + workaround 5"),
                ("D", "C + workaround 3"),
            ]
        )
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wilson_interval_has_known_values() {
        // 60 tasks × 3 seeds at 50 %: about ±7 points (brief, step 5).
        let (lo, hi) = wilson(90, 180);
        assert!((lo - 42.8).abs() < 0.1, "{lo}");
        assert!((hi - 57.2).abs() < 0.1, "{hi}");
        let (lo, hi) = wilson(0, 90);
        assert_eq!(lo, 0.0);
        assert!((hi - 4.1).abs() < 0.1, "{hi}");
        let (lo, hi) = wilson(90, 90);
        assert!((lo - 95.9).abs() < 0.1, "{lo}");
        assert!((hi - 100.0).abs() < 1e-9, "{hi}");
        assert_eq!(rate(0, 0), "—");
    }

    #[test]
    fn the_median_takes_the_middle_value() {
        assert_eq!(median(vec![3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(Vec::new()), None);
    }
}
