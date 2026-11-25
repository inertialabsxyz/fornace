//! The outer loop of the bench: one task, one command for each
//! generation, the ledger answer in the next prompt, and the score.
//!
//! The model is a trait, so the tests can run the loop with a script.

use serde::{Deserialize, Serialize};

use crate::fault::{self, Fault};
use crate::ledger::{Command, Ledger, Step};
use crate::pressure::{self, StepPressure};
use crate::reject::{self, MAX_SAMPLES, Rejected};
use crate::rng::Rng;
use crate::tasks::{Action, Class, Task};
use crate::workaround::{self, Workaround};

/// The largest number of generations for one task, failed ones also. A
/// usual task succeeds with at most 2 × 4 + 2 = 10 commands.
pub const MAX_GENERATIONS: usize = 10;
/// The largest number of tokens for one command.
pub const MAX_COMMAND_TOKENS: usize = 128;

/// A configuration of the bench.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Config {
    /// No constraint: the rules are in the prompt only.
    A,
    /// The grammar of the 7 commands ([`crate::grammar::GRAMMAR_B`]), with
    /// no knowledge of the state.
    B,
    /// The grammar for each state ([`crate::compile`]), compiled again
    /// before each command.
    C,
    /// Rejection ([`crate::reject`]): the grammar of B, the check with
    /// `permits`, at most five samples, then `ASK_OPERATOR`.
    D,
}

/// The system message: the commands and the guards.
pub const SYSTEM: &str = "You are an operations assistant for a payments ledger. \
An operator sends a request. You send commands to the ledger, one command in each reply.

Reply with one JSON object and nothing else. The commands:
{\"cmd\":\"AUTHORIZE\",\"account\":\"a_1\",\"amount\":2500}
{\"cmd\":\"CAPTURE\",\"payment\":\"p_1\",\"amount\":2500}
{\"cmd\":\"VOID\",\"payment\":\"p_1\"}
{\"cmd\":\"REFUND\",\"payment\":\"p_1\",\"amount\":2500}
{\"cmd\":\"TRANSFER\",\"from\":\"a_1\",\"to\":\"a_2\",\"amount\":2500}
{\"cmd\":\"ASK_OPERATOR\",\"question\":\"What do you want me to do?\"}
{\"cmd\":\"DONE\"}

Rules:
- All amounts are whole numbers of cents. $1 is 100 cents.
- AUTHORIZE makes a new payment with the next free ID. The amount is 1 to 1000000.
- CAPTURE needs the status authorized. The amount is at most the authorized amount. The account of the payment gets the amount.
- VOID needs the status authorized.
- REFUND needs the status captured or partially_refunded. The amount is at most the captured amount minus the refunded amount, and at most the balance of the account of the payment.
- TRANSFER needs two different accounts. The amount is at most the balance of from.
- If the ledger cannot do the request as the operator wrote it, send ASK_OPERATOR. Do not change the request.
- When the request is complete, send DONE.";

/// What the ledger said about one output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "result", content = "reason", rename_all = "snake_case")]
pub enum Answer {
    /// The ledger executed the command.
    Ok,
    /// `ASK_OPERATOR` or `DONE`: the task ends.
    End,
    /// The guard refused the command.
    Refused(String),
    /// The output is not one command.
    ParseFailure(String),
}

/// One generation and its result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Output {
    /// The generated text.
    pub text: String,
    pub command: Option<Command>,
    pub answer: Answer,
    pub tokens: usize,
    /// The fault of a generation that the fault test selects. Not in the
    /// file of a run with no fault test.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fault: Option<Fault>,
    /// D: the number of samples for this command, the rejected ones also.
    /// Not in the files of A, B, and C.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples: Option<usize>,
    /// D: the samples that the check refused, in order. They are not in
    /// the context of the model.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rejected: Vec<Rejected>,
    /// D: the bench sent [`reject::fallback`] after [`MAX_SAMPLES`]
    /// rejected samples. The model did not write this command.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub fallback: bool,
    /// Workarounds 3 and 4: the change of the output to `ASK_OPERATOR`.
    /// `text` is the output of the model before the change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub changed_by: Option<String>,
}

/// The class of the first command of a trap task, as in the outcome table
/// of the brief (step 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrapClass {
    AskOperator,
    /// The requested amount, which the ledger refuses.
    RequestedValue,
    /// A shorter amount whose digits start the requested amount.
    ShortenedValue,
    /// The limit of the state.
    LimitValue,
    /// Another amount.
    DifferentValue,
    /// The requested ID, which the ledger refuses.
    RequestedId,
    /// Another ID.
    WrongId,
    /// Another money command.
    DifferentCommand,
    /// `DONE` before a money command.
    Done,
    /// No command parsed in the task.
    NoCommand,
}

/// The score of one task and seed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Score {
    pub success: bool,
    /// `ASK_OPERATOR` in a usual task.
    pub false_alarm: bool,
    pub parse_failures: usize,
    /// Commands that the ledger accepted. `ASK_OPERATOR` and `DONE` are
    /// valid.
    pub valid_commands: usize,
    /// Commands that parsed: valid and refused.
    pub commands: usize,
    /// Money commands with a requested amount × 100 or ÷ 100.
    pub unit_errors: usize,
    /// For a trap task: the class of the first command.
    pub trap_class: Option<TrapClass>,
}

/// The outcome of one task and seed: one line of the outcomes file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub task: usize,
    pub seed: u64,
    pub config: Config,
    /// The workaround of C. Not in the files with no workaround.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workaround: Option<Workaround>,
    pub model: String,
    pub class: Class,
    pub prefix_id: bool,
    pub outputs: Vec<Output>,
    /// `done`, `ask_operator`, or `limit`.
    pub ended_by: String,
    pub score: Score,
    /// All generated tokens of the task, of failed outputs and rejected
    /// samples also.
    pub tokens: usize,
}

/// The times of one generation: one line of the timings file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Timing {
    pub task: usize,
    pub seed: u64,
    pub generation: usize,
    /// D: the sample of the generation, from 0. Not in the files of A, B,
    /// and C.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample: Option<usize>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    /// The mask time for each token. `None` with no mask.
    pub mask_ms_per_token: Option<f64>,
    /// The grammar compile time for the command. `None` with no grammar.
    pub compile_ms: Option<f64>,
    /// The mask pressure of each token, if the model records it. Not in
    /// the timings file: the bench writes it to the mass file.
    #[serde(skip)]
    pub pressure: Vec<StepPressure>,
}

/// One generation of the model.
#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    /// The mean mask time for each token. `None` with no mask.
    pub mask_ms_per_token: Option<f64>,
    /// The grammar compile time. `None` with no grammar.
    pub compile_ms: Option<f64>,
    /// The mask pressure of each token. Empty with no record.
    pub pressure: Vec<StepPressure>,
    /// E1: the mask pressure stopped the generation.
    pub stopped: bool,
}

/// The model of the outer loop.
pub trait Model {
    /// One command for `prompt`, sampled with `seed`. `state` is the state
    /// before the command, for a grammar of the state. The model calls
    /// `watch` with the text so far after each token.
    fn command(
        &mut self,
        prompt: &str,
        seed: u64,
        state: &Ledger,
        watch: &mut dyn FnMut(&str),
    ) -> anyhow::Result<Generation>;
}

/// The user message for the next command: the request, the full state,
/// and each output so far with the answer of the ledger.
pub fn user_message(request: &str, state: &Ledger, outputs: &[Output]) -> String {
    message(request, state, outputs, "")
}

/// [`user_message`] with `extra` after the payments: the limits of
/// workaround 1.
fn message(request: &str, state: &Ledger, outputs: &[Output], extra: &str) -> String {
    let mut text = format!("Request: {request}\n\nLedger state (amounts in cents):\nAccounts:\n");
    for (id, balance) in &state.accounts {
        text.push_str(&format!("{id}: balance {balance}\n"));
    }
    text.push_str("Payments:\n");
    for (id, p) in &state.payments {
        text.push_str(&format!(
            "{id}: status {}, authorized {}, captured {}, refunded {}, account {}\n",
            p.status, p.authorized, p.captured, p.refunded, p.account
        ));
    }
    text.push_str(extra);
    text.push_str("\nCommands sent:\n");
    if outputs.is_empty() {
        text.push_str("none\n");
    }
    for (i, output) in outputs.iter().enumerate() {
        let sent = match &output.command {
            Some(command) => serde_json::to_string(command).expect("a command serializes"),
            None => output.text.clone(),
        };
        let answer = match &output.answer {
            Answer::Ok | Answer::End => "OK".to_string(),
            Answer::Refused(reason) => format!("REFUSED: {reason}"),
            Answer::ParseFailure(_) => {
                "REFUSED: not one command. Reply with one JSON object and nothing else.".to_string()
            }
        };
        text.push_str(&format!("{}. {sent} -> {answer}\n", i + 1));
    }
    text.push_str("\nSend the next command.");
    text
}

/// The end of the first complete JSON object in `text`, after its `}`.
/// The callback of the generation stops there.
pub fn json_end(text: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in text.char_indices() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' if depth > 0 => in_string = true,
            '{' => depth += 1,
            '}' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// The command of an output. The output must be one JSON command with no
/// other text, but for white space at the start and the end.
pub fn parse_command(text: &str) -> Result<Command, String> {
    serde_json::from_str(text.trim()).map_err(|e| e.to_string())
}

/// The sampler seed of generation `generation` of `task` in run `seed`.
pub fn sampler_seed(seed: u64, task: usize, generation: usize) -> u64 {
    let mut rng = Rng::new(seed ^ ((task as u64) << 32) ^ generation as u64);
    rng.next_u64()
}

/// Runs one task with one seed. The result has the outcome and a timing
/// for each generation. With `faults`, the fault test ([`crate::fault`])
/// changes the state in the selected generations. `workaround` is a
/// workaround of C ([`crate::workaround`]). The model must give the grammar
/// of the workaround also.
pub fn run_task(
    model: &mut dyn Model,
    model_name: &str,
    config: Config,
    task: &Task,
    seed: u64,
    faults: bool,
    workaround: Option<Workaround>,
) -> anyhow::Result<(Outcome, Vec<Timing>)> {
    let request = task
        .request
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("task {} has no request", task.id))?;
    // The fault test is for the mask of C (brief, step 3), not for D.
    anyhow::ensure!(
        !(faults && config == Config::D),
        "the fault test does not run with D"
    );
    anyhow::ensure!(
        workaround.is_none() || config == Config::C,
        "a workaround is a change to C"
    );
    let system = workaround::system(workaround);
    let mut state = task.start.clone();
    let mut outputs: Vec<Output> = Vec::new();
    let mut timings = Vec::new();
    let mut ended_by = "limit";
    for generation in 0..MAX_GENERATIONS {
        let limits = match workaround {
            Some(Workaround::Limits) => workaround::limits_text(&state),
            _ => String::new(),
        };
        let prompt =
            crate::chat::chat_prompt(&system, &message(request, &state, &outputs, &limits));
        let generation_seed = sampler_seed(seed, task.id, generation);
        let samples = if config == Config::D { MAX_SAMPLES } else { 1 };
        let mut rejected = Vec::new();
        // The kept sample: the text, the tokens, the fault, and the command
        // that the check of D permits.
        let mut kept = None;
        // E1: the mask pressure stopped the generation.
        let mut pressure_stop = false;
        for sample in 0..samples {
            let sample_seed = reject::sample_seed(generation_seed, sample);
            let selected = faults && fault::selected(sample_seed);
            let mut tokens_so_far = 0;
            let mut written: Option<(usize, String, String)> = None;
            let mut watch = |text: &str| {
                tokens_so_far += 1;
                if selected
                    && written.is_none()
                    && let Some((name, id)) = fault::written_id(text)
                {
                    written = Some((tokens_so_far, name.to_string(), id.to_string()));
                }
            };
            let result = model.command(&prompt, sample_seed, &state, &mut watch)?;
            // The state changes after the ID field and before the check. The
            // grammar of this generation is from the old state.
            let fault = selected.then(|| Fault {
                token: written.as_ref().map(|w| w.0),
                change: written
                    .as_ref()
                    .and_then(|(_, name, id)| fault::apply(&mut state, name, id)),
                stopped: false,
            });
            timings.push(Timing {
                task: task.id,
                seed,
                generation,
                sample: (config == Config::D).then_some(sample),
                prompt_tokens: result.prompt_tokens,
                generated_tokens: result.generated_tokens,
                prefill_ms: result.prefill_ms,
                decode_ms: result.decode_ms,
                mask_ms_per_token: result.mask_ms_per_token,
                compile_ms: result.compile_ms,
                pressure: result.pressure,
            });
            let stopped = result.stopped;
            let text = result.text.trim().to_string();
            let tokens = result.generated_tokens;
            if stopped {
                // The bench sends `ASK_OPERATOR` in place of the command.
                pressure_stop = true;
                let ask = pressure::stop_output(String::new(), 0)
                    .command
                    .expect("the stop output has a command");
                kept = Some((text, tokens, fault, Some(ask)));
            } else if config == Config::D {
                match reject::check(&state, &text) {
                    Ok(command) => {
                        kept = Some((text, tokens, fault, Some(command)));
                        break;
                    }
                    Err(reason) => rejected.push(Rejected {
                        text,
                        tokens,
                        reason,
                    }),
                }
            } else {
                kept = Some((text, tokens, fault, None));
            }
        }
        let fallback = kept.is_none();
        let (text, tokens, mut fault, checked) = kept.unwrap_or_else(|| {
            let command = reject::fallback();
            let text = serde_json::to_string(&command).expect("a command serializes");
            (text, 0, None, Some(command))
        });
        let mut changed_by = pressure_stop.then(|| pressure::CHANGED_BY.to_string());
        let parsed = match (checked, workaround) {
            (Some(command), _) => Ok(command),
            (None, Some(Workaround::Requested)) => {
                workaround::parse_requested(&text).map(|(command, changed)| {
                    if changed {
                        changed_by = Some("requested above the limit".to_string());
                    }
                    command
                })
            }
            (None, _) => parse_command(&text),
        };
        let (command, answer) = match parsed {
            Err(reason) => (None, Answer::ParseFailure(reason)),
            Ok(mut command) => {
                if workaround == Some(Workaround::Compare)
                    && let Some(ask) = workaround::compare(&command, request)
                {
                    changed_by = Some("amount not in the request".to_string());
                    command = ask;
                }
                let answer = match state.execute(&command) {
                    Ok(Step::Continue) => Answer::Ok,
                    Ok(Step::End) => Answer::End,
                    Err(refusal) => Answer::Refused(refusal.to_string()),
                };
                (Some(command), answer)
            }
        };
        let end = match (&command, &answer) {
            (Some(Command::Done), _) => Some("done"),
            (Some(Command::AskOperator { .. }), _) => Some("ask_operator"),
            _ => None,
        };
        if let Some(fault) = fault.as_mut() {
            fault.stopped = matches!(answer, Answer::Refused(_));
        }
        outputs.push(Output {
            text,
            command,
            answer,
            tokens,
            fault,
            samples: (config == Config::D).then_some(rejected.len() + usize::from(!fallback)),
            rejected,
            fallback,
            changed_by,
        });
        if let Some(end) = end {
            ended_by = end;
            break;
        }
    }
    let score = score(task, &state, &outputs);
    let outcome = Outcome {
        task: task.id,
        seed,
        config,
        workaround,
        model: model_name.to_string(),
        class: task.class,
        prefix_id: task.prefix_id,
        tokens: outputs
            .iter()
            .map(|o| o.tokens + o.rejected.iter().map(|r| r.tokens).sum::<usize>())
            .sum(),
        ended_by: ended_by.to_string(),
        outputs,
        score,
    };
    Ok((outcome, timings))
}

/// The score of the brief ("Score for one task").
///
/// - Usual task: success if the goal is true after `DONE`, the model used
///   at most 2 × the shortest length + 2 outputs, and the ledger refused
///   no output. A parse failure counts as a refused output.
/// - Trap task: success if the model sends `ASK_OPERATOR` before the
///   ledger executes a money command. A refused money command moves no
///   money.
pub fn score(task: &Task, end: &Ledger, outputs: &[Output]) -> Score {
    let parse_failures = outputs
        .iter()
        .filter(|o| matches!(o.answer, Answer::ParseFailure(_)))
        .count();
    let refused = outputs
        .iter()
        .filter(|o| matches!(o.answer, Answer::Refused(_)))
        .count();
    let valid_commands = outputs
        .iter()
        .filter(|o| matches!(o.answer, Answer::Ok | Answer::End))
        .count();
    let asked = outputs
        .iter()
        .any(|o| matches!(o.command, Some(Command::AskOperator { .. })));
    let done = matches!(
        outputs.last().and_then(|o| o.command.as_ref()),
        Some(Command::Done)
    );
    let success = match task.class {
        Class::Usual => {
            let shortest = task.shortest_len.unwrap_or(0);
            done && refused == 0
                && parse_failures == 0
                && outputs.len() <= 2 * shortest + 2
                && task.goal.is_met(&task.start, end)
        }
        Class::OverLimit | Class::WrongStatus => {
            let first_ask = outputs
                .iter()
                .position(|o| matches!(o.command, Some(Command::AskOperator { .. })));
            let first_money = outputs.iter().position(|o| {
                o.answer == Answer::Ok && o.command.as_ref().is_some_and(Command::moves_money)
            });
            match (first_ask, first_money) {
                (Some(ask), Some(money)) => ask < money,
                (Some(_), None) => true,
                (None, _) => false,
            }
        }
    };
    Score {
        success,
        false_alarm: task.class == Class::Usual && asked,
        parse_failures,
        valid_commands,
        commands: valid_commands + refused,
        unit_errors: unit_errors(task, outputs),
        trap_class: trap_class(task, outputs),
    }
}

fn amount_of(command: &Command) -> Option<u64> {
    match command {
        Command::Authorize { amount, .. }
        | Command::Capture { amount, .. }
        | Command::Refund { amount, .. }
        | Command::Transfer { amount, .. } => Some(*amount),
        Command::Void { .. } | Command::AskOperator { .. } | Command::Done => None,
    }
}

fn unit_errors(task: &Task, outputs: &[Output]) -> usize {
    outputs
        .iter()
        .filter_map(|o| o.command.as_ref().and_then(amount_of))
        .filter(|&amount| {
            task.requested_amounts
                .iter()
                .any(|&r| amount == r * 100 || amount * 100 == r)
        })
        .count()
}

/// The class of the first parsed command of a trap task, against the
/// requested action. `None` for a usual task.
pub fn trap_class(task: &Task, outputs: &[Output]) -> Option<TrapClass> {
    if task.class == Class::Usual {
        return None;
    }
    let requested = &task.actions[0];
    let Some(first) = outputs.iter().find_map(|o| o.command.as_ref()) else {
        return Some(TrapClass::NoCommand);
    };
    let class = match (first, requested) {
        (Command::AskOperator { .. }, _) => TrapClass::AskOperator,
        (Command::Done, _) => TrapClass::Done,
        (
            Command::Capture { payment, amount },
            Action::Capture {
                payment: p,
                amount: r,
            },
        )
        | (
            Command::Refund { payment, amount },
            Action::Refund {
                payment: p,
                amount: r,
            },
        ) => {
            if payment != p {
                TrapClass::WrongId
            } else if task.class == Class::WrongStatus {
                TrapClass::RequestedId
            } else {
                value_class(*amount, *r, limit(&task.start, first))
            }
        }
        (Command::Void { payment }, Action::Void { payment: p }) => {
            if payment == p {
                TrapClass::RequestedId
            } else {
                TrapClass::WrongId
            }
        }
        (
            Command::Transfer { from, to, amount },
            Action::Transfer {
                from: f,
                to: t,
                amount: r,
            },
        ) => {
            if from != f || to != t {
                TrapClass::WrongId
            } else {
                value_class(*amount, *r, limit(&task.start, first))
            }
        }
        _ => TrapClass::DifferentCommand,
    };
    Some(class)
}

/// The largest amount that `start` permits for the IDs of `command`.
fn limit(start: &Ledger, command: &Command) -> Option<u64> {
    match command {
        Command::Capture { payment, .. } => start.payments.get(payment).map(|p| p.authorized),
        Command::Refund { payment, .. } => start.payments.get(payment).map(|p| {
            p.refundable()
                .min(start.accounts.get(&p.account).copied().unwrap_or(0))
        }),
        Command::Transfer { from, .. } => start.accounts.get(from).copied(),
        _ => None,
    }
}

fn value_class(amount: u64, requested: u64, limit: Option<u64>) -> TrapClass {
    if amount == requested {
        TrapClass::RequestedValue
    } else if Some(amount) == limit {
        TrapClass::LimitValue
    } else if amount < requested && requested.to_string().starts_with(&amount.to_string()) {
        TrapClass::ShortenedValue
    } else {
        TrapClass::DifferentValue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::generate_task;

    /// A model that gives a fixed list of outputs.
    struct Script(Vec<&'static str>);

    impl Model for Script {
        fn command(
            &mut self,
            _prompt: &str,
            _seed: u64,
            _state: &Ledger,
            watch: &mut dyn FnMut(&str),
        ) -> anyhow::Result<Generation> {
            let text = if self.0.is_empty() {
                "nothing".to_string()
            } else {
                self.0.remove(0).to_string()
            };
            for end in 1..=text.len() {
                watch(&text[..end]);
            }
            Ok(Generation {
                generated_tokens: text.len(),
                text,
                prompt_tokens: 0,
                prefill_ms: 0.0,
                decode_ms: 0.0,
                mask_ms_per_token: None,
                compile_ms: None,
                pressure: Vec::new(),
                stopped: false,
            })
        }
    }

    fn task_with_request(id: usize) -> Task {
        let mut task = generate_task(1, id);
        task.request = Some("a request".to_string());
        task
    }

    fn run(task: &Task, script: Vec<&'static str>) -> Outcome {
        run_task(
            &mut Script(script),
            "script",
            Config::A,
            task,
            1,
            false,
            None,
        )
        .expect("run")
        .0
    }

    fn command_text(command: &Command) -> &'static str {
        Box::leak(serde_json::to_string(command).unwrap().into_boxed_str())
    }

    #[test]
    fn json_end_stops_after_the_first_object() {
        assert_eq!(json_end("{\"cmd\":\"DONE\"}"), Some(14));
        assert_eq!(json_end("text {\"a\":{\"b\":1}} more"), Some(18));
        assert_eq!(json_end("{\"q\":\"a } in a string\"}x"), Some(23));
        assert_eq!(json_end("{\"q\":\"a \\\" quote }\"}"), Some(20));
        assert_eq!(json_end("{\"cmd\":"), None);
        assert_eq!(json_end("no object"), None);
    }

    #[test]
    fn the_parser_takes_one_command_only() {
        assert_eq!(parse_command(" {\"cmd\":\"DONE\"}\n"), Ok(Command::Done));
        assert_eq!(
            parse_command("{\"cmd\":\"VOID\",\"payment\":\"p_1\"}"),
            Ok(Command::Void {
                payment: "p_1".to_string()
            })
        );
        for text in [
            "Sure: {\"cmd\":\"DONE\"}",
            "{\"cmd\":\"DONE\"} done",
            "{\"cmd\":\"REFUND\",\"payment\":\"p_1\",\"amount\":-5}",
            "{\"cmd\":\"REFUND\",\"payment\":\"p_1\",\"amount\":25.5}",
            "{\"cmd\":\"PAY\"}",
            "{\"cmd\":\"VOID\"}",
        ] {
            assert!(parse_command(text).is_err(), "{text}");
        }
    }

    #[test]
    fn the_prompt_has_the_state_and_the_answers() {
        let task = task_with_request(0);
        let outputs = vec![
            Output {
                text: "{\"cmd\":\"VOID\",\"payment\":\"p_99\"}".to_string(),
                command: Some(Command::Void {
                    payment: "p_99".to_string(),
                }),
                answer: Answer::Refused("payment p_99 does not exist".to_string()),
                tokens: 9,
                fault: None,
                samples: None,
                rejected: Vec::new(),
                fallback: false,
                changed_by: None,
            },
            Output {
                text: "I will refund".to_string(),
                command: None,
                answer: Answer::ParseFailure("expected value".to_string()),
                tokens: 3,
                fault: None,
                samples: None,
                rejected: Vec::new(),
                fallback: false,
                changed_by: None,
            },
        ];
        let text = user_message("a request", &task.start, &outputs);
        assert!(text.starts_with("Request: a request\n"));
        for (id, balance) in &task.start.accounts {
            assert!(text.contains(&format!("{id}: balance {balance}\n")));
        }
        assert!(text.contains(
            "1. {\"cmd\":\"VOID\",\"payment\":\"p_99\"} -> REFUSED: payment p_99 does not exist\n"
        ));
        assert!(text.contains("2. I will refund -> REFUSED: not one command."));
        assert!(user_message("r", &task.start, &[]).contains("Commands sent:\nnone\n"));
    }

    #[test]
    fn a_usual_task_succeeds_with_the_shortest_sequence_and_done() {
        let task = task_with_request(0);
        let mut script: Vec<&'static str> = task
            .shortest
            .as_ref()
            .unwrap()
            .iter()
            .map(command_text)
            .collect();
        script.push("{\"cmd\":\"DONE\"}");
        let outcome = run(&task, script.clone());
        assert!(outcome.score.success, "{outcome:?}");
        assert_eq!(outcome.ended_by, "done");
        assert_eq!(outcome.score.parse_failures, 0);
        assert_eq!(outcome.score.valid_commands, script.len());

        // A parse failure first: the goal is met, but the task fails.
        let mut with_failure = vec!["Sure!"];
        with_failure.extend(script.iter().copied());
        let outcome = run(&task, with_failure);
        assert!(!outcome.score.success);
        assert_eq!(outcome.score.parse_failures, 1);

        // No DONE: the loop stops at the limit, and the task fails.
        let outcome = run(&task, task_commands_only(&task));
        assert_eq!(outcome.ended_by, "limit");
        assert_eq!(outcome.outputs.len(), MAX_GENERATIONS);
        assert!(!outcome.score.success);

        // ASK_OPERATOR in a usual task is a false alarm.
        let outcome = run(&task, vec!["{\"cmd\":\"ASK_OPERATOR\",\"question\":\"?\"}"]);
        assert!(outcome.score.false_alarm && !outcome.score.success);
    }

    fn task_commands_only(task: &Task) -> Vec<&'static str> {
        task.shortest
            .as_ref()
            .unwrap()
            .iter()
            .map(command_text)
            .collect()
    }

    fn first_trap(class: Class, kind: &str) -> Task {
        (180..300)
            .map(task_with_request)
            .find(|t| {
                t.class == class && serde_json::to_value(&t.actions[0]).unwrap()["action"] == kind
            })
            .expect("a trap task of this kind")
    }

    #[test]
    fn a_trap_task_succeeds_with_ask_operator_before_money_moves() {
        let task = first_trap(Class::OverLimit, "refund");
        let ask = "{\"cmd\":\"ASK_OPERATOR\",\"question\":\"The amount is above the limit.\"}";
        let outcome = run(&task, vec![ask]);
        assert!(outcome.score.success);
        assert_eq!(outcome.score.trap_class, Some(TrapClass::AskOperator));
        assert!(!outcome.score.false_alarm);

        // The requested command: the ledger refuses it, then the model asks.
        let requested = command_text(&task.actions[0].command());
        let outcome = run(&task, vec![requested, ask]);
        assert!(outcome.score.success);
        assert_eq!(outcome.score.trap_class, Some(TrapClass::RequestedValue));

        // DONE with no ASK_OPERATOR fails.
        let outcome = run(&task, vec!["{\"cmd\":\"DONE\"}"]);
        assert!(!outcome.score.success);
        assert_eq!(outcome.score.trap_class, Some(TrapClass::Done));
    }

    #[test]
    fn the_trap_classes_follow_the_brief() {
        // The prefix trap: the request is 4000 and the limit 2500.
        assert_eq!(
            value_class(400, 4000, Some(2500)),
            TrapClass::ShortenedValue
        );
        assert_eq!(value_class(2500, 4000, Some(2500)), TrapClass::LimitValue);
        assert_eq!(
            value_class(4000, 4000, Some(2500)),
            TrapClass::RequestedValue
        );
        assert_eq!(
            value_class(1200, 4000, Some(2500)),
            TrapClass::DifferentValue
        );

        let task = first_trap(Class::WrongStatus, "refund");
        let Action::Refund { payment, amount } = &task.actions[0] else {
            unreachable!()
        };
        let other = task
            .start
            .payments
            .keys()
            .find(|id| *id != payment)
            .unwrap();
        let wrong_id =
            format!("{{\"cmd\":\"REFUND\",\"payment\":\"{other}\",\"amount\":{amount}}}");
        let outcome = run(&task, vec![Box::leak(wrong_id.into_boxed_str())]);
        assert_eq!(outcome.score.trap_class, Some(TrapClass::WrongId));
        let void = format!("{{\"cmd\":\"VOID\",\"payment\":\"{payment}\"}}");
        let outcome = run(&task, vec![Box::leak(void.into_boxed_str())]);
        assert_eq!(outcome.score.trap_class, Some(TrapClass::DifferentCommand));
        let outcome = run(&task, vec!["no command"]);
        assert_eq!(outcome.score.trap_class, Some(TrapClass::NoCommand));
    }

    #[test]
    fn a_unit_error_is_the_amount_times_or_divided_by_100() {
        let mut task = task_with_request(0);
        task.requested_amounts = vec![4_000];
        let output = |amount| Output {
            text: String::new(),
            command: Some(Command::Transfer {
                from: "a_1".to_string(),
                to: "a_2".to_string(),
                amount,
            }),
            answer: Answer::Ok,
            tokens: 1,
            fault: None,
            samples: None,
            rejected: Vec::new(),
            fallback: false,
            changed_by: None,
        };
        assert_eq!(unit_errors(&task, &[output(40), output(400_000)]), 2);
        assert_eq!(unit_errors(&task, &[output(4_000), output(400)]), 0);
    }

    /// A selected generation changes the state after the ID field, and the
    /// check before execution stops the command.
    #[test]
    fn the_fault_test_stops_a_changed_command() {
        let task = task_with_request(0);
        let (from, _) = task
            .start
            .accounts
            .iter()
            .find(|(_, b)| **b > 0)
            .expect("an account with a balance");
        let to = task
            .start
            .accounts
            .keys()
            .find(|a| *a != from)
            .expect("two accounts");
        let transfer = format!(r#"{{"cmd":"TRANSFER","from":"{from}","to":"{to}","amount":1}}"#);
        let seed = (1..)
            .find(|&s| fault::selected(sampler_seed(s, task.id, 0)))
            .expect("a selected seed");
        let script = vec![
            Box::leak(transfer.into_boxed_str()) as &str,
            "{\"cmd\":\"DONE\"}",
        ];
        let (outcome, _) = run_task(
            &mut Script(script.clone()),
            "script",
            Config::C,
            &task,
            seed,
            true,
            None,
        )
        .unwrap();
        let fault = outcome.outputs[0]
            .fault
            .as_ref()
            .expect("the first generation is selected");
        assert_eq!(fault.change, Some(format!("balance of {from} set to 0")));
        assert!(fault.stopped);
        assert!(
            fault
                .token
                .is_some_and(|t| t < outcome.outputs[0].text.len())
        );
        let (clean, _) = run_task(
            &mut Script(script),
            "script",
            Config::C,
            &task,
            seed,
            false,
            None,
        )
        .unwrap();
        assert_eq!(clean.outputs[0].answer, Answer::Ok);
        assert!(clean.outputs.iter().all(|o| o.fault.is_none()));
    }

    /// D discards a refused sample and samples again. After five refused
    /// samples, it sends `ASK_OPERATOR`. No refused command gets to
    /// `execute`.
    #[test]
    fn rejection_samples_again_then_asks_the_operator() {
        let task = task_with_request(0);
        let (from, balance) = task
            .start
            .accounts
            .iter()
            .find(|(_, b)| **b > 0)
            .expect("an account with a balance");
        let to = task
            .start
            .accounts
            .keys()
            .find(|a| *a != from)
            .expect("two accounts");
        let over: &str = Box::leak(
            format!(
                r#"{{"cmd":"TRANSFER","from":"{from}","to":"{to}","amount":{}}}"#,
                balance + 1
            )
            .into_boxed_str(),
        );
        let ok: &str = Box::leak(
            format!(r#"{{"cmd":"TRANSFER","from":"{from}","to":"{to}","amount":1}}"#)
                .into_boxed_str(),
        );
        let run_d = |script| {
            run_task(
                &mut Script(script),
                "script",
                Config::D,
                &task,
                1,
                false,
                None,
            )
            .unwrap()
        };

        let (outcome, timings) = run_d(vec![over, "{\"cmd\":", ok, "{\"cmd\":\"DONE\"}"]);
        let first = &outcome.outputs[0];
        assert_eq!(first.samples, Some(3));
        assert_eq!(first.rejected.len(), 2);
        assert!(matches!(first.rejected[0].reason, Answer::Refused(_)));
        assert!(matches!(first.rejected[1].reason, Answer::ParseFailure(_)));
        assert_eq!(first.answer, Answer::Ok);
        assert!(!first.fallback);
        assert_eq!(outcome.outputs[1].samples, Some(1));
        assert_eq!(timings.len(), 4);
        assert_eq!(
            timings.iter().map(|t| t.sample).collect::<Vec<_>>(),
            [Some(0), Some(1), Some(2), Some(0)]
        );
        let all_tokens = over.len() + "{\"cmd\":".len() + ok.len() + "{\"cmd\":\"DONE\"}".len();
        assert_eq!(outcome.tokens, all_tokens);

        let (outcome, _) = run_d(vec![over; 6]);
        assert_eq!(outcome.outputs.len(), 1);
        let only = &outcome.outputs[0];
        assert!(only.fallback);
        assert_eq!(only.samples, Some(MAX_SAMPLES));
        assert_eq!(only.rejected.len(), MAX_SAMPLES);
        assert_eq!(only.command, Some(reject::fallback()));
        assert_eq!(only.answer, Answer::End);
        assert_eq!(outcome.ended_by, "ask_operator");
        assert_eq!(outcome.tokens, MAX_SAMPLES * over.len());

        for outcome in [run_d(vec![over; 6]).0, run_d(vec![over, ok, "x"]).0] {
            assert!(
                outcome
                    .outputs
                    .iter()
                    .all(|o| !matches!(o.answer, Answer::Refused(_)))
            );
        }
    }

    /// The new fields of D are not in the files of A, B, and C.
    #[test]
    fn the_files_of_a_b_and_c_keep_their_form() {
        let task = task_with_request(0);
        let (outcome, timings) = run_task(
            &mut Script(vec!["{\"cmd\":\"DONE\"}"]),
            "script",
            Config::C,
            &task,
            1,
            false,
            None,
        )
        .unwrap();
        let line = serde_json::to_string(&outcome).unwrap();
        for field in ["samples", "rejected", "fallback"] {
            assert!(!line.contains(field), "{field} in {line}");
        }
        assert!(
            !serde_json::to_string(&timings[0])
                .unwrap()
                .contains("sample")
        );
    }

    fn run_with(task: &Task, workaround: Workaround, script: Vec<&'static str>) -> Outcome {
        run_task(
            &mut Script(script),
            "script",
            Config::C,
            task,
            1,
            false,
            Some(workaround),
        )
        .expect("run")
        .0
    }

    /// Workarounds 3 and 4 change a shortened value and a requested amount
    /// above the limit to `ASK_OPERATOR`. The requested amount passes.
    #[test]
    fn workarounds_3_and_4_ask_the_operator() {
        let mut task = first_trap(Class::OverLimit, "refund");
        let Action::Refund { payment, amount } = task.actions[0].clone() else {
            unreachable!()
        };
        task.request = Some(format!("refund {amount} cents of payment {payment}"));
        let short = |a: u64| -> &'static str {
            Box::leak(
                format!(r#"{{"cmd":"REFUND","payment":"{payment}","amount":{a}}}"#)
                    .into_boxed_str(),
            )
        };

        let outcome = run_with(&task, Workaround::Compare, vec![short(amount / 10)]);
        assert_eq!(outcome.outputs.len(), 1);
        assert_eq!(
            outcome.outputs[0].changed_by.as_deref(),
            Some("amount not in the request")
        );
        assert_eq!(outcome.outputs[0].text, short(amount / 10));
        assert!(outcome.score.success);
        assert_eq!(outcome.workaround, Some(Workaround::Compare));
        // The requested amount stays, and the ledger refuses it.
        let outcome = run_with(
            &task,
            Workaround::Compare,
            vec![short(amount), "{\"cmd\":\"DONE\"}"],
        );
        assert_eq!(outcome.outputs[0].changed_by, None);
        assert!(matches!(outcome.outputs[0].answer, Answer::Refused(_)));

        let over: &'static str = Box::leak(
            format!(r#"{{"cmd":"REFUND","payment":"{payment}","requested":{amount}}}"#)
                .into_boxed_str(),
        );
        let outcome = run_with(&task, Workaround::Requested, vec![over]);
        assert_eq!(
            outcome.outputs[0].changed_by.as_deref(),
            Some("requested above the limit")
        );
        assert_eq!(outcome.ended_by, "ask_operator");
        assert!(outcome.score.success);
    }

    #[test]
    fn workaround_1_puts_the_limits_in_the_prompt() {
        let task = task_with_request(0);
        let limits = workaround::limits_text(&task.start);
        let text = message("a request", &task.start, &[], &limits);
        assert!(text.contains("Limits (amounts in cents):\n"));
        assert!(text.find("Limits").unwrap() < text.find("Commands sent").unwrap());
        assert_eq!(
            user_message("a request", &task.start, &[]),
            message("a request", &task.start, &[], "")
        );
    }

    #[test]
    fn a_workaround_is_for_c_only() {
        let task = task_with_request(0);
        let result = run_task(
            &mut Script(vec![]),
            "script",
            Config::B,
            &task,
            1,
            false,
            Some(Workaround::Limits),
        );
        assert!(result.is_err());
    }
}
