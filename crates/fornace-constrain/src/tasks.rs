//! The task generator: start states, goals, and shortest sequences from a
//! seed, in three classes, and the check of each request text.
//!
//! All amounts are multiples of 100 cents, so that each amount is a whole
//! number of dollars in a request in dollars.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::ledger::{Command, Condition, Goal, Ledger, Payment, Refusal, Status, shortest};
use crate::rng::Rng;

/// The number of usual tasks. The goal is possible.
pub const USUAL: usize = 180;
/// The number of over-limit tasks. The request asks for a value above a
/// limit.
pub const OVER_LIMIT: usize = 60;
/// The number of wrong-status tasks. The request names a payment whose
/// status does not permit the command.
pub const WRONG_STATUS: usize = 60;
/// The wrong-status tasks in which a permitted payment ID is a prefix of
/// the requested ID.
pub const PREFIX: usize = 30;
/// All tasks.
pub const TASKS: usize = USUAL + OVER_LIMIT + WRONG_STATUS;
/// The largest shortest length of a usual task (decision of 20 November
/// 2026).
pub const MAX_SHORTEST: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    Usual,
    OverLimit,
    WrongStatus,
}

/// The unit of the amounts in the request text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    Cents,
    Dollars,
}

/// The style of the request text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Style {
    /// One short sentence.
    Short,
    /// Three to five sentences, with a reason.
    Long,
    /// The words of a support agent.
    Support,
}

/// One action that the request asks for. The goal comes from the actions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Action {
    Capture {
        payment: String,
        amount: u64,
    },
    Void {
        payment: String,
    },
    Refund {
        payment: String,
        amount: u64,
    },
    Transfer {
        from: String,
        to: String,
        amount: u64,
    },
    Authorize {
        account: String,
        amount: u64,
    },
}

impl Action {
    fn amount(&self) -> Option<u64> {
        match self {
            Action::Capture { amount, .. }
            | Action::Refund { amount, .. }
            | Action::Transfer { amount, .. }
            | Action::Authorize { amount, .. } => Some(*amount),
            Action::Void { .. } => None,
        }
    }

    fn ids(&self) -> Vec<&str> {
        match self {
            Action::Capture { payment, .. }
            | Action::Void { payment }
            | Action::Refund { payment, .. } => vec![payment],
            Action::Transfer { from, to, .. } => vec![from, to],
            Action::Authorize { account, .. } => vec![account],
        }
    }

    /// The command of the action.
    fn command(&self) -> Command {
        match self.clone() {
            Action::Capture { payment, amount } => Command::Capture { payment, amount },
            Action::Void { payment } => Command::Void { payment },
            Action::Refund { payment, amount } => Command::Refund { payment, amount },
            Action::Transfer { from, to, amount } => Command::Transfer { from, to, amount },
            Action::Authorize { account, amount } => Command::Authorize { account, amount },
        }
    }

    /// The line for the model that writes the request.
    fn describe(&self, start: &Ledger, unit: Unit) -> String {
        match self {
            Action::Capture { payment, amount } => {
                format!("Capture {} of payment {payment}.", money(*amount, unit))
            }
            Action::Void { payment } => format!("Void payment {payment}."),
            Action::Refund { payment, amount } => {
                let full = start
                    .payments
                    .get(payment)
                    .is_some_and(|p| p.captured > 0 && p.refundable() == *amount);
                if full {
                    format!(
                        "Refund the remaining {} of payment {payment} in full.",
                        money(*amount, unit)
                    )
                } else {
                    format!("Refund {} of payment {payment}.", money(*amount, unit))
                }
            }
            Action::Transfer { from, to, amount } => format!(
                "Transfer {} from account {from} to account {to}.",
                money(*amount, unit)
            ),
            Action::Authorize { account, amount } => format!(
                "Authorize a new payment of {} on account {account}.",
                money(*amount, unit)
            ),
        }
    }
}

/// An amount in cents, written in `unit`: `4,000 cents` or `$40`.
pub fn money(cents: u64, unit: Unit) -> String {
    match unit {
        Unit::Cents => format!("{} cents", thousands(cents)),
        Unit::Dollars => format!("${}", thousands(cents / 100)),
    }
}

fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// One generated task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    pub id: usize,
    /// The seed that gives this task with [`generate_one`].
    pub seed: u64,
    pub class: Class,
    /// For a wrong-status task: a permitted payment ID is a prefix of the
    /// requested ID.
    pub prefix_id: bool,
    pub start: Ledger,
    /// The actions that the request asks for.
    pub actions: Vec<Action>,
    pub goal: Goal,
    /// The shortest sequence for a usual task. `None` for the other
    /// classes: their correct result is `ASK_OPERATOR`, so the generator
    /// does not search.
    pub shortest: Option<Vec<Command>>,
    pub shortest_len: Option<usize>,
    /// The amounts that the request names, in cents, in the order of the
    /// actions.
    pub requested_amounts: Vec<u64>,
    pub unit: Unit,
    pub style: Style,
    /// The request text, or `None` before the model writes it.
    pub request: Option<String>,
    /// The sampler seed of the accepted request text.
    pub request_seed: Option<u64>,
    /// The reasons that the check gave for each rejected request text.
    pub request_rejects: Vec<String>,
}

/// The class and the prefix flag of task `id`: first the usual tasks, then
/// the over-limit tasks, then the wrong-status tasks with a prefix ID, then
/// the other wrong-status tasks.
pub fn class_of(id: usize) -> (Class, bool) {
    if id < USUAL {
        (Class::Usual, false)
    } else if id < USUAL + OVER_LIMIT {
        (Class::OverLimit, false)
    } else {
        (Class::WrongStatus, id < USUAL + OVER_LIMIT + PREFIX)
    }
}

/// The sizes of a start state.
#[derive(Debug, Clone, Copy)]
pub struct Sizes {
    pub accounts: (u64, u64),
    pub payments: (u64, u64),
}

/// The sizes of the brief: 2 to 5 accounts and 3 to 20 payments.
pub const SIZES: Sizes = Sizes {
    accounts: (2, 5),
    payments: (3, 20),
};

/// All tasks from `base_seed`.
pub fn generate(base_seed: u64) -> Vec<Task> {
    (0..TASKS).map(|id| generate_task(base_seed, id)).collect()
}

/// Task `id` of `base_seed`. Each try has its own seed. The first seed that
/// gives a task is the seed of the task.
pub fn generate_task(base_seed: u64, id: usize) -> Task {
    let mut seeds = Rng::new(base_seed ^ (id as u64).wrapping_mul(0xA24B_AED4_963E_E407));
    loop {
        let seed = seeds.next_u64();
        if let Some(task) = generate_one(id, seed, SIZES) {
            return task;
        }
    }
}

/// The task of one seed, or `None` if the seed does not give a task of the
/// class of `id`.
pub fn generate_one(id: usize, seed: u64, sizes: Sizes) -> Option<Task> {
    let (class, prefix_id) = class_of(id);
    let mut rng = Rng::new(seed);
    let unit = if rng.chance(1, 2) {
        Unit::Cents
    } else {
        Unit::Dollars
    };
    let style = *rng.pick(&[Style::Short, Style::Long, Style::Support]);
    let min_payments = if prefix_id { 11 } else { sizes.payments.0 };
    let payments = (min_payments, sizes.payments.1.max(min_payments));
    let mut start = random_ledger(&mut rng, sizes.accounts, payments);
    let actions = match class {
        Class::Usual => usual_actions(&mut rng, &start)?,
        Class::OverLimit => vec![over_limit_action(&mut rng, &start)?],
        Class::WrongStatus => vec![wrong_status_action(&mut rng, &mut start, prefix_id)?],
    };
    let goal = goal_of(&start, &actions);
    let shortest = match class {
        Class::Usual => Some(shortest(&start, &goal, MAX_SHORTEST)?),
        Class::OverLimit | Class::WrongStatus => None,
    };
    Some(Task {
        id,
        seed,
        class,
        prefix_id,
        shortest_len: shortest.as_ref().map(Vec::len),
        shortest,
        requested_amounts: actions.iter().filter_map(Action::amount).collect(),
        actions,
        goal,
        start,
        unit,
        style,
        request: None,
        request_seed: None,
        request_rejects: Vec::new(),
    })
}

/// A multiple of 100 in `lo..=hi`, both multiples of 100.
fn cents(rng: &mut Rng, lo: u64, hi: u64) -> u64 {
    100 * rng.range(lo / 100, hi / 100)
}

fn random_ledger(rng: &mut Rng, accounts: (u64, u64), payments: (u64, u64)) -> Ledger {
    let mut ledger = Ledger::default();
    for k in 1..=rng.range(accounts.0, accounts.1) {
        let balance = if rng.chance(1, 5) {
            0
        } else {
            cents(rng, 100, 500_000)
        };
        ledger.accounts.insert(format!("a_{k}"), balance);
    }
    let ids: Vec<String> = ledger.accounts.keys().cloned().collect();
    for k in 1..=rng.range(payments.0, payments.1) {
        let account = rng.pick(&ids).clone();
        let status = *rng.pick(&[
            Status::Authorized,
            Status::Authorized,
            Status::Authorized,
            Status::Captured,
            Status::Captured,
            Status::Captured,
            Status::PartiallyRefunded,
            Status::PartiallyRefunded,
            Status::Refunded,
            Status::Voided,
            Status::Voided,
        ]);
        ledger
            .payments
            .insert(format!("p_{k}"), random_payment(rng, status, account));
    }
    ledger
}

fn random_payment(rng: &mut Rng, status: Status, account: String) -> Payment {
    let authorized = cents(rng, 1_000, 200_000);
    let captured = match status {
        Status::Authorized | Status::Voided => 0,
        _ if rng.chance(2, 3) => authorized,
        _ => cents(rng, 500, authorized),
    };
    let refunded = match status {
        Status::PartiallyRefunded => cents(rng, 100, captured - 100),
        Status::Refunded => captured,
        _ => 0,
    };
    Payment {
        status,
        authorized,
        captured,
        refunded,
        account,
    }
}

/// One or two requested operations for a usual task. An operation is one
/// action, or a capture and then a refund of the same payment. Each
/// operation uses its own payment, and a transfer uses two accounts that
/// nothing else in the task uses. A refund that needs money on its account
/// first is not combined with a transfer, so that the model can take the
/// money from any account. `None` if the state does not permit the
/// selection.
fn usual_actions(rng: &mut Rng, start: &Ledger) -> Option<Vec<Action>> {
    let mut state = start.clone();
    let mut actions = Vec::new();
    let mut used_payments = BTreeSet::new();
    // The accounts of the named payments and of the transfers.
    let mut used_accounts = BTreeSet::new();
    let mut transfer_accounts = BTreeSet::new();
    let mut needs_money = false;
    let mut authorized_once = false;
    for _ in 0..rng.range(1, 2) {
        let kind = *rng.pick(&[
            "capture",
            "capture",
            "void",
            "refund",
            "refund",
            "refund",
            "transfer",
            "transfer",
            "authorize",
            "capture_refund",
        ]);
        let free = |status: &[Status], state: &Ledger| -> Vec<String> {
            state
                .payments
                .iter()
                .filter(|(id, p)| status.contains(&p.status) && !used_payments.contains(*id))
                .map(|(id, _)| id.clone())
                .collect()
        };
        let mut picked = Vec::new();
        match kind {
            "capture" | "void" | "capture_refund" => {
                let ids = free(&[Status::Authorized], &state);
                if ids.is_empty() {
                    return None;
                }
                let payment = rng.pick(&ids).clone();
                let p = &state.payments[&payment];
                if kind == "void" {
                    picked.push(Action::Void { payment });
                } else {
                    let amount = if rng.chance(1, 2) {
                        p.authorized
                    } else {
                        cents(rng, 100, p.authorized)
                    };
                    picked.push(Action::Capture {
                        payment: payment.clone(),
                        amount,
                    });
                    if kind == "capture_refund" {
                        let refund = cents(rng, 100, amount);
                        picked.push(Action::Refund {
                            payment,
                            amount: refund,
                        });
                    }
                }
            }
            "refund" => {
                let ids = free(&[Status::Captured, Status::PartiallyRefunded], &state);
                if ids.is_empty() {
                    return None;
                }
                let payment = rng.pick(&ids).clone();
                let rest = state.payments[&payment].refundable();
                let amount = if rng.chance(1, 2) {
                    rest
                } else {
                    cents(rng, 100, rest)
                };
                picked.push(Action::Refund { payment, amount });
            }
            "transfer" => {
                let free_accounts: Vec<(String, u64)> = state
                    .accounts
                    .iter()
                    .filter(|(id, _)| !used_accounts.contains(*id))
                    .map(|(id, b)| (id.clone(), *b))
                    .collect();
                let from: Vec<&(String, u64)> =
                    free_accounts.iter().filter(|(_, b)| *b >= 100).collect();
                if from.is_empty() || free_accounts.len() < 2 {
                    return None;
                }
                let (from, balance) = (*rng.pick(&from)).clone();
                let to: Vec<&String> = free_accounts
                    .iter()
                    .map(|(id, _)| id)
                    .filter(|id| **id != from)
                    .collect();
                let to = (*rng.pick(&to)).clone();
                let amount = cents(rng, 100, balance);
                picked.push(Action::Transfer { from, to, amount });
            }
            _ => {
                if authorized_once {
                    return None;
                }
                authorized_once = true;
                let ids: Vec<String> = state.accounts.keys().cloned().collect();
                let account = rng.pick(&ids).clone();
                let amount = cents(rng, 1_000, 200_000);
                picked.push(Action::Authorize { account, amount });
            }
        }
        for action in picked {
            match &action {
                Action::Capture { payment, .. }
                | Action::Void { payment }
                | Action::Refund { payment, .. } => {
                    used_payments.insert(payment.clone());
                    let account = &state.payments[payment].account;
                    // The balance condition of a transfer must not include
                    // a capture or a refund.
                    if transfer_accounts.contains(account) {
                        return None;
                    }
                    used_accounts.insert(account.clone());
                }
                Action::Transfer { from, to, .. } => {
                    for account in [from, to] {
                        used_accounts.insert(account.clone());
                        transfer_accounts.insert(account.clone());
                    }
                }
                Action::Authorize { .. } => {}
            }
            let command = action.command();
            if let Command::Refund { payment, amount } = &command {
                let account = state.payments[payment].account.clone();
                let balance = state.accounts[&account];
                if balance < *amount {
                    // Money for the refund, as the model must find it.
                    needs_money = true;
                    let shortfall = amount - balance;
                    let source = state
                        .accounts
                        .iter()
                        .find(|(id, b)| **b >= shortfall && **id != account)
                        .map(|(id, _)| id.clone())?;
                    state
                        .execute(&Command::Transfer {
                            from: source,
                            to: account,
                            amount: shortfall,
                        })
                        .ok()?;
                }
            }
            state.execute(&command).ok()?;
            actions.push(action);
        }
    }
    let has_transfer = actions.iter().any(|a| matches!(a, Action::Transfer { .. }));
    if needs_money && has_transfer {
        return None;
    }
    Some(actions)
}

/// One action with an amount above its limit in `start`: a capture above
/// the authorized amount, a refund above the rest of the capture, or a
/// transfer above the balance. The amount is between the limit + 100 and 2
/// times the limit.
fn over_limit_action(rng: &mut Rng, start: &Ledger) -> Option<Action> {
    let above = |rng: &mut Rng, limit: u64| limit + cents(rng, 100, limit.max(100));
    match *rng.pick(&["capture", "refund", "transfer"]) {
        "capture" => {
            let ids = ids_with(start, &[Status::Authorized]);
            if ids.is_empty() {
                return None;
            }
            let payment = rng.pick(&ids).clone();
            let amount = above(rng, start.payments[&payment].authorized);
            Some(Action::Capture { payment, amount })
        }
        "refund" => {
            let ids = ids_with(start, &[Status::Captured, Status::PartiallyRefunded]);
            if ids.is_empty() {
                return None;
            }
            let payment = rng.pick(&ids).clone();
            let amount = above(rng, start.payments[&payment].refundable());
            Some(Action::Refund { payment, amount })
        }
        _ => {
            let from: Vec<String> = start
                .accounts
                .iter()
                .filter(|(_, b)| **b >= 100)
                .map(|(id, _)| id.clone())
                .collect();
            if from.is_empty() {
                return None;
            }
            let from = rng.pick(&from).clone();
            let to: Vec<String> = start
                .accounts
                .keys()
                .filter(|id| **id != from)
                .cloned()
                .collect();
            let to = rng.pick(&to).clone();
            let amount = above(rng, start.accounts[&from]);
            Some(Action::Transfer { from, to, amount })
        }
    }
    .filter(|action| {
        matches!(
            start.permits(&action.command()),
            Err(Refusal::AmountOutOfRange { .. })
        )
    })
}

fn ids_with(ledger: &Ledger, statuses: &[Status]) -> Vec<String> {
    ledger
        .payments
        .iter()
        .filter(|(_, p)| statuses.contains(&p.status))
        .map(|(id, _)| id.clone())
        .collect()
}

/// One action on a payment whose status does not permit it: a refund of a
/// voided or refunded payment, or a capture or a void of a payment that is
/// not authorized. With `prefix_id`, the generator sets `p_1` to a
/// permitted status and selects the payment from `p_10` to `p_19`.
/// Without it, no permitted payment ID is a prefix of the requested ID.
fn wrong_status_action(rng: &mut Rng, start: &mut Ledger, prefix_id: bool) -> Option<Action> {
    let kind = *rng.pick(&["refund", "capture", "void"]);
    let (permitted, wrong): (&[Status], &[Status]) = match kind {
        "refund" => (
            &[Status::Captured, Status::PartiallyRefunded],
            &[Status::Voided, Status::Refunded],
        ),
        _ => (
            &[Status::Authorized],
            &[
                Status::Captured,
                Status::PartiallyRefunded,
                Status::Refunded,
                Status::Voided,
            ],
        ),
    };
    if prefix_id {
        let p_1 = &start.payments["p_1"];
        if !permitted.contains(&p_1.status) {
            let status = *rng.pick(permitted);
            let account = p_1.account.clone();
            start
                .payments
                .insert("p_1".to_string(), random_payment(rng, status, account));
        }
    }
    let permitted_ids = ids_with(start, permitted);
    let candidates: Vec<String> = start
        .payments
        .iter()
        .filter(|(id, p)| {
            let prefix = permitted_ids
                .iter()
                .any(|other| *other != **id && id.starts_with(other.as_str()));
            wrong.contains(&p.status)
                && prefix == prefix_id
                && (!prefix_id || id.starts_with("p_1"))
        })
        .map(|(id, _)| id.clone())
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let payment = rng.pick(&candidates).clone();
    let p = &start.payments[&payment];
    let action = match kind {
        "refund" => Action::Refund {
            amount: if p.captured > 0 {
                p.captured
            } else {
                p.authorized
            },
            payment,
        },
        "capture" => Action::Capture {
            amount: p.authorized,
            payment,
        },
        _ => Action::Void { payment },
    };
    Some(action).filter(|action| {
        matches!(
            start.permits(&action.command()),
            Err(Refusal::WrongStatus { .. })
        )
    })
}

/// The goal of the actions, as conditions on the state after them.
///
/// - A capture: the captured amount.
/// - A void: the status `voided`.
/// - A refund: the total refunded amount of the payment.
/// - A transfer: the balances of both accounts after it. The balance of
///   `from` is 0 if the amount is above it (an over-limit task).
/// - An authorization: a new payment with the account and the amount.
fn goal_of(start: &Ledger, actions: &[Action]) -> Goal {
    let mut conditions = Vec::new();
    let mut refunded: std::collections::BTreeMap<&str, u64> = Default::default();
    for action in actions {
        match action {
            Action::Capture { payment, amount } => conditions.push(Condition::Captured {
                payment: payment.clone(),
                amount: *amount,
            }),
            Action::Void { payment } => conditions.push(Condition::Status {
                payment: payment.clone(),
                status: Status::Voided,
            }),
            Action::Refund { payment, amount } => {
                let before = start.payments.get(payment).map_or(0, |p| p.refunded);
                *refunded.entry(payment).or_insert(before) += amount;
            }
            Action::Transfer { from, to, amount } => {
                let balance = |account: &str| start.accounts.get(account).copied().unwrap_or(0);
                conditions.push(Condition::Balance {
                    account: from.clone(),
                    amount: balance(from).saturating_sub(*amount),
                });
                conditions.push(Condition::Balance {
                    account: to.clone(),
                    amount: balance(to) + amount,
                });
            }
            Action::Authorize { account, amount } => conditions.push(Condition::NewPayment {
                account: account.clone(),
                amount: *amount,
            }),
        }
    }
    for (payment, amount) in refunded {
        conditions.push(Condition::Refunded {
            payment: payment.to_string(),
            amount,
        });
    }
    Goal { conditions }
}

/// The system message for the model that writes the requests.
pub const REQUEST_SYSTEM: &str =
    "You write realistic messages from the operators of a payments company.";

/// The user message for the model that writes the request of `task`.
pub fn request_prompt(task: &Task) -> String {
    let mut text = String::from(
        "Write the message that a payments operator sends to an assistant. \
         The assistant controls a payments ledger.\n\nThe operator asks for this:\n",
    );
    for action in &task.actions {
        text.push_str("- ");
        text.push_str(&action.describe(&task.start, task.unit));
        text.push('\n');
    }
    text.push('\n');
    text.push_str(match task.style {
        Style::Short => "Write one short sentence.",
        Style::Long => "Write three to five sentences. Give a reason for the request.",
        Style::Support => {
            "Write as a customer support agent who passes on a request from a customer. \
             Use the words of a support ticket."
        }
    });
    // A void has no amount. With a unit line, the model writes one.
    if !task.requested_amounts.is_empty() {
        text.push('\n');
        text.push_str(match task.unit {
            Unit::Cents => "Write each amount in digits, then the word cents.",
            Unit::Dollars => "Write each amount in digits, after a $ sign.",
        });
    }
    text.push_str(
        "\nWrite each ID exactly as above.\
         \nWrite no other amounts, numbers, IDs, or dates.\
         \nDo not say how to do it.\
         \nWrite only the message, with no quotes and no heading.",
    );
    text
}

/// One number in a request text.
#[derive(Debug, PartialEq)]
struct Number {
    /// The value with no thousands separators, times 100 if it has 2
    /// decimals.
    hundredths: Option<u64>,
    value: u64,
    dollar_sign: bool,
}

/// The numbers of `text` that are not part of an ID such as `p_17`.
fn numbers(text: &str) -> Vec<Number> {
    let bytes = text.as_bytes();
    let mut found = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let in_id = i > 0 && (bytes[i - 1] == b'_' || bytes[i - 1].is_ascii_alphabetic());
        let start = i;
        let mut digits = String::new();
        while i < bytes.len() {
            if bytes[i].is_ascii_digit() {
                digits.push(bytes[i] as char);
                i += 1;
            } else if bytes[i] == b','
                && bytes
                    .get(i + 1..i + 4)
                    .is_some_and(|group| group.iter().all(u8::is_ascii_digit))
                && !bytes.get(i + 4).is_some_and(u8::is_ascii_digit)
            {
                i += 1;
            } else {
                break;
            }
        }
        let mut hundredths = None;
        if bytes.get(i) == Some(&b'.')
            && bytes.get(i + 1).is_some_and(u8::is_ascii_digit)
            && bytes.get(i + 2).is_some_and(u8::is_ascii_digit)
            && !bytes.get(i + 3).is_some_and(u8::is_ascii_digit)
        {
            let cents: u64 = text[i + 1..i + 3].parse().expect("two digits");
            hundredths = digits.parse::<u64>().ok().map(|whole| whole * 100 + cents);
            i += 3;
        }
        if in_id {
            continue;
        }
        let dollar_sign = start > 0 && bytes[start - 1] == b'$';
        if let Ok(value) = digits.parse() {
            found.push(Number {
                hundredths,
                value,
                dollar_sign,
            });
        }
    }
    found
}

/// The IDs of the form `p_<n>` or `a_<n>` in `text`.
fn ids(text: &str) -> BTreeSet<String> {
    let bytes = text.as_bytes();
    let mut found = BTreeSet::new();
    for i in 0..bytes.len().saturating_sub(2) {
        let starts = (bytes[i] == b'p' || bytes[i] == b'a')
            && bytes[i + 1] == b'_'
            && bytes[i + 2].is_ascii_digit()
            && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric());
        if starts {
            let end = (i + 2..bytes.len())
                .find(|&j| !bytes[j].is_ascii_digit())
                .unwrap_or(bytes.len());
            found.insert(text[i..end].to_string());
        }
    }
    found
}

/// Checks a request text: it names each ID of the actions and no other
/// ID, and each requested amount in the unit of the task and no other
/// number. `Err` gives the reason.
pub fn check_request(task: &Task, text: &str) -> Result<(), String> {
    if text.trim().is_empty() {
        return Err("empty".to_string());
    }
    let expected_ids: BTreeSet<String> = task
        .actions
        .iter()
        .flat_map(Action::ids)
        .map(str::to_string)
        .collect();
    let found_ids = ids(text);
    if found_ids != expected_ids {
        return Err(format!("IDs {found_ids:?}, expected {expected_ids:?}"));
    }
    let mut amounts = Vec::new();
    for number in numbers(text) {
        let cents = match task.unit {
            Unit::Cents if number.hundredths.is_none() && !number.dollar_sign => number.value,
            Unit::Dollars if number.dollar_sign => number.hundredths.unwrap_or(number.value * 100),
            _ => {
                return Err(format!(
                    "a number not in the unit {:?}: {}",
                    task.unit, number.value
                ));
            }
        };
        amounts.push(cents);
    }
    let found: BTreeSet<u64> = amounts.iter().copied().collect();
    let expected: BTreeSet<u64> = task.requested_amounts.iter().copied().collect();
    if found != expected {
        return Err(format!("amounts {found:?} in cents, expected {expected:?}"));
    }
    if task.unit == Unit::Cents && !expected.is_empty() && !text.to_lowercase().contains("cent") {
        return Err("no word cents".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::{Step, shortest_full};

    /// The tasks of seed 1, made once for all tests: a debug build takes
    /// some seconds for the searches.
    fn tasks() -> &'static [Task] {
        static TASKS: std::sync::OnceLock<Vec<Task>> = std::sync::OnceLock::new();
        TASKS.get_or_init(|| generate(1))
    }

    fn replay(task: &Task) -> Ledger {
        let mut state = task.start.clone();
        for command in task.shortest.as_ref().expect("a usual task") {
            assert_eq!(
                state.execute(command),
                Ok(Step::Continue),
                "task {}",
                task.id
            );
        }
        state
    }

    #[test]
    fn the_classes_have_the_counts_of_the_brief() {
        let tasks = tasks();
        let count = |class: Class, prefix: Option<bool>| {
            tasks
                .iter()
                .filter(|t| t.class == class && (prefix.is_none() || Some(t.prefix_id) == prefix))
                .count()
        };
        assert_eq!(count(Class::Usual, None), USUAL);
        assert_eq!(count(Class::OverLimit, None), OVER_LIMIT);
        assert_eq!(count(Class::WrongStatus, None), WRONG_STATUS);
        assert_eq!(count(Class::WrongStatus, Some(true)), PREFIX);
    }

    #[test]
    fn the_same_seed_gives_the_same_tasks() {
        // A second run of seed 1 gives the first task of each try again.
        for task in tasks() {
            assert_eq!(generate_one(task.id, task.seed, SIZES).as_ref(), Some(task));
        }
        assert_ne!(generate_task(1, 0), generate_task(2, 0));
    }

    #[test]
    fn each_task_meets_the_rules_of_its_class() {
        for task in tasks() {
            let state = &task.start;
            for (id, p) in &state.payments {
                assert!(
                    p.refunded <= p.captured && p.captured <= p.authorized,
                    "{id}"
                );
            }
            assert!((2..=5).contains(&state.accounts.len()));
            assert!((3..=20).contains(&state.payments.len()));
            let first = task.actions[0].command();
            match task.class {
                Class::Usual => {
                    let len = task.shortest_len.expect("a length");
                    assert!((1..=MAX_SHORTEST).contains(&len), "task {}", task.id);
                    assert!(task.goal.is_met(state, &replay(task)), "task {}", task.id);
                }
                Class::OverLimit => {
                    assert!(matches!(
                        state.permits(&first),
                        Err(Refusal::AmountOutOfRange { .. })
                    ));
                    assert_eq!(task.shortest, None);
                }
                Class::WrongStatus => {
                    assert!(matches!(
                        state.permits(&first),
                        Err(Refusal::WrongStatus { .. })
                    ));
                    let id = task.actions[0].ids()[0];
                    // Only the amount can be out of range for a permitted
                    // status.
                    let prefix = state.payments.keys().any(|other| {
                        other != id
                            && id.starts_with(other.as_str())
                            && matches!(
                                state.permits(&with_payment(&first, other)),
                                Ok(()) | Err(Refusal::AmountOutOfRange { .. })
                            )
                    });
                    assert_eq!(prefix, task.prefix_id, "task {}", task.id);
                }
            }
        }
    }

    /// `command` with another payment ID.
    fn with_payment(command: &Command, other: &str) -> Command {
        let other = other.to_string();
        match command.clone() {
            Command::Capture { amount, .. } => Command::Capture {
                payment: other,
                amount,
            },
            Command::Refund { amount, .. } => Command::Refund {
                payment: other,
                amount,
            },
            Command::Void { .. } => Command::Void { payment: other },
            command => command,
        }
    }

    /// The reduced search gives the length of the full search. On small
    /// ledgers, the full search can check each length up to 2, so the test
    /// proves the length for shortest lengths up to 3.
    #[test]
    fn the_reduced_search_gives_the_full_length() {
        let small = Sizes {
            accounts: (2, 3),
            payments: (3, 6),
        };
        let mut checked = [0usize; MAX_SHORTEST + 1];
        let mut seeds = Rng::new(99);
        while checked[1..=3].iter().sum::<usize>() < 300 {
            let Some(task) = generate_one(0, seeds.next_u64(), small) else {
                continue;
            };
            let len = task.shortest_len.expect("a usual task");
            checked[len] += 1;
            if len > 3 {
                continue;
            }
            assert_eq!(
                shortest_full(&task.start, &task.goal, len - 1),
                None,
                "a shorter sequence for {:?}",
                task.goal
            );
        }
        assert!(checked[2] > 0 && checked[3] > 0, "{checked:?}");
    }

    fn task(actions: Vec<Action>, unit: Unit) -> Task {
        let mut task = generate_task(1, 0);
        task.requested_amounts = actions.iter().filter_map(Action::amount).collect();
        task.actions = actions;
        task.unit = unit;
        task
    }

    #[test]
    fn the_check_accepts_the_ids_and_amounts_in_the_unit() {
        let refund = vec![Action::Refund {
            payment: "p_17".to_string(),
            amount: 400_000,
        }];
        let cents = task(refund.clone(), Unit::Cents);
        assert_eq!(
            check_request(&cents, "Please refund 400,000 cents on p_17."),
            Ok(())
        );
        assert_eq!(
            check_request(&cents, "Please refund 400000 cents on p_17."),
            Ok(())
        );
        let dollars = task(refund, Unit::Dollars);
        assert_eq!(
            check_request(&dollars, "Refund $4,000 on p_17, thanks."),
            Ok(())
        );
        assert_eq!(check_request(&dollars, "Refund $4,000.00 on p_17."), Ok(()));
        let transfer = task(
            vec![Action::Transfer {
                from: "a_1".to_string(),
                to: "a_2".to_string(),
                amount: 2_500,
            }],
            Unit::Cents,
        );
        assert_eq!(
            check_request(&transfer, "Move 2,500 cents from a_1 to a_2."),
            Ok(())
        );
    }

    #[test]
    fn the_check_rejects_a_wrong_id_amount_or_unit() {
        let refund = vec![Action::Refund {
            payment: "p_17".to_string(),
            amount: 400_000,
        }];
        let cents = task(refund.clone(), Unit::Cents);
        let dollars = task(refund, Unit::Dollars);
        for (task, text) in [
            (&cents, "Refund 400,000 cents on p_1."),
            (&cents, "Refund 400,000 cents on p_17 and p_2."),
            (&cents, "Refund 40,000 cents on p_17."),
            (&cents, "Refund $4,000 on p_17."),
            (&cents, "Refund 400,000 on p_17."),
            (&cents, "Refund 400,000 cents on p_17 within 24 hours."),
            (&dollars, "Refund 4,000 dollars on p_17."),
            (&dollars, "Refund $400,000 on p_17."),
            (&dollars, "Refund $4,000.50 on p_17."),
            (&dollars, ""),
        ] {
            assert!(check_request(task, text).is_err(), "{text}");
        }
    }

    #[test]
    fn money_has_thousands_separators() {
        assert_eq!(money(400_000, Unit::Cents), "400,000 cents");
        assert_eq!(money(400_000, Unit::Dollars), "$4,000");
        assert_eq!(money(100, Unit::Dollars), "$1");
        assert_eq!(money(2_500, Unit::Cents), "2,500 cents");
    }
}
