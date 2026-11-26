//! The simulated payments ledger: the state, the 7 commands, their guards
//! and effects, and a breadth-first search for the shortest command
//! sequence to a goal.
//!
//! All amounts are integers in cents.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};

/// The largest amount of one `AUTHORIZE`, in cents.
pub const MAX_AUTHORIZE: u64 = 1_000_000;

/// The status of a payment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Authorized,
    Captured,
    Voided,
    Refunded,
    PartiallyRefunded,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Status::Authorized => "authorized",
            Status::Captured => "captured",
            Status::Voided => "voided",
            Status::Refunded => "refunded",
            Status::PartiallyRefunded => "partially_refunded",
        })
    }
}

/// One payment record.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Payment {
    pub status: Status,
    /// The authorized amount.
    pub authorized: u64,
    /// The captured amount, 0 before capture.
    pub captured: u64,
    /// The total refunded amount.
    pub refunded: u64,
    /// The merchant account that receives the capture.
    pub account: String,
}

impl Payment {
    /// The amount that a refund can still return: captured minus refunded.
    pub fn refundable(&self) -> u64 {
        self.captured - self.refunded
    }
}

/// The state of the ledger.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Ledger {
    /// Account ID to balance.
    pub accounts: BTreeMap<String, u64>,
    /// Payment ID to payment record.
    pub payments: BTreeMap<String, Payment>,
}

/// One command of the model. The JSON form is the output format of the
/// model, for example `{"cmd":"REFUND","payment":"p_17","amount":2500}`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Command {
    Authorize {
        account: String,
        amount: u64,
    },
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
    AskOperator {
        question: String,
    },
    Done,
}

impl Command {
    /// True for the commands that move money or change a payment.
    pub fn moves_money(&self) -> bool {
        !matches!(self, Command::AskOperator { .. } | Command::Done)
    }
}

/// The reason that a guard refuses a command.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Refusal {
    #[error("account {0} does not exist")]
    UnknownAccount(String),
    #[error("payment {0} does not exist")]
    UnknownPayment(String),
    #[error("payment {payment} is {status}")]
    WrongStatus { payment: String, status: Status },
    #[error("amount {amount} is not in the range 1 to {max}")]
    AmountOutOfRange { amount: u64, max: u64 },
    #[error("a transfer from account {0} to the same account")]
    SameAccount(String),
}

/// What the task does after a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// The task continues with the next command.
    Continue,
    /// `ASK_OPERATOR` or `DONE`: the task ends.
    End,
}

fn in_range(amount: u64, max: u64) -> Result<(), Refusal> {
    if (1..=max).contains(&amount) {
        Ok(())
    } else {
        Err(Refusal::AmountOutOfRange { amount, max })
    }
}

impl Ledger {
    fn balance(&self, account: &str) -> Result<u64, Refusal> {
        self.accounts
            .get(account)
            .copied()
            .ok_or_else(|| Refusal::UnknownAccount(account.to_string()))
    }

    fn payment(&self, payment: &str) -> Result<&Payment, Refusal> {
        self.payments
            .get(payment)
            .ok_or_else(|| Refusal::UnknownPayment(payment.to_string()))
    }

    /// Payment `payment` with one of `statuses`.
    fn payment_in(&self, payment: &str, statuses: &[Status]) -> Result<&Payment, Refusal> {
        let record = self.payment(payment)?;
        if statuses.contains(&record.status) {
            Ok(record)
        } else {
            Err(Refusal::WrongStatus {
                payment: payment.to_string(),
                status: record.status,
            })
        }
    }

    /// The ID that the next `AUTHORIZE` gives: the lowest `p_<k>`, from
    /// k = 1, that is not in use.
    pub fn next_payment_id(&self) -> String {
        (1..)
            .map(|k| format!("p_{k}"))
            .find(|id| !self.payments.contains_key(id))
            .expect("a free payment ID")
    }

    /// Checks the guard of `command` in this state.
    pub fn permits(&self, command: &Command) -> Result<(), Refusal> {
        match command {
            Command::Authorize { account, amount } => {
                self.balance(account)?;
                in_range(*amount, MAX_AUTHORIZE)
            }
            Command::Capture { payment, amount } => {
                let record = self.payment_in(payment, &[Status::Authorized])?;
                in_range(*amount, record.authorized)
            }
            Command::Void { payment } => {
                self.payment_in(payment, &[Status::Authorized]).map(|_| ())
            }
            Command::Refund { payment, amount } => {
                let record =
                    self.payment_in(payment, &[Status::Captured, Status::PartiallyRefunded])?;
                let balance = self.balance(&record.account)?;
                in_range(*amount, record.refundable().min(balance))
            }
            Command::Transfer { from, to, amount } => {
                let balance = self.balance(from)?;
                self.balance(to)?;
                if from == to {
                    return Err(Refusal::SameAccount(from.clone()));
                }
                in_range(*amount, balance)
            }
            Command::AskOperator { .. } | Command::Done => Ok(()),
        }
    }

    /// Checks the guard of `command`, then applies its effect. A refused
    /// command does not change the state.
    pub fn execute(&mut self, command: &Command) -> Result<Step, Refusal> {
        self.permits(command)?;
        match command {
            Command::Authorize { account, amount } => {
                let id = self.next_payment_id();
                self.payments.insert(
                    id,
                    Payment {
                        status: Status::Authorized,
                        authorized: *amount,
                        captured: 0,
                        refunded: 0,
                        account: account.clone(),
                    },
                );
            }
            Command::Capture { payment, amount } => {
                let record = self.payments.get_mut(payment).expect("permitted");
                record.captured = *amount;
                record.status = Status::Captured;
                let account = record.account.clone();
                *self.accounts.get_mut(&account).expect("permitted") += amount;
            }
            Command::Void { payment } => {
                self.payments.get_mut(payment).expect("permitted").status = Status::Voided;
            }
            Command::Refund { payment, amount } => {
                let record = self.payments.get_mut(payment).expect("permitted");
                record.refunded += amount;
                record.status = if record.refunded == record.captured {
                    Status::Refunded
                } else {
                    Status::PartiallyRefunded
                };
                let account = record.account.clone();
                *self.accounts.get_mut(&account).expect("permitted") -= amount;
            }
            Command::Transfer { from, to, amount } => {
                *self.accounts.get_mut(from).expect("permitted") -= amount;
                *self.accounts.get_mut(to).expect("permitted") += amount;
            }
            Command::AskOperator { .. } | Command::Done => return Ok(Step::End),
        }
        Ok(Step::Continue)
    }
}

/// One condition of a goal, on the state after the commands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Condition {
    /// The payment has this status.
    Status { payment: String, status: Status },
    /// The captured amount of the payment is `amount`.
    Captured { payment: String, amount: u64 },
    /// The total refunded amount of the payment is `amount`.
    Refunded { payment: String, amount: u64 },
    /// The balance of the account is `amount`.
    Balance { account: String, amount: u64 },
    /// A payment that is not in the start state has this account and this
    /// authorized amount.
    NewPayment { account: String, amount: u64 },
}

/// A goal: all its conditions are true.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Goal {
    pub conditions: Vec<Condition>,
}

impl Goal {
    /// True if each condition is true in `state`. `start` is the state at
    /// the start of the task, for [`Condition::NewPayment`].
    pub fn is_met(&self, start: &Ledger, state: &Ledger) -> bool {
        self.conditions.iter().all(|condition| match condition {
            Condition::Status { payment, status } => state
                .payments
                .get(payment)
                .is_some_and(|p| p.status == *status),
            Condition::Captured { payment, amount } => state
                .payments
                .get(payment)
                .is_some_and(|p| p.captured == *amount),
            Condition::Refunded { payment, amount } => state
                .payments
                .get(payment)
                .is_some_and(|p| p.refunded == *amount),
            Condition::Balance { account, amount } => state.accounts.get(account) == Some(amount),
            Condition::NewPayment { account, amount } => state.payments.iter().any(|(id, p)| {
                !start.payments.contains_key(id) && p.account == *account && p.authorized == *amount
            }),
        })
    }

    /// The amounts that the goal names, and the amounts that it needs from
    /// `start`: the change of each balance it names, and the rest of each
    /// refund total it names.
    fn amounts(&self, start: &Ledger) -> BTreeSet<u64> {
        let mut amounts = BTreeSet::new();
        for condition in &self.conditions {
            match condition {
                Condition::Status { .. } => {}
                Condition::Captured { amount, .. } | Condition::NewPayment { amount, .. } => {
                    amounts.insert(*amount);
                }
                Condition::Refunded { payment, amount } => {
                    amounts.insert(*amount);
                    if let Some(p) = start.payments.get(payment) {
                        amounts.insert(amount.saturating_sub(p.refunded));
                    }
                }
                Condition::Balance { account, amount } => {
                    amounts.insert(*amount);
                    if let Some(balance) = start.accounts.get(account) {
                        amounts.insert(amount.abs_diff(*balance));
                    }
                }
            }
        }
        amounts.remove(&0);
        amounts
    }

    /// The payments and accounts that the goal names, for the reduced
    /// search.
    fn focus(&self, start: &Ledger) -> Focus {
        let mut focus = Focus::default();
        for condition in &self.conditions {
            match condition {
                Condition::Status { payment, .. }
                | Condition::Captured { payment, .. }
                | Condition::Refunded { payment, .. } => {
                    focus.payments.insert(payment.clone());
                    if let Some(p) = start.payments.get(payment) {
                        focus.accounts.insert(p.account.clone());
                    }
                }
                Condition::Balance { account, .. } => {
                    focus.accounts.insert(account.clone());
                }
                Condition::NewPayment { account, amount } => {
                    focus.accounts.insert(account.clone());
                    focus.new_payments.insert((account.clone(), *amount));
                }
            }
        }
        focus
    }
}

/// The objects that a goal names.
#[derive(Debug, Default)]
struct Focus {
    payments: BTreeSet<String>,
    accounts: BTreeSet<String>,
    /// The account and the amount of each `NewPayment` condition.
    new_payments: BTreeSet<(String, u64)>,
}

/// The money commands that the full search tries in `state`: each command with
/// each amount of `amounts`, and the amounts that `state` gives (the
/// authorized amount of each authorized payment, and the refundable amount
/// of each refundable payment). Only the permitted commands.
fn moves(state: &Ledger, amounts: &BTreeSet<u64>) -> Vec<Command> {
    let mut amounts = amounts.clone();
    for p in state.payments.values() {
        match p.status {
            Status::Authorized => {
                amounts.insert(p.authorized);
            }
            Status::Captured | Status::PartiallyRefunded => {
                amounts.insert(p.refundable());
            }
            Status::Voided | Status::Refunded => {}
        }
    }
    let mut moves = Vec::new();
    for payment in state.payments.keys() {
        moves.push(Command::Void {
            payment: payment.clone(),
        });
        for &amount in &amounts {
            let payment = payment.clone();
            moves.push(Command::Capture {
                payment: payment.clone(),
                amount,
            });
            moves.push(Command::Refund { payment, amount });
        }
    }
    for account in state.accounts.keys() {
        for &amount in &amounts {
            moves.push(Command::Authorize {
                account: account.clone(),
                amount,
            });
            for to in state.accounts.keys() {
                moves.push(Command::Transfer {
                    from: account.clone(),
                    to: to.clone(),
                    amount,
                });
            }
        }
    }
    moves.retain(|command| state.permits(command).is_ok());
    moves
}

/// The money commands that the reduced search tries in `state`. They are
/// the commands of [`moves`] that can change what the goal names:
///
/// - `CAPTURE`, `VOID`, and `REFUND` only on the payments that the goal
///   names, and on the payments that are not in `start`.
/// - `AUTHORIZE` only with the account and the amount of a `NewPayment`
///   condition.
/// - `TRANSFER` only to or from an account that the goal names, which
///   includes the account of each payment that it names.
///
/// The amounts are those of [`moves`], from these payments only, and the
/// amount that each refund needs on top of the balance of its account.
fn reduced_moves(
    state: &Ledger,
    start: &Ledger,
    focus: &Focus,
    amounts: &BTreeSet<u64>,
) -> Vec<Command> {
    let payments: Vec<&String> = state
        .payments
        .keys()
        .filter(|id| focus.payments.contains(*id) || !start.payments.contains_key(*id))
        .collect();
    let mut amounts = amounts.clone();
    for id in &payments {
        let p = &state.payments[*id];
        match p.status {
            Status::Authorized => {
                amounts.insert(p.authorized);
            }
            Status::Captured | Status::PartiallyRefunded => {
                amounts.insert(p.refundable());
                let balance = state.accounts.get(&p.account).copied().unwrap_or(0);
                if p.refundable() > balance {
                    amounts.insert(p.refundable() - balance);
                }
            }
            Status::Voided | Status::Refunded => {}
        }
    }
    let mut moves = Vec::new();
    for payment in payments {
        moves.push(Command::Void {
            payment: payment.clone(),
        });
        for &amount in &amounts {
            moves.push(Command::Capture {
                payment: payment.clone(),
                amount,
            });
            moves.push(Command::Refund {
                payment: payment.clone(),
                amount,
            });
        }
    }
    for (account, amount) in &focus.new_payments {
        moves.push(Command::Authorize {
            account: account.clone(),
            amount: *amount,
        });
    }
    for from in state.accounts.keys() {
        for to in state.accounts.keys() {
            if !focus.accounts.contains(from) && !focus.accounts.contains(to) {
                continue;
            }
            for &amount in &amounts {
                moves.push(Command::Transfer {
                    from: from.clone(),
                    to: to.clone(),
                    amount,
                });
            }
        }
    }
    moves.retain(|command| state.permits(command).is_ok());
    moves
}

/// The shortest sequence of money commands from `start` to `goal`, with at
/// most `max_len` commands, by breadth-first search. The sequence does not
/// include `DONE`. `None` if no sequence of that length gets to the goal.
///
/// The search tries only the commands of [`reduced_moves`]. The full
/// search ([`shortest_full`]) does not end in useful time on a ledger with 5
/// accounts and 20 payments at a length of 3. A sequence with other
/// commands can be shorter, so the length is an upper bound in general.
/// The tests compare the two searches up to a length of 3. Of two
/// sequences with the same length, the search gives the first in the order
/// of the moves.
pub fn shortest(start: &Ledger, goal: &Goal, max_len: usize) -> Option<Vec<Command>> {
    let amounts = goal.amounts(start);
    let focus = goal.focus(start);
    search(start, goal, max_len, |state| {
        reduced_moves(state, start, &focus, &amounts)
    })
}

/// [`shortest`] with all the commands of [`moves`]. The length is exact for
/// the amounts of [`moves`], but the search is slow on a large ledger.
pub fn shortest_full(start: &Ledger, goal: &Goal, max_len: usize) -> Option<Vec<Command>> {
    let amounts = goal.amounts(start);
    search(start, goal, max_len, |state| moves(state, &amounts))
}

fn search(
    start: &Ledger,
    goal: &Goal,
    max_len: usize,
    moves: impl Fn(&Ledger) -> Vec<Command>,
) -> Option<Vec<Command>> {
    if goal.is_met(start, start) {
        return Some(Vec::new());
    }
    // Each node: the state, the node before it, the command, the length.
    let mut nodes: Vec<(Ledger, usize, Command, usize)> = Vec::new();
    let mut seen = HashSet::from([start.clone()]);
    let mut queue: VecDeque<Option<usize>> = VecDeque::from([None]);
    while let Some(parent) = queue.pop_front() {
        let (state, len) = match parent {
            None => (start, 0),
            Some(i) => (&nodes[i].0, nodes[i].3),
        };
        if len == max_len {
            continue;
        }
        let mut found = None;
        let mut next = Vec::new();
        for command in moves(state) {
            let mut after = state.clone();
            after
                .execute(&command)
                .expect("moves gives permitted commands");
            if !seen.insert(after.clone()) {
                continue;
            }
            let met = goal.is_met(start, &after);
            next.push((after, parent.unwrap_or(usize::MAX), command, len + 1));
            if met {
                found = Some(nodes.len() + next.len() - 1);
                break;
            }
        }
        let first = nodes.len();
        nodes.extend(next);
        queue.extend((first..nodes.len()).map(Some));
        if let Some(mut i) = found {
            let mut sequence = Vec::new();
            loop {
                sequence.push(nodes[i].2.clone());
                match nodes[i].1 {
                    usize::MAX => break,
                    before => i = before,
                }
            }
            sequence.reverse();
            return Some(sequence);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payment(
        status: Status,
        authorized: u64,
        captured: u64,
        refunded: u64,
        account: &str,
    ) -> Payment {
        Payment {
            status,
            authorized,
            captured,
            refunded,
            account: account.to_string(),
        }
    }

    /// Three accounts and six payments, one or more in each status.
    fn ledger() -> Ledger {
        Ledger {
            accounts: BTreeMap::from([
                ("a_1".to_string(), 50_000),
                ("a_2".to_string(), 10_000),
                ("a_3".to_string(), 0),
            ]),
            payments: BTreeMap::from([
                (
                    "p_1".to_string(),
                    payment(Status::Authorized, 3_000, 0, 0, "a_1"),
                ),
                (
                    "p_2".to_string(),
                    payment(Status::Captured, 5_000, 5_000, 0, "a_2"),
                ),
                (
                    "p_3".to_string(),
                    payment(Status::Voided, 2_000, 0, 0, "a_1"),
                ),
                (
                    "p_4".to_string(),
                    payment(Status::PartiallyRefunded, 4_000, 4_000, 1_000, "a_1"),
                ),
                (
                    "p_5".to_string(),
                    payment(Status::Refunded, 1_000, 1_000, 1_000, "a_2"),
                ),
                (
                    "p_6".to_string(),
                    payment(Status::Captured, 2_000, 2_000, 0, "a_3"),
                ),
            ]),
        }
    }

    fn s(text: &str) -> String {
        text.to_string()
    }

    fn authorize(account: &str, amount: u64) -> Command {
        Command::Authorize {
            account: s(account),
            amount,
        }
    }
    fn capture(payment: &str, amount: u64) -> Command {
        Command::Capture {
            payment: s(payment),
            amount,
        }
    }
    fn void(payment: &str) -> Command {
        Command::Void {
            payment: s(payment),
        }
    }
    fn refund(payment: &str, amount: u64) -> Command {
        Command::Refund {
            payment: s(payment),
            amount,
        }
    }
    fn transfer(from: &str, to: &str, amount: u64) -> Command {
        Command::Transfer {
            from: s(from),
            to: s(to),
            amount,
        }
    }

    fn out_of_range(amount: u64, max: u64) -> Result<(), Refusal> {
        Err(Refusal::AmountOutOfRange { amount, max })
    }

    fn wrong_status(payment: &str, status: Status) -> Result<(), Refusal> {
        Err(Refusal::WrongStatus {
            payment: s(payment),
            status,
        })
    }

    // The guards: one pass and the refusals of each command.

    #[test]
    fn authorize_needs_an_account_and_1_to_1_000_000() {
        let l = ledger();
        assert_eq!(l.permits(&authorize("a_3", 1)), Ok(()));
        assert_eq!(l.permits(&authorize("a_1", MAX_AUTHORIZE)), Ok(()));
        assert_eq!(
            l.permits(&authorize("a_9", 100)),
            Err(Refusal::UnknownAccount(s("a_9")))
        );
        assert_eq!(
            l.permits(&authorize("a_1", 0)),
            out_of_range(0, MAX_AUTHORIZE)
        );
        assert_eq!(
            l.permits(&authorize("a_1", MAX_AUTHORIZE + 1)),
            out_of_range(MAX_AUTHORIZE + 1, MAX_AUTHORIZE)
        );
    }

    #[test]
    fn capture_needs_authorized_and_at_most_the_authorized_amount() {
        let l = ledger();
        assert_eq!(l.permits(&capture("p_1", 3_000)), Ok(()));
        assert_eq!(
            l.permits(&capture("p_1", 3_001)),
            out_of_range(3_001, 3_000)
        );
        assert_eq!(l.permits(&capture("p_1", 0)), out_of_range(0, 3_000));
        assert_eq!(
            l.permits(&capture("p_2", 100)),
            wrong_status("p_2", Status::Captured)
        );
        assert_eq!(
            l.permits(&capture("p_9", 100)),
            Err(Refusal::UnknownPayment(s("p_9")))
        );
    }

    #[test]
    fn void_needs_authorized() {
        let l = ledger();
        assert_eq!(l.permits(&void("p_1")), Ok(()));
        assert_eq!(l.permits(&void("p_3")), wrong_status("p_3", Status::Voided));
        assert_eq!(
            l.permits(&void("p_2")),
            wrong_status("p_2", Status::Captured)
        );
        assert_eq!(
            l.permits(&void("p_9")),
            Err(Refusal::UnknownPayment(s("p_9")))
        );
    }

    #[test]
    fn refund_needs_captured_and_at_most_the_rest_and_the_balance() {
        let l = ledger();
        assert_eq!(l.permits(&refund("p_2", 5_000)), Ok(()));
        assert_eq!(l.permits(&refund("p_4", 3_000)), Ok(()));
        // The rest of the capture limits it.
        assert_eq!(l.permits(&refund("p_2", 5_001)), out_of_range(5_001, 5_000));
        assert_eq!(l.permits(&refund("p_4", 3_001)), out_of_range(3_001, 3_000));
        // The balance of the merchant account limits it.
        assert_eq!(l.permits(&refund("p_6", 1)), out_of_range(1, 0));
        assert_eq!(l.permits(&refund("p_2", 0)), out_of_range(0, 5_000));
        assert_eq!(
            l.permits(&refund("p_1", 100)),
            wrong_status("p_1", Status::Authorized)
        );
        assert_eq!(
            l.permits(&refund("p_3", 100)),
            wrong_status("p_3", Status::Voided)
        );
        assert_eq!(
            l.permits(&refund("p_5", 100)),
            wrong_status("p_5", Status::Refunded)
        );
    }

    #[test]
    fn transfer_needs_two_different_accounts_and_at_most_the_balance() {
        let l = ledger();
        assert_eq!(l.permits(&transfer("a_2", "a_3", 10_000)), Ok(()));
        assert_eq!(
            l.permits(&transfer("a_2", "a_3", 10_001)),
            out_of_range(10_001, 10_000)
        );
        assert_eq!(l.permits(&transfer("a_3", "a_1", 1)), out_of_range(1, 0));
        assert_eq!(
            l.permits(&transfer("a_1", "a_1", 100)),
            Err(Refusal::SameAccount(s("a_1")))
        );
        assert_eq!(
            l.permits(&transfer("a_9", "a_1", 100)),
            Err(Refusal::UnknownAccount(s("a_9")))
        );
        assert_eq!(
            l.permits(&transfer("a_1", "a_9", 100)),
            Err(Refusal::UnknownAccount(s("a_9")))
        );
    }

    #[test]
    fn ask_operator_and_done_are_always_permitted_and_end_the_task() {
        let mut l = Ledger::default();
        let ask = Command::AskOperator {
            question: s("Which payment?"),
        };
        assert_eq!(l.permits(&ask), Ok(()));
        assert_eq!(l.permits(&Command::Done), Ok(()));
        assert_eq!(l.execute(&ask), Ok(Step::End));
        assert_eq!(l.execute(&Command::Done), Ok(Step::End));
        assert_eq!(l, Ledger::default());
        assert!(!ask.moves_money());
        assert!(transfer("a_1", "a_2", 1).moves_money());
    }

    // The effects of each command.

    #[test]
    fn authorize_makes_a_payment_with_the_lowest_free_id() {
        let mut l = ledger();
        l.payments.remove("p_3");
        assert_eq!(l.execute(&authorize("a_2", 700)), Ok(Step::Continue));
        assert_eq!(
            l.payments["p_3"],
            payment(Status::Authorized, 700, 0, 0, "a_2")
        );
        assert_eq!(l.execute(&authorize("a_2", 800)), Ok(Step::Continue));
        assert_eq!(
            l.payments["p_7"],
            payment(Status::Authorized, 800, 0, 0, "a_2")
        );
        assert_eq!(l.accounts, ledger().accounts);
    }

    #[test]
    fn capture_sets_the_amount_and_pays_the_account() {
        let mut l = ledger();
        assert_eq!(l.execute(&capture("p_1", 2_500)), Ok(Step::Continue));
        assert_eq!(
            l.payments["p_1"],
            payment(Status::Captured, 3_000, 2_500, 0, "a_1")
        );
        assert_eq!(l.accounts["a_1"], 52_500);
        // The rest of the authorization is released: no second capture.
        assert_eq!(
            l.execute(&capture("p_1", 500)),
            Err(Refusal::WrongStatus {
                payment: s("p_1"),
                status: Status::Captured
            })
        );
    }

    #[test]
    fn void_sets_the_status_only() {
        let mut l = ledger();
        assert_eq!(l.execute(&void("p_1")), Ok(Step::Continue));
        assert_eq!(
            l.payments["p_1"],
            payment(Status::Voided, 3_000, 0, 0, "a_1")
        );
        assert_eq!(l.accounts, ledger().accounts);
    }

    #[test]
    fn refund_adds_to_refunded_and_takes_from_the_account() {
        let mut l = ledger();
        assert_eq!(l.execute(&refund("p_2", 2_000)), Ok(Step::Continue));
        assert_eq!(
            l.payments["p_2"],
            payment(Status::PartiallyRefunded, 5_000, 5_000, 2_000, "a_2")
        );
        assert_eq!(l.accounts["a_2"], 8_000);
        assert_eq!(l.execute(&refund("p_2", 3_000)), Ok(Step::Continue));
        assert_eq!(
            l.payments["p_2"],
            payment(Status::Refunded, 5_000, 5_000, 5_000, "a_2")
        );
        assert_eq!(l.accounts["a_2"], 5_000);
    }

    #[test]
    fn transfer_moves_the_amount() {
        let mut l = ledger();
        assert_eq!(
            l.execute(&transfer("a_1", "a_3", 7_000)),
            Ok(Step::Continue)
        );
        assert_eq!(l.accounts["a_1"], 43_000);
        assert_eq!(l.accounts["a_3"], 7_000);
        assert_eq!(l.payments, ledger().payments);
    }

    #[test]
    fn a_refused_command_changes_nothing() {
        let mut l = ledger();
        assert!(l.execute(&refund("p_6", 1)).is_err());
        assert!(l.execute(&transfer("a_2", "a_1", 10_001)).is_err());
        assert_eq!(l, ledger());
    }

    // The search.

    fn status(payment: &str, status: Status) -> Condition {
        Condition::Status {
            payment: s(payment),
            status,
        }
    }
    fn balance(account: &str, amount: u64) -> Condition {
        Condition::Balance {
            account: s(account),
            amount,
        }
    }

    fn goal(conditions: Vec<Condition>) -> Goal {
        Goal { conditions }
    }

    /// The search gives `expected`, the only shortest sequence, and a
    /// replay of it gets to the goal.
    fn check(goal: &Goal, expected: &[Command]) {
        let found = check_len(goal, expected.len());
        assert_eq!(found, expected, "{goal:?}");
    }

    /// For a goal with more than one shortest sequence: the search gives a
    /// sequence of `len` commands, and a replay of it gets to the goal.
    fn check_len(goal: &Goal, len: usize) -> Vec<Command> {
        let start = ledger();
        let found = shortest(&start, goal, 4).expect("a sequence");
        assert_eq!(found.len(), len, "{goal:?}: {found:?}");
        let full = shortest_full(&start, goal, 4).expect("a sequence");
        assert_eq!(full.len(), len, "the full search: {goal:?}: {full:?}");
        let mut state = start.clone();
        for command in &found {
            assert_eq!(state.execute(command), Ok(Step::Continue));
        }
        assert!(goal.is_met(&start, &state));
        found
    }

    #[test]
    fn the_search_finds_the_known_shortest_sequences() {
        // 1. Already met: no commands.
        check(&goal(vec![status("p_5", Status::Refunded)]), &[]);
        // 2. A capture. Each amount from 1 to 3,000 is a shortest sequence.
        check_len(&goal(vec![status("p_1", Status::Captured)]), 1);
        // 3. A partial capture, with the amount that the goal names.
        check(
            &goal(vec![Condition::Captured {
                payment: s("p_1"),
                amount: 2_000,
            }]),
            &[capture("p_1", 2_000)],
        );
        // 4. A void.
        check(&goal(vec![status("p_1", Status::Voided)]), &[void("p_1")]);
        // 5. A full refund, with the rest of the capture from the state.
        check(
            &goal(vec![status("p_2", Status::Refunded)]),
            &[refund("p_2", 5_000)],
        );
        // 6. A refund total: 2,500 needs 1,500 more on top of 1,000.
        check(
            &goal(vec![Condition::Refunded {
                payment: s("p_4"),
                amount: 2_500,
            }]),
            &[refund("p_4", 1_500)],
        );
        // 7. A transfer, with the change of the balances.
        check(
            &goal(vec![balance("a_1", 43_000), balance("a_3", 7_000)]),
            &[transfer("a_1", "a_3", 7_000)],
        );
        // 8. A new payment.
        check(
            &goal(vec![Condition::NewPayment {
                account: s("a_2"),
                amount: 1_200,
            }]),
            &[authorize("a_2", 1_200)],
        );
        // 9. Capture, then refund in full: two commands in this order. The
        // amount of the capture is free.
        let found = check_len(&goal(vec![status("p_1", Status::Refunded)]), 2);
        assert!(matches!(found[0], Command::Capture { .. }));
        // 10. The example of the brief: p_2 fully refunded and a_2 has
        // received 10,000 from a_1. The two commands go in either order.
        let found = check_len(
            &goal(vec![
                status("p_2", Status::Refunded),
                balance("a_1", 40_000),
                balance("a_2", 15_000),
            ]),
            2,
        );
        assert!(found.contains(&refund("p_2", 5_000)));
        assert!(found.contains(&transfer("a_1", "a_2", 10_000)));
        // 11. The balance of a_3 is 0, so the refund of p_6 needs money
        // first: a transfer to a_3, from a_1 or a_2, then the refund.
        let found = check_len(&goal(vec![status("p_6", Status::Refunded)]), 2);
        assert!(matches!(&found[0], Command::Transfer { to, .. } if to == "a_3"));
        assert_eq!(found[1], refund("p_6", 2_000));
    }

    #[test]
    fn the_search_gives_none_for_a_goal_it_cannot_get_to() {
        let start = ledger();
        // A voided payment cannot be captured.
        assert_eq!(
            shortest(&start, &goal(vec![status("p_3", Status::Captured)]), 4),
            None
        );
        // Two commands are needed, and the limit is one.
        assert_eq!(
            shortest(&start, &goal(vec![status("p_1", Status::Refunded)]), 1),
            None
        );
        for search in [shortest, shortest_full] {
            assert_eq!(
                search(&start, &goal(vec![status("p_3", Status::Captured)]), 2),
                None
            );
        }
    }
}
