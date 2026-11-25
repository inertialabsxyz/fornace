//! E3 of the changes brief: check the request against the state before the
//! model runs. The reader finds the commands in the request text. If it can
//! read all of them, and the state does not permit one of them, the bench
//! sends `ASK_OPERATOR` and the model does not run. If it cannot read the
//! request, the bench raises no alarm and the model runs as before.
//!
//! The reader uses only the request text and the state: no goal and no task
//! class.

use crate::bench::{Answer, Output};
use crate::ledger::{Command, Ledger};
use crate::workaround::amount_spans;

/// The question of the `ASK_OPERATOR` that the bench sends after the check.
pub const PRECHECK_QUESTION: &str =
    "The ledger state does not permit a command of the request. What do you want me to do?";

/// The value of [`Output::changed_by`] for an alarm of the check.
pub const CHANGED_BY: &str = "request check";

/// The result of the reader for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reading {
    /// The commands of the request, in order.
    Commands(Vec<Command>),
    /// The reader could not link the IDs and the amounts to the commands.
    NotRead(String),
}

/// The result of the check for one request and one state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Check {
    /// The state permits each command, in order. The model runs.
    Permitted,
    /// The state refuses the command with this index. The bench asks.
    Refused(usize, String),
    /// The reader could not read the request. The model runs.
    NotRead(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verb {
    Authorize,
    Capture,
    Void,
    Refund,
    Transfer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Item {
    Verb(Verb),
    Payment(String),
    Account(String),
    Amount(u64),
}

/// The verbs, the IDs, and the amounts of `text`, in order.
fn items(text: &str) -> Vec<Item> {
    let mut found: Vec<(usize, Item)> = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_alphabetic() || (i > 0 && bytes[i - 1].is_ascii_alphanumeric()) {
            i += 1;
            continue;
        }
        let start = i;
        while bytes
            .get(i)
            .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
        {
            i += 1;
        }
        let word = &text[start..i];
        let lower = word.to_ascii_lowercase();
        let verb = match lower.as_str() {
            "authorize" | "authorise" | "authorizes" | "authorizing" => Some(Verb::Authorize),
            "capture" | "captures" | "capturing" => Some(Verb::Capture),
            "void" | "voids" | "voiding" => Some(Verb::Void),
            "refund" | "refunds" | "refunding" => Some(Verb::Refund),
            "transfer" | "transfers" | "transferring" => Some(Verb::Transfer),
            _ => None,
        };
        if let Some(verb) = verb {
            found.push((start, Item::Verb(verb)));
            continue;
        }
        let id = |prefix: &str| {
            lower
                .strip_prefix(prefix)
                .filter(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
                .is_some()
        };
        if id("p_") {
            found.push((start, Item::Payment(lower.clone())));
        } else if id("a_") {
            found.push((start, Item::Account(lower.clone())));
        }
    }
    for (position, amount) in amount_spans(text) {
        found.push((position, Item::Amount(amount)));
    }
    found.sort_by_key(|(position, _)| *position);
    found.into_iter().map(|(_, item)| item).collect()
}

/// The commands of a request.
///
/// Each verb starts a part that continues to the next verb. A part must
/// have the IDs and the amounts of its commands, in order: for each amount,
/// one payment for `capture` and `refund`, one account for `authorize`, and
/// two accounts (from, then to) for `transfer`. `void` has payments and no
/// amount. A part with no
/// ID and no amount is a word in the text, not a command, and the reader
/// ignores it. IDs before the first verb go to each part that has no ID of
/// its own. Any other part makes the request not read.
pub fn read(request: &str) -> Reading {
    let items = items(request);
    let first_verb = items.iter().position(|i| matches!(i, Item::Verb(_)));
    let Some(first_verb) = first_verb else {
        return Reading::NotRead("no command verb".to_string());
    };
    let lead = &items[..first_verb];
    if lead.iter().any(|i| matches!(i, Item::Amount(_))) {
        return Reading::NotRead("an amount before the first verb".to_string());
    }
    let mut commands = Vec::new();
    let mut i = first_verb;
    while i < items.len() {
        let Item::Verb(verb) = items[i] else {
            unreachable!("a part starts at a verb");
        };
        let end = items[i + 1..]
            .iter()
            .position(|x| matches!(x, Item::Verb(_)))
            .map_or(items.len(), |p| i + 1 + p);
        let part = &items[i + 1..end];
        i = end;
        let mut payments: Vec<&str> = Vec::new();
        let mut accounts: Vec<&str> = Vec::new();
        let mut amounts: Vec<u64> = Vec::new();
        for item in part {
            match item {
                Item::Payment(p) => payments.push(p),
                Item::Account(a) => accounts.push(a),
                Item::Amount(a) => amounts.push(*a),
                Item::Verb(_) => unreachable!("a part has one verb"),
            }
        }
        if payments.is_empty() && accounts.is_empty() && amounts.is_empty() {
            continue;
        }
        let lead_ids = |payment: bool| -> Vec<&str> {
            lead.iter()
                .filter_map(|x| match (x, payment) {
                    (Item::Payment(p), true) => Some(p.as_str()),
                    (Item::Account(a), false) => Some(a.as_str()),
                    _ => None,
                })
                .collect()
        };
        if payments.is_empty() && matches!(verb, Verb::Capture | Verb::Refund | Verb::Void) {
            payments = lead_ids(true);
        }
        if accounts.is_empty() && matches!(verb, Verb::Authorize | Verb::Transfer) {
            accounts = lead_ids(false);
        }
        // One verb can have several commands: `capture $5 of p_1 and $6 of
        // p_2`. The IDs and the amounts go together in order.
        let n = amounts.len();
        let not_read = || {
            Reading::NotRead(format!(
                "{verb:?} with {} payments, {} accounts, and {n} amounts",
                payments.len(),
                accounts.len()
            ))
        };
        match verb {
            Verb::Void => {
                if n > 0 || payments.is_empty() || !accounts.is_empty() {
                    return not_read();
                }
                for p in &payments {
                    commands.push(Command::Void {
                        payment: p.to_string(),
                    });
                }
            }
            Verb::Capture | Verb::Refund => {
                if n == 0 || payments.len() != n || !accounts.is_empty() {
                    return not_read();
                }
                for (p, &amount) in payments.iter().zip(&amounts) {
                    let payment = p.to_string();
                    commands.push(if verb == Verb::Capture {
                        Command::Capture { payment, amount }
                    } else {
                        Command::Refund { payment, amount }
                    });
                }
            }
            Verb::Authorize => {
                if n == 0 || accounts.len() != n || !payments.is_empty() {
                    return not_read();
                }
                for (a, &amount) in accounts.iter().zip(&amounts) {
                    commands.push(Command::Authorize {
                        account: a.to_string(),
                        amount,
                    });
                }
            }
            Verb::Transfer => {
                if n == 0 || accounts.len() != 2 * n || !payments.is_empty() {
                    return not_read();
                }
                for (pair, &amount) in accounts.chunks(2).zip(&amounts) {
                    commands.push(Command::Transfer {
                        from: pair[0].to_string(),
                        to: pair[1].to_string(),
                        amount,
                    });
                }
            }
        }
    }
    if commands.is_empty() {
        return Reading::NotRead("no command with an ID or an amount".to_string());
    }
    Reading::Commands(commands)
}

/// The check of the request against the state: each command of the
/// request, in order, on a copy of the state.
pub fn check(request: &str, state: &Ledger) -> Check {
    let commands = match read(request) {
        Reading::Commands(commands) => commands,
        Reading::NotRead(reason) => return Check::NotRead(reason),
    };
    let mut state = state.clone();
    for (index, command) in commands.iter().enumerate() {
        if let Err(refusal) = state.execute(command) {
            return Check::Refused(index, refusal.to_string());
        }
    }
    Check::Permitted
}

/// The output that the bench sends after an alarm of the check. The model
/// does not run, so the output has no text and no tokens.
pub fn alarm_output() -> Output {
    let command = Command::AskOperator {
        question: PRECHECK_QUESTION.to_string(),
    };
    Output {
        text: serde_json::to_string(&command).expect("a command serializes"),
        command: Some(command),
        answer: Answer::End,
        tokens: 0,
        fault: None,
        samples: None,
        rejected: Vec::new(),
        fallback: false,
        changed_by: Some(CHANGED_BY.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::generate_task;

    fn commands(request: &str) -> Vec<Command> {
        match read(request) {
            Reading::Commands(c) => c,
            Reading::NotRead(reason) => panic!("{request:?} not read: {reason}"),
        }
    }

    #[test]
    fn the_reader_links_ids_and_amounts_to_the_commands() {
        assert_eq!(
            commands("capture 21100 cents of payment p_1 refund 8900 cents of payment p_11"),
            vec![
                Command::Capture {
                    payment: "p_1".into(),
                    amount: 21100
                },
                Command::Refund {
                    payment: "p_11".into(),
                    amount: 8900
                },
            ]
        );
        assert_eq!(
            commands("Payment p_2 - capture $1,277, refund $903"),
            vec![
                Command::Capture {
                    payment: "p_2".into(),
                    amount: 127_700
                },
                Command::Refund {
                    payment: "p_2".into(),
                    amount: 90_300
                },
            ]
        );
        assert_eq!(
            commands(
                "Can you please transfer $8,104 from account a_2 to account a_1? The funds are required."
            ),
            vec![Command::Transfer {
                from: "a_2".into(),
                to: "a_1".into(),
                amount: 810_400
            }]
        );
        assert_eq!(
            commands("Please authorize a new payment of $115 on account a_2 and void payment p_6."),
            vec![
                Command::Authorize {
                    account: "a_2".into(),
                    amount: 11_500
                },
                Command::Void {
                    payment: "p_6".into()
                },
            ]
        );
        assert_eq!(
            commands("Please capture $967 for payment p_8 and $1,824 for payment p_10."),
            vec![
                Command::Capture {
                    payment: "p_8".into(),
                    amount: 96_700
                },
                Command::Capture {
                    payment: "p_10".into(),
                    amount: 182_400
                },
            ]
        );
        assert_eq!(
            commands(
                "Please process the following transfers: $164 from a_3 to a_4 and $1,492 from a_2 to a_1."
            ),
            vec![
                Command::Transfer {
                    from: "a_3".into(),
                    to: "a_4".into(),
                    amount: 16_400
                },
                Command::Transfer {
                    from: "a_2".into(),
                    to: "a_1".into(),
                    amount: 149_200
                },
            ]
        );
        // The second "refund" has no ID and no amount: it is not a command.
        assert_eq!(
            commands(
                "Please process a refund of $79 for payment p_3. The customer has requested a refund due to a defect."
            ),
            vec![Command::Refund {
                payment: "p_3".into(),
                amount: 7_900
            }]
        );
    }

    #[test]
    fn the_reader_does_not_guess() {
        for request in [
            "please do the usual",
            "capture payment p_3",
            "capture $5 and $6 of payment p_3",
            "transfer $5 and $6 from a_1 to a_2",
            "refund $5 of payment p_3 or p_4",
            "transfer $5 to account a_1",
            "void $5 of payment p_3",
            "$5: capture payment p_3",
        ] {
            assert!(
                matches!(read(request), Reading::NotRead(_)),
                "{request:?} was read"
            );
        }
    }

    #[test]
    fn the_check_refuses_what_the_state_refuses() {
        let state = generate_task(1, 0).start;
        let (id, payment) = state.payments.iter().next().expect("a payment");
        let too_much = payment.authorized + 1;
        assert!(matches!(
            check(&format!("void payment {id}"), &state),
            Check::Permitted | Check::Refused(0, _)
        ));
        let mut voided = state.clone();
        let _ = voided.execute(&Command::Void {
            payment: id.clone(),
        });
        assert!(matches!(
            check(&format!("void payment {id}"), &voided),
            Check::Refused(0, _)
        ));
        assert!(matches!(
            check(&format!("capture {too_much} cents of payment {id}"), &state),
            Check::Refused(0, _)
        ));
        assert!(matches!(
            check("please do the usual", &state),
            Check::NotRead(_)
        ));
        let out = alarm_output();
        assert_eq!(out.changed_by.as_deref(), Some(CHANGED_BY));
        assert!(
            state
                .permits(out.command.as_ref().expect("a command"))
                .is_ok()
        );
    }
}
