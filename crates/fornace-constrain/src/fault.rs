//! The fault test of configuration C (brief, step 3): the state changes
//! while the model writes a command, after the ID field. The grammar is
//! from the old state, so the mask cannot stop the command. The check
//! before execution must stop it.
//!
//! A seeded draw selects 1 in 20 generations. In a selected generation,
//! the first ID field of a money command gives the change:
//!
//! | Command | Change |
//! |---|---|
//! | `TRANSFER` | The balance of `from` becomes 0 |
//! | `CAPTURE` | The payment is voided |
//! | `REFUND` | The balance of the account of the payment becomes 0 |
//! | `VOID` | The payment is captured, for its authorized amount |
//!
//! `AUTHORIZE`, `ASK_OPERATOR`, and `DONE` get no change.

use serde::{Deserialize, Serialize};

use crate::ledger::{Ledger, Status};
use crate::rng::Rng;

/// 1 in 20 generations: 5 % (brief, step 3).
pub const FAULT_RATE: (u64, u64) = (1, 20);

/// The fault of one selected generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fault {
    /// The number of tokens when the ID field was complete. `None` if the
    /// output has no ID field of a money command.
    pub token: Option<usize>,
    /// The change to the state. `None` if the command gets no change.
    pub change: Option<String>,
    /// The check before execution refused the command.
    pub stopped: bool,
}

/// True if the fault test selects this generation. `sampler_seed` is the
/// seed of the generation, so the draw is the same in each run.
pub fn selected(sampler_seed: u64) -> bool {
    Rng::new(sampler_seed ^ 0xFA17_FA17).chance(FAULT_RATE.0, FAULT_RATE.1)
}

/// The command name and the first ID, when `text` holds the complete ID
/// field, for example `{"cmd":"REFUND","payment":"p_17"`.
pub fn written_id(text: &str) -> Option<(&str, &str)> {
    let rest = text.strip_prefix(r#"{"cmd":""#)?;
    let (name, rest) = rest.split_once('"')?;
    let rest = rest.strip_prefix(r#",""#)?;
    let (field, rest) = rest.split_once('"')?;
    if !["account", "payment", "from"].contains(&field) {
        return None;
    }
    let rest = rest.strip_prefix(r#":""#)?;
    let (id, _) = rest.split_once('"')?;
    Some((name, id))
}

/// Changes `state` for a command `name` on `id`, as in the table of the
/// module. The result describes the change, or is `None` if the command
/// gets no change or the ID does not exist.
pub fn apply(state: &mut Ledger, name: &str, id: &str) -> Option<String> {
    match name {
        "TRANSFER" => {
            *state.accounts.get_mut(id)? = 0;
            Some(format!("balance of {id} set to 0"))
        }
        "CAPTURE" => {
            state.payments.get_mut(id)?.status = Status::Voided;
            Some(format!("{id} voided"))
        }
        "REFUND" => {
            let account = state.payments.get(id)?.account.clone();
            *state.accounts.get_mut(&account)? = 0;
            Some(format!("balance of {account} set to 0"))
        }
        "VOID" => {
            let p = state.payments.get_mut(id)?;
            p.status = Status::Captured;
            p.captured = p.authorized;
            let (account, amount) = (p.account.clone(), p.authorized);
            *state.accounts.get_mut(&account)? += amount;
            Some(format!("{id} captured for {amount}"))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::{Command, Payment};

    #[test]
    fn the_id_field_is_found_only_when_complete() {
        let text = r#"{"cmd":"REFUND","payment":"p_17","amount":25"#;
        assert_eq!(written_id(text), Some(("REFUND", "p_17")));
        assert_eq!(written_id(r#"{"cmd":"REFUND","payment":"p_1"#), None);
        assert_eq!(
            written_id(r#"{"cmd":"TRANSFER","from":"a_2""#),
            Some(("TRANSFER", "a_2"))
        );
        assert_eq!(
            written_id(r#"{"cmd":"ASK_OPERATOR","question":"Why""#),
            None
        );
        assert_eq!(written_id(r#"{"cmd":"DONE"}"#), None);
    }

    #[test]
    fn each_change_makes_the_ledger_refuse_the_command() {
        let state = Ledger {
            accounts: [("a_1".to_string(), 5_000), ("a_2".to_string(), 5_000)].into(),
            payments: [
                (
                    "p_1".to_string(),
                    Payment {
                        status: Status::Authorized,
                        authorized: 1_000,
                        captured: 0,
                        refunded: 0,
                        account: "a_1".to_string(),
                    },
                ),
                (
                    "p_2".to_string(),
                    Payment {
                        status: Status::Captured,
                        authorized: 1_000,
                        captured: 1_000,
                        refunded: 0,
                        account: "a_2".to_string(),
                    },
                ),
            ]
            .into(),
        };
        let commands = [
            (
                "TRANSFER",
                "a_1",
                Command::Transfer {
                    from: "a_1".into(),
                    to: "a_2".into(),
                    amount: 1,
                },
            ),
            (
                "CAPTURE",
                "p_1",
                Command::Capture {
                    payment: "p_1".into(),
                    amount: 1,
                },
            ),
            (
                "REFUND",
                "p_2",
                Command::Refund {
                    payment: "p_2".into(),
                    amount: 1,
                },
            ),
            (
                "VOID",
                "p_1",
                Command::Void {
                    payment: "p_1".into(),
                },
            ),
        ];
        for (name, id, command) in commands {
            let mut changed = state.clone();
            assert!(
                changed.permits(&command).is_ok(),
                "{name} before the change"
            );
            assert!(apply(&mut changed, name, id).is_some());
            assert!(
                changed.permits(&command).is_err(),
                "{name} after the change"
            );
        }
        let mut changed = state.clone();
        assert_eq!(apply(&mut changed, "AUTHORIZE", "a_1"), None);
        assert_eq!(apply(&mut changed, "CAPTURE", "p_9"), None);
        assert_eq!(changed, state);
    }

    #[test]
    fn about_one_in_twenty_is_selected() {
        let count = (0..20_000u64).filter(|&s| selected(s)).count();
        assert!((800..1_200).contains(&count), "{count}");
    }
}
