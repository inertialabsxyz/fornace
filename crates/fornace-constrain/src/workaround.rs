//! The workarounds for the prefix trap and the ID trap (brief, step 4).
//! Each one is a change to configuration C. Workaround 2 is configuration
//! D ([`crate::reject`]).
//!
//! | Flag | Workaround | Change |
//! |---|---|---|
//! | `limits` | 1 | The prompt has the amount limit of each ID |
//! | `compare` | 3 | An amount that is not in the request text becomes `ASK_OPERATOR` |
//! | `requested` | 4 | A `"requested"` field with no limit before the amount |
//! | `all-ids` | 5 | The mask permits each existing payment ID |

use serde::{Deserialize, Serialize};

use crate::bench::SYSTEM;
use crate::compile::{Branch, GrammarParams, LedgerRules, MAX_WRITTEN, Rules};
use crate::ledger::{Command, Ledger};

/// A workaround of configuration C.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Workaround {
    /// 1: the limit of each ID in the prompt.
    Limits,
    /// 3: compare the amount with the amounts in the request text.
    Compare,
    /// 4: a `"requested"` field before the amount.
    Requested,
    /// 5: each existing payment ID in the mask.
    AllIds,
}

impl Workaround {
    /// The number of the workaround in the brief.
    pub fn number(self) -> u8 {
        match self {
            Workaround::Limits => 1,
            Workaround::Compare => 3,
            Workaround::Requested => 4,
            Workaround::AllIds => 5,
        }
    }
}

/// The system message. Workaround 4 adds `"requested"` to the examples
/// and one rule.
pub fn system(workaround: Option<Workaround>) -> String {
    if workaround != Some(Workaround::Requested) {
        return SYSTEM.to_string();
    }
    SYSTEM
        .replace(
            "\"amount\":2500}",
            "\"requested\":2500,\"amount\":2500}",
        )
        .replace(
            "- All amounts are whole numbers of cents. $1 is 100 cents.\n",
            "- All amounts are whole numbers of cents. $1 is 100 cents.\n\
             - requested is the amount that the operator asked for. amount is the amount to send.\n",
        )
}

/// Workaround 1: the amount limit of each ID that the state permits, from
/// the rules of the mask. The model gets the limits and does not have to
/// calculate them.
pub fn limits_text(state: &Ledger) -> String {
    let params = LedgerRules.params(state);
    let mut text = String::from("Limits (amounts in cents):\n");
    for (name, verb) in [
        ("CAPTURE", "capture"),
        ("REFUND", "refund"),
        ("TRANSFER", "transfer"),
    ] {
        for branch in params.branches.get(name).into_iter().flatten() {
            if let Some((_, hi)) = branch.amount {
                text.push_str(&format!("{}: {verb} up to {hi}\n", branch.id));
            }
        }
    }
    text
}

/// The amounts in a request text, in cents. `$40` and `$40.50` are
/// dollars. Another number is in cents. A number after `_` is part of an
/// ID, and is not an amount. Commas between digits are part of the number.
pub fn request_amounts(text: &str) -> Vec<u64> {
    amount_spans(text)
        .into_iter()
        .map(|(_, amount)| amount)
        .collect()
}

/// The amounts of [`request_amounts`], each with the byte position of its
/// first digit.
pub fn amount_spans(text: &str) -> Vec<(usize, u64)> {
    let bytes = text.as_bytes();
    let mut amounts = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        if i > 0 && bytes[i - 1] == b'_' {
            while bytes.get(i).is_some_and(u8::is_ascii_digit) {
                i += 1;
            }
            continue;
        }
        let dollars = i > 0 && bytes[i - 1] == b'$';
        let start = i;
        let mut whole = 0u64;
        while i < bytes.len() {
            match bytes[i] {
                b @ b'0'..=b'9' => {
                    whole = whole.saturating_mul(10).saturating_add(u64::from(b - b'0'))
                }
                b',' if bytes.get(i + 1).is_some_and(u8::is_ascii_digit) => {}
                _ => break,
            }
            i += 1;
        }
        let mut cents = None;
        if dollars
            && bytes.get(i) == Some(&b'.')
            && bytes.get(i + 1).is_some_and(u8::is_ascii_digit)
            && bytes.get(i + 2).is_some_and(u8::is_ascii_digit)
        {
            cents = Some(u64::from(bytes[i + 1] - b'0') * 10 + u64::from(bytes[i + 2] - b'0'));
            i += 3;
        }
        amounts.push((
            start,
            if dollars {
                whole.saturating_mul(100) + cents.unwrap_or(0)
            } else {
                whole
            },
        ));
    }
    amounts
}

/// Workaround 3: `ASK_OPERATOR` in place of a command whose amount is not
/// one of the amounts in the request. `None` if the command stays.
pub fn compare(command: &Command, request: &str) -> Option<Command> {
    let amount = match command {
        Command::Authorize { amount, .. }
        | Command::Capture { amount, .. }
        | Command::Refund { amount, .. }
        | Command::Transfer { amount, .. } => *amount,
        _ => return None,
    };
    (!request_amounts(request).contains(&amount)).then(|| Command::AskOperator {
        question: format!(
            "The amount {amount} is not an amount in the request. What do you want me to do?"
        ),
    })
}

/// Workaround 4: the command of an output with a `"requested"` field. An
/// amount command with `requested` and no `amount` becomes `ASK_OPERATOR`:
/// the grammar permits this form only when `requested` is above the limit.
/// The second value is `true` for this change.
pub fn parse_requested(text: &str) -> Result<(Command, bool), String> {
    let mut value: serde_json::Value =
        serde_json::from_str(text.trim()).map_err(|e| e.to_string())?;
    let object = value.as_object_mut().ok_or("not a JSON object")?;
    let requested = object.remove("requested");
    if let Some(requested) = requested
        && !object.contains_key("amount")
    {
        return Ok((
            Command::AskOperator {
                question: format!(
                    "The requested amount {requested} is above the limit. What do you want me to do?"
                ),
            },
            true,
        ));
    }
    let command = serde_json::from_value(value).map_err(|e| e.to_string())?;
    Ok((command, false))
}

/// Workaround 5: the parameters of the mask with each existing payment ID
/// in `CAPTURE`, `VOID`, and `REFUND`. An ID that the state permits keeps
/// its range. Another ID gets the amounts of the grammar of B, 1 to
/// 9,999,999, and the check before execution refuses its command.
pub fn params_all_ids(state: &Ledger) -> GrammarParams {
    let mut params = LedgerRules.params(state);
    for name in ["CAPTURE", "VOID", "REFUND"] {
        let permitted = params.branches.remove(name).unwrap_or_default();
        let amount = (name != "VOID").then_some((1, MAX_WRITTEN));
        let branches = state
            .payments
            .keys()
            .map(|id| {
                permitted
                    .iter()
                    .find(|b| &b.id == id)
                    .cloned()
                    .unwrap_or_else(|| Branch {
                        id: id.clone(),
                        second_ids: Vec::new(),
                        amount,
                    })
            })
            .collect::<Vec<_>>();
        if !branches.is_empty() {
            params.branches.insert(name, branches);
        }
    }
    let mut commands: Vec<&'static str> = ["AUTHORIZE", "CAPTURE", "VOID", "REFUND", "TRANSFER"]
        .into_iter()
        .filter(|name| params.branches.contains_key(name))
        .collect();
    commands.extend(["ASK_OPERATOR", "DONE"]);
    params.commands = commands;
    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{lark, lark_requested};
    use crate::grammar::Grammars;
    use crate::ledger::{Payment, Status};
    use crate::tasks::generate_task;
    use fornace_engine::LogitsHook;

    #[test]
    fn amounts_come_from_the_request_text() {
        assert_eq!(
            request_amounts("refund 158800 cents of payment p_1"),
            [158_800]
        );
        assert_eq!(
            request_amounts("Transfer $5,384 from a_2 to a_1."),
            [538_400]
        );
        assert_eq!(request_amounts("refund p_12 $17.18 now"), [1_718]);
        assert_eq!(
            request_amounts("refund 16,500 cents, then $4."),
            [16_500, 400]
        );
        assert_eq!(
            request_amounts("Request to void payment p_13."),
            Vec::<u64>::new()
        );
        assert_eq!(request_amounts("$1588, of p_2"), [158_800]);
    }

    #[test]
    fn compare_keeps_only_the_requested_amount() {
        let refund = |amount| Command::Refund {
            payment: "p_1".to_string(),
            amount,
        };
        let request = "refund 4,000 cents of payment p_1";
        assert_eq!(compare(&refund(4_000), request), None);
        assert!(matches!(
            compare(&refund(400), request),
            Some(Command::AskOperator { .. })
        ));
        let void = Command::Void {
            payment: "p_1".to_string(),
        };
        assert_eq!(compare(&void, request), None);
        assert_eq!(compare(&Command::Done, request), None);
    }

    #[test]
    fn the_requested_form_without_an_amount_asks_the_operator() {
        let (command, changed) =
            parse_requested(r#"{"cmd":"REFUND","payment":"p_1","requested":4000,"amount":2500}"#)
                .unwrap();
        assert_eq!(
            command,
            Command::Refund {
                payment: "p_1".to_string(),
                amount: 2_500
            }
        );
        assert!(!changed);
        let (command, changed) =
            parse_requested(r#"{"cmd":"REFUND","payment":"p_1","requested":4000}"#).unwrap();
        assert!(matches!(command, Command::AskOperator { .. }));
        assert!(changed);
        assert_eq!(
            parse_requested(r#"{"cmd":"DONE"}"#).unwrap(),
            (Command::Done, false)
        );
        assert!(parse_requested("{\"cmd\":").is_err());
    }

    #[test]
    fn the_system_message_of_workaround_4_has_requested() {
        let text = system(Some(Workaround::Requested));
        assert_eq!(
            text.matches("\"requested\":2500,\"amount\":2500}").count(),
            4
        );
        assert!(text.contains("- requested is the amount"));
        assert_eq!(system(None), SYSTEM);
        assert_eq!(system(Some(Workaround::Compare)), SYSTEM);
    }

    fn ledger() -> Ledger {
        let payment = |status, authorized, captured, account: &str| Payment {
            status,
            authorized,
            captured,
            refunded: 0,
            account: account.to_string(),
        };
        Ledger {
            accounts: [("a_1".to_string(), 3_000), ("a_2".to_string(), 0)].into(),
            payments: [
                (
                    "p_1".to_string(),
                    payment(Status::Captured, 2_500, 2_500, "a_1"),
                ),
                ("p_17".to_string(), payment(Status::Voided, 4_000, 0, "a_1")),
            ]
            .into(),
        }
    }

    #[test]
    fn limits_follow_the_mask() {
        assert_eq!(
            limits_text(&ledger()),
            "Limits (amounts in cents):\np_1: refund up to 2500\na_1: transfer up to 3000\n"
        );
    }

    /// The masks of workarounds 4 and 5 take the outputs of the brief, and
    /// block the others.
    #[test]
    fn the_grammars_of_workarounds_4_and_5() {
        let mut eos = vec![llguidance::toktrie::TokTrie::SPECIAL_TOKEN_MARKER];
        eos.extend_from_slice(b"<|im_end|>");
        let vocab: Vec<Vec<u8>> = std::iter::once(eos)
            .chain((0..=255u8).map(|b| vec![b]))
            .collect();
        let grammars = Grammars::new(vocab.clone(), &[0], vocab.len()).expect("the env builds");
        // Each byte of `text` is permitted, one byte token at a time.
        let accepts = |grammar: &str, text: &str| {
            let mut mask = grammars.mask(grammar).expect("the grammar compiles");
            for (step, byte) in text.bytes().enumerate() {
                let mut logits = vec![0.0f32; vocab.len()];
                mask.on_logits(step, &mut logits);
                let id = usize::from(byte) + 1;
                if logits[id] == f32::NEG_INFINITY {
                    return false;
                }
                mask.on_token(step, id as u32);
            }
            assert_eq!(mask.error(), None);
            true
        };
        let state = ledger();
        let requested = lark_requested(&LedgerRules.params(&state));
        for (text, ok) in [
            (r#"{"cmd":"REFUND","payment":"p_1","requested":4000}"#, true),
            (
                r#"{"cmd":"REFUND","payment":"p_1","requested":2000,"amount":2000}"#,
                true,
            ),
            (
                r#"{"cmd":"REFUND","payment":"p_1","requested":2000}"#,
                false,
            ),
            (
                r#"{"cmd":"REFUND","payment":"p_1","requested":4000,"amount":400}"#,
                false,
            ),
            (
                r#"{"cmd":"REFUND","payment":"p_1","requested":2500,"amount":2600}"#,
                false,
            ),
            (
                r#"{"cmd":"TRANSFER","from":"a_1","to":"a_2","requested":3001}"#,
                true,
            ),
            (
                r#"{"cmd":"REFUND","payment":"p_17","requested":4000}"#,
                false,
            ),
        ] {
            assert_eq!(accepts(&requested, text), ok, "{text}");
        }
        let all_ids = lark(&params_all_ids(&state));
        let only_state = lark(&LedgerRules.params(&state));
        let p17 = r#"{"cmd":"REFUND","payment":"p_17","amount":4000}"#;
        assert!(accepts(&all_ids, p17));
        assert!(!accepts(&only_state, p17));
        assert!(accepts(&all_ids, r#"{"cmd":"VOID","payment":"p_1"}"#));
        assert!(!accepts(
            &all_ids,
            r#"{"cmd":"REFUND","payment":"p_1","amount":2600}"#
        ));
        assert!(!accepts(
            &all_ids,
            r#"{"cmd":"REFUND","payment":"p_9","amount":1}"#
        ));
    }

    #[test]
    fn all_ids_keeps_the_ranges_of_the_state() {
        for id in 0..30 {
            let state = generate_task(1, id).start;
            let base = LedgerRules.params(&state);
            let wide = params_all_ids(&state);
            for (name, branches) in &base.branches {
                for b in branches {
                    assert!(wide.branches[name].contains(b), "{name} {}", b.id);
                }
            }
            if !state.payments.is_empty() {
                assert_eq!(wide.branches["VOID"].len(), state.payments.len());
            }
        }
    }
}
