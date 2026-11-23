//! The grammar for each state (brief, step 3). Before each command, the
//! rules read the state and give the permitted commands, one branch for
//! each permitted ID, and the amount range of each branch. [`lark`] makes
//! the grammar text from them. The grammar engine compiles the full text
//! again for each command, because the state does not repeat.
//!
//! The [`Rules`] trait keeps the state logic out of the mask:
//! [`crate::grammar::GrammarMask`] gets only the grammar text.

use std::collections::BTreeMap;

use crate::ledger::{Command, Ledger, MAX_AUTHORIZE, Status};
use crate::range::range_pattern;

/// The parts of the grammar that depend on the state. The maps are
/// ordered, so that one state always gives the same grammar text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarParams {
    /// The command names that the state permits.
    pub commands: Vec<&'static str>,
    /// For each command, one branch for each permitted ID. Each branch has
    /// the permitted values of the other ID fields and the amount range.
    pub branches: BTreeMap<&'static str, Vec<Branch>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Branch {
    pub id: String,
    /// For `TRANSFER`: the permitted `to` accounts for this `from`.
    pub second_ids: Vec<String>,
    /// The inclusive amount range, or `None` for a command with no amount.
    pub amount: Option<(u32, u32)>,
}

/// The rule set of an external system.
pub trait Rules {
    type State;
    type Command;

    /// The permitted commands, the ID branches, and the value ranges.
    fn params(&self, state: &Self::State) -> GrammarParams;

    /// The check before execution. The mask does not replace it.
    fn permits(&self, state: &Self::State, cmd: &Self::Command) -> bool;
}

/// The rule set of the ledger. It agrees with [`Ledger::permits`].
pub struct LedgerRules;

/// A range `1..=hi` as a branch amount, or `None` if `hi` is 0: then the
/// state permits no amount, and the branch is not in the grammar.
fn amount_range(hi: u64) -> Option<(u32, u32)> {
    (hi > 0).then(|| {
        (
            1,
            u32::try_from(hi).expect("a ledger amount above u32::MAX"),
        )
    })
}

impl Rules for LedgerRules {
    type State = Ledger;
    type Command = Command;

    fn params(&self, state: &Ledger) -> GrammarParams {
        let mut branches: BTreeMap<&'static str, Vec<Branch>> = BTreeMap::new();
        let mut add = |name: &'static str, id: &str, second_ids: Vec<String>, amount| {
            branches.entry(name).or_default().push(Branch {
                id: id.to_string(),
                second_ids,
                amount,
            });
        };
        for account in state.accounts.keys() {
            add(
                "AUTHORIZE",
                account,
                Vec::new(),
                amount_range(MAX_AUTHORIZE),
            );
        }
        for (id, p) in &state.payments {
            match p.status {
                Status::Authorized => {
                    if let Some(range) = amount_range(p.authorized) {
                        add("CAPTURE", id, Vec::new(), Some(range));
                    }
                    add("VOID", id, Vec::new(), None);
                }
                Status::Captured | Status::PartiallyRefunded => {
                    let balance = state.accounts.get(&p.account).copied().unwrap_or(0);
                    if let Some(range) = amount_range(p.refundable().min(balance)) {
                        add("REFUND", id, Vec::new(), Some(range));
                    }
                }
                Status::Voided | Status::Refunded => {}
            }
        }
        for (from, &balance) in &state.accounts {
            let to: Vec<String> = state
                .accounts
                .keys()
                .filter(|a| *a != from)
                .cloned()
                .collect();
            if let (Some(range), false) = (amount_range(balance), to.is_empty()) {
                add("TRANSFER", from, to, Some(range));
            }
        }
        let mut commands: Vec<&'static str> =
            ["AUTHORIZE", "CAPTURE", "VOID", "REFUND", "TRANSFER"]
                .into_iter()
                .filter(|name| branches.contains_key(name))
                .collect();
        commands.extend(["ASK_OPERATOR", "DONE"]);
        GrammarParams { commands, branches }
    }

    fn permits(&self, state: &Ledger, cmd: &Command) -> bool {
        state.permits(cmd).is_ok()
    }
}

/// The question of `ASK_OPERATOR`, as in [`crate::grammar::GRAMMAR_B`].
const QUESTION: &str = r#"/[^"\\\x00-\x1f]{1,100}/"#;

/// The largest amount that the grammar of B permits: 7 digits.
pub const MAX_WRITTEN: u32 = 9_999_999;

/// The Lark grammar of `params`, in the byte form of the system message.
/// Each branch is one rule: the fixed bytes up to the amount, then the
/// range pattern of the branch.
pub fn lark(params: &GrammarParams) -> String {
    lark_with(params, false)
}

/// The grammar of workaround 4 (brief, step 4). Each command with an amount
/// has a `"requested"` field before the amount, with no range limit. If
/// `requested` is in the range of the branch, the amount follows, in the
/// range. If it is above the range, the grammar permits only the end of
/// the object: `{"cmd":"REFUND","payment":"p_17","requested":4000}`. The
/// bench changes this form to `ASK_OPERATOR`
/// ([`crate::workaround::parse_requested`]).
pub fn lark_requested(params: &GrammarParams) -> String {
    lark_with(params, true)
}

fn lark_with(params: &GrammarParams, requested: bool) -> String {
    let mut rules = Vec::new();
    // The end of the rule after the ID: the rest of the literal, the range,
    // and the `}`. With `requested`, one pattern holds both continuations,
    // so that the lexer has no two terminals with the same prefix.
    let amount = |branch: &Branch| {
        let (lo, hi) = branch.amount.expect("a command with an amount");
        if !requested {
            return format!(r#"\",\"amount\":" /({})/ "}}""#, range_pattern(lo, hi));
        }
        let range = range_pattern(lo, hi);
        let mut tail = format!(r#"({range}),"amount":({range})\}}"#);
        if hi < MAX_WRITTEN {
            tail.push_str(&format!(r#"|({})\}}"#, range_pattern(hi + 1, MAX_WRITTEN)));
        }
        format!(r#"\",\"requested\":" /{tail}/"#)
    };
    for &name in &params.commands {
        let branches = params.branches.get(name).map(Vec::as_slice).unwrap_or(&[]);
        match name {
            "AUTHORIZE" | "CAPTURE" | "REFUND" => {
                let field = if name == "AUTHORIZE" {
                    "account"
                } else {
                    "payment"
                };
                for b in branches {
                    rules.push(format!(
                        r#""{{\"cmd\":\"{name}\",\"{field}\":\"{}{}"#,
                        b.id,
                        amount(b)
                    ));
                }
            }
            "VOID" => {
                for b in branches {
                    rules.push(format!(
                        r#""{{\"cmd\":\"VOID\",\"payment\":\"{}\"}}""#,
                        b.id
                    ));
                }
            }
            "TRANSFER" => {
                for b in branches {
                    for to in &b.second_ids {
                        rules.push(format!(
                            r#""{{\"cmd\":\"TRANSFER\",\"from\":\"{}\",\"to\":\"{to}{}"#,
                            b.id,
                            amount(b)
                        ));
                    }
                }
            }
            "ASK_OPERATOR" => rules.push(format!(
                r#""{{\"cmd\":\"ASK_OPERATOR\",\"question\":\"" {QUESTION} "\"}}""#
            )),
            "DONE" => rules.push(r#""{\"cmd\":\"DONE\"}""#.to_string()),
            other => panic!("no grammar form for the command {other}"),
        }
    }
    let mut text = String::from("start: ");
    text.push_str(
        &(0..rules.len())
            .map(|i| format!("b{i}"))
            .collect::<Vec<_>>()
            .join(" | "),
    );
    text.push('\n');
    for (i, rule) in rules.iter().enumerate() {
        text.push_str(&format!("b{i}: {rule}\n"));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::parse_command;
    use crate::grammar::Grammars;
    use crate::ledger::Payment;
    use crate::rng::Rng;
    use crate::tasks::generate_task;
    use fornace_engine::LogitsHook;
    use llguidance::toktrie::TokTrie;

    const EOS: u32 = 0;

    fn vocab() -> Vec<Vec<u8>> {
        let mut eos = vec![TokTrie::SPECIAL_TOKEN_MARKER];
        eos.extend_from_slice(b"<|im_end|>");
        let mut vocab = vec![eos];
        vocab.extend((0..=255u8).map(|b| vec![b]));
        for piece in [
            "{\"", "cmd", "\":\"", "\",\"", "\":", "\"}", "p_1", "a_", "00", "25",
        ] {
            vocab.push(piece.as_bytes().to_vec());
        }
        vocab
    }

    fn ledger() -> Ledger {
        let payment = |status, authorized, captured, refunded, account: &str| Payment {
            status,
            authorized,
            captured,
            refunded,
            account: account.to_string(),
        };
        Ledger {
            accounts: [("a_1".to_string(), 0), ("a_2".to_string(), 2_000)].into(),
            payments: [
                (
                    "p_1".to_string(),
                    payment(Status::Authorized, 5_000, 0, 0, "a_1"),
                ),
                (
                    "p_17".to_string(),
                    payment(Status::PartiallyRefunded, 3_000, 3_000, 500, "a_2"),
                ),
                (
                    "p_3".to_string(),
                    payment(Status::Captured, 900, 900, 0, "a_1"),
                ),
                ("p_4".to_string(), payment(Status::Voided, 100, 0, 0, "a_2")),
            ]
            .into(),
        }
    }

    #[test]
    fn params_follow_the_guards() {
        let params = LedgerRules.params(&ledger());
        assert_eq!(
            params.commands,
            [
                "AUTHORIZE",
                "CAPTURE",
                "VOID",
                "REFUND",
                "TRANSFER",
                "ASK_OPERATOR",
                "DONE"
            ]
        );
        let ids = |name| {
            params.branches[name]
                .iter()
                .map(|b| (b.id.as_str(), b.amount))
                .collect::<Vec<_>>()
        };
        assert_eq!(ids("CAPTURE"), [("p_1", Some((1, 5_000)))]);
        assert_eq!(ids("VOID"), [("p_1", None)]);
        // p_17: 2,500 refundable, balance 2,000. p_3: balance 0, so no branch.
        assert_eq!(ids("REFUND"), [("p_17", Some((1, 2_000)))]);
        // a_1 has a balance of 0, so it cannot be `from`.
        assert_eq!(ids("TRANSFER"), [("a_2", Some((1, 2_000)))]);
        assert_eq!(params.branches["TRANSFER"][0].second_ids, ["a_1"]);
    }

    #[test]
    fn no_authorized_payment_gives_no_capture_or_void() {
        let mut state = ledger();
        state.payments.remove("p_1");
        let params = LedgerRules.params(&state);
        assert!(!params.commands.contains(&"CAPTURE"));
        assert!(!params.commands.contains(&"VOID"));
        assert!(!lark(&params).contains("CAPTURE"));
    }

    /// Feeds `text` byte by byte. False at the first byte that the mask
    /// does not permit.
    fn grammar_permits(grammars: &Grammars, grammar: &str, vocab: &[Vec<u8>], text: &str) -> bool {
        let mut mask = grammars.mask(grammar).expect("the grammar compiles");
        let len = vocab.len();
        for (step, byte) in text.bytes().enumerate() {
            let mut logits = vec![0.0f32; len];
            mask.on_logits(step, &mut logits);
            let id = vocab
                .iter()
                .position(|t| t == &[byte])
                .expect("each byte is a token");
            if logits[id] == f32::NEG_INFINITY {
                return false;
            }
            mask.on_token(step, id as u32);
        }
        assert_eq!(mask.error(), None);
        true
    }

    #[test]
    fn the_grammar_takes_the_permitted_values_only() {
        let vocab = vocab();
        let grammars = Grammars::new(vocab.clone(), &[EOS], vocab.len()).expect("the env builds");
        let grammar = lark(&LedgerRules.params(&ledger()));
        for (text, expected) in [
            (r#"{"cmd":"REFUND","payment":"p_17","amount":2000}"#, true),
            (r#"{"cmd":"REFUND","payment":"p_17","amount":2001}"#, false),
            (r#"{"cmd":"REFUND","payment":"p_17","amount":0"#, false),
            (r#"{"cmd":"REFUND","payment":"p_3""#, false),
            (r#"{"cmd":"REFUND","payment":"p_1""#, false),
            (r#"{"cmd":"CAPTURE","payment":"p_1","amount":5000}"#, true),
            (r#"{"cmd":"CAPTURE","payment":"p_1","amount":5001"#, false),
            (r#"{"cmd":"VOID","payment":"p_1"}"#, true),
            (r#"{"cmd":"VOID","payment":"p_4""#, false),
            (
                r#"{"cmd":"TRANSFER","from":"a_2","to":"a_1","amount":1}"#,
                true,
            ),
            (r#"{"cmd":"TRANSFER","from":"a_2","to":"a_2""#, false),
            (r#"{"cmd":"TRANSFER","from":"a_1""#, false),
            (
                r#"{"cmd":"AUTHORIZE","account":"a_1","amount":1000000}"#,
                true,
            ),
            (
                r#"{"cmd":"AUTHORIZE","account":"a_1","amount":1000001"#,
                false,
            ),
            (r#"{"cmd":"AUTHORIZE","account":"a_3""#, false),
            (
                r#"{"cmd":"ASK_OPERATOR","question":"Which payment?"}"#,
                true,
            ),
            (r#"{"cmd":"DONE"}"#, true),
        ] {
            assert_eq!(
                grammar_permits(&grammars, &grammar, &vocab, text),
                expected,
                "{text}"
            );
        }
    }

    /// Random logits through the grammar of each start state of the
    /// generated tasks: each output parses, and the ledger permits it.
    #[test]
    fn random_outputs_of_task_states_are_valid() {
        let vocab = vocab();
        let len = vocab.len();
        let grammars = Grammars::new(vocab.clone(), &[EOS], len).expect("the env builds");
        let mut rng = Rng::new(11);
        let mut money = 0;
        for id in (0..300).step_by(10) {
            let state = generate_task(1, id).start;
            let grammar = lark(&LedgerRules.params(&state));
            for _ in 0..10 {
                let mut mask = grammars.mask(&grammar).expect("the grammar compiles");
                let mut text = Vec::new();
                for step in 0..400 {
                    let mut logits: Vec<f32> =
                        (0..len).map(|_| rng.range(0, 1000) as f32).collect();
                    mask.on_logits(step, &mut logits);
                    let token = (0..len)
                        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                        .expect("logits") as u32;
                    assert!(
                        logits[token as usize].is_finite(),
                        "the mask permits no token"
                    );
                    mask.on_token(step, token);
                    if token == EOS {
                        break;
                    }
                    text.extend_from_slice(&vocab[token as usize]);
                    if crate::bench::json_end(&String::from_utf8_lossy(&text)).is_some() {
                        break;
                    }
                }
                assert_eq!(mask.error(), None);
                let text = String::from_utf8(text).expect("UTF-8");
                let command = parse_command(&text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
                assert!(LedgerRules.permits(&state, &command), "task {id}: {text}");
                money += usize::from(command.moves_money());
            }
        }
        assert!(money > 100, "only {money} money commands in 300 outputs");
    }
}
