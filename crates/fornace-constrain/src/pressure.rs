//! Mask pressure (E1 of the changes brief): the probability mass that the
//! mask of C removes at each token. In a trap, the model can put a high
//! probability on a token that the state does not permit. This module
//! measures that mass. It does not change the generation.

use serde::{Deserialize, Serialize};

/// The field of the command that a token starts or continues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Field {
    /// The command name.
    Cmd,
    /// An ID: `account`, `payment`, `from`, or `to`.
    Id,
    Amount,
    /// The question of `ASK_OPERATOR`.
    Question,
    /// The fixed bytes of the grammar, between the fields.
    Other,
}

impl Field {
    /// An ID or an amount: a field whose value the state limits.
    pub fn is_value(self) -> bool {
        matches!(self, Field::Id | Field::Amount)
    }
}

/// The mask pressure of one token.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StepPressure {
    pub field: Field,
    /// The sum of the probabilities of the tokens that the mask blocks.
    pub blocked: f32,
    /// The mask blocks the token with the highest probability.
    pub top_blocked: bool,
}

/// The mask pressure of one generation: one line of the mass file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PressureLine {
    pub task: usize,
    pub seed: u64,
    pub generation: usize,
    pub steps: Vec<StepPressure>,
}

impl PressureLine {
    /// The largest blocked mass in an ID or amount field. 0 if the
    /// generation has no value field.
    pub fn max_value_mass(&self) -> f32 {
        self.steps
            .iter()
            .filter(|s| s.field.is_value())
            .map(|s| s.blocked)
            .fold(0.0, f32::max)
    }
}

/// The field of the next token, from the bytes of the command before it.
/// The token is in a field if the text ends after `"<key>":`, after
/// `"<key>":"`, or inside the value that follows.
pub fn field(text: &[u8]) -> Field {
    let Some(colon) = text.windows(2).rposition(|w| w == b"\":") else {
        return Field::Other;
    };
    let Some(key_start) = text[..colon].iter().rposition(|&b| b == b'"') else {
        return Field::Other;
    };
    let key = &text[key_start + 1..colon];
    let rest = &text[colon + 2..];
    let open = match rest.first() {
        // A string value: open until its closing quote.
        Some(b'"') => !rest[1..].contains(&b'"'),
        // A number value: open until `,` or `}`.
        Some(_) => !rest.iter().any(|&b| b == b',' || b == b'}'),
        None => true,
    };
    if !open {
        return Field::Other;
    }
    match key {
        b"cmd" => Field::Cmd,
        b"account" | b"payment" | b"from" | b"to" => Field::Id,
        b"amount" => Field::Amount,
        b"question" => Field::Question,
        _ => Field::Other,
    }
}

/// The blocked mass of the softmax of `logits` at `temperature`, and if the
/// top token is blocked. `logits` are the values before the mask.
/// `blocked(id)` is true for a token that the mask blocks. A temperature of
/// 0 gives the mass of the top token only.
pub fn blocked_mass(
    logits: &[f32],
    temperature: f64,
    blocked: impl Fn(usize) -> bool,
) -> (f32, bool) {
    let (top, &max) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("logits are not empty");
    let top_blocked = blocked(top);
    if temperature <= 0.0 {
        return (if top_blocked { 1.0 } else { 0.0 }, top_blocked);
    }
    let t = temperature as f32;
    let (mut all, mut removed) = (0.0f64, 0.0f64);
    for (id, &logit) in logits.iter().enumerate() {
        let p = f64::from(((logit - max) / t).exp());
        all += p;
        if blocked(id) {
            removed += p;
        }
    }
    ((removed / all) as f32, top_blocked)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_field_follows_the_text() {
        let cases: &[(&str, Field)] = &[
            ("", Field::Other),
            ("{\"cmd", Field::Other),
            ("{\"cmd\":", Field::Cmd),
            ("{\"cmd\":\"CAP", Field::Cmd),
            ("{\"cmd\":\"CAPTURE\"", Field::Other),
            ("{\"cmd\":\"CAPTURE\",\"payment\":", Field::Id),
            ("{\"cmd\":\"CAPTURE\",\"payment\":\"", Field::Id),
            ("{\"cmd\":\"CAPTURE\",\"payment\":\"p_1", Field::Id),
            ("{\"cmd\":\"CAPTURE\",\"payment\":\"p_12\"", Field::Other),
            (
                "{\"cmd\":\"CAPTURE\",\"payment\":\"p_12\",\"amount\":",
                Field::Amount,
            ),
            (
                "{\"cmd\":\"CAPTURE\",\"payment\":\"p_12\",\"amount\":25",
                Field::Amount,
            ),
            (
                "{\"cmd\":\"CAPTURE\",\"payment\":\"p_12\",\"amount\":2500}",
                Field::Other,
            ),
            (
                "{\"cmd\":\"TRANSFER\",\"from\":\"a_1\",\"to\":\"a_",
                Field::Id,
            ),
            (
                "{\"cmd\":\"ASK_OPERATOR\",\"question\":\"Why",
                Field::Question,
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(field(text.as_bytes()), *expected, "{text:?}");
        }
        assert!(Field::Id.is_value() && Field::Amount.is_value());
        assert!(!Field::Cmd.is_value() && !Field::Other.is_value());
    }

    #[test]
    fn the_blocked_mass_is_a_part_of_the_softmax() {
        // Two equal logits: each has half the mass. A tie has no top token.
        let (mass, _) = blocked_mass(&[0.0, 0.0], 1.0, |id| id == 1);
        assert!((mass - 0.5).abs() < 1e-6);
        let (mass, top) = blocked_mass(&[1.0, 3.0, 0.0], 0.7, |id| id == 1);
        let p = |x: f32| (x / 0.7).exp();
        let expected = p(3.0) / (p(1.0) + p(3.0) + p(0.0));
        assert!((mass - expected).abs() < 1e-6);
        assert!(top);
        assert_eq!(blocked_mass(&[1.0, 3.0], 1.0, |_| false).0, 0.0);
        assert_eq!(blocked_mass(&[1.0, 3.0], 0.0, |id| id == 1), (1.0, true));
    }

    #[test]
    fn the_largest_value_mass_ignores_other_fields() {
        let step = |field, blocked| StepPressure {
            field,
            blocked,
            top_blocked: false,
        };
        let line = PressureLine {
            task: 0,
            seed: 1,
            generation: 0,
            steps: vec![
                step(Field::Cmd, 0.9),
                step(Field::Id, 0.2),
                step(Field::Amount, 0.4),
                step(Field::Other, 0.8),
            ],
        };
        assert_eq!(line.max_value_mass(), 0.4);
    }
}
