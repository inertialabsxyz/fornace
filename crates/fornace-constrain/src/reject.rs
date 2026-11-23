//! Configuration D, rejection (brief, step 5). The model writes a full
//! command with the grammar of B only. Then `permits` checks the command
//! against the state. If the state does not permit it, the bench discards
//! it and samples again with a new seed. After [`MAX_SAMPLES`] rejected
//! samples, the bench sends [`fallback`], an `ASK_OPERATOR`.
//!
//! A rejected sample does not go into the context of the next sample. The
//! model sees only the commands that the ledger executed.

use serde::{Deserialize, Serialize};

use crate::bench::{Answer, parse_command};
use crate::ledger::{Command, Ledger};
use crate::rng::Rng;

/// The largest number of samples for one command (brief, step 5).
pub const MAX_SAMPLES: usize = 5;

/// The question of the `ASK_OPERATOR` after [`MAX_SAMPLES`] rejections.
pub const FALLBACK_QUESTION: &str =
    "The ledger does not permit the commands that I wrote. What do you want me to do?";

/// One sample that the check refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rejected {
    pub text: String,
    pub tokens: usize,
    /// `refused` (the state does not permit the command) or
    /// `parse_failure`.
    pub reason: Answer,
}

/// The sampler seed of sample `sample` of a command. Sample 0 uses the
/// seed of the generation, as in B.
pub fn sample_seed(generation_seed: u64, sample: usize) -> u64 {
    if sample == 0 {
        generation_seed
    } else {
        Rng::new(generation_seed ^ (sample as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)).next_u64()
    }
}

/// The command of `text` if the state permits it, or the reason to reject
/// the sample.
pub fn check(state: &Ledger, text: &str) -> Result<Command, Answer> {
    let command = parse_command(text).map_err(Answer::ParseFailure)?;
    state
        .permits(&command)
        .map_err(|refusal| Answer::Refused(refusal.to_string()))?;
    Ok(command)
}

/// The command after [`MAX_SAMPLES`] rejected samples.
pub fn fallback() -> Command {
    Command::AskOperator {
        question: FALLBACK_QUESTION.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::generate_task;

    #[test]
    fn the_check_takes_only_permitted_commands() {
        let state = generate_task(1, 0).start;
        let (from, balance) = state
            .accounts
            .iter()
            .find(|(_, b)| **b > 0)
            .expect("an account with a balance");
        let to = state
            .accounts
            .keys()
            .find(|a| *a != from)
            .expect("two accounts");
        let ok = format!(r#"{{"cmd":"TRANSFER","from":"{from}","to":"{to}","amount":1}}"#);
        let over = format!(
            r#"{{"cmd":"TRANSFER","from":"{from}","to":"{to}","amount":{}}}"#,
            balance + 1
        );
        assert!(check(&state, &ok).is_ok());
        assert!(matches!(check(&state, &over), Err(Answer::Refused(_))));
        assert!(matches!(
            check(&state, "{\"cmd\":"),
            Err(Answer::ParseFailure(_))
        ));
        assert!(check(&state, "{\"cmd\":\"DONE\"}").is_ok());
        assert!(state.permits(&fallback()).is_ok());
    }

    #[test]
    fn each_sample_has_its_own_seed() {
        let seeds: std::collections::HashSet<u64> =
            (0..MAX_SAMPLES).map(|s| sample_seed(42, s)).collect();
        assert_eq!(seeds.len(), MAX_SAMPLES);
        assert_eq!(sample_seed(42, 0), 42);
    }
}
