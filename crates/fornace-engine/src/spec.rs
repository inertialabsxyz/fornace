//! The pure parts of speculative decoding: the greedy acceptance rule, the
//! count of the guesses that the target checked, the cut of a round, and
//! the expected tokens of a round.
//!
//! In a round, the draft guesses K tokens d₁ to d_K after the last accepted
//! token. The target runs one forward pass of `[last, d₁, …, d_K]` and gives
//! its own token after each of the K + 1 positions.

/// Accepts the guesses while they equal the target's tokens. `target[i]` is
/// the target's token after position i of the verify input, so `target` has
/// one more entry than `drafts`. Returns j, the number of accepted guesses,
/// and the next token, `target[j]`: the target's token at the first reject,
/// or the bonus token when all guesses are accepted. A round thus gives
/// j + 1 tokens.
pub(crate) fn accept_greedy(drafts: &[u32], target: &[u32]) -> (usize, u32) {
    assert_eq!(
        target.len(),
        drafts.len() + 1,
        "the target gives one token after each verify position"
    );
    let j = drafts
        .iter()
        .zip(target)
        .take_while(|(d, t)| d == t)
        .count();
    (j, target[j])
}

/// The guesses that the target checked in a round with j accepts out of k:
/// j + 1 if a guess was rejected, else k. The guesses after the first reject
/// are not checked, so they do not count.
pub(crate) fn tested(j: usize, k: usize) -> usize {
    if j < k { j + 1 } else { k }
}

/// The tokens that a round adds: the accepted guesses, then `next`. The
/// round stops before the first token in `eos`, and after `room` tokens.
/// Returns the tokens, and `true` if the generation must stop at an EOS.
pub(crate) fn round_tokens(
    drafts: &[u32],
    j: usize,
    next: u32,
    eos: &[u32],
    room: usize,
) -> (Vec<u32>, bool) {
    let mut tokens = Vec::with_capacity(j + 1);
    for &t in drafts[..j].iter().chain([&next]) {
        if eos.contains(&t) {
            return (tokens, true);
        }
        if tokens.len() == room {
            break;
        }
        tokens.push(t);
    }
    (tokens, false)
}

/// The mean tokens of a round, E(α, K) = (1 − α^(K+1)) / (1 − α), if each
/// guess is accepted with probability α, independent of the others.
pub fn expected_tokens(alpha: f64, k: usize) -> f64 {
    if alpha >= 1.0 {
        return (k + 1) as f64;
    }
    (1.0 - alpha.powi(k as i32 + 1)) / (1.0 - alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_guesses_accepted_give_the_bonus_token() {
        assert_eq!(accept_greedy(&[5, 6, 7], &[5, 6, 7, 8]), (3, 8));
    }

    #[test]
    fn a_reject_at_the_first_guess_gives_the_target_token() {
        assert_eq!(accept_greedy(&[5, 6, 7], &[9, 6, 7, 8]), (0, 9));
    }

    #[test]
    fn a_reject_in_the_middle_stops_the_round_there() {
        // The third guess agrees again, but it comes after a reject.
        assert_eq!(accept_greedy(&[5, 6, 7], &[5, 9, 7, 8]), (1, 9));
    }

    #[test]
    fn tested_counts_the_checked_guesses_only() {
        assert_eq!(tested(0, 4), 1);
        assert_eq!(tested(2, 4), 3);
        assert_eq!(tested(4, 4), 4);
        // The example of the tutorial: j = 4, 1, 2 at K = 4.
        let (accepted, checked) = [4, 1, 2]
            .iter()
            .fold((0, 0), |(a, t), &j| (a + j, t + tested(j, 4)));
        assert_eq!((accepted, checked), (7, 9));
    }

    #[test]
    fn expected_tokens_follow_the_formula() {
        assert!((expected_tokens(0.8, 4) - 3.3616).abs() < 1e-4);
        assert!((expected_tokens(0.7, 3) - 2.533).abs() < 1e-3);
        assert_eq!(expected_tokens(0.0, 4), 1.0);
        assert_eq!(expected_tokens(1.0, 4), 5.0);
    }

    #[test]
    fn a_round_stops_before_eos_and_at_the_room() {
        assert_eq!(
            round_tokens(&[5, 6, 7], 3, 8, &[], 10),
            (vec![5, 6, 7, 8], false)
        );
        assert_eq!(
            round_tokens(&[5, 6, 7], 2, 9, &[], 10),
            (vec![5, 6, 9], false)
        );
        assert_eq!(round_tokens(&[5, 0, 7], 3, 8, &[0], 10), (vec![5], true));
        assert_eq!(round_tokens(&[5, 6, 7], 2, 0, &[0], 10), (vec![5, 6], true));
        assert_eq!(round_tokens(&[5, 6, 7], 3, 8, &[], 2), (vec![5, 6], false));
        assert_eq!(round_tokens(&[5, 6, 7], 3, 8, &[], 0), (vec![], false));
    }
}
