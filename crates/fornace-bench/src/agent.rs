//! The agent load: the requests of one agent harness on one machine.
//!
//! Every request starts with the same system prompt. Then it has its own
//! part, from one of two source texts in turn, with a length from a cycle.
//! The output length has a cycle too. It moves one step for each full
//! length cycle, so every 9 requests have all 9 (length, output) pairs. The
//! list is the same in every run and in every engine.

use tokenizers::Tokenizer;

/// The tokens of the system prompt: the start of `alice-2048.txt`.
pub const SYSTEM_TOKENS: usize = 256;
/// The lengths of the own part, in turn.
pub const OWN_CYCLE: [usize; 3] = [64, 256, 1024];
/// `max_tokens`. One step for each full cycle of [`OWN_CYCLE`].
pub const MAX_TOKENS_CYCLE: [usize; 3] = [32, 128, 256];
/// The tokens between the starts of two own parts from one source. 31 has
/// no common factor with the source lengths, so the starts are different.
const STRIDE: usize = 31;

/// One request of the list, before the text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Planned {
    pub index: usize,
    /// The index of the source text.
    pub source: usize,
    /// The first token of the own part in the source.
    pub offset: usize,
    pub own_tokens: usize,
    pub max_tokens: usize,
}

/// The first `count` requests. `source_lens` are the token counts of the
/// source texts.
pub fn plan(count: usize, source_lens: &[usize]) -> Vec<Planned> {
    (0..count)
        .map(|index| {
            let source = index % source_lens.len();
            let nth = index / source_lens.len();
            Planned {
                index,
                source,
                offset: (nth * STRIDE) % source_lens[source],
                own_tokens: OWN_CYCLE[index % OWN_CYCLE.len()],
                max_tokens: MAX_TOKENS_CYCLE[(index / OWN_CYCLE.len()) % MAX_TOKENS_CYCLE.len()],
            }
        })
        .collect()
}

/// One request with its text.
#[derive(Debug, Clone)]
pub struct AgentRequest {
    pub plan: Planned,
    pub prompt: String,
    /// The Fornace token count of `prompt`. A server must give the same
    /// count in its usage.
    pub prompt_tokens: usize,
}

/// Builds the text of each request: the system tokens, then the own
/// tokens, decoded. The own part wraps to the start of its source. An own
/// part that starts inside a character starts at the next token, and an
/// incomplete character at the end is removed.
pub fn build(
    plan: &[Planned],
    system: &[u32],
    sources: &[Vec<u32>],
    tokenizer: &Tokenizer,
) -> Result<Vec<AgentRequest>, String> {
    let decode = |ids: &[u32]| tokenizer.decode(ids, false).map_err(|e| e.to_string());
    plan.iter()
        .map(|p| {
            let src = &sources[p.source];
            let mut p = p.clone();
            while decode(&src[p.offset..=p.offset])?.starts_with(char::REPLACEMENT_CHARACTER) {
                p.offset = (p.offset + 1) % src.len();
            }
            let own = (0..p.own_tokens).map(|j| src[(p.offset + j) % src.len()]);
            let ids: Vec<u32> = system.iter().copied().chain(own).collect();
            let prompt = decode(&ids)?
                .trim_end_matches(char::REPLACEMENT_CHARACTER)
                .to_string();
            let prompt_tokens = tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| e.to_string())?
                .len();
            Ok(AgentRequest {
                plan: p,
                prompt,
                prompt_tokens,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plan_is_the_same_in_two_runs() {
        assert_eq!(plan(32, &[512, 513]), plan(32, &[512, 513]));
    }

    #[test]
    fn the_lengths_follow_the_cycles_and_the_sources_alternate() {
        let list = plan(10, &[512, 513]);
        let own: Vec<usize> = list.iter().map(|p| p.own_tokens).collect();
        let max: Vec<usize> = list.iter().map(|p| p.max_tokens).collect();
        let source: Vec<usize> = list.iter().map(|p| p.source).collect();
        assert_eq!(own, [64, 256, 1024, 64, 256, 1024, 64, 256, 1024, 64]);
        assert_eq!(max, [32, 32, 32, 128, 128, 128, 256, 256, 256, 32]);
        assert_eq!(source, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn every_9_requests_have_all_9_pairs() {
        let list = plan(9, &[512, 513]);
        let mut pairs: Vec<(usize, usize)> =
            list.iter().map(|p| (p.own_tokens, p.max_tokens)).collect();
        pairs.sort();
        pairs.dedup();
        assert_eq!(pairs.len(), 9);
    }

    #[test]
    fn no_two_requests_of_one_source_start_at_the_same_token() {
        let list = plan(64, &[512, 513]);
        for s in 0..2 {
            let mut offsets: Vec<usize> = list
                .iter()
                .filter(|p| p.source == s)
                .map(|p| p.offset)
                .collect();
            let n = offsets.len();
            offsets.sort();
            offsets.dedup();
            assert_eq!(offsets.len(), n, "source {s}");
        }
    }
}
