//! The grammar mask (brief, step 2): a [`LogitsHook`] that permits only
//! the tokens that keep the output in a grammar. The grammar engine is
//! `llguidance` (decision 4). This module does not compile grammars.
//!
//! The token bytes come from `tokenizer.json`, through
//! `llguidance::token_bytes_from_tokenizer_json`. That function marks the
//! special tokens, so a grammar cannot match `<|im_end|>` as text. Thus
//! `toktrie_hf_tokenizers` is not necessary. It needs `tokenizers` 0.21,
//! and the workspace has 0.23.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use fornace_engine::LogitsHook;
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use llguidance::{Matcher, ParserFactory};

/// Configuration B: the 7 commands in the exact byte form of the system
/// message, with no knowledge of the state. An ID is the prefix of its
/// field and 1 to 3 digits. An amount is 1 to 9,999,999, as in the brief.
/// A question is 1 to 100 characters with no quote, backslash, or control
/// character, so that a command always ends in [`crate::bench::MAX_COMMAND_TOKENS`].
pub const GRAMMAR_B: &str = r#"
start: authorize | capture | void | refund | transfer | ask_operator | done
authorize: "{\"cmd\":\"AUTHORIZE\",\"account\":\"" ACCOUNT "\",\"amount\":" AMOUNT "}"
capture: "{\"cmd\":\"CAPTURE\",\"payment\":\"" PAYMENT "\",\"amount\":" AMOUNT "}"
void: "{\"cmd\":\"VOID\",\"payment\":\"" PAYMENT "\"}"
refund: "{\"cmd\":\"REFUND\",\"payment\":\"" PAYMENT "\",\"amount\":" AMOUNT "}"
transfer: "{\"cmd\":\"TRANSFER\",\"from\":\"" ACCOUNT "\",\"to\":\"" ACCOUNT "\",\"amount\":" AMOUNT "}"
ask_operator: "{\"cmd\":\"ASK_OPERATOR\",\"question\":\"" QUESTION "\"}"
done: "{\"cmd\":\"DONE\"}"
ACCOUNT: /a_[0-9]{1,3}/
PAYMENT: /p_[0-9]{1,3}/
AMOUNT: /[1-9][0-9]{0,6}/
QUESTION: /[^"\\\x00-\x1f]{1,100}/
"#;

/// A token environment from token bytes. Tokenization of text is greedy
/// on the token trie.
struct Env {
    trie: TokTrie,
}

impl TokenizerEnv for Env {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.trie.greedy_tokenize(s)
    }

    /// The greedy tokenization is not the tokenization of the model. With
    /// false, the grammar engine does not force one token for the fixed
    /// bytes of the grammar, and the model can write them in any tokens.
    fn tokenize_is_canonical(&self) -> bool {
        false
    }
}

/// The parser factory for one vocabulary. Build it one time for a run.
pub struct Grammars {
    factory: ParserFactory,
}

impl Grammars {
    /// From token bytes, with special tokens marked by
    /// `TokTrie::SPECIAL_TOKEN_MARKER`. The vocabulary gets placeholder
    /// special tokens up to `logits_len`. `eos[0]` is the main EOS token.
    pub fn new(mut token_bytes: Vec<Vec<u8>>, eos: &[u32], logits_len: usize) -> Result<Self> {
        if token_bytes.len() > logits_len {
            bail!(
                "{} tokens, but the logits have {logits_len}",
                token_bytes.len()
            );
        }
        while token_bytes.len() < logits_len {
            let mut name = format!(".<[{}]>", token_bytes.len()).into_bytes();
            name[0] = TokTrie::SPECIAL_TOKEN_MARKER;
            token_bytes.push(name);
        }
        let info = TokRxInfo::new(logits_len as u32, *eos.first().context("no EOS token")?);
        let mut trie = TokTrie::from(&info, &token_bytes);
        if eos.len() > 1 {
            trie = trie.with_eos_tokens(eos);
        }
        let env: TokEnv = Arc::new(Env { trie });
        let mut factory = ParserFactory::new_simple(&env)?;
        factory.quiet();
        Ok(Grammars { factory })
    }

    /// From the `tokenizer.json` of a model directory. `eos` holds the
    /// names of the end tokens, the main one first. The result has their
    /// IDs also.
    pub fn from_model_dir(
        model: &Path,
        eos: &[&str],
        logits_len: usize,
    ) -> Result<(Self, Vec<u32>)> {
        let path = model.join("tokenizer.json");
        let json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&path).with_context(|| format!("read {path:?}"))?,
        )?;
        let bytes = llguidance::token_bytes_from_tokenizer_json(&json)?;
        let ids = eos
            .iter()
            .map(|name| {
                let mut marked = vec![TokTrie::SPECIAL_TOKEN_MARKER];
                marked.extend_from_slice(name.as_bytes());
                bytes
                    .iter()
                    .position(|b| *b == marked)
                    .map(|id| id as u32)
                    .with_context(|| format!("no special token {name}"))
            })
            .collect::<Result<Vec<u32>>>()?;
        Ok((Grammars::new(bytes, &ids, logits_len)?, ids))
    }

    /// A mask for one generation, from a Lark grammar. The compile time is
    /// in [`GrammarMask::compile_ms`].
    pub fn mask(&self, lark: &str) -> Result<GrammarMask> {
        let start = Instant::now();
        let parser = self
            .factory
            .create_parser(TopLevelGrammar::from_lark(lark.to_string()));
        let matcher = Matcher::new(parser);
        if let Some(error) = matcher.get_error() {
            bail!("the grammar does not compile: {error}");
        }
        Ok(GrammarMask {
            matcher,
            compile: start.elapsed(),
            mask_time: Duration::ZERO,
            mask_calls: 0,
            error: None,
        })
    }
}

/// A [`LogitsHook`] for one generation with one grammar.
pub struct GrammarMask {
    matcher: Matcher,
    compile: Duration,
    mask_time: Duration,
    mask_calls: usize,
    error: Option<String>,
}

impl GrammarMask {
    /// The time to compile the grammar and make the parser.
    pub fn compile_ms(&self) -> f64 {
        self.compile.as_secs_f64() * 1e3
    }

    /// The mean time of [`LogitsHook::on_logits`], in milliseconds. 0 with
    /// no call.
    pub fn mask_ms_per_token(&self) -> f64 {
        if self.mask_calls == 0 {
            return 0.0;
        }
        self.mask_time.as_secs_f64() * 1e3 / self.mask_calls as f64
    }

    /// The first error of the grammar engine in this generation. After an
    /// error, the mask permits only EOS.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

impl LogitsHook for GrammarMask {
    fn on_logits(&mut self, _step: usize, logits: &mut [f32]) {
        let start = Instant::now();
        let mask = match self.matcher.compute_mask_or_eos() {
            Ok(mask) => mask,
            Err(error) => {
                self.error.get_or_insert(error.to_string());
                self.matcher.tok_env().map_or_else(
                    |_| llguidance::toktrie::SimpleVob::alloc(logits.len()),
                    |env| env.tok_trie().eos_token_set(),
                )
            }
        };
        // `SimpleVob::apply_to` sets the permitted logits to 0, so it is
        // not a mask. Set each token that is not permitted to -inf.
        mask.iter_unset_entries(|id| {
            if let Some(logit) = logits.get_mut(id) {
                *logit = f32::NEG_INFINITY;
            }
        });
        self.mask_time += start.elapsed();
        self.mask_calls += 1;
    }

    fn on_token(&mut self, _step: usize, token: u32) {
        if self.error.is_some() {
            return;
        }
        if let Err(error) = self.matcher.consume_token(token) {
            self.error = Some(error.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::parse_command;
    use crate::ledger::Command;
    use crate::rng::Rng;

    const EOS: u32 = 0;

    /// A small vocabulary: EOS, each byte, and some longer pieces of the
    /// commands, also pieces that go past the end of a field.
    fn vocab() -> Vec<Vec<u8>> {
        let mut eos = vec![TokTrie::SPECIAL_TOKEN_MARKER];
        eos.extend_from_slice(b"<|im_end|>");
        let mut vocab = vec![eos];
        vocab.extend((0..=255u8).map(|b| vec![b]));
        for piece in [
            "{\"", "cmd", "\":\"", "\",\"", "\":", "\"}", "}", "CAPTURE", "REFUND", "DONE", "VOID",
            "payment", "amount", "account", "p_", "a_", "25", "00", "\"}\n", "```", " ",
        ] {
            vocab.push(piece.as_bytes().to_vec());
        }
        vocab
    }

    fn grammars(vocab: &[Vec<u8>]) -> Grammars {
        Grammars::new(vocab.to_vec(), &[EOS], vocab.len() + 4).expect("the env builds")
    }

    /// Feeds `text` byte by byte, and gives false at the first byte that
    /// the mask does not permit.
    fn permits(grammars: &Grammars, vocab: &[Vec<u8>], text: &str) -> bool {
        let mut mask = grammars.mask(GRAMMAR_B).expect("the grammar compiles");
        let len = vocab.len() + 4;
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
    fn the_grammar_takes_each_example_of_the_system_message() {
        let vocab = vocab();
        let grammars = grammars(&vocab);
        for line in crate::bench::SYSTEM.lines().filter(|l| l.starts_with('{')) {
            assert!(permits(&grammars, &vocab, line), "{line}");
            parse_command(line).expect("each example parses");
        }
    }

    #[test]
    fn the_grammar_blocks_other_forms() {
        let vocab = vocab();
        let grammars = grammars(&vocab);
        for text in [
            "```json",
            " {",
            "{\"cmd\": \"DONE\"}",
            "{\"cmd\":\"REFUND\",\"payment\":\"a_1\"",
            "{\"cmd\":\"REFUND\",\"payment\":\"p_1\",\"amount\":-",
            "{\"cmd\":\"REFUND\",\"payment\":\"p_1\",\"amount\":0",
            "{\"cmd\":\"REFUND\",\"payment\":\"p_1\",\"amount\":12345678",
            "{\"cmd\":\"VOID\",\"payment\":\"p_1234",
            "{\"cmd\":\"ASK_OPERATOR\",\"question\":\"\"",
        ] {
            assert!(!permits(&grammars, &vocab, text), "{text}");
        }
    }

    /// Random logits and a greedy sampler, through the hook: each output
    /// ends with EOS or after its `}`, and parses as one command.
    #[test]
    fn random_logits_give_commands_that_parse() {
        let vocab = vocab();
        let len = vocab.len() + 4;
        let grammars = grammars(&vocab);
        let mut rng = Rng::new(3);
        let mut names = std::collections::HashSet::new();
        for _ in 0..300 {
            let mut mask = grammars.mask(GRAMMAR_B).expect("the grammar compiles");
            let mut text = Vec::new();
            for step in 0..400 {
                let mut logits: Vec<f32> = (0..len).map(|_| rng.range(0, 1000) as f32).collect();
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
            let text = String::from_utf8(text).expect("the grammar gives UTF-8");
            let command = parse_command(&text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
            names.insert(std::mem::discriminant(&command));
            if let Command::AskOperator { question } = &command {
                assert!(!question.is_empty() && question.chars().count() <= 100);
            }
        }
        assert!(
            names.len() >= 5,
            "only {} commands in 300 runs",
            names.len()
        );
        assert!(mask_times_are_kept(&grammars));
    }

    fn mask_times_are_kept(grammars: &Grammars) -> bool {
        let mut mask = grammars.mask(GRAMMAR_B).expect("the grammar compiles");
        let mut logits = vec![0.0f32; vocab().len() + 4];
        mask.on_logits(0, &mut logits);
        mask.compile_ms() > 0.0 && mask.mask_ms_per_token() > 0.0
    }
}
