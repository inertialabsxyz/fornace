//! The static mask: the model can write only one string of a fixed list
//! (brief, step 1).
//!
//! The mask operates on bytes, because one token can hold a part of a name
//! or the end of one name and more bytes. A token is permitted at a trie
//! node if all its bytes follow a path from that node. At a node where a
//! string ends, EOS is also permitted. Thus the model can stop.

use std::time::{Duration, Instant};

use fornace_engine::LogitsHook;
use serde::{Deserialize, Serialize};

/// The 7 command names of the ledger.
pub const COMMAND_NAMES: [&str; 7] = [
    "AUTHORIZE",
    "CAPTURE",
    "VOID",
    "REFUND",
    "TRANSFER",
    "ASK_OPERATOR",
    "DONE",
];

/// No child for this byte.
const NONE: u32 = u32::MAX;

/// A byte trie of a list of strings. Node 0 is the root.
#[derive(Debug, Clone)]
pub struct Trie {
    next: Vec<[u32; 256]>,
    end: Vec<bool>,
}

impl Trie {
    /// The trie of `strings`. An empty string is not permitted, because
    /// then the model can stop before it writes a byte.
    pub fn new(strings: &[&[u8]]) -> Self {
        let mut trie = Trie {
            next: vec![[NONE; 256]],
            end: vec![false],
        };
        for string in strings {
            assert!(!string.is_empty(), "an empty string in the trie");
            let mut node = 0;
            for &byte in *string {
                let child = trie.next[node][byte as usize];
                node = if child == NONE {
                    trie.next.push([NONE; 256]);
                    trie.end.push(false);
                    let child = trie.next.len() - 1;
                    trie.next[node][byte as usize] = child as u32;
                    child
                } else {
                    child as usize
                };
            }
            trie.end[node] = true;
        }
        trie
    }

    /// The number of nodes.
    pub fn len(&self) -> usize {
        self.next.len()
    }

    /// True if the trie has only the root. [`Trie::new`] does not permit
    /// this with one string or more.
    pub fn is_empty(&self) -> bool {
        self.next.len() == 1
    }

    /// True if a string ends at `node`.
    pub fn is_end(&self, node: u32) -> bool {
        self.end[node as usize]
    }

    /// The node after `bytes` from `node`, or `None` if the bytes leave the
    /// trie. An empty `bytes` gives `None`: a token with no bytes does not
    /// move the model forward.
    pub fn walk(&self, node: u32, bytes: &[u8]) -> Option<u32> {
        if bytes.is_empty() {
            return None;
        }
        let mut node = node;
        for &byte in bytes {
            node = self.next[node as usize][byte as usize];
            if node == NONE {
                return None;
            }
        }
        Some(node)
    }
}

/// A set of token IDs, one bit for each ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bitmask {
    words: Vec<u64>,
}

impl Bitmask {
    /// An empty set for the IDs 0 to `len - 1`.
    pub fn new(len: usize) -> Self {
        Bitmask {
            words: vec![0; len.div_ceil(64)],
        }
    }

    pub fn insert(&mut self, id: usize) {
        self.words[id / 64] |= 1 << (id % 64);
    }

    pub fn contains(&self, id: usize) -> bool {
        self.words
            .get(id / 64)
            .is_some_and(|word| word >> (id % 64) & 1 == 1)
    }

    /// The number of IDs in the set.
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }
}

/// How the mask finds the permitted tokens at each step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// A walk of the trie for each token of the vocabulary, at each step.
    Scan,
    /// A read of the bitmask of the node. The bitmasks are calculated one
    /// time, when the mask is built.
    Bitmask,
}

/// A [`LogitsHook`] that permits only the strings of a [`Trie`].
pub struct TrieMask {
    trie: Trie,
    vocab: Vec<Vec<u8>>,
    eos: Vec<u32>,
    /// The length of the logits. It can be longer than `vocab`.
    len: usize,
    mode: Mode,
    /// One bitmask for each node. Empty in [`Mode::Scan`].
    allowed: Vec<Bitmask>,
    node: u32,
    mask_time: Duration,
    mask_calls: usize,
}

impl TrieMask {
    /// The mask for `strings`. `vocab` holds the bytes of each token ID
    /// (`Engine::token_bytes`), `eos` the end tokens, and `len` the length
    /// of the logits. An ID at `vocab.len()` or above is never permitted.
    pub fn new(
        strings: &[&[u8]],
        vocab: Vec<Vec<u8>>,
        eos: Vec<u32>,
        len: usize,
        mode: Mode,
    ) -> Self {
        let mut mask = TrieMask {
            trie: Trie::new(strings),
            vocab,
            eos,
            len,
            mode,
            allowed: Vec::new(),
            node: 0,
            mask_time: Duration::ZERO,
            mask_calls: 0,
        };
        if mode == Mode::Bitmask {
            mask.allowed = (0..mask.trie.len() as u32)
                .map(|node| {
                    let mut set = Bitmask::new(len);
                    for id in 0..len {
                        if mask.permitted(node, id) {
                            set.insert(id);
                        }
                    }
                    set
                })
                .collect();
        }
        mask
    }

    /// The mask for the 7 command names.
    pub fn command_names(vocab: Vec<Vec<u8>>, eos: Vec<u32>, len: usize, mode: Mode) -> Self {
        let names: Vec<&[u8]> = COMMAND_NAMES.iter().map(|n| n.as_bytes()).collect();
        TrieMask::new(&names, vocab, eos, len, mode)
    }

    /// True if the token `id` is permitted at `node`. This is the check of
    /// the scan, and the source of each bitmask.
    pub fn permitted(&self, node: u32, id: usize) -> bool {
        if self.eos.contains(&(id as u32)) {
            return self.trie.is_end(node);
        }
        self.vocab
            .get(id)
            .is_some_and(|bytes| self.trie.walk(node, bytes).is_some())
    }

    /// The permitted tokens at `node`, from the bitmask. `None` in
    /// [`Mode::Scan`].
    pub fn allowed(&self, node: u32) -> Option<&Bitmask> {
        self.allowed.get(node as usize)
    }

    pub fn trie(&self) -> &Trie {
        &self.trie
    }

    /// Back to the root for a new generation. The times stay.
    pub fn reset(&mut self) {
        self.node = 0;
    }

    /// The mean time of [`LogitsHook::on_logits`], in milliseconds, and the
    /// number of calls. The time is 0 with no call.
    pub fn mask_ms_per_token(&self) -> (f64, usize) {
        if self.mask_calls == 0 {
            return (0.0, 0);
        }
        let ms = self.mask_time.as_secs_f64() * 1e3 / self.mask_calls as f64;
        (ms, self.mask_calls)
    }

    /// Sets the time and the count of the calls to 0.
    pub fn clear_times(&mut self) {
        self.mask_time = Duration::ZERO;
        self.mask_calls = 0;
    }
}

impl LogitsHook for TrieMask {
    fn on_logits(&mut self, _step: usize, logits: &mut [f32]) {
        assert_eq!(logits.len(), self.len, "the logits length of the mask");
        let start = Instant::now();
        match self.mode {
            Mode::Scan => {
                for (id, logit) in logits.iter_mut().enumerate() {
                    if !self.permitted(self.node, id) {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
            Mode::Bitmask => {
                let allowed = &self.allowed[self.node as usize];
                for (id, logit) in logits.iter_mut().enumerate() {
                    if !allowed.contains(id) {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
        }
        self.mask_time += start.elapsed();
        self.mask_calls += 1;
    }

    fn on_token(&mut self, _step: usize, token: u32) {
        if self.eos.contains(&token) {
            return;
        }
        let bytes = &self.vocab[token as usize];
        self.node = self
            .trie
            .walk(self.node, bytes)
            .expect("the mask permits only tokens that follow the trie");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    const EOS: u32 = 0;

    /// A small vocabulary: EOS, each byte, and some longer pieces, also
    /// pieces that go past the end of a name.
    fn vocab() -> Vec<Vec<u8>> {
        let mut vocab = vec![b"<|im_end|>".to_vec(), Vec::new()];
        vocab.extend((0..=255u8).map(|b| vec![b]));
        for piece in [
            "CAP",
            "TURE",
            "VO",
            "ID",
            "VOID",
            "VOIDX",
            "RE",
            "FUND",
            "TRANS",
            "FER",
            "AUTH",
            "ORIZE",
            "ASK",
            "_",
            "OPER",
            "ATOR",
            "DONE",
            "DONE\"",
            " DONE",
            "{\"",
            "ASK_OPERATOR",
        ] {
            vocab.push(piece.as_bytes().to_vec());
        }
        vocab
    }

    fn id(vocab: &[Vec<u8>], piece: &str) -> usize {
        vocab
            .iter()
            .position(|t| t == piece.as_bytes())
            .expect("the piece is in the vocabulary")
    }

    #[test]
    fn walk_follows_bytes_across_tokens() {
        let names: Vec<&[u8]> = COMMAND_NAMES.iter().map(|n| n.as_bytes()).collect();
        let trie = Trie::new(&names);
        let node = trie.walk(0, b"CAP").expect("CAP is a prefix");
        assert!(!trie.is_end(node));
        let node = trie.walk(node, b"TURE").expect("CAPTURE is a name");
        assert!(trie.is_end(node));
        assert_eq!(trie.walk(node, b"X"), None);
        assert_eq!(trie.walk(0, b"VOIDX"), None);
        assert_eq!(trie.walk(0, b" DONE"), None);
        assert_eq!(trie.walk(0, b""), None);
        assert!(trie.walk(0, b"A").is_some_and(|n| !trie.is_end(n)));
    }

    #[test]
    fn eos_only_where_a_name_ends() {
        let vocab = vocab();
        let len = vocab.len() + 5;
        let mask = TrieMask::command_names(vocab.clone(), vec![EOS], len, Mode::Bitmask);
        let root = mask.allowed(0).expect("bitmask mode");
        assert!(!root.contains(EOS as usize));
        assert!(root.contains(id(&vocab, "VOID")));
        assert!(root.contains(id(&vocab, "ASK_OPERATOR")));
        assert!(!root.contains(id(&vocab, "VOIDX")));
        assert!(!root.contains(id(&vocab, "DONE\"")));
        assert!(!root.contains(1), "a token with no bytes");
        for padded in vocab.len()..len {
            assert!(!root.contains(padded), "a padded ID");
        }
        let end = mask.trie().walk(0, b"DONE").expect("DONE is a name");
        let at_end = mask.allowed(end).expect("bitmask mode");
        assert_eq!(at_end.count(), 1);
        assert!(at_end.contains(EOS as usize));
    }

    #[test]
    fn scan_and_bitmask_agree_at_each_node() {
        let vocab = vocab();
        let len = vocab.len() + 3;
        let bitmask = TrieMask::command_names(vocab.clone(), vec![EOS], len, Mode::Bitmask);
        let scan = TrieMask::command_names(vocab, vec![EOS], len, Mode::Scan);
        for node in 0..bitmask.trie().len() as u32 {
            let allowed = bitmask.allowed(node).expect("bitmask mode");
            assert!(allowed.count() > 0, "node {node} permits no token");
            for id in 0..len {
                assert_eq!(
                    allowed.contains(id),
                    scan.permitted(node, id),
                    "node {node}, ID {id}"
                );
            }
        }
    }

    /// Random logits and a greedy sampler, through the hook: each output is
    /// one of the names, and the two modes give the same outputs.
    #[test]
    fn random_logits_give_only_names() {
        let vocab = vocab();
        let len = vocab.len() + 3;
        let mut outputs = Vec::new();
        for mode in [Mode::Scan, Mode::Bitmask] {
            let mut mask = TrieMask::command_names(vocab.clone(), vec![EOS], len, mode);
            let mut rng = Rng::new(7);
            let mut texts = Vec::new();
            for _ in 0..500 {
                mask.reset();
                let mut text = Vec::new();
                for step in 0..20 {
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
                }
                let text = String::from_utf8(text).expect("names are ASCII");
                assert!(COMMAND_NAMES.contains(&text.as_str()), "{text:?}");
                texts.push(text);
            }
            let (ms, calls) = mask.mask_ms_per_token();
            assert!(calls > 0 && ms >= 0.0);
            outputs.push(texts);
        }
        assert_eq!(outputs[0], outputs[1]);
        let seen: std::collections::BTreeSet<_> = outputs[0].iter().collect();
        assert_eq!(seen.len(), COMMAND_NAMES.len(), "{seen:?}");
    }
}
