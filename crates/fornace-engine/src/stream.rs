//! Text for each new token of a stream.
//!
//! The tokenizer works on bytes, so one token can end inside a UTF-8
//! character. Its text alone is then not valid. [`TextStream`] decodes all
//! generated ids after each token, and gives only the text after the part
//! that it gave before. The decoder writes U+FFFD for an incomplete
//! character, so a U+FFFD at the end waits for the next token. The text
//! before it goes out now.

use tokenizers::Tokenizer;

use crate::{Error, Result};

/// The text of a stream that was already given out.
#[derive(Debug, Default)]
pub struct TextStream {
    /// The length in bytes of the text given out so far.
    sent: usize,
}

impl TextStream {
    /// Decodes `generated` (all ids of the stream so far), and returns the
    /// new complete text. It is empty if the last id ends inside a
    /// character.
    pub fn push(&mut self, tokenizer: &Tokenizer, generated: &[u32]) -> Result<String> {
        let full = tokenizer
            .decode(generated, false)
            .map_err(Error::TokenDecoding)?;
        let text = new_text(&full, self.sent);
        self.sent += text.len();
        Ok(text.to_string())
    }
}

/// The text of `full` after the first `sent` bytes, without the U+FFFD
/// characters at its end (an incomplete character).
pub fn new_text(full: &str, sent: usize) -> &str {
    let complete = full.trim_end_matches(char::REPLACEMENT_CHARACTER);
    complete.get(sent..).unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_text_gives_the_part_after_the_sent_bytes() {
        assert_eq!(new_text("Alice was", 5), " was");
        assert_eq!(new_text("Alice", 0), "Alice");
        assert_eq!(new_text("Alice", 5), "");
    }

    #[test]
    fn a_character_split_over_two_tokens_waits_for_the_second() {
        // "é" is 0xC3 0xA9. After the first byte, the decoder writes U+FFFD.
        let first = format!("caf{}", char::REPLACEMENT_CHARACTER);
        assert_eq!(new_text(&first, 3), "");
        // After the second byte, the whole character comes at once.
        assert_eq!(new_text("café", 3), "é");
    }

    #[test]
    fn the_text_before_an_incomplete_character_goes_out_now() {
        // One token: a space and the first byte of an emoji.
        let full = format!("a, {}", char::REPLACEMENT_CHARACTER);
        assert_eq!(new_text(&full, 2), " ");
        assert_eq!(new_text("a, 🎉", 3), "🎉");
    }

    #[test]
    fn a_replacement_character_inside_the_text_does_not_wait() {
        let full = format!("a{}b", char::REPLACEMENT_CHARACTER);
        assert_eq!(new_text(&full, 0), full.as_str());
    }
}
