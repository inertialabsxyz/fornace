//! The same-tokens check: where the text of another engine first differs
//! from the tokens of Fornace.

/// The first Fornace token that `other` does not continue.
///
/// `prefixes[k]` is the decoded text of the first k + 1 Fornace tokens.
/// `None` if `other` is the full Fornace text. `Some(prefixes.len())` if
/// `other` has all Fornace tokens and more text after them.
pub fn first_difference(prefixes: &[String], other: &str) -> Option<usize> {
    let clean = |s: &str| s.trim_end_matches(char::REPLACEMENT_CHARACTER).to_string();
    match prefixes.iter().position(|p| !other.starts_with(&clean(p))) {
        Some(k) => Some(k),
        None if prefixes.last().map(|p| clean(p)) == Some(other.to_string()) => None,
        None if prefixes.is_empty() && other.is_empty() => None,
        None => Some(prefixes.len()),
    }
}

/// Up to `chars` characters of `text` after its first `start` bytes.
/// `start` is moved back to a character boundary.
pub fn snippet(text: &str, start: usize, chars: usize) -> String {
    let mut start = start.min(text.len());
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    text[start..].chars().take(chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefixes(parts: &[&str]) -> Vec<String> {
        (1..=parts.len()).map(|k| parts[..k].concat()).collect()
    }

    #[test]
    fn equal_text_has_no_difference() {
        let p = prefixes(&[" Alice", " was", " tired"]);
        assert_eq!(first_difference(&p, " Alice was tired"), None);
    }

    #[test]
    fn the_first_token_that_differs_is_found() {
        let p = prefixes(&[" Alice", " was", " tired"]);
        assert_eq!(first_difference(&p, " Alice is tired"), Some(1));
        assert_eq!(first_difference(&p, " Bob"), Some(0));
        assert_eq!(first_difference(&p, " Alice was tired and"), Some(3));
    }

    #[test]
    fn an_incomplete_last_character_is_not_a_difference() {
        let p = vec![
            " a".to_string(),
            format!(" a {}", char::REPLACEMENT_CHARACTER),
        ];
        assert_eq!(first_difference(&p, " a "), None);
        assert_eq!(first_difference(&p, " a 🦀"), Some(2));
    }

    #[test]
    fn snippet_starts_at_a_character_boundary() {
        assert_eq!(snippet("café au lait", 4, 3), "é a");
        assert_eq!(snippet("abc", 9, 3), "");
    }
}
