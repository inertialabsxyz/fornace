//! An integer range as a pattern on decimal digits.
//!
//! The grammar engine operates on bytes, and Qwen2.5 writes each digit as
//! one token. A range in the form of a digit pattern lets the mask apply a
//! value guard at each digit, for all token divisions.

/// A pattern that matches each integer from `lo` to `hi` and no other.
///
/// The pattern is an alternation of branches with `|`, in the regular
/// expression syntax of the grammar engine. Each branch is a sequence of
/// digits, digit classes such as `[0-4]`, and `[0-9]{k}`. The pattern
/// matches no leading zero. Put it in a group before you join it with other
/// text.
///
/// For `range_pattern(1, 2500)` the result is
/// `[1-9]|[1-9][0-9]|[1-9][0-9]{2}|1[0-9]{3}|2[0-4][0-9]{2}|2500`.
///
/// # Panics
///
/// If `lo` is 0, or if `lo` is above `hi`. The amounts of the ledger start
/// at 1.
pub fn range_pattern(lo: u32, hi: u32) -> String {
    assert!(lo >= 1, "range_pattern: lo must be 1 or more, not {lo}");
    assert!(lo <= hi, "range_pattern: lo {lo} is above hi {hi}");
    let mut branches = Vec::new();
    let mut start = lo as u64;
    let hi = hi as u64;
    // One part for each number of digits, because a pattern of one length
    // cannot match a number of another length.
    while start <= hi {
        let digits = start.to_string().len() as u32;
        let end = hi.min(10u64.pow(digits) - 1);
        branches.extend(same_length(
            start.to_string().as_bytes(),
            end.to_string().as_bytes(),
        ));
        start = end + 1;
    }
    branches.join("|")
}

/// The branches for the digit strings from `a` to `b`. Both have the same
/// length, and `a` is not above `b`.
fn same_length(a: &[u8], b: &[u8]) -> Vec<String> {
    let Some((&a0, a_rest)) = a.split_first() else {
        return vec![String::new()];
    };
    let (&b0, b_rest) = b.split_first().expect("same length");
    if a0 == b0 {
        return same_length(a_rest, b_rest)
            .into_iter()
            .map(|tail| format!("{}{tail}", a0 as char))
            .collect();
    }
    let n = a_rest.len();
    let a_rest_all_zero = a_rest.iter().all(|&d| d == b'0');
    let b_rest_all_nine = b_rest.iter().all(|&d| d == b'9');
    let mut branches = Vec::new();
    // The first digit of `a`, if the rest of `a` is not the lowest value.
    if !a_rest_all_zero {
        let nines = vec![b'9'; n];
        for tail in same_length(a_rest, &nines) {
            branches.push(format!("{}{tail}", a0 as char));
        }
    }
    // The first digits that permit each tail.
    let mid_lo = if a_rest_all_zero { a0 } else { a0 + 1 };
    let mid_hi = if b_rest_all_nine { b0 } else { b0 - 1 };
    if mid_lo <= mid_hi {
        branches.push(format!("{}{}", class(mid_lo, mid_hi), any_digits(n)));
    }
    // The first digit of `b`, if the rest of `b` is not the highest value.
    if !b_rest_all_nine {
        let zeros = vec![b'0'; n];
        for tail in same_length(&zeros, b_rest) {
            branches.push(format!("{}{tail}", b0 as char));
        }
    }
    branches
}

fn class(lo: u8, hi: u8) -> String {
    if lo == hi {
        (lo as char).to_string()
    } else {
        format!("[{}-{}]", lo as char, hi as char)
    }
}

fn any_digits(n: usize) -> String {
    match n {
        0 => String::new(),
        1 => "[0-9]".to_string(),
        _ => format!("[0-9]{{{n}}}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One element of a branch: a digit class, `count` times.
    struct Atom {
        lo: u8,
        hi: u8,
        count: usize,
    }

    /// A matcher for the narrow pattern form that `range_pattern` gives.
    /// It panics on any other syntax, so a change of form fails the test.
    struct Matcher {
        branches: Vec<Vec<Atom>>,
    }

    impl Matcher {
        fn new(pattern: &str) -> Matcher {
            let branches = pattern.split('|').map(parse_branch).collect();
            Matcher { branches }
        }

        fn is_match(&self, text: &str) -> bool {
            self.is_match_bytes(text.as_bytes())
        }

        fn is_match_bytes(&self, text: &[u8]) -> bool {
            self.branches.iter().any(|branch| {
                let mut pos = 0;
                for atom in branch {
                    for _ in 0..atom.count {
                        match text.get(pos) {
                            Some(&c) if (atom.lo..=atom.hi).contains(&c) => pos += 1,
                            _ => return false,
                        }
                    }
                }
                pos == text.len()
            })
        }
    }

    fn parse_branch(branch: &str) -> Vec<Atom> {
        let bytes = branch.as_bytes();
        let mut atoms = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let (lo, hi) = match bytes[i] {
                b'0'..=b'9' => {
                    i += 1;
                    (bytes[i - 1], bytes[i - 1])
                }
                b'[' => {
                    assert!(
                        bytes[i + 2] == b'-' && bytes[i + 4] == b']',
                        "class in {branch}"
                    );
                    i += 5;
                    (bytes[i - 4], bytes[i - 2])
                }
                other => panic!("unexpected byte {:?} in {branch}", other as char),
            };
            assert!(
                lo.is_ascii_digit() && hi.is_ascii_digit() && lo <= hi,
                "class in {branch}"
            );
            let mut count = 1;
            if bytes.get(i) == Some(&b'{') {
                let close = i + branch[i..].find('}').expect("closed quantifier");
                count = branch[i + 1..close].parse().expect("quantifier count");
                i = close + 1;
            }
            atoms.push(Atom { lo, hi, count });
        }
        atoms
    }

    /// SplitMix64: a seeded generator, so that each run checks the same
    /// ranges.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// An integer in `lo..=hi`.
        fn range(&mut self, lo: u64, hi: u64) -> u64 {
            lo + self.next() % (hi - lo + 1)
        }
    }

    fn check(matcher: &Matcher, pattern: &str, lo: u32, hi: u32, n: u64) {
        let expected = (lo as u64..=hi as u64).contains(&n);
        // The decimal digits of `n`, with no allocation, because the test
        // checks about 70 million numbers.
        let mut buf = [0u8; 20];
        let mut start = buf.len();
        let mut rest = n;
        loop {
            start -= 1;
            buf[start] = b'0' + (rest % 10) as u8;
            rest /= 10;
            if rest == 0 {
                break;
            }
        }
        assert_eq!(
            matcher.is_match_bytes(&buf[start..]),
            expected,
            "{n} in {lo}..={hi}, pattern {pattern}"
        );
    }

    #[test]
    fn the_pattern_of_the_brief() {
        assert_eq!(
            range_pattern(1, 2500),
            "[1-9]|[1-9][0-9]|[1-9][0-9]{2}|1[0-9]{3}|2[0-4][0-9]{2}|2500"
        );
    }

    #[test]
    fn single_values_and_full_lengths() {
        assert_eq!(range_pattern(7, 7), "7");
        assert_eq!(range_pattern(2500, 2500), "2500");
        assert_eq!(range_pattern(1, 9), "[1-9]");
        assert_eq!(range_pattern(10, 99), "[1-9][0-9]");
        assert_eq!(
            range_pattern(1, 1_000_000),
            "[1-9]|[1-9][0-9]|[1-9][0-9]{2}|[1-9][0-9]{3}|[1-9][0-9]{4}|[1-9][0-9]{5}|1000000"
        );
    }

    #[test]
    fn no_leading_zero_and_no_other_text() {
        let pattern = range_pattern(1, 2500);
        let matcher = Matcher::new(&pattern);
        for text in ["0", "00", "05", "0100", "", "-1", "1.0", "+5", "2500 "] {
            assert!(!matcher.is_match(text), "{text:?} matched {pattern}");
        }
    }

    #[test]
    #[should_panic(expected = "lo must be 1 or more")]
    fn lo_of_zero_panics() {
        range_pattern(0, 10);
    }

    #[test]
    #[should_panic(expected = "is above hi")]
    fn lo_above_hi_panics() {
        range_pattern(11, 10);
    }

    /// The property test of the brief (step 4, "Code example").
    #[test]
    fn pattern_is_exact() {
        let mut rng = Rng(0x5EED);
        for _ in 0..1000 {
            // A number of digits for `hi` first, so that short ranges get
            // checked as often as long ones.
            let digits = rng.range(1, 7) as u32;
            let hi = rng.range(
                10u64.pow(digits - 1),
                (10u64.pow(digits) - 1).min(1_000_000),
            ) as u32;
            let lo = rng.range(1, hi as u64) as u32;
            let pattern = range_pattern(lo, hi);
            let matcher = Matcher::new(&pattern);
            for n in [lo as u64 - 1, lo as u64, hi as u64, hi as u64 + 1] {
                check(&matcher, &pattern, lo, hi, n);
            }
            for p in 0..=7 {
                check(&matcher, &pattern, lo, hi, 10u64.pow(p));
            }
            for _ in 0..1000 {
                check(&matcher, &pattern, lo, hi, rng.range(0, 10_000_000));
            }
            if hi <= 99_999 {
                for n in 0..=99_999 {
                    check(&matcher, &pattern, lo, hi, n);
                }
            }
        }
    }

    /// Each range with `hi` up to 100, against each integer up to 1,000.
    #[test]
    fn small_ranges_are_exact() {
        for hi in 1..=100 {
            for lo in 1..=hi {
                let pattern = range_pattern(lo, hi);
                let matcher = Matcher::new(&pattern);
                for n in 0..=1000 {
                    check(&matcher, &pattern, lo, hi, n);
                }
            }
        }
    }
}
