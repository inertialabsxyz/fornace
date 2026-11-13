//! The HTTP client: one streaming request, timed by the client.
//!
//! The client never uses a time that a server reports about itself. It
//! records the time before it writes the request, and the arrival time of
//! each SSE line with text.

use std::io::{BufRead, BufReader};
use std::time::Instant;

use serde_json::Value;

/// The token counts that a server gives in `usage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// One `data:` line of an SSE stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Line {
    /// A chunk. `text` can be empty, and can hold more than one token.
    Chunk {
        text: String,
        finish: Option<String>,
        usage: Option<Usage>,
    },
    /// `data: [DONE]`.
    Done,
}

/// Parses one line of the stream. `None` for a line that is not `data:`
/// (an empty line, or a comment).
pub fn parse_line(line: &str) -> Result<Option<Line>, String> {
    let Some(data) = line.strip_prefix("data:") else {
        return Ok(None);
    };
    let data = data.trim();
    if data == "[DONE]" {
        return Ok(Some(Line::Done));
    }
    let v: Value = serde_json::from_str(data).map_err(|e| format!("bad SSE data {data:?}: {e}"))?;
    if let Some(error) = v.get("error") {
        return Err(format!("the server sent an error: {error}"));
    }
    let choice = &v["choices"][0];
    let text = choice["text"]
        .as_str()
        .or_else(|| choice["delta"]["content"].as_str())
        .unwrap_or("")
        .to_string();
    Ok(Some(Line::Chunk {
        text,
        finish: choice["finish_reason"].as_str().map(str::to_string),
        usage: usage(&v),
    }))
}

/// `usage` of a response or a chunk, if it has one.
pub fn usage(v: &Value) -> Option<Usage> {
    let u = v.get("usage")?;
    Some(Usage {
        prompt_tokens: u["prompt_tokens"].as_u64()? as usize,
        completion_tokens: u["completion_tokens"].as_u64()? as usize,
    })
}

/// The times and the counts of one streaming request.
#[derive(Debug, Clone)]
pub struct Timing {
    pub t_send: Instant,
    /// The arrival of each line with text.
    pub text_lines: Vec<Instant>,
    pub text: String,
    pub finish: Option<String>,
    pub usage: Option<Usage>,
}

impl Timing {
    /// The time to the first line with text, in milliseconds.
    pub fn ttft_ms(&self) -> Option<f64> {
        let first = self.text_lines.first()?;
        Some(first.duration_since(self.t_send).as_secs_f64() * 1e3)
    }

    /// (n − 1) / (t_last − t_first): the first token belongs to the prefill.
    pub fn decode_tok_s(&self, n: usize) -> Option<f64> {
        let (first, last) = (self.text_lines.first()?, self.text_lines.last()?);
        let secs = last.duration_since(*first).as_secs_f64();
        (n > 1 && secs > 0.0).then(|| (n - 1) as f64 / secs)
    }

    /// The mean gap between the last 64 lines with text, in milliseconds,
    /// if each line holds one token (n lines). Otherwise `None`.
    pub fn decode_ms_mean_last64(&self, n: usize) -> Option<f64> {
        if self.text_lines.len() != n || n < 2 {
            return None;
        }
        let gaps: Vec<f64> = self
            .text_lines
            .windows(2)
            .map(|w| w[1].duration_since(w[0]).as_secs_f64() * 1e3)
            .collect();
        let last = &gaps[gaps.len().saturating_sub(64)..];
        Some(last.iter().sum::<f64>() / last.len() as f64)
    }
}

/// Sends one streaming request and reads the stream to `[DONE]`.
pub fn stream(agent: &ureq::Agent, url: &str, body: &Value) -> Result<Timing, String> {
    let t_send = Instant::now();
    let response = agent
        .post(url)
        .send_json(body)
        .map_err(|e| format!("POST {url}: {e}"))?;
    let reader = BufReader::new(response.into_body().into_reader());
    let mut timing = Timing {
        t_send,
        text_lines: Vec::new(),
        text: String::new(),
        finish: None,
        usage: None,
    };
    for line in reader.lines() {
        let line = line.map_err(|e| format!("read the stream of {url}: {e}"))?;
        match parse_line(&line)? {
            None => {}
            Some(Line::Done) => return Ok(timing),
            Some(Line::Chunk {
                text,
                finish,
                usage,
            }) => {
                if !text.is_empty() {
                    timing.text_lines.push(Instant::now());
                    timing.text.push_str(&text);
                }
                timing.finish = finish.or(timing.finish);
                timing.usage = usage.or(timing.usage);
            }
        }
    }
    Err(format!("the stream of {url} ended without [DONE]"))
}

/// Sends one request without a stream, and returns the usage.
pub fn complete(agent: &ureq::Agent, url: &str, body: &Value) -> Result<(Value, Usage), String> {
    let v: Value = agent
        .post(url)
        .send_json(body)
        .map_err(|e| format!("POST {url}: {e}"))?
        .body_mut()
        .read_json()
        .map_err(|e| format!("read the response of {url}: {e}"))?;
    let u = usage(&v).ok_or_else(|| format!("the response of {url} has no usage: {v}"))?;
    Ok((v, u))
}

/// The value at percentile `p` (0 to 100), by the nearest rank. `None` for
/// no values.
pub fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    Some(sorted[rank.clamp(1, sorted.len()) - 1])
}

/// All generated tokens of a level over the wall time of the level.
pub fn total_tok_s(tokens: usize, wall_secs: f64) -> f64 {
    if wall_secs > 0.0 {
        tokens as f64 / wall_secs
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn chunk(text: &str) -> Line {
        Line::Chunk {
            text: text.into(),
            finish: None,
            usage: None,
        }
    }

    #[test]
    fn a_text_line_gives_its_text() {
        let line = r#"data: {"choices":[{"text":" Madrid","index":0,"finish_reason":null}]}"#;
        assert_eq!(parse_line(line), Ok(Some(chunk(" Madrid"))));
        let chat = r#"data: {"choices":[{"delta":{"content":"Hi"},"index":0}]}"#;
        assert_eq!(parse_line(chat), Ok(Some(chunk("Hi"))));
    }

    #[test]
    fn a_line_with_two_tokens_is_one_chunk() {
        // One line can hold more than one token, so the client takes n from
        // the usage and does not count lines.
        let line = r#"data: {"choices":[{"text":" 🦀 Rust","index":0}]}"#;
        assert_eq!(parse_line(line), Ok(Some(chunk(" 🦀 Rust"))));
    }

    #[test]
    fn the_usage_line_gives_the_counts_and_the_finish_reason() {
        let line = r#"data: {"choices":[{"text":"","finish_reason":"length"}],"usage":{"prompt_tokens":513,"completion_tokens":128,"total_tokens":641}}"#;
        assert_eq!(
            parse_line(line),
            Ok(Some(Line::Chunk {
                text: String::new(),
                finish: Some("length".into()),
                usage: Some(Usage {
                    prompt_tokens: 513,
                    completion_tokens: 128,
                }),
            }))
        );
    }

    #[test]
    fn done_other_lines_and_errors() {
        assert_eq!(parse_line("data: [DONE]"), Ok(Some(Line::Done)));
        assert_eq!(parse_line(""), Ok(None));
        assert_eq!(parse_line(": keep-alive"), Ok(None));
        assert!(parse_line(r#"data: {"error":{"message":"x"}}"#).is_err());
        assert!(parse_line("data: {").is_err());
    }

    #[test]
    fn timing_uses_the_first_and_the_last_text_line() {
        let t = Instant::now();
        let ms = |n| t + Duration::from_millis(n);
        let timing = Timing {
            t_send: t,
            text_lines: vec![ms(100), ms(110), ms(120), ms(130), ms(140)],
            text: String::new(),
            finish: None,
            usage: None,
        };
        assert_eq!(timing.ttft_ms(), Some(100.0));
        // 4 tokens after the first in 40 ms.
        let tok_s = timing.decode_tok_s(5).unwrap();
        assert!((tok_s - 100.0).abs() < 1e-9, "{tok_s}");
        assert!((timing.decode_ms_mean_last64(5).unwrap() - 10.0).abs() < 1e-9);
        // 6 tokens in 5 lines: one line held two tokens.
        assert_eq!(timing.decode_ms_mean_last64(6), None);
    }

    #[test]
    fn percentile_is_the_nearest_rank() {
        let v = [5.0, 1.0, 4.0, 2.0, 3.0];
        assert_eq!(percentile(&v, 50.0), Some(3.0));
        assert_eq!(percentile(&v, 100.0), Some(5.0));
        assert_eq!(percentile(&v, 0.0), Some(1.0));
        assert_eq!(percentile(&[7.0, 9.0], 50.0), Some(7.0));
        assert_eq!(percentile(&[], 50.0), None);
    }

    #[test]
    fn total_tok_s_is_all_tokens_over_the_wall_time() {
        assert_eq!(total_tok_s(512, 2.0), 256.0);
        assert_eq!(total_tok_s(512, 0.0), 0.0);
    }
}
