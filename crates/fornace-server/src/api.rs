//! The OpenAI request and response shapes, with no I/O.

use serde::Deserialize;
use serde_json::{Value, json};

/// The endpoint of a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `/v1/completions`: a raw prompt.
    Completion,
    /// `/v1/chat/completions`: messages in the Qwen chat template.
    Chat,
}

/// The body of a request to either endpoint. Fields that the server does
/// not know are ignored.
#[derive(Debug, Default, Deserialize)]
pub struct Request {
    pub prompt: Option<Value>,
    pub messages: Option<Value>,
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub n: Option<usize>,
    pub logprobs: Option<Value>,
    pub stop: Option<Value>,
    /// An extension, as in llama.cpp: do not stop at the end-of-sequence
    /// token.
    #[serde(default)]
    pub ignore_eos: bool,
    /// An extension: the repeat penalty. 1.0 turns it off.
    pub repeat_penalty: Option<f32>,
}

/// A message of a chat request.
#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// A request that passed the checks.
#[derive(Debug, PartialEq)]
pub struct Params {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub stream: bool,
    pub stop: Vec<String>,
    pub ignore_eos: bool,
    pub repeat_penalty: Option<f32>,
}

/// A request that the server cannot do. It names the field.
#[derive(Debug, PartialEq)]
pub struct ApiError {
    pub field: &'static str,
    pub message: String,
}

impl ApiError {
    fn new(field: &'static str, message: impl Into<String>) -> ApiError {
        ApiError {
            field,
            message: message.into(),
        }
    }

    /// The error body, in the OpenAI shape.
    pub fn body(&self) -> Value {
        json!({"error": {
            "message": format!("{}: {}", self.field, self.message),
            "type": "invalid_request_error",
            "param": self.field,
        }})
    }
}

/// Checks the request. Greedy decoding is the only sampler, so a field that
/// asks for another sampler is an error.
pub fn validate(req: &Request, kind: Kind) -> Result<Params, ApiError> {
    if req.temperature.is_some_and(|t| t > 0.0) {
        return Err(ApiError::new("temperature", "only 0 (greedy) is supported"));
    }
    if req.top_p.is_some_and(|p| p < 1.0) {
        return Err(ApiError::new("top_p", "only 1 is supported"));
    }
    if req.n.is_some_and(|n| n != 1) {
        return Err(ApiError::new("n", "only 1 is supported"));
    }
    if req
        .logprobs
        .as_ref()
        .is_some_and(|v| !v.is_null() && *v != Value::Bool(false))
    {
        return Err(ApiError::new("logprobs", "is not supported"));
    }
    if req.max_tokens == Some(0) {
        return Err(ApiError::new("max_tokens", "must be 1 or more"));
    }
    if req
        .repeat_penalty
        .is_some_and(|p| p <= 0.0 || !p.is_finite())
    {
        return Err(ApiError::new("repeat_penalty", "must be above 0"));
    }
    let prompt = match kind {
        Kind::Completion => match &req.prompt {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(_)) => {
                return Err(ApiError::new("prompt", "a list is not supported"));
            }
            _ => return Err(ApiError::new("prompt", "must be a string")),
        },
        Kind::Chat => {
            let messages: Vec<Message> = req
                .messages
                .clone()
                .and_then(|v| serde_json::from_value(v).ok())
                .ok_or_else(|| {
                    ApiError::new("messages", "must be a list of {role, content} strings")
                })?;
            if messages.is_empty() {
                return Err(ApiError::new("messages", "must not be empty"));
            }
            chat_prompt(&messages)
        }
    };
    let stop = match &req.stop {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(v) => serde_json::from_value::<Vec<String>>(v.clone())
            .map_err(|_| ApiError::new("stop", "must be a string or a list of strings"))?,
    };
    Ok(Params {
        prompt,
        max_tokens: req.max_tokens,
        stream: req.stream,
        stop: stop.into_iter().filter(|s| !s.is_empty()).collect(),
        ignore_eos: req.ignore_eos,
        repeat_penalty: req.repeat_penalty,
    })
}

/// The tokens to generate. Without `max_tokens`, the rest of the context.
pub fn token_budget(
    prompt_tokens: usize,
    max_tokens: Option<usize>,
    max_context: usize,
) -> Result<usize, ApiError> {
    let room = max_context.saturating_sub(prompt_tokens);
    if room == 0 {
        return Err(ApiError::new(
            "prompt",
            format!("{prompt_tokens} tokens fill the context of {max_context}"),
        ));
    }
    match max_tokens {
        Some(n) if n > room => Err(ApiError::new(
            "max_tokens",
            format!(
                "{prompt_tokens} prompt tokens and {n} new tokens are above the context of {max_context}"
            ),
        )),
        Some(n) => Ok(n),
        None => Ok(room),
    }
}

const DEFAULT_SYSTEM: &str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

/// The fixed Qwen2.5 chat template, without tools. As in the template of
/// `tokenizer_config.json`, a chat with no system message first gets the
/// default system message.
pub fn chat_prompt(messages: &[Message]) -> String {
    let mut prompt = String::new();
    if messages.first().is_none_or(|m| m.role != "system") {
        prompt.push_str(&format!("<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n"));
    }
    for m in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            m.role, m.content
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// The token counts of a response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

impl Usage {
    fn json(self) -> Value {
        json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        })
    }
}

/// The fields that all responses of one request have in common.
#[derive(Debug, Clone)]
pub struct Shape {
    pub kind: Kind,
    pub id: String,
    pub created: u64,
    pub model: String,
}

impl Shape {
    fn object(&self, chunk: bool) -> &'static str {
        match (self.kind, chunk) {
            (Kind::Completion, _) => "text_completion",
            (Kind::Chat, false) => "chat.completion",
            (Kind::Chat, true) => "chat.completion.chunk",
        }
    }

    fn wrap(&self, chunk: bool, choice: Value) -> Value {
        json!({
            "id": self.id,
            "object": self.object(chunk),
            "created": self.created,
            "model": self.model,
            "choices": [choice],
        })
    }

    fn delta_choice(&self, delta: Value, finish: Option<&str>) -> Value {
        match self.kind {
            Kind::Completion => json!({
                "index": 0,
                "text": delta.get("content").and_then(Value::as_str).unwrap_or(""),
                "logprobs": null,
                "finish_reason": finish,
            }),
            Kind::Chat => json!({"index": 0, "delta": delta, "finish_reason": finish}),
        }
    }

    /// The first chunk of a chat stream: the role, with no text.
    pub fn role_chunk(&self) -> Value {
        let delta = json!({"role": "assistant", "content": ""});
        self.wrap(true, self.delta_choice(delta, None))
    }

    /// A chunk with new text.
    pub fn chunk(&self, text: &str) -> Value {
        let choice = self.delta_choice(json!({ "content": text }), None);
        self.wrap(true, choice)
    }

    /// The last chunk of a stream: no text, the finish reason, and the usage.
    pub fn last_chunk(&self, finish: &str, usage: Usage) -> Value {
        let delta = match self.kind {
            Kind::Completion => json!({"content": ""}),
            Kind::Chat => json!({}),
        };
        let mut v = self.wrap(true, self.delta_choice(delta, Some(finish)));
        v["usage"] = usage.json();
        v
    }

    /// The response without a stream.
    pub fn full(&self, text: &str, finish: &str, usage: Usage) -> Value {
        let choice = match self.kind {
            Kind::Completion => {
                json!({"index": 0, "text": text, "logprobs": null, "finish_reason": finish})
            }
            Kind::Chat => json!({
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }),
        };
        let mut v = self.wrap(false, choice);
        v["usage"] = usage.json();
        v
    }
}

/// Holds back the text that can be the start of a stop string, and finds
/// the first stop string.
#[derive(Debug, Default)]
pub struct StopFilter {
    stops: Vec<String>,
    held: String,
}

impl StopFilter {
    pub fn new(stops: Vec<String>) -> StopFilter {
        StopFilter {
            stops,
            held: String::new(),
        }
    }

    /// Adds `text`. Returns the text to send now, and true if a stop string
    /// was found. The stop string and the text after it are not sent.
    pub fn push(&mut self, text: &str) -> (String, bool) {
        self.held.push_str(text);
        if let Some(at) = self.stops.iter().filter_map(|s| self.held.find(s)).min() {
            let out = self.held[..at].to_string();
            self.held.clear();
            return (out, true);
        }
        let keep = self
            .stops
            .iter()
            .flat_map(|s| {
                (1..s.len())
                    .filter(|&k| s.is_char_boundary(k))
                    .map(|k| &s[..k])
            })
            .filter(|start| self.held.ends_with(start))
            .map(str::len)
            .max()
            .unwrap_or(0);
        let out: String = self.held.drain(..self.held.len() - keep).collect();
        (out, false)
    }

    /// The text that is still held back, at the end of the generation.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(v: Value) -> Request {
        serde_json::from_value(v).unwrap()
    }

    fn shape(kind: Kind) -> Shape {
        Shape {
            kind,
            id: "cmpl-1".into(),
            created: 7,
            model: "m".into(),
        }
    }

    #[test]
    fn a_greedy_completion_passes() {
        let p = validate(
            &request(json!({"prompt": "Hi", "max_tokens": 4, "temperature": 0, "stop": "\n"})),
            Kind::Completion,
        )
        .unwrap();
        assert_eq!(p.prompt, "Hi");
        assert_eq!(p.max_tokens, Some(4));
        assert_eq!(p.stop, vec!["\n".to_string()]);
    }

    #[test]
    fn each_unsupported_field_is_named() {
        let cases = [
            (json!({"prompt": "a", "temperature": 0.7}), "temperature"),
            (json!({"prompt": "a", "top_p": 0.9}), "top_p"),
            (json!({"prompt": "a", "n": 2}), "n"),
            (json!({"prompt": "a", "logprobs": 1}), "logprobs"),
            (json!({"prompt": ["a", "b"]}), "prompt"),
            (json!({}), "prompt"),
            (json!({"prompt": "a", "max_tokens": 0}), "max_tokens"),
            (json!({"prompt": "a", "stop": 3}), "stop"),
        ];
        for (body, field) in cases {
            let err = validate(&request(body.clone()), Kind::Completion).unwrap_err();
            assert_eq!(err.field, field, "{body}");
        }
        let err = validate(&request(json!({"messages": []})), Kind::Chat).unwrap_err();
        assert_eq!(err.field, "messages");
    }

    #[test]
    fn the_budget_is_the_context_less_the_prompt() {
        assert_eq!(token_budget(10, Some(5), 16), Ok(5));
        assert_eq!(token_budget(10, None, 16), Ok(6));
        assert_eq!(
            token_budget(10, Some(7), 16).unwrap_err().field,
            "max_tokens"
        );
        assert_eq!(token_budget(16, None, 16).unwrap_err().field, "prompt");
    }

    #[test]
    fn the_chat_template_adds_the_default_system_message() {
        let user = Message {
            role: "user".into(),
            content: "Hi".into(),
        };
        assert_eq!(
            chat_prompt(std::slice::from_ref(&user)),
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful \
             assistant.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
        let system = Message {
            role: "system".into(),
            content: "Be brief.".into(),
        };
        assert_eq!(
            chat_prompt(&[system, user]),
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn completion_chunks_have_the_openai_shape() {
        let s = shape(Kind::Completion);
        assert_eq!(
            s.chunk("Hi"),
            json!({"id": "cmpl-1", "object": "text_completion", "created": 7, "model": "m",
                   "choices": [{"index": 0, "text": "Hi", "logprobs": null, "finish_reason": null}]})
        );
        let usage = Usage {
            prompt_tokens: 3,
            completion_tokens: 2,
        };
        let last = s.last_chunk("length", usage);
        assert_eq!(last["choices"][0]["finish_reason"], "length");
        assert_eq!(last["choices"][0]["text"], "");
        assert_eq!(
            last["usage"],
            json!({"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
        );
        assert_eq!(s.full("Hi", "stop", usage)["choices"][0]["text"], "Hi");
    }

    #[test]
    fn chat_chunks_have_the_openai_shape() {
        let s = shape(Kind::Chat);
        assert_eq!(
            s.role_chunk()["choices"][0]["delta"],
            json!({"role": "assistant", "content": ""})
        );
        let c = s.chunk("Hi");
        assert_eq!(c["object"], "chat.completion.chunk");
        assert_eq!(c["choices"][0]["delta"], json!({"content": "Hi"}));
        let usage = Usage {
            prompt_tokens: 3,
            completion_tokens: 2,
        };
        assert_eq!(
            s.last_chunk("stop", usage)["choices"][0]["delta"],
            json!({})
        );
        let full = s.full("Hi", "stop", usage);
        assert_eq!(full["object"], "chat.completion");
        assert_eq!(
            full["choices"][0]["message"],
            json!({"role": "assistant", "content": "Hi"})
        );
    }

    #[test]
    fn the_stop_filter_cuts_at_the_first_stop_string() {
        let mut f = StopFilter::new(vec!["END".into()]);
        assert_eq!(f.push("one E"), ("one ".to_string(), false));
        assert_eq!(f.push("N"), (String::new(), false));
        assert_eq!(f.push("D two"), (String::new(), true));
        assert_eq!(f.finish(), "");
    }

    #[test]
    fn the_stop_filter_gives_back_held_text_that_is_not_a_stop() {
        let mut f = StopFilter::new(vec!["END".into()]);
        assert_eq!(f.push("a E"), ("a ".to_string(), false));
        assert_eq!(f.push("x"), ("Ex".to_string(), false));
        assert_eq!(f.push("EN"), (String::new(), false));
        assert_eq!(f.finish(), "EN");
        let mut none = StopFilter::new(Vec::new());
        assert_eq!(none.push("abc"), ("abc".to_string(), false));
    }
}
