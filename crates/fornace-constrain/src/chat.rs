//! The chat template of Qwen2.5. A copy of `chat_prompt` in
//! `fornace-server`, so that this crate does not depend on the server.

/// The system message of `fornace-server` when the request has none.
pub const DEFAULT_SYSTEM: &str =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

/// The prompt for one system message and one user message, up to the start
/// of the reply of the assistant.
pub fn chat_prompt(system: &str, user: &str) -> String {
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )
}
