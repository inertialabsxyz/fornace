use std::path::PathBuf;

use fornace_engine::{DeviceKind, Dtype, Engine, LoadConfig};
use tokenizers::Tokenizer;

/// For 1,000 random token IDs, the bytes of `token_bytes` equal the UTF-8
/// bytes of the decoded token, where the token is complete UTF-8.
///
/// Set `FORNACE_MODEL_DIR` to the absolute path of a Qwen2.5-0.5B directory
/// to run this test.
#[test]
fn token_bytes_equal_the_decoded_text() {
    let Ok(model_dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return;
    };
    let model_dir = PathBuf::from(model_dir);
    let config = LoadConfig {
        model_dir: model_dir.clone(),
        dtype: Dtype::F16,
        device: DeviceKind::Cpu,
        max_new_tokens: 1,
        batch: 1,
        kv_cache: Default::default(),
        block_size: fornace_engine::DEFAULT_BLOCK_SIZE,
        kv_capacity: None,
        share_prefix: false,
        attention: Default::default(),
        sync_spans: false,
        ignore_eos: false,
        repeat_penalty: fornace_engine::DEFAULT_REPEAT_PENALTY,
        draft: None,
    };
    let engine = Engine::load(&config).expect("load the engine");
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("tokenizer");
    let bytes = engine.token_bytes();
    assert_eq!(bytes.len(), tokenizer.get_vocab_size(true));

    // A fixed linear congruential generator, so the IDs are the same in each run.
    let mut state = 42u64;
    let mut compared = 0;
    for _ in 0..1000 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let id = ((state >> 33) % bytes.len() as u64) as u32;
        let text = tokenizer.decode(&[id], false).expect("decode");
        if text.contains('\u{FFFD}') {
            continue;
        }
        assert_eq!(bytes[id as usize], text.as_bytes(), "token {id}");
        compared += 1;
    }
    assert!(compared > 900, "only {compared} complete tokens");

    // Known tokens of Qwen2.5: a space before a word, the newline, a special
    // token, and one byte that is not complete UTF-8.
    let id = |t: &str| tokenizer.token_to_id(t).expect(t) as usize;
    assert_eq!(bytes[id("Ġthe")], b" the");
    assert_eq!(bytes[id("Ċ")], b"\n");
    assert_eq!(bytes[id("<|im_end|>")], b"<|im_end|>");
    assert_eq!(bytes[id("Ã")], [0xC3]);
}
