//! The client against `fornace-server` with the model, one request.
//!
//! Set `FORNACE_MODEL_DIR` to the absolute path of a Qwen2.5-0.5B directory
//! to run this test.

use std::path::PathBuf;

use fornace_bench::http;
use fornace_bench::servers::{self, ServerEngine};
use fornace_engine::{DeviceKind, Dtype, Engine, KvCacheKind, LoadConfig};
use tokenizers::Tokenizer;

#[test]
fn the_client_times_one_request_of_fornace_server() {
    let Ok(model_dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return;
    };
    let model_dir = PathBuf::from(model_dir);
    let engine = Engine::load(&LoadConfig {
        model_dir: model_dir.clone(),
        dtype: Dtype::F16,
        device: DeviceKind::Cpu,
        max_new_tokens: 16,
        batch: 1,
        kv_cache: KvCacheKind::Prealloc,
        block_size: fornace_engine::DEFAULT_BLOCK_SIZE,
        kv_capacity: None,
        share_prefix: false,
        attention: Default::default(),
        sync_spans: false,
        ignore_eos: true,
        repeat_penalty: 1.0,
        draft: None,
    })
    .expect("load the engine");
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let prompt = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/prompts/alice-128.txt"),
    )
    .unwrap();
    let prompt_tokens = tokenizer.encode(prompt.as_str(), true).unwrap().len();
    let config = fornace_server::Config {
        model_name: "test".into(),
        max_context: 4096,
        repeat_penalty: 1.0,
        max_batch: 8,
    };
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async move {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                tx.send(listener.local_addr().unwrap()).unwrap();
                let app = fornace_server::app(config, tokenizer, move || Ok(engine));
                axum::serve(listener, app).await.unwrap();
            });
    });
    let base = format!("http://{}", rx.recv().unwrap());
    let agent = ureq::Agent::new_with_defaults();
    let body = servers::request_body(ServerEngine::Fornace, &prompt, 16, 1.0, true, false, true);
    let timing = http::stream(&agent, &format!("{base}/v1/completions"), &body).expect("stream");

    let usage = timing.usage.expect("usage in the last chunk");
    assert_eq!(usage.prompt_tokens, prompt_tokens);
    assert_eq!(usage.completion_tokens, 16);
    assert_eq!(timing.finish.as_deref(), Some("length"));
    let ttft = timing.ttft_ms().unwrap();
    assert!(ttft > 0.0, "ttft {ttft}");
    assert!(timing.decode_tok_s(16).unwrap() > 0.0);
    // Fornace sends one line for each token with text.
    assert!(timing.text_lines.len() <= 16 && !timing.text.is_empty());
}
