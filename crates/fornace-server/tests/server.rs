//! The server with the model, over HTTP on a free port.
//!
//! Set `FORNACE_MODEL_DIR` to the absolute path of a Qwen2.5-0.5B directory
//! to run these tests.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use fornace_engine::{DeviceKind, Dtype, Engine, KvCacheKind, LoadConfig};
use fornace_server::Config;
use serde_json::{Value, json};
use tokenizers::Tokenizer;

/// The last token of this prompt, after 32 new tokens, ends inside a
/// character (see the stream tests of the engine).
const PROMPT: &str = "Write a short list of words with accents and emoji: café, 🦀,";

fn load_config(max_new_tokens: usize) -> Option<LoadConfig> {
    let Ok(model_dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return None;
    };
    Some(LoadConfig {
        model_dir: PathBuf::from(model_dir),
        dtype: Dtype::F16,
        device: DeviceKind::Cpu,
        max_new_tokens,
        batch: 1,
        kv_cache: KvCacheKind::Prealloc,
        block_size: fornace_engine::DEFAULT_BLOCK_SIZE,
        kv_capacity: None,
        share_prefix: false,
        attention: Default::default(),
        sync_spans: false,
        ignore_eos: true,
        repeat_penalty: fornace_engine::DEFAULT_REPEAT_PENALTY,
        draft: None,
    })
}

/// Starts the server with `engine` on a free port, waits until `/health`
/// gives 200, and returns the base URL.
fn start(engine: Engine, model_dir: PathBuf, max_batch: usize) -> String {
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let config = Config {
        model_name: "test".into(),
        max_context: 4096,
        repeat_penalty: fornace_engine::DEFAULT_REPEAT_PENALTY,
        max_batch,
    };
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(listener.local_addr().unwrap()).unwrap();
            let app = fornace_server::app(config, tokenizer, move || Ok(engine));
            axum::serve(listener, app).await.unwrap();
        });
    });
    let base = format!("http://{}", rx.recv().unwrap());
    for _ in 0..100 {
        if ureq::get(format!("{base}/health")).call().is_ok() {
            return base;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    panic!("the server is not ready");
}

/// One streaming request: the text of each chunk with its arrival time, and
/// the last chunk (with the usage).
struct Streamed {
    parts: Vec<(Instant, String)>,
    last: Value,
}

fn stream(base: &str, body: Value) -> Streamed {
    let response = ureq::post(format!("{base}/v1/completions"))
        .send_json(body)
        .expect("stream request");
    let reader = BufReader::new(response.into_body().into_reader());
    let mut parts = Vec::new();
    let mut last = Value::Null;
    let mut done = false;
    for line in reader.lines() {
        let line = line.unwrap();
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            done = true;
            break;
        }
        let chunk: Value = serde_json::from_str(data).unwrap();
        if chunk.get("usage").is_some() {
            last = chunk;
            continue;
        }
        let text = chunk["choices"][0]["text"].as_str().unwrap().to_string();
        if !text.is_empty() {
            parts.push((Instant::now(), text));
        }
    }
    assert!(done, "the stream has no [DONE]");
    Streamed { parts, last }
}

/// A stream and a response without a stream give the text of
/// `Engine::generate` with the same options.
#[test]
fn stream_and_full_response_equal_generate() {
    let Some(config) = load_config(32) else {
        return;
    };
    let mut engine = Engine::load(&config).expect("load the engine");
    let expected = engine.generate(PROMPT).expect("generate").texts.remove(0);
    let base = start(engine, config.model_dir, 8);

    let body = json!({"prompt": PROMPT, "max_tokens": 32, "ignore_eos": true});
    let full: Value = ureq::post(format!("{base}/v1/completions"))
        .send_json(&body)
        .expect("request")
        .body_mut()
        .read_json()
        .unwrap();
    assert_eq!(full["choices"][0]["text"], expected);
    assert_eq!(full["choices"][0]["finish_reason"], "length");
    assert_eq!(full["usage"]["completion_tokens"], 32);

    let mut body = body;
    body["stream"] = json!(true);
    let streamed = stream(&base, body);
    let text: String = streamed.parts.iter().map(|(_, t)| t.as_str()).collect();
    assert_eq!(text, expected);
    assert_eq!(streamed.last["choices"][0]["finish_reason"], "length");
    assert_eq!(streamed.last["usage"]["completion_tokens"], 32);
}

/// With `max_batch` 1 (a queue), two requests at the same time both
/// complete, one after the other: the first token of the second comes
/// after the last token of the first.
#[test]
fn two_requests_run_one_after_the_other() {
    let Some(config) = load_config(16) else {
        return;
    };
    let engine = Engine::load(&config).expect("load the engine");
    let base = start(engine, config.model_dir, 1);
    let body = json!({"prompt": PROMPT, "max_tokens": 16, "ignore_eos": true, "stream": true});
    let threads: Vec<_> = (0..2)
        .map(|_| {
            let (base, body) = (base.clone(), body.clone());
            std::thread::spawn(move || stream(&base, body))
        })
        .collect();
    let mut runs: Vec<Streamed> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    for run in &runs {
        assert_eq!(run.last["usage"]["completion_tokens"], 16);
    }
    runs.sort_by_key(|r| r.parts[0].0);
    let first_last = runs[0].parts.last().unwrap().0;
    let second_first = runs[1].parts[0].0;
    assert!(
        second_first > first_last,
        "the second request started before the first ended"
    );
}

/// A request above the context gives 400 and names the field.
#[test]
fn a_request_above_the_context_names_max_tokens() {
    let Some(config) = load_config(16) else {
        return;
    };
    let engine = Engine::load(&config).expect("load the engine");
    let base = start(engine, config.model_dir, 8);
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into();
    let mut response = agent
        .post(format!("{base}/v1/completions"))
        .send_json(json!({"prompt": PROMPT, "max_tokens": 5000}))
        .unwrap();
    assert_eq!(response.status(), 400);
    let body: Value = response.body_mut().read_json().unwrap();
    assert_eq!(body["error"]["param"], "max_tokens");
}

/// With a batch, two requests at the same time run together: the first
/// token of the second comes before the last token of the first. Both give
/// the text of `Engine::generate`.
#[test]
fn two_requests_run_together_in_one_batch() {
    let Some(config) = load_config(24) else {
        return;
    };
    let mut engine = Engine::load(&config).expect("load the engine");
    let second = "The capital of Spain is";
    let expected = [
        engine.generate(PROMPT).expect("generate").texts.remove(0),
        engine.generate(second).expect("generate").texts.remove(0),
    ];
    let base = start(engine, config.model_dir, 8);
    let threads: Vec<_> = [PROMPT, second]
        .into_iter()
        .enumerate()
        .map(|(i, prompt)| {
            let base = base.clone();
            let body =
                json!({"prompt": prompt, "max_tokens": 24, "ignore_eos": true, "stream": true});
            std::thread::spawn(move || {
                // The second request starts a little later, so that it joins
                // a running batch.
                std::thread::sleep(std::time::Duration::from_millis(150 * i as u64));
                stream(&base, body)
            })
        })
        .collect();
    let runs: Vec<Streamed> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    for (run, want) in runs.iter().zip(&expected) {
        let text: String = run.parts.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(&text, want);
        assert_eq!(run.last["usage"]["completion_tokens"], 24);
    }
    let first_last = runs[0].parts.last().unwrap().0;
    let second_first = runs[1].parts[0].0;
    assert!(
        second_first < first_last,
        "the second request waited for the first to end"
    );
}
