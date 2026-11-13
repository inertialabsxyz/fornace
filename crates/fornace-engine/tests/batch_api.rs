//! The batch API: requests that join and leave one batch give the tokens of
//! each request alone.
//!
//! Set `FORNACE_MODEL_DIR` to the absolute path of a Qwen2.5-0.5B directory
//! to run these tests. They run on the CPU in F32.

use std::cell::RefCell;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::rc::Rc;

use fornace_engine::batch::{Batch, Done, Finish, OnToken};
use fornace_engine::{DeviceKind, Dtype, Engine, GenerateOptions, KvCacheKind, LoadConfig};

const PROMPTS: [&str; 3] = [
    "The capital of Spain is",
    "Write a short list of words with accents and emoji: café, 🦀,",
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading,",
];

fn engine() -> Option<Engine> {
    let Ok(model_dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return None;
    };
    Some(
        Engine::load(&LoadConfig {
            model_dir: PathBuf::from(model_dir),
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
            repeat_penalty: fornace_engine::DEFAULT_REPEAT_PENALTY,
            draft: None,
        })
        .expect("load the engine"),
    )
}

fn options(max_new_tokens: usize, ignore_eos: bool) -> GenerateOptions {
    GenerateOptions {
        max_new_tokens,
        ignore_eos,
        repeat_penalty: fornace_engine::DEFAULT_REPEAT_PENALTY,
    }
}

/// The tokens and the text of `prompt` alone, from `generate_stream`.
fn alone(engine: &mut Engine, prompt: &str, options: &GenerateOptions) -> (Vec<u32>, String) {
    let report = engine
        .generate_stream(prompt, options, &mut |_| ControlFlow::Continue(()))
        .expect("stream");
    (report.tokens[0].clone(), report.texts[0].clone())
}

/// The ids and the text of the events of one row.
type Seen = Rc<RefCell<(Vec<u32>, String)>>;

/// A callback that records the ids and the text of its events, and breaks
/// after `stop_after` events.
fn recorder(stop_after: Option<usize>) -> (OnToken, Seen) {
    let seen = Rc::new(RefCell::new((Vec::new(), String::new())));
    let inner = seen.clone();
    let callback: OnToken = Box::new(move |event| {
        let mut s = inner.borrow_mut();
        s.0.push(event.id);
        s.1.push_str(&event.text);
        if stop_after == Some(s.0.len()) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    (callback, seen)
}

/// Steps until the batch is empty, and returns all rows that left.
fn drain(batch: &mut Batch, engine: &mut Engine, mut done: Vec<Done>) -> Vec<Done> {
    while !batch.is_empty() {
        done.extend(batch.step(engine).expect("step"));
    }
    done.sort_by_key(|d| d.id);
    done
}

/// Three requests of different lengths join at different steps, and give
/// the tokens and the text of each request alone.
#[test]
fn rows_that_join_at_different_steps_give_the_tokens_alone() {
    let Some(mut engine) = engine() else {
        return;
    };
    let opts = options(12, true);
    let expected: Vec<_> = PROMPTS
        .iter()
        .map(|p| alone(&mut engine, p, &opts))
        .collect();

    let mut batch = Batch::new();
    let mut seen = Vec::new();
    let mut done = Vec::new();
    for (i, prompt) in PROMPTS.iter().enumerate() {
        if i == 2 {
            // The third request joins after two steps.
            done.extend(batch.step(&mut engine).unwrap());
            done.extend(batch.step(&mut engine).unwrap());
        }
        let (callback, record) = recorder(None);
        let (_, left) = batch.join(&mut engine, prompt, &opts, callback).unwrap();
        assert!(left.is_none());
        seen.push(record);
    }
    let done = drain(&mut batch, &mut engine, done);
    assert_eq!(done.len(), 3);
    for (i, row) in done.iter().enumerate() {
        assert_eq!(row.finish, Finish::Length, "row {i}");
        assert_eq!(row.tokens, expected[i].0, "row {i}");
        assert_eq!(row.text, expected[i].1, "row {i}");
        assert_eq!(seen[i].borrow().0, expected[i].0, "row {i}: the events");
    }
}

/// A row whose callback breaks leaves the batch with its tokens so far. The
/// other rows go on with the tokens of each request alone.
#[test]
fn a_row_that_breaks_leaves_and_the_others_go_on() {
    let Some(mut engine) = engine() else {
        return;
    };
    let opts = options(10, true);
    let expected: Vec<_> = PROMPTS
        .iter()
        .map(|p| alone(&mut engine, p, &opts))
        .collect();
    let mut batch = Batch::new();
    for (i, prompt) in PROMPTS.iter().enumerate() {
        let (callback, _) = recorder((i == 1).then_some(3));
        batch.join(&mut engine, prompt, &opts, callback).unwrap();
    }
    let done = drain(&mut batch, &mut engine, Vec::new());
    assert_eq!(done[1].finish, Finish::Stopped);
    assert_eq!(done[1].tokens, expected[1].0[..3]);
    for i in [0, 2] {
        assert_eq!(done[i].finish, Finish::Length);
        assert_eq!(done[i].tokens, expected[i].0, "row {i}");
    }
}

/// EOS ends a row (EOS is not in its tokens), as in `generate_stream`. A
/// request of one token leaves at its join.
#[test]
fn eos_and_max_tokens_end_a_row() {
    let Some(mut engine) = engine() else {
        return;
    };
    let chat = "<|im_start|>user\nName the capital of Spain in one word.<|im_end|>\n<|im_start|>assistant\n";
    let opts = options(32, false);
    let expected = alone(&mut engine, chat, &opts);
    assert!(
        expected.0.len() < 32,
        "the prompt must end before 32 tokens"
    );

    let mut batch = Batch::new();
    let (callback, _) = recorder(None);
    batch.join(&mut engine, chat, &opts, callback).unwrap();
    let (callback, _) = recorder(None);
    batch
        .join(&mut engine, PROMPTS[0], &options(32, true), callback)
        .unwrap();
    let (callback, _) = recorder(None);
    let (_, one) = batch
        .join(&mut engine, PROMPTS[2], &options(1, true), callback)
        .unwrap();
    let one = one.expect("a request of one token leaves at its join");
    assert_eq!(one.finish, Finish::Length);
    assert_eq!(one.tokens.len(), 1);

    let done = drain(&mut batch, &mut engine, Vec::new());
    assert_eq!(done[0].finish, Finish::Eos);
    assert_eq!(done[0].tokens, expected.0);
    assert_eq!(done[1].finish, Finish::Length);
    assert_eq!(done[1].tokens.len(), 32);
}
