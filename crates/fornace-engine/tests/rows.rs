//! Rows with their own cache and position (`ModelForCausalLM::forward_rows`)
//! give the same tokens in one batch as alone.
//!
//! Set `FORNACE_MODEL_DIR` to the absolute path of a Qwen2.5-0.5B directory
//! to run these tests. They run on the CPU in F32, where a different token
//! is a bug, not a near tie.

use std::path::PathBuf;

use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;
use fornace_engine::KvCacheKind;
use fornace_engine::model::{Config, ModelForCausalLM, RowCache};
use tokenizers::Tokenizer;

const STEPS: usize = 8;

fn setup() -> Option<(ModelForCausalLM, Vec<Vec<u32>>)> {
    let Ok(dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return None;
    };
    let dir = PathBuf::from(dir);
    let config: Config =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    // SAFETY: the test does not change the weights file while it is mapped.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[dir.join("model.safetensors")],
            DType::F32,
            &Device::Cpu,
        )
    }
    .unwrap();
    let model = ModelForCausalLM::new(&config, vb, false).unwrap();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/prompts/alice-512.txt"),
    )
    .unwrap();
    let ids = tokenizer
        .encode(text.as_str(), true)
        .unwrap()
        .get_ids()
        .to_vec();
    // Three prompts of different lengths, from different parts of the text.
    let prompts = vec![
        ids[..20].to_vec(),
        ids[40..75].to_vec(),
        ids[100..150].to_vec(),
    ];
    Some((model, prompts))
}

fn argmax(logits: &Tensor) -> Vec<u32> {
    logits
        .squeeze(1)
        .unwrap()
        .argmax(D::Minus1)
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Prefills `prompt` into a new cache, and returns the cache and the first
/// token.
fn prefill(model: &mut ModelForCausalLM, prompt: &[u32]) -> (RowCache, u32) {
    let mut cache = model.new_row_cache(prompt.len() + STEPS + 1);
    let input = Tensor::new(prompt, &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let logits = model.forward_rows(&input, &mut [&mut cache]).unwrap();
    (cache, argmax(&logits)[0])
}

/// `STEPS` greedy tokens of `prompt`, decoded alone.
fn alone(model: &mut ModelForCausalLM, prompt: &[u32]) -> Vec<u32> {
    let (mut cache, mut next) = prefill(model, prompt);
    let mut tokens = vec![next];
    for _ in 1..STEPS {
        let input = Tensor::new(&[[next]], &Device::Cpu).unwrap();
        next = argmax(&model.forward_rows(&input, &mut [&mut cache]).unwrap())[0];
        tokens.push(next);
    }
    tokens
}

/// One row alone with `forward_rows` gives the tokens of the old path,
/// `forward` with one batch cache.
#[test]
fn one_row_gives_the_tokens_of_forward() {
    let Some((mut model, prompts)) = setup() else {
        return;
    };
    let prompt = &prompts[1];
    model.reset_kv_cache(KvCacheKind::Prealloc, prompt.len() + STEPS);
    let input = Tensor::new(prompt.as_slice(), &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let mut next = argmax(&model.forward(&input, 0).unwrap())[0];
    let mut old = vec![next];
    for i in 1..STEPS {
        let input = Tensor::new(&[[next]], &Device::Cpu).unwrap();
        next = argmax(&model.forward(&input, prompt.len() + i - 1).unwrap())[0];
        old.push(next);
    }
    assert_eq!(alone(&mut model, prompt), old);
}

/// Three rows of 20, 35, and 50 tokens in one batch give the tokens of each
/// row alone. The third row joins after two steps, and the first row leaves
/// after five.
#[test]
fn rows_of_different_lengths_give_the_tokens_of_each_row_alone() {
    let Some((mut model, prompts)) = setup() else {
        return;
    };
    let expected: Vec<Vec<u32>> = prompts.iter().map(|p| alone(&mut model, p)).collect();

    struct Row {
        id: usize,
        cache: RowCache,
        tokens: Vec<u32>,
    }
    let mut rows: Vec<Row> = Vec::new();
    let mut done: Vec<Row> = Vec::new();
    for (id, prompt) in prompts.iter().enumerate().take(2) {
        let (cache, first) = prefill(&mut model, prompt);
        rows.push(Row {
            id,
            cache,
            tokens: vec![first],
        });
    }
    let mut step = 0;
    while !rows.is_empty() {
        if step == 2 {
            let (cache, first) = prefill(&mut model, &prompts[2]);
            rows.push(Row {
                id: 2,
                cache,
                tokens: vec![first],
            });
        }
        // A row leaves after five steps, or when it has all its tokens.
        let (leave, stay): (Vec<Row>, Vec<Row>) = rows
            .into_iter()
            .partition(|r| r.tokens.len() == STEPS || (r.id == 0 && step == 5));
        done.extend(leave);
        rows = stay;
        if rows.is_empty() {
            break;
        }
        let next: Vec<u32> = rows.iter().map(|r| *r.tokens.last().unwrap()).collect();
        let input = Tensor::from_vec(next, (rows.len(), 1), &Device::Cpu).unwrap();
        let mut caches: Vec<&mut RowCache> = rows.iter_mut().map(|r| &mut r.cache).collect();
        let logits = model.forward_rows(&input, &mut caches).unwrap();
        for (row, token) in rows.iter_mut().zip(argmax(&logits)) {
            row.tokens.push(token);
        }
        step += 1;
    }
    done.sort_by_key(|r| r.id);
    for row in &done {
        let want = &expected[row.id];
        assert_eq!(row.tokens, want[..row.tokens.len()], "row {}", row.id);
        assert_eq!(
            row.cache.len(),
            prompts[row.id].len() + row.tokens.len() - 1
        );
    }
    // Row 0 left early, the others have all their tokens.
    assert_eq!(done[0].tokens.len(), 6);
    assert_eq!(done[1].tokens.len(), STEPS);
    assert_eq!(done[2].tokens.len(), STEPS);
}
