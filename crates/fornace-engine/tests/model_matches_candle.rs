//! Fornace's copy of Qwen2 gives the same output as Candle's original.
//!
//! Set `FORNACE_MODEL_DIR` to a Qwen2.5 directory to run this test. It runs
//! on the CPU in F32, so the result is deterministic.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2 as candle_qwen2;
use fornace_engine::KvCacheKind;
use fornace_engine::model as fornace_qwen2;
use tokenizers::Tokenizer;

const PROMPT: &str = "The capital of Spain is";
const DECODE_STEPS: usize = 32;

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// Runs a prefill, then greedy decode steps. Returns the prefill logits and
/// the generated token ids.
fn run(
    mut forward: impl FnMut(&Tensor, usize) -> candle_core::Result<Tensor>,
    prompt: &[u32],
    device: &Device,
) -> (Vec<f32>, Vec<u32>) {
    let input = Tensor::new(prompt, device).unwrap().unsqueeze(0).unwrap();
    let prefill: Vec<f32> = forward(&input, 0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let mut tokens = vec![argmax(&prefill)];
    for pos in prompt.len()..prompt.len() + DECODE_STEPS - 1 {
        let input = Tensor::new(&[*tokens.last().unwrap()], device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let logits: Vec<f32> = forward(&input, pos)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        tokens.push(argmax(&logits));
    }
    (prefill, tokens)
}

#[test]
fn fornace_qwen2_matches_candle_qwen2() {
    let Ok(model_dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return;
    };
    let dir = Path::new(&model_dir);
    let device = Device::Cpu;
    let config_json = std::fs::read(dir.join("config.json")).expect("read config.json");
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).expect("load the tokenizer");
    let prompt = tokenizer
        .encode(PROMPT, true)
        .expect("encode the prompt")
        .get_ids()
        .to_vec();
    // SAFETY: the test does not change the weights file while it is mapped.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &device)
    }
    .expect("map the weights");

    let config: candle_qwen2::Config = serde_json::from_slice(&config_json).unwrap();
    let mut candle = candle_qwen2::ModelForCausalLM::new(&config, vb.clone()).unwrap();
    let (candle_prefill, candle_tokens) = run(|x, pos| candle.forward(x, pos), &prompt, &device);

    let config: fornace_qwen2::Config = serde_json::from_slice(&config_json).unwrap();
    let mut fornace = fornace_qwen2::ModelForCausalLM::new(&config, vb, false).unwrap();
    for kind in [
        KvCacheKind::Cat,
        KvCacheKind::CatContiguous,
        KvCacheKind::Prealloc,
    ] {
        // A small capacity, so that the prealloc cache must also grow.
        fornace.reset_kv_cache(kind, prompt.len() + 4);
        let (fornace_prefill, fornace_tokens) =
            run(|x, pos| fornace.forward(x, pos), &prompt, &device);

        assert!(
            fornace_prefill == candle_prefill,
            "{kind}: the prefill logits are different"
        );
        assert_eq!(fornace_tokens, candle_tokens, "{kind}");
    }
}
