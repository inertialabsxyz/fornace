//! Fused attention gives almost the same logits as naive attention.
//!
//! Set `FORNACE_MODEL_DIR` to a Qwen2.5 directory, and build with the
//! `metal` feature, to run this test. For each prompt in `bench/prompts/`,
//! it decodes greedily with naive attention in BF16 on Metal. Then it gives
//! the same tokens to the model with fused attention, and compares the
//! logits at each step. It prints the largest difference.
//!
//! BF16 rounds differently in the two kernels, so the top token can change
//! where the two best logits are almost equal. The test fails if the top
//! token changes at a step where the naive margin between the two best
//! logits is larger than the difference between the two logit vectors.

#![cfg(feature = "metal")]

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use fornace_engine::model::{Config, ModelForCausalLM};
use fornace_engine::{AttentionKind, KvCacheKind};
use tokenizers::Tokenizer;

const PROMPTS: [&str; 3] = ["alice-512.txt", "alice-2048.txt", "alice-4096.txt"];
const STEPS: usize = 32;

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best
}

/// The top logit minus the second logit.
fn margin(logits: &[f32]) -> f32 {
    let best = argmax(logits);
    let second = logits
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != best)
        .map(|(_, &v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    logits[best] - second
}

/// Runs a prefill and then `STEPS - 1` decode steps. Each step uses the
/// next token of `forced`, or the top token if `forced` is `None`. Returns
/// the logits of each step and the tokens.
fn run(
    model: &mut ModelForCausalLM,
    attention: AttentionKind,
    prompt: &[u32],
    forced: Option<&[u32]>,
    device: &Device,
) -> (Vec<Vec<f32>>, Vec<u32>) {
    model.set_attention(attention);
    model.reset_kv_cache(KvCacheKind::Prealloc, prompt.len() + STEPS);
    let to_vec = |t: Tensor| -> Vec<f32> {
        t.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    let input = Tensor::new(prompt, device).unwrap().unsqueeze(0).unwrap();
    let mut logits = vec![to_vec(model.forward(&input, 0).unwrap())];
    let mut tokens = Vec::new();
    for step in 0..STEPS - 1 {
        let token = match forced {
            Some(forced) => forced[step],
            None => argmax(logits.last().unwrap()) as u32,
        };
        tokens.push(token);
        let input = Tensor::new(&[token], device).unwrap().unsqueeze(0).unwrap();
        logits.push(to_vec(model.forward(&input, prompt.len() + step).unwrap()));
    }
    (logits, tokens)
}

#[test]
fn fused_logits_are_close_to_naive() {
    let Ok(model_dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return;
    };
    let dir = Path::new(&model_dir);
    let prompts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/prompts");
    let device = Device::new_metal(0).expect("open the metal device");
    let config: Config =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    // SAFETY: the test does not change the weights file while it is mapped.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::BF16, &device)
    }
    .unwrap();
    let mut model = ModelForCausalLM::new(&config, vb, false).unwrap();

    for name in PROMPTS {
        // Read as the CLI does with "$(cat FILE)": without the final newline.
        let text = std::fs::read_to_string(prompts_dir.join(name)).unwrap();
        let ids = tokenizer.encode(text.trim_end_matches('\n'), true).unwrap();
        let prompt = ids.get_ids();
        let (naive, tokens) = run(&mut model, AttentionKind::Naive, prompt, None, &device);
        let (fused, _) = run(
            &mut model,
            AttentionKind::Fused,
            prompt,
            Some(&tokens),
            &device,
        );
        let mut max_diff = [0f32; 2];
        for (step, (n, f)) in naive.iter().zip(&fused).enumerate() {
            let diff = n
                .iter()
                .zip(f)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let phase = usize::from(step > 0);
            max_diff[phase] = max_diff[phase].max(diff);
            if argmax(n) != argmax(f) {
                eprintln!(
                    "{name}: step {step}: the top token changes, naive margin {:.4}, max diff {diff:.4}",
                    margin(n)
                );
                assert!(
                    margin(n) <= diff,
                    "{name}: step {step}: the top token changes, but the logits are not a near-tie"
                );
            }
        }
        eprintln!(
            "{name}: {} tokens, max |naive - fused| = {:.4} after the prefill, {:.4} in {} decode steps",
            prompt.len(),
            max_diff[0],
            max_diff[1],
            STEPS - 1
        );
    }
}
