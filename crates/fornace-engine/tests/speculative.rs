//! Speculative decoding: the verify pass against one-token steps, and the
//! text with a draft against the text of the target alone.
//!
//! Set `FORNACE_MODEL_DIR` to the Qwen2.5-0.5B directory to run these
//! tests. Tests with the 1.5B use the directory `qwen2.5-1.5b-instruct`
//! next to it, and skip with a message if it is not there.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use fornace_engine::model::{Config, ModelForCausalLM};
use fornace_engine::{AttentionKind, KvCacheKind};
use tokenizers::Tokenizer;

/// The verify pass length: K = 4.
const M: usize = 5;
const WINDOWS: usize = 8;

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

fn prompt(dir: &Path) -> Vec<u32> {
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let prompts = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/prompts");
    // Read as the CLI does with "$(cat FILE)": without the final newline.
    let text = std::fs::read_to_string(prompts.join("alice-512.txt")).unwrap();
    tokenizer
        .encode(text.trim_end_matches('\n'), true)
        .unwrap()
        .get_ids()
        .to_vec()
}

fn dense(dir: &Path, dtype: DType, attention: AttentionKind, device: &Device) -> ModelForCausalLM {
    let config: Config =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    // SAFETY: the test does not change the weights file while it is mapped.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], dtype, device)
    }
    .unwrap();
    let mut model = ModelForCausalLM::new(&config, vb, false).unwrap();
    model.set_attention(attention);
    model
}

fn rows(t: Tensor) -> Vec<Vec<f32>> {
    t.to_dtype(DType::F32).unwrap().to_vec2().unwrap()
}

/// Decodes `WINDOWS × M` tokens greedily, one token for each step. Then
/// feeds the same tokens again, `M` at a time, as verify passes. Prints the
/// largest logit difference, the top token changes, and the top-2 margin
/// of the one-token step at each change. Returns the number of changes.
fn compare(label: &str, model: &mut ModelForCausalLM, prompt: &[u32], device: &Device) -> usize {
    let n = WINDOWS * M;
    let t = prompt.len();
    let input = Tensor::new(prompt, device).unwrap().unsqueeze(0).unwrap();

    model.reset_kv_cache(KvCacheKind::Prealloc, t + n);
    let first = model.forward(&input, 0).unwrap().squeeze(0).unwrap();
    let mut token = argmax(&rows(first)[0]) as u32;
    let mut tokens = Vec::with_capacity(n);
    let mut steps = Vec::with_capacity(n);
    for i in 0..n {
        tokens.push(token);
        let input = Tensor::new(&[token], device).unwrap().unsqueeze(0).unwrap();
        let logits = rows(model.forward(&input, t + i).unwrap().squeeze(0).unwrap()).remove(0);
        token = argmax(&logits) as u32;
        steps.push(logits);
    }

    model.reset_kv_cache(KvCacheKind::Prealloc, t + n);
    model.hidden_states(&input, 0).unwrap();
    let mut max_diff = 0f32;
    let mut changes = Vec::new();
    for w in 0..WINDOWS {
        let input = Tensor::new(&tokens[w * M..(w + 1) * M], device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let hidden = model
            .hidden_states(&input, t + w * M)
            .unwrap()
            .squeeze(0)
            .unwrap();
        let verify = rows(model.lm_head(&hidden).unwrap());
        for (i, logits) in verify.iter().enumerate() {
            let step = &steps[w * M + i];
            let diff = step
                .iter()
                .zip(logits)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            max_diff = max_diff.max(diff);
            if argmax(step) != argmax(logits) {
                changes.push(format!("{} ({:.3})", w * M + i, margin(step)));
            }
        }
    }
    eprintln!(
        "{label}: {n} positions, verify passes of {M}: max |step - verify| = {max_diff:.4}; \
         top token changes at (one-token margin): {}",
        if changes.is_empty() {
            "none".to_string()
        } else {
            changes.join(", ")
        }
    );
    changes.len()
}

/// As `compare`, but each verify pass starts at a new offset o = 0, 1, 2,
/// …: the cache of the one-token run is truncated to the prompt and o
/// tokens, and the pass feeds tokens o to o + M − 1. This is the rollback
/// of speculative decoding, at every alignment of the cache length.
/// Prints the offsets with a top token change, and returns their count.
fn compare_truncated(
    label: &str,
    model: &mut ModelForCausalLM,
    prompt: &[u32],
    device: &Device,
) -> usize {
    let n = WINDOWS * M;
    let t = prompt.len();
    let input = Tensor::new(prompt, device).unwrap().unsqueeze(0).unwrap();
    model.reset_kv_cache(KvCacheKind::Prealloc, t + n + M);
    let first = model.forward(&input, 0).unwrap().squeeze(0).unwrap();
    let mut token = argmax(&rows(first)[0]) as u32;
    let mut tokens = Vec::with_capacity(n);
    let mut steps = Vec::with_capacity(n);
    for i in 0..n {
        tokens.push(token);
        let input = Tensor::new(&[token], device).unwrap().unsqueeze(0).unwrap();
        let logits = rows(model.forward(&input, t + i).unwrap().squeeze(0).unwrap()).remove(0);
        token = argmax(&logits) as u32;
        steps.push(logits);
    }
    let mut max_diff = 0f32;
    let mut bad = Vec::new();
    for o in 0..n - M {
        model.truncate_kv_cache(t + o).unwrap();
        let input = Tensor::new(&tokens[o..o + M], device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let hidden = model
            .hidden_states(&input, t + o)
            .unwrap()
            .squeeze(0)
            .unwrap();
        let verify = rows(model.lm_head(&hidden).unwrap());
        let diff = verify
            .iter()
            .zip(&steps[o..o + M])
            .flat_map(|(v, s)| v.iter().zip(s).map(|(a, b)| (a - b).abs()))
            .fold(0f32, f32::max);
        max_diff = max_diff.max(diff);
        if verify
            .iter()
            .zip(&steps[o..o + M])
            .any(|(v, s)| argmax(v) != argmax(s))
        {
            bad.push(format!("{} (kv {}, diff {diff:.3})", o, t + o + M));
        }
    }
    eprintln!(
        "{label}: truncate and verify at {} offsets: max |step - verify| = {max_diff:.4}; \
         offsets with a top token change: {}",
        n - M,
        if bad.is_empty() {
            "none".to_string()
        } else {
            bad.join(", ")
        }
    );
    bad.len()
}

#[test]
fn verify_logits_equal_one_token_steps_on_the_cpu() {
    let Ok(dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return;
    };
    let dir = Path::new(&dir);
    let prompt = prompt(dir);
    let mut model = dense(dir, DType::F32, AttentionKind::Naive, &Device::Cpu);
    assert_eq!(
        compare_truncated("cpu f32 naive", &mut model, &prompt, &Device::Cpu),
        0
    );
    assert_eq!(
        compare("cpu f32 naive", &mut model, &prompt, &Device::Cpu),
        0
    );
}

#[cfg(feature = "metal")]
#[test]
fn verify_logits_track_one_token_steps_on_metal() {
    let Ok(dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return;
    };
    let dir = Path::new(&dir);
    let prompt = prompt(dir);
    let device = Device::new_metal(0).unwrap();
    for (dtype, attention) in [
        (DType::F32, AttentionKind::Naive),
        (DType::F32, AttentionKind::Fused),
        (DType::BF16, AttentionKind::Naive),
        (DType::BF16, AttentionKind::Fused),
    ] {
        let mut model = dense(dir, dtype, attention, &device);
        let label = format!("metal {dtype:?} {attention}");
        compare(&label, &mut model, &prompt, &device);
        compare_truncated(&label, &mut model, &prompt, &device);
    }
}
