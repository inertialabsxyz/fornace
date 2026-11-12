//! Speculative decoding: the verify pass against one-token steps, and the
//! text with a draft against the text of the target alone.
//!
//! Set `FORNACE_MODEL_DIR` to the Qwen2.5-0.5B directory to run these
//! tests. Tests with the 1.5B use the directory `qwen2.5-1.5b-instruct`
//! next to it, and skip with a message if it is not there. The `q8` draft
//! needs `model-q8_0.gguf` from `make quantize`.
//!
//! With both models, the CPU tests take about 4 minutes and the Metal test
//! about 2 minutes on an M4 Pro, in release mode:
//!
//! ```sh
//! FORNACE_MODEL_DIR=$PWD/models/qwen2.5-0.5b-instruct cargo test --release \
//!     --features metal -p fornace-engine --test speculative -- --nocapture
//! ```

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use fornace_engine::model::{Config, ModelForCausalLM};
use fornace_engine::{
    AttentionKind, DEFAULT_BLOCK_SIZE, DeviceKind, DraftConfig, Dtype, Engine, GenerationReport,
    KvCacheKind, LoadConfig,
};
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

// The text with a draft against the text of the target alone.

/// The tokens that each engine test generates.
const NEW_TOKENS: usize = 128;

/// The 1.5B directory next to the 0.5B, or `None` with a message.
fn larger(dir: &Path) -> Option<PathBuf> {
    let larger = dir.parent()?.join("qwen2.5-1.5b-instruct");
    if larger.join("config.json").exists() {
        Some(larger)
    } else {
        eprintln!("skipped the 1.5B: no model in {}", larger.display());
        None
    }
}

fn prompt_text() -> String {
    let prompts = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/prompts");
    let text = std::fs::read_to_string(prompts.join("alice-512.txt")).unwrap();
    text.trim_end_matches('\n').to_string()
}

/// Greedy generation of `NEW_TOKENS` tokens with EOS ignored. Fused
/// attention on Metal, naive on the CPU.
fn generate(
    target: &Path,
    dtype: Dtype,
    device: DeviceKind,
    draft: Option<(&Path, Dtype, usize)>,
) -> GenerationReport {
    let mut engine = Engine::load(&LoadConfig {
        model_dir: target.to_path_buf(),
        dtype,
        device,
        max_new_tokens: NEW_TOKENS,
        batch: 1,
        kv_cache: KvCacheKind::Prealloc,
        block_size: DEFAULT_BLOCK_SIZE,
        kv_capacity: None,
        share_prefix: false,
        attention: match device {
            DeviceKind::Metal => AttentionKind::Fused,
            DeviceKind::Cpu => AttentionKind::Naive,
        },
        sync_spans: false,
        ignore_eos: true,
        draft: draft.map(|(dir, dtype, k)| DraftConfig {
            model_dir: dir.to_path_buf(),
            dtype,
            k,
        }),
    })
    .unwrap();
    engine.generate(&prompt_text()).unwrap()
}

/// The first position where the two token lists differ, if any.
fn first_difference(a: &[u32], b: &[u32]) -> Option<usize> {
    assert_eq!(a.len(), b.len(), "both runs make NEW_TOKENS tokens");
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// The acceptance rate of a report with a draft, and a label for it.
fn alpha(report: &GenerationReport) -> String {
    let spec = report.speculative.as_ref().expect("a run with a draft");
    format!(
        "alpha {:.3} ({}/{}), {} rounds",
        spec.alpha(),
        spec.accepted,
        spec.tested,
        spec.rounds
    )
}

/// The distance between two BF16 values next to `x`: BF16 keeps 8
/// significant bits.
#[cfg(feature = "metal")]
fn bf16_step(x: f32) -> f32 {
    2f32.powf(x.abs().log2().floor() - 7.0)
}

/// The top-2 margin and the top logit of the target alone before
/// generated token `p`, as the engine computes them: a prefill of the
/// prompt, one-token steps for `generated[..p]`, and the repeat penalty over
/// the last 64 generated tokens. The first token has no penalty.
#[cfg(feature = "metal")]
fn margin_before(
    dir: &Path,
    dtype: DType,
    device: &Device,
    generated: &[u32],
    p: usize,
) -> (f32, f32) {
    let prompt = prompt(dir);
    let attention = if device.is_cpu() {
        AttentionKind::Naive
    } else {
        AttentionKind::Fused
    };
    let mut model = dense(dir, dtype, attention, device);
    model.reset_kv_cache(KvCacheKind::Prealloc, prompt.len() + p + 1);
    let input = Tensor::new(prompt.as_slice(), device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let mut logits = model.forward(&input, 0).unwrap();
    for (i, &token) in generated[..p].iter().enumerate() {
        let input = Tensor::new(&[token], device).unwrap().unsqueeze(0).unwrap();
        logits = model.forward(&input, prompt.len() + i).unwrap();
    }
    let logits = logits
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let logits = if p == 0 {
        logits
    } else {
        let context = &generated[p.saturating_sub(64)..p];
        candle_transformers::utils::apply_repeat_penalty(&logits, 1.1, context).unwrap()
    };
    let values: Vec<f32> = logits.to_vec1().unwrap();
    (margin(&values), values[argmax(&values)])
}

#[test]
fn the_model_as_its_own_draft_gives_its_text_on_the_cpu() {
    let Ok(dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return;
    };
    let dir = Path::new(&dir);
    let alone = generate(dir, Dtype::F32, DeviceKind::Cpu, None);
    for k in [1, 4] {
        let report = generate(dir, Dtype::F32, DeviceKind::Cpu, Some((dir, Dtype::F32, k)));
        let spec = report.speculative.as_ref().unwrap();
        eprintln!("cpu f32, own draft, K = {k}: {}", alpha(&report));
        assert_eq!(report.tokens, alone.tokens, "K = {k}: the text changed");
        assert_eq!(spec.accepted, spec.tested, "K = {k}: a guess was rejected");
    }
}

#[test]
fn a_draft_gives_the_text_of_the_target_alone_on_the_cpu() {
    let Ok(dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return;
    };
    let small = Path::new(&dir).to_path_buf();
    let mut pairs = vec![(small.clone(), Dtype::Q8, "0.5B, 0.5B q8 draft")];
    if let Some(large) = larger(&small) {
        pairs.push((large, Dtype::F32, "1.5B, 0.5B f32 draft"));
    }
    for (target, draft_dtype, label) in pairs {
        let alone = generate(&target, Dtype::F32, DeviceKind::Cpu, None);
        for k in [1, 3, 8] {
            let report = generate(
                &target,
                Dtype::F32,
                DeviceKind::Cpu,
                Some((&small, draft_dtype, k)),
            );
            eprintln!("cpu f32, {label}, K = {k}: {}", alpha(&report));
            assert_eq!(
                first_difference(&alone.tokens[0], &report.tokens[0]),
                None,
                "{label}, K = {k}: the text changed"
            );
        }
    }
}

/// On Metal, F32 must give the text of the target alone. BF16 may leave it
/// only at a near tie: at the first difference, the top-2 margin of the
/// target alone must be at most 2 BF16 steps at its top logit. The verify
/// pass runs GEMM and the full `sdpa` kernel, and one-token decode runs GEMV
/// and the vector kernel, so BF16 rounds them differently.
#[cfg(feature = "metal")]
#[test]
fn a_draft_gives_the_text_of_the_target_alone_or_a_near_tie_on_metal() {
    let Ok(dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return;
    };
    let small = Path::new(&dir).to_path_buf();
    let device = Device::new_metal(0).unwrap();
    let mut pairs = vec![(small.clone(), Dtype::Q8, "0.5B, 0.5B q8 draft")];
    if let Some(large) = larger(&small) {
        pairs.push((large, Dtype::F16, "1.5B, 0.5B bf16 draft"));
    }
    for (target, draft_dtype, label) in pairs {
        for (dtype, compute) in [(Dtype::F32, DType::F32), (Dtype::F16, DType::BF16)] {
            let alone = generate(&target, dtype, DeviceKind::Metal, None);
            for k in [1, 3, 8] {
                let report = generate(
                    &target,
                    dtype,
                    DeviceKind::Metal,
                    Some((&small, draft_dtype, k)),
                );
                let difference = first_difference(&alone.tokens[0], &report.tokens[0]);
                eprintln!(
                    "metal {compute:?}, {label}, K = {k}: {}, first difference: {difference:?}",
                    alpha(&report)
                );
                let Some(p) = difference else { continue };
                assert_eq!(
                    compute,
                    DType::BF16,
                    "{label}, K = {k}: F32 changed the text"
                );
                let (margin, top) = margin_before(&target, compute, &device, &alone.tokens[0], p);
                let limit = 2.0 * bf16_step(top);
                eprintln!(
                    "  at token {p}: margin {margin:.4}, top logit {top:.3}, limit {limit:.4}"
                );
                assert!(
                    margin <= limit,
                    "{label}, K = {k}: the text changed at token {p} with margin {margin}, \
                     above 2 BF16 steps ({limit})"
                );
            }
        }
    }
}
