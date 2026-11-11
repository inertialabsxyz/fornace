//! A batch of B rows gives almost the same logits and tokens as each row
//! alone.
//!
//! Set `FORNACE_MODEL_DIR` to a Qwen2.5 directory to run these tests. The
//! Metal test also needs the `metal` feature.
//!
//! The model tests decode each prompt greedily at batch 1. Then they give
//! the same tokens to a batch of all prompts, and compare the logits of each
//! row at each step. There are two batches: 4 copies of one prompt, and 4
//! different prompts of 512 tokens each.
//!
//! - Level 1: after the prefill, the logits of each row are within a
//!   tolerance of the batch-1 logits. A failure here is a bug: a row mixup,
//!   an offset, or a mask.
//! - Level 2: at each step, the top token of each row equals the batch-1
//!   top token. A matmul with B rows can add in a different order from a
//!   matmul with one row, so the top token can change where the two best
//!   logits are almost equal. The test fails if the top token changes at a
//!   step where the batch-1 margin between the two best logits is larger
//!   than the difference between the two logit vectors.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use fornace_engine::model::{Config, ModelForCausalLM};
use fornace_engine::{AttentionKind, DeviceKind, Dtype, Engine, KvCacheKind, LoadConfig};
use tokenizers::Tokenizer;

const BATCH: usize = 4;
const PROMPT_TOKENS: usize = 512;
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Runs a prefill of all rows and then `STEPS - 1` decode steps. At each
/// step, row b takes the next token of `forced[b]`, or its top token if
/// `forced` is `None`. Returns the logits as `[step][row]` and the tokens
/// of each row.
fn run(
    model: &mut ModelForCausalLM,
    prompts: &[&[u32]],
    forced: Option<&[&[u32]]>,
    device: &Device,
) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<u32>>) {
    let (batch, len) = (prompts.len(), prompts[0].len());
    model.reset_kv_cache(KvCacheKind::Prealloc, len + STEPS);
    let rows = |logits: Tensor| -> Vec<Vec<f32>> {
        let logits = logits.squeeze(1).unwrap().to_dtype(DType::F32).unwrap();
        (0..batch)
            .map(|b| logits.i(b).unwrap().to_vec1().unwrap())
            .collect()
    };
    let input = Tensor::from_vec(prompts.concat(), (batch, len), device).unwrap();
    let mut logits = vec![rows(model.forward(&input, 0).unwrap())];
    let mut tokens = vec![Vec::new(); batch];
    for step in 0..STEPS - 1 {
        let next: Vec<u32> = (0..batch)
            .map(|b| match forced {
                Some(forced) => forced[b][step],
                None => argmax(&logits.last().unwrap()[b]) as u32,
            })
            .collect();
        for (row, &t) in tokens.iter_mut().zip(&next) {
            row.push(t);
        }
        let input = Tensor::from_vec(next, (batch, 1), device).unwrap();
        logits.push(rows(model.forward(&input, len + step).unwrap()));
    }
    (logits, tokens)
}

/// The two batches: 4 copies of `alice-512.txt`, and 4 different windows
/// of 512 tokens from `alice-2048.txt`.
fn batches(tokenizer: &Tokenizer) -> Vec<(&'static str, Vec<Vec<u32>>)> {
    let prompts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/prompts");
    let ids = |name: &str| -> Vec<u32> {
        // Read as the CLI does with "$(cat FILE)": without the final newline.
        let text = std::fs::read_to_string(prompts_dir.join(name)).unwrap();
        let ids = tokenizer.encode(text.trim_end_matches('\n'), true).unwrap();
        ids.get_ids().to_vec()
    };
    let one = ids("alice-512.txt");
    assert_eq!(one.len(), PROMPT_TOKENS);
    let long = ids("alice-2048.txt");
    let windows = long
        .as_chunks::<PROMPT_TOKENS>()
        .0
        .iter()
        .take(BATCH)
        .map(|w| w.to_vec())
        .collect::<Vec<_>>();
    assert_eq!(windows.len(), BATCH);
    vec![
        ("copies of one prompt", vec![one; BATCH]),
        ("different prompts", windows),
    ]
}

/// Compares a batch of each set of prompts with batch 1, for each attention
/// kind. `tolerance` is the largest allowed difference of the logits after
/// the prefill (level 1).
fn batch_matches_batch_one(
    dir: &Path,
    device: &Device,
    dtype: DType,
    attentions: &[AttentionKind],
    tolerance: f32,
) {
    let config: Config =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    // SAFETY: the test does not change the weights file while it is mapped.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], dtype, device)
    }
    .unwrap();
    let mut model = ModelForCausalLM::new(&config, vb, false).unwrap();

    for &attention in attentions {
        model.set_attention(attention);
        for (name, prompts) in batches(&tokenizer) {
            let device_name = if device.is_cpu() { "cpu" } else { "metal" };
            let label = format!("{device_name} {attention} {name}");
            let singles: Vec<_> = prompts
                .iter()
                .map(|p| run(&mut model, &[p], None, device))
                .collect();
            let prompt_refs: Vec<&[u32]> = prompts.iter().map(Vec::as_slice).collect();
            let forced: Vec<&[u32]> = singles.iter().map(|(_, t)| t[0].as_slice()).collect();
            let (batched, _) = run(&mut model, &prompt_refs, Some(&forced), device);

            for (b, (single, _)) in singles.iter().enumerate() {
                let first = max_abs_diff(&single[0][0], &batched[0][b]);
                assert!(
                    first <= tolerance,
                    "{label}: row {b}: level 1: the logits after the prefill differ by {first:.4}, more than {tolerance}"
                );
                let mut max_decode = 0f32;
                let mut first_change = None;
                for step in 0..STEPS {
                    let (s, x) = (&single[step][0], &batched[step][b]);
                    let diff = max_abs_diff(s, x);
                    if step > 0 {
                        max_decode = max_decode.max(diff);
                    }
                    if argmax(s) != argmax(x) {
                        eprintln!(
                            "{label}: row {b}: step {step}: the top token changes, batch-1 margin {:.4}, max diff {diff:.4}",
                            margin(s)
                        );
                        assert!(
                            margin(s) <= diff,
                            "{label}: row {b}: step {step}: the top token changes, but the logits are not a near-tie"
                        );
                        first_change.get_or_insert(step);
                    }
                }
                let tokens = match first_change {
                    None => format!("all {STEPS} top tokens equal"),
                    Some(step) => format!("the first top token change is at step {step}"),
                };
                eprintln!(
                    "{label}: row {b}: max |batch 1 - batch {BATCH}| = {first:.4} after the prefill, {max_decode:.4} in {} decode steps; {tokens}",
                    STEPS - 1
                );
            }
        }
    }
}

fn model_dir() -> Option<PathBuf> {
    match std::env::var("FORNACE_MODEL_DIR") {
        Ok(dir) => Some(PathBuf::from(dir)),
        Err(_) => {
            eprintln!("skipped: FORNACE_MODEL_DIR is not set");
            None
        }
    }
}

#[test]
fn batch_matches_batch_one_on_the_cpu_in_f32() {
    let Some(dir) = model_dir() else { return };
    batch_matches_batch_one(
        &dir,
        &Device::Cpu,
        DType::F32,
        &[AttentionKind::Naive],
        1e-3,
    );
}

#[cfg(feature = "metal")]
#[test]
fn batch_matches_batch_one_on_metal_in_bf16() {
    let Some(dir) = model_dir() else { return };
    let device = Device::new_metal(0).expect("open the metal device");
    batch_matches_batch_one(
        &dir,
        &device,
        DType::BF16,
        &[AttentionKind::Naive, AttentionKind::Fused],
        0.5,
    );
}

/// Each row of `generate_batch` gives the text of `generate` for its own
/// prompt. This checks that the engine keeps each row with its prompt, its
/// sampler, and its repeat penalty. It runs on the CPU in F32, where the
/// model test shows no top token change.
#[test]
fn generate_batch_rows_equal_generate_on_the_cpu() {
    let Some(dir) = model_dir() else { return };
    let mut engine = Engine::load(&LoadConfig {
        model_dir: dir,
        dtype: Dtype::F16,
        device: DeviceKind::Cpu,
        max_new_tokens: 16,
        batch: BATCH,
        kv_cache: KvCacheKind::Prealloc,
        block_size: fornace_engine::DEFAULT_BLOCK_SIZE,
        kv_capacity: None,
        share_prefix: false,
        attention: AttentionKind::Naive,
        sync_spans: false,
        ignore_eos: true,
    })
    .unwrap();
    let prompts = [
        "The capital of Spain is",
        "The capital of France is",
        "The capital of Italy is",
        "The capital of Japan is",
    ];
    let batch = engine.generate_batch(&prompts).unwrap();
    assert_eq!(batch.batch, BATCH);
    assert_eq!(batch.texts.len(), BATCH);
    for (prompt, text) in prompts.iter().zip(&batch.texts) {
        let single = engine.generate(prompt).unwrap();
        assert_eq!(&single.texts[0], text, "prompt {prompt:?}");
    }
}
