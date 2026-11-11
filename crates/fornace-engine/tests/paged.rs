//! The paged KV cache gives the same logits as `prealloc`.
//!
//! Set `FORNACE_MODEL_DIR` to a Qwen2.5 directory to run these tests. The
//! Metal tests also need the `metal` feature. The `q4` test also needs
//! `model-q4_0.gguf` (`make quantize`), and skips without it.
//!
//! Each test decodes greedily with `prealloc`. Then it gives the same tokens
//! to the model with the paged cache, and compares the logits at each step.
//! The paged cache copies the same values into the same layout before
//! attention, so the logits must be equal, not only close.
//!
//! The shared-prefix test runs one prefill and forks it into 4 rows. It
//! compares the rows with 4 rows that each run their own prefill.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
#[cfg(feature = "metal")]
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;
use fornace_engine::model::{Config, ModelForCausalLM};
#[cfg(feature = "metal")]
use fornace_engine::quantize::gguf_path;
use fornace_engine::{
    AttentionKind, DEFAULT_BLOCK_SIZE, DeviceKind, Dtype, Engine, Error, KvCacheKind, LoadConfig,
};
use tokenizers::Tokenizer;

const PROMPT_TOKENS: usize = 512;
const STEPS: usize = 16;
/// 1 gives one block for each position. 7 does not divide 512, so the last
/// block of the prompt is not full.
#[cfg(feature = "metal")]
const BLOCK_SIZES: [usize; 3] = [1, DEFAULT_BLOCK_SIZE, 7];

#[derive(Debug, Clone, Copy)]
enum Cache {
    Prealloc,
    Paged(usize),
    /// Paged, with one prefill of the first prompt, forked into all rows.
    Shared(usize),
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Runs a prefill and then `STEPS - 1` decode steps. At each step, row b
/// takes the next token of `forced[b]`, or its top token if `forced` is
/// `None`. Returns the logits as `[step][row]`, and the tokens of each row.
fn run(
    model: &mut ModelForCausalLM,
    cache: Cache,
    prompts: &[&[u32]],
    forced: Option<&[Vec<u32>]>,
    device: &Device,
) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<u32>>) {
    let (batch, len) = (prompts.len(), prompts[0].len());
    let capacity = len + STEPS;
    let rows = |logits: Tensor, n: usize| -> Vec<Vec<f32>> {
        let logits = logits.squeeze(1).unwrap().to_dtype(DType::F32).unwrap();
        (0..n)
            .map(|b| logits.i(b).unwrap().to_vec1().unwrap())
            .collect()
    };
    let first = match cache {
        Cache::Prealloc => {
            model.reset_kv_cache(KvCacheKind::Prealloc, capacity);
            let input = Tensor::from_vec(prompts.concat(), (batch, len), device).unwrap();
            rows(model.forward(&input, 0).unwrap(), batch)
        }
        Cache::Paged(block_size) => {
            model.reset_paged_kv_cache(block_size, batch * capacity.div_ceil(block_size));
            let input = Tensor::from_vec(prompts.concat(), (batch, len), device).unwrap();
            rows(model.forward(&input, 0).unwrap(), batch)
        }
        Cache::Shared(block_size) => {
            model.reset_paged_kv_cache(block_size, batch * capacity.div_ceil(block_size));
            let input = Tensor::new(prompts[0], device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let one = rows(model.forward(&input, 0).unwrap(), 1);
            model.fork_kv_cache(batch).unwrap();
            vec![one[0].clone(); batch]
        }
    };
    let mut logits = vec![first];
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
        logits.push(rows(model.forward(&input, len + step).unwrap(), batch));
    }
    (logits, tokens)
}

/// The largest logit difference between two runs, over all steps and rows.
fn max_diff(a: &[Vec<Vec<f32>>], b: &[Vec<Vec<f32>>]) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(r, s)| max_abs_diff(r, s)))
        .fold(0f32, f32::max)
}

/// One prompt of 512 tokens, and 4 different windows of 512 tokens.
fn prompts(tokenizer: &Tokenizer) -> Vec<(&'static str, Vec<Vec<u32>>)> {
    let prompts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/prompts");
    let ids = |name: &str| -> Vec<u32> {
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
        .take(4)
        .map(|w| w.to_vec())
        .collect::<Vec<_>>();
    vec![("batch 1", vec![one]), ("batch 4", windows)]
}

/// For each attention kind, prompt set, and block size: the paged logits
/// equal the `prealloc` logits at every step.
fn paged_equals_prealloc(
    label: &str,
    model: &mut ModelForCausalLM,
    tokenizer: &Tokenizer,
    device: &Device,
    attentions: &[AttentionKind],
    block_sizes: &[usize],
) {
    for &attention in attentions {
        model.set_attention(attention);
        for (name, prompts) in prompts(tokenizer) {
            let refs: Vec<&[u32]> = prompts.iter().map(Vec::as_slice).collect();
            let (want, tokens) = run(model, Cache::Prealloc, &refs, None, device);
            for &block_size in block_sizes {
                let (got, _) = run(
                    model,
                    Cache::Paged(block_size),
                    &refs,
                    Some(&tokens),
                    device,
                );
                let diff = max_diff(&want, &got);
                eprintln!(
                    "{label} {attention} {name} block {block_size}: max |prealloc - paged| = {diff} over {STEPS} steps"
                );
                assert_eq!(
                    diff, 0.0,
                    "{label} {attention} {name} block {block_size}: the logits differ"
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

fn config(dir: &Path) -> Config {
    serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap()
}

fn tokenizer(dir: &Path) -> Tokenizer {
    Tokenizer::from_file(dir.join("tokenizer.json")).unwrap()
}

fn dense(dir: &Path, dtype: DType, device: &Device) -> ModelForCausalLM {
    // SAFETY: the test does not change the weights file while it is mapped.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], dtype, device)
    }
    .unwrap();
    ModelForCausalLM::new(&config(dir), vb, false).unwrap()
}

#[test]
fn paged_equals_prealloc_on_the_cpu_in_f32() {
    let Some(dir) = model_dir() else { return };
    let mut model = dense(&dir, DType::F32, &Device::Cpu);
    paged_equals_prealloc(
        "cpu f32",
        &mut model,
        &tokenizer(&dir),
        &Device::Cpu,
        &[AttentionKind::Naive],
        &[DEFAULT_BLOCK_SIZE, 7],
    );
}

#[cfg(feature = "metal")]
#[test]
fn paged_equals_prealloc_on_metal_in_bf16() {
    let Some(dir) = model_dir() else { return };
    let device = Device::new_metal(0).expect("open the metal device");
    let mut model = dense(&dir, DType::BF16, &device);
    paged_equals_prealloc(
        "metal bf16",
        &mut model,
        &tokenizer(&dir),
        &device,
        &[AttentionKind::Naive, AttentionKind::Fused],
        &BLOCK_SIZES,
    );
}

#[cfg(feature = "metal")]
#[test]
fn paged_equals_prealloc_on_metal_in_q4() {
    let Some(dir) = model_dir() else { return };
    let path = gguf_path(&dir, Dtype::Q4).unwrap();
    if !path.exists() {
        eprintln!("skipped q4: {} is missing", path.display());
        return;
    }
    let device = Device::new_metal(0).expect("open the metal device");
    let vb = QVarBuilder::from_gguf(&path, &device).unwrap();
    let mut model = ModelForCausalLM::new_quantized(&config(&dir), vb, false).unwrap();
    paged_equals_prealloc(
        "metal q4",
        &mut model,
        &tokenizer(&dir),
        &device,
        &[AttentionKind::Fused],
        &[DEFAULT_BLOCK_SIZE, 7],
    );
}

/// One prefill forked into 4 rows against 4 prefills of the same prompt.
/// The prefill of one row and of 4 rows can round differently, so this
/// compares with a tolerance, and asserts the block counts.
fn shared_prefix_matches_four_prefills(
    label: &str,
    model: &mut ModelForCausalLM,
    tokenizer: &Tokenizer,
    device: &Device,
    tolerance: f32,
) {
    let (_, one) = prompts(tokenizer).remove(0);
    let copies: Vec<&[u32]> = vec![one[0].as_slice(); 4];
    for block_size in [DEFAULT_BLOCK_SIZE, 7] {
        let (want, tokens) = run(model, Cache::Paged(block_size), &copies, None, device);
        let unshared = model.kv_block_usage().unwrap();
        let (got, _) = run(
            model,
            Cache::Shared(block_size),
            &copies,
            Some(&tokens),
            device,
        );
        let shared = model.kv_block_usage().unwrap();
        let diff = max_diff(&want, &got);
        eprintln!(
            "{label} block {block_size}: max |4 prefills - 1 shared prefill| = {diff}; blocks in use {} unshared, {} shared",
            unshared.in_use, shared.in_use
        );
        assert!(diff <= tolerance, "{label} block {block_size}: diff {diff}");
        // Each row holds 512 + 15 positions. Shared: the full prompt blocks
        // once, and a copy of the last block and the new blocks per row.
        let per_row = (PROMPT_TOKENS + STEPS - 1).div_ceil(block_size);
        let full_prompt = PROMPT_TOKENS / block_size;
        assert_eq!(unshared.in_use, 4 * per_row);
        assert_eq!(shared.in_use, full_prompt + 4 * (per_row - full_prompt));
    }
}

#[test]
fn shared_prefix_matches_four_prefills_on_the_cpu() {
    let Some(dir) = model_dir() else { return };
    let mut model = dense(&dir, DType::F32, &Device::Cpu);
    shared_prefix_matches_four_prefills(
        "cpu f32",
        &mut model,
        &tokenizer(&dir),
        &Device::Cpu,
        1e-3,
    );
}

#[cfg(feature = "metal")]
#[test]
fn shared_prefix_matches_four_prefills_on_metal() {
    let Some(dir) = model_dir() else { return };
    let device = Device::new_metal(0).expect("open the metal device");
    let mut model = dense(&dir, DType::BF16, &device);
    model.set_attention(AttentionKind::Fused);
    shared_prefix_matches_four_prefills(
        "metal bf16 fused",
        &mut model,
        &tokenizer(&dir),
        &device,
        0.5,
    );
}

/// Through the engine: each row of a shared-prefix batch gives the text of
/// `generate` with the same prompt, and the batch uses fewer blocks than
/// the same batch without sharing.
#[test]
fn engine_shared_prefix_rows_equal_generate_on_the_cpu() {
    let Some(dir) = model_dir() else { return };
    let load = |share_prefix: bool| {
        Engine::load(&LoadConfig {
            model_dir: dir.clone(),
            dtype: Dtype::F16,
            device: DeviceKind::Cpu,
            max_new_tokens: 16,
            batch: 4,
            kv_cache: KvCacheKind::Paged,
            block_size: DEFAULT_BLOCK_SIZE,
            kv_capacity: None,
            share_prefix,
            attention: AttentionKind::Naive,
            sync_spans: false,
            ignore_eos: true,
        })
        .unwrap()
    };
    // Longer than one block, so that the rows share at least one block.
    let prompt = "The capital of France is Paris. The capital of Germany is Berlin. \
                  The capital of Spain is";
    let mut engine = load(true);
    let shared = engine.generate_batch(&[prompt; 4]).unwrap();
    let single = engine.generate(prompt).unwrap();
    for text in &shared.texts {
        assert_eq!(text, &single.texts[0]);
    }
    let unshared = load(false).generate_batch(&[prompt; 4]).unwrap();
    let (s, u) = (shared.kv_blocks.unwrap(), unshared.kv_blocks.unwrap());
    eprintln!("blocks in use: {} shared, {} unshared", s.in_use, u.in_use);
    assert!(s.in_use < u.in_use);
    assert!(matches!(
        engine.generate_batch(&["The capital of Spain is", "The capital of Italy is"]),
        Err(Error::SharePrefixNeedsOnePrompt)
    ));
}
