//! Q8_0 and Q4_0 weights give logits near the dense model.
//!
//! Set `FORNACE_MODEL_DIR` to a Qwen2.5 directory, and make the GGUF files
//! with `make quantize`, to run these tests. If a GGUF file is missing, the
//! test skips that format.
//!
//! Each test decodes `alice-512.txt` greedily with the dense F32 model. Then
//! it gives the same tokens to the quantized model, and compares the logits
//! at each step, as `tests/fused_attention.rs` does. It also compares the
//! top token at every prompt position (top-1 agreement). The reference is
//! F32, because the quantized models compute in F32. On Metal, dense BF16 is
//! compared with the same reference, to show the effect of BF16
//! activations alone.
//!
//! Quantization changes the text, so the tests do not assert equal tokens.
//! They assert only what a bug would break: the logits shape, no NaN, and a
//! top-1 agreement above a floor. The floors are below the measured values.

use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;
use fornace_engine::model::{Config, ModelForCausalLM};
use fornace_engine::quantize::gguf_path;
use fornace_engine::{Dtype, KvCacheKind};
use tokenizers::Tokenizer;

const STEPS: usize = 32;
/// Positions for each `lm_head` chunk in the top-1 agreement.
const CHUNK: usize = 128;

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

/// Runs a prefill and then `STEPS - 1` decode steps. Each step uses the next
/// token of `forced`, or the top token if `forced` is `None`. Returns the
/// logits of each step and the tokens.
fn run(
    model: &mut ModelForCausalLM,
    prompt: &[u32],
    forced: Option<&[u32]>,
    device: &Device,
) -> (Vec<Vec<f32>>, Vec<u32>) {
    model.reset_kv_cache(KvCacheKind::Prealloc, prompt.len() + STEPS);
    let vocab = |t: Tensor| -> Vec<f32> {
        assert_eq!(t.dims().len(), 3, "logits are [batch, 1, vocab]");
        assert_eq!(&t.dims()[..2], &[1, 1]);
        t.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    let input = Tensor::new(prompt, device).unwrap().unsqueeze(0).unwrap();
    let mut logits = vec![vocab(model.forward(&input, 0).unwrap())];
    let mut tokens = Vec::new();
    for step in 0..STEPS - 1 {
        let token = match forced {
            Some(forced) => forced[step],
            None => argmax(logits.last().unwrap()) as u32,
        };
        tokens.push(token);
        let input = Tensor::new(&[token], device).unwrap().unsqueeze(0).unwrap();
        logits.push(vocab(model.forward(&input, prompt.len() + step).unwrap()));
    }
    (logits, tokens)
}

/// The top token at every prompt position, from one forward pass.
fn top_tokens(model: &mut ModelForCausalLM, prompt: &[u32], device: &Device) -> Vec<u32> {
    model.reset_kv_cache(KvCacheKind::Prealloc, prompt.len());
    let input = Tensor::new(prompt, device).unwrap().unsqueeze(0).unwrap();
    let hidden = model.hidden_states(&input, 0).unwrap().squeeze(0).unwrap();
    let mut top = Vec::with_capacity(prompt.len());
    for start in (0..prompt.len()).step_by(CHUNK) {
        let len = CHUNK.min(prompt.len() - start);
        // The quantized `lm_head` copies a view with an offset to offset
        // 0. See `metal_quantized_matmul_needs_a_chunk_at_offset_zero`.
        let rows = hidden.narrow(0, start, len).unwrap();
        let logits = model.lm_head(&rows).unwrap().to_dtype(DType::F32).unwrap();
        top.extend(logits.argmax(D::Minus1).unwrap().to_vec1::<u32>().unwrap());
    }
    top
}

/// Compares a quantized model with the dense reference on `alice-512.txt`.
/// Returns the top-1 agreement over the prompt positions.
fn compare(
    label: &str,
    dense: &mut ModelForCausalLM,
    quantized: &mut ModelForCausalLM,
    prompt: &[u32],
    device: &Device,
) -> f64 {
    let (reference, tokens) = run(dense, prompt, None, device);
    let (logits, _) = run(quantized, prompt, Some(&tokens), device);
    let mut max_diff = [0f32; 2];
    let mut changes = Vec::new();
    for (step, (r, q)) in reference.iter().zip(&logits).enumerate() {
        assert_eq!(r.len(), q.len(), "{label}: step {step}: vocab size");
        assert!(
            q.iter().all(|v| v.is_finite()),
            "{label}: step {step}: a logit is not finite"
        );
        let diff = r
            .iter()
            .zip(q)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let phase = usize::from(step > 0);
        max_diff[phase] = max_diff[phase].max(diff);
        if argmax(r) != argmax(q) {
            changes.push(format!("{step} ({:.3})", margin(r)));
        }
    }
    let dense_top = top_tokens(dense, prompt, device);
    let quantized_top = top_tokens(quantized, prompt, device);
    let same = dense_top
        .iter()
        .zip(&quantized_top)
        .filter(|(a, b)| a == b)
        .count();
    let agreement = same as f64 / prompt.len() as f64;
    let changes = if changes.is_empty() {
        "none".to_string()
    } else {
        changes.join(", ")
    };
    eprintln!(
        "{label}: max |dense - quantized| = {:.4} after the prefill, {:.4} in {} decode steps; \
         top token changes at steps (dense margin): {changes}; \
         top-1 agreement over {} prompt positions: {same}/{} = {:.4}",
        max_diff[0],
        max_diff[1],
        STEPS - 1,
        prompt.len(),
        prompt.len(),
        agreement,
    );
    agreement
}

struct Setup {
    config: Config,
    prompt: Vec<u32>,
}

fn setup(dir: &Path) -> Setup {
    let config: Config =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let prompts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/prompts");
    // Read as the CLI does with "$(cat FILE)": without the final newline.
    let text = std::fs::read_to_string(prompts_dir.join("alice-512.txt")).unwrap();
    let prompt = tokenizer
        .encode(text.trim_end_matches('\n'), true)
        .unwrap()
        .get_ids()
        .to_vec();
    Setup { config, prompt }
}

fn dense(dir: &Path, config: &Config, dtype: DType, device: &Device) -> ModelForCausalLM {
    // SAFETY: the test does not change the weights file while it is mapped.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], dtype, device)
    }
    .unwrap();
    ModelForCausalLM::new(config, vb, false).unwrap()
}

/// The quantized model, or `None` if its GGUF file is missing.
fn quantized(
    dir: &Path,
    config: &Config,
    dtype: Dtype,
    device: &Device,
) -> Option<ModelForCausalLM> {
    let path = gguf_path(dir, dtype).unwrap();
    if !path.exists() {
        eprintln!("skipped {dtype}: {} is missing", path.display());
        return None;
    }
    // The file holds only the quantized tensors, with their safetensors
    // names.
    let mut file = std::fs::File::open(&path).unwrap();
    let content = gguf_file::Content::read(&mut file).unwrap();
    assert!(
        content
            .tensor_infos
            .contains_key("model.embed_tokens.weight")
    );
    let vb = QVarBuilder::from_gguf(&path, device).unwrap();
    Some(ModelForCausalLM::new_quantized(config, vb, false).unwrap())
}

#[cfg(feature = "metal")]
#[test]
fn quantized_logits_track_f32_on_metal() {
    use fornace_engine::AttentionKind;

    let Ok(dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return;
    };
    let dir = Path::new(&dir);
    let device = Device::new_metal(0).expect("open the metal device");
    let Setup { config, prompt } = setup(dir);
    // The reference is dense F32, because the quantized models compute in
    // F32. Dense BF16 is compared too: it shows the effect of the BF16
    // activations alone.
    let mut reference = dense(dir, &config, DType::F32, &device);
    reference.set_attention(AttentionKind::Fused);
    let mut bf16 = dense(dir, &config, DType::BF16, &device);
    bf16.set_attention(AttentionKind::Fused);
    compare(
        "metal f32 against bf16",
        &mut reference,
        &mut bf16,
        &prompt,
        &device,
    );
    drop(bf16);
    // Measured 2026-11-09 on alice-512 (512 positions): Q8 0.9863 and Q4
    // 0.6934. Dense BF16 gave 0.9824 (0.8223 before the RoPE tables were
    // computed in F32). The floors are a few points below.
    for (dtype, floor) in [(Dtype::Q8, 0.95), (Dtype::Q4, 0.65)] {
        let Some(mut model) = quantized(dir, &config, dtype, &device) else {
            continue;
        };
        model.set_attention(AttentionKind::Fused);
        let label = format!("metal f32 against {dtype}");
        let agreement = compare(&label, &mut reference, &mut model, &prompt, &device);
        assert!(
            agreement >= floor,
            "{label}: top-1 agreement {agreement:.4} is below {floor}"
        );
    }
}

#[test]
fn q8_logits_track_f32_on_the_cpu() {
    let Ok(dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return;
    };
    let dir = Path::new(&dir);
    let device = Device::Cpu;
    let Setup { config, prompt } = setup(dir);
    let mut reference = dense(dir, &config, DType::F32, &device);
    let Some(mut model) = quantized(dir, &config, Dtype::Q8, &device) else {
        return;
    };
    let label = "cpu f32 against q8";
    let agreement = compare(label, &mut reference, &mut model, &prompt, &device);
    // Measured 2026-11-09 on alice-512 (512 positions): 0.9766.
    let floor = 0.95;
    assert!(
        agreement >= floor,
        "{label}: top-1 agreement {agreement:.4} is below {floor}"
    );
}

/// Candle's quantized Metal matmul for more than one row builds a new
/// layout for its input, and so ignores the start offset of a narrow view
/// (`candle-core/src/quantized/metal.rs`, `fwd`, at fork `79fe58df`).
/// `Tensor::copy` keeps the offset. The quantized projection in `model.rs`
/// uses `force_contiguous`, which copies to a new buffer at offset 0. This test
/// checks that such a chunk gives the rows of the full matmul. It prints
/// whether a narrow view still gives the first rows. No model needed.
#[cfg(feature = "metal")]
#[test]
fn metal_quantized_matmul_needs_a_chunk_at_offset_zero() {
    use candle_core::Module;
    use candle_core::quantized::{GgmlDType, QMatMul, QTensor};

    let device = Device::new_metal(0).expect("open the metal device");
    let (rows, k, n) = (8, 64, 32);
    let values = |count: usize, scale: f32| -> Vec<f32> {
        (0..count).map(|i| (i as f32 * scale).sin()).collect()
    };
    let weight = Tensor::from_vec(values(n * k, 0.37), (n, k), &device).unwrap();
    let matmul =
        QMatMul::from_qtensor(QTensor::quantize(&weight, GgmlDType::Q8_0).unwrap()).unwrap();
    let input = Tensor::from_vec(values(rows * k, 0.11), (rows, k), &device).unwrap();
    let run = |x: &Tensor| -> Vec<Vec<f32>> { matmul.forward(x).unwrap().to_vec2().unwrap() };
    let near = |a: &[Vec<f32>], b: &[Vec<f32>]| {
        a.iter()
            .flatten()
            .zip(b.iter().flatten())
            .all(|(x, y)| (x - y).abs() < 1e-2)
    };
    let full = run(&input);
    let (start, len) = (4, 4);
    let chunk = input.narrow(0, start, len).unwrap();
    assert!(near(
        &run(&chunk.force_contiguous().unwrap()),
        &full[start..]
    ));
    let view = run(&chunk);
    if near(&view, &full[..len]) {
        eprintln!("candle still ignores the input offset: a view of rows 4 to 7 gave rows 0 to 3");
    } else if near(&view, &full[start..]) {
        eprintln!("candle now uses the input offset: the copy in the quantized projection can go");
    }
}

/// A quantized batch runs, and each row gives the text of batch 1. At
/// decode, `lm_head` gets the last position of each row as a strided view,
/// which Candle's quantized matmul does not accept as it is.
#[cfg(feature = "metal")]
#[test]
fn a_quantized_batch_gives_the_text_of_batch_one() {
    use fornace_engine::{AttentionKind, DeviceKind, Engine, LoadConfig};

    let Ok(dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return;
    };
    let dir = Path::new(&dir);
    for dtype in [Dtype::Q8, Dtype::Q4] {
        if !gguf_path(dir, dtype).unwrap().exists() {
            eprintln!("skipped {dtype}: the GGUF file is missing");
            continue;
        }
        let mut engine = Engine::load(&LoadConfig {
            model_dir: dir.to_path_buf(),
            dtype,
            device: DeviceKind::Metal,
            max_new_tokens: 16,
            batch: 4,
            kv_cache: KvCacheKind::Prealloc,
            block_size: fornace_engine::DEFAULT_BLOCK_SIZE,
            kv_capacity: None,
            share_prefix: false,
            attention: AttentionKind::Fused,
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
        for (prompt, text) in prompts.iter().zip(&batch.texts) {
            let single = engine.generate(prompt).unwrap();
            assert_eq!(&single.texts[0], text, "{dtype} {prompt:?}");
        }
    }
}
