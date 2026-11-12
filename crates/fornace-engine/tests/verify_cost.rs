//! Times the verify pass of speculative decoding: one forward of M tokens
//! after a prompt, with the logits of all M positions on the CPU.
//!
//! This is a measurement, not a check, so it is ignored by default. Run it
//! with:
//!
//! ```sh
//! cargo test --release --features metal -p fornace-engine \
//!     --test verify_cost -- --ignored --nocapture
//! ```
//!
//! For each model in `models/qwen2.5-0.5b-instruct` and
//! `models/qwen2.5-1.5b-instruct` (a missing model or GGUF file is
//! skipped), and for each format (`bf16`, `q8`, `q4`), the test prefills
//! the 513 tokens of `alice-512.txt`. Then it times a forward of M tokens
//! at offset 513, for M = 1 to 9, with fused attention. The cache is reset
//! and the prompt prefilled again before each call, so each call starts
//! from 513 positions. The prefill is not timed.
//!
//! M = 1 is one decode step without the sampling. M = K + 1 is the verify
//! pass for K guesses. The ratio to M = 1 is r in the speculative decoding
//! brief.

#![cfg(feature = "metal")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;
use fornace_engine::model::{Config, ModelForCausalLM};
use fornace_engine::quantize::gguf_path;
use fornace_engine::{AttentionKind, Dtype, KvCacheKind};
use tokenizers::Tokenizer;

const MAX_M: usize = 9;
const WARMUP: usize = 3;
const TIMED: usize = 10;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn prompt(dir: &Path) -> Vec<u32> {
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    // Read as the CLI does with "$(cat FILE)": without the final newline.
    let text = std::fs::read_to_string(repo().join("bench/prompts/alice-512.txt")).unwrap();
    tokenizer
        .encode(text.trim_end_matches('\n'), true)
        .unwrap()
        .get_ids()
        .to_vec()
}

/// The model in `format`, or `None` if its GGUF file is missing.
fn load(dir: &Path, config: &Config, format: Dtype, device: &Device) -> Option<ModelForCausalLM> {
    let mut model = match gguf_path(dir, format) {
        Some(path) => {
            if !path.exists() {
                return None;
            }
            let vb = QVarBuilder::from_gguf(&path, device).unwrap();
            ModelForCausalLM::new_quantized(config, vb, false).unwrap()
        }
        None => {
            // SAFETY: the test does not change the weights file while it is
            // mapped.
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[dir.join("model.safetensors")],
                    DType::BF16,
                    device,
                )
            }
            .unwrap();
            ModelForCausalLM::new(config, vb, false).unwrap()
        }
    };
    model.set_attention(AttentionKind::Fused);
    Some(model)
}

/// Prefills `prompt`, then returns the time in milliseconds of one forward
/// of `m` tokens at the end of it, with the logits of all `m` positions
/// copied to the CPU.
fn verify_ms(
    model: &mut ModelForCausalLM,
    prompt: &[u32],
    m: usize,
    device: &Device,
) -> Result<f64> {
    model.reset_kv_cache(KvCacheKind::Prealloc, prompt.len() + MAX_M);
    let input = Tensor::new(prompt, device)?.unsqueeze(0)?;
    model.hidden_states(&input, 0)?;
    device.synchronize()?;
    // The token values do not change the time.
    let input = Tensor::new(&prompt[..m], device)?.unsqueeze(0)?;
    let start = Instant::now();
    let hidden = model.hidden_states(&input, prompt.len())?.squeeze(0)?;
    let logits: Vec<Vec<f32>> = model.lm_head(&hidden)?.to_dtype(DType::F32)?.to_vec2()?;
    let ms = start.elapsed().as_secs_f64() * 1e3;
    assert_eq!(logits.len(), m);
    Ok(ms)
}

#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn time_the_verify_pass_on_metal() -> Result<()> {
    let device = Device::new_metal(0)?;
    println!("| model | format | M | median, ms | min, ms | max, ms | median / median at M = 1 |");
    println!("|---|---|---:|---:|---:|---:|---:|");
    for name in ["qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct"] {
        let dir = repo().join("models").join(name);
        if !dir.join("config.json").exists() {
            eprintln!("skip {name}: no model in {}", dir.display());
            continue;
        }
        let config: Config =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let prompt = prompt(&dir);
        for format in [Dtype::F16, Dtype::Q8, Dtype::Q4] {
            let Some(mut model) = load(&dir, &config, format, &device) else {
                eprintln!("skip {name} {format}: no GGUF file");
                continue;
            };
            let label = if format == Dtype::F16 {
                "bf16".to_string()
            } else {
                format.to_string()
            };
            let mut base = 0.0;
            for m in 1..=MAX_M {
                for _ in 0..WARMUP {
                    verify_ms(&mut model, &prompt, m, &device)?;
                }
                let mut times = (0..TIMED)
                    .map(|_| verify_ms(&mut model, &prompt, m, &device))
                    .collect::<Result<Vec<f64>>>()?;
                times.sort_by(f64::total_cmp);
                let median = (times[TIMED / 2 - 1] + times[TIMED / 2]) / 2.0;
                if m == 1 {
                    base = median;
                }
                println!(
                    "| {name} | {label} | {m} | {median:.2} | {:.2} | {:.2} | {:.2} |",
                    times[0],
                    times[TIMED - 1],
                    median / base,
                );
            }
        }
    }
    Ok(())
}
