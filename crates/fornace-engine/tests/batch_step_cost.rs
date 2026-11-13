//! Times one decode step with B rows, for the scheduler brief (step 1).
//!
//! This is a measurement, not a check, so it is ignored by default. Run it
//! with:
//!
//! ```sh
//! cargo test --release --features metal -p fornace-engine \
//!     --test batch_step_cost -- --ignored --nocapture --test-threads=1
//! ```
//!
//! For Qwen2.5-0.5B in `bf16`, `q8`, and `q4` (a missing GGUF file is
//! skipped), and B = 1, 2, 4, 8: prefill B copies of `alice-512.txt` (513
//! tokens) as one batch, then time 64 greedy decode steps with fused
//! attention. Each step reads the next token of each row back to the CPU,
//! as a real step does.
//!
//! Two layouts for the quantized matmuls: the rows on the batch axis
//! (`[B, 1, K]`, one matrix-vector launch for each row, as today) and the
//! rows on the M axis (`[B, K]`, one matmul launch). Dense weights always
//! use the M axis. The test also compares the logits of the two layouts.
//!
//! Then it times the attention part alone: 24 layers of one fused attention
//! launch for B rows, against B launches of one row each, with each row in
//! its own cache tensor.
//!
//! `row_caches_step_cost` times the same steps with one cache for each row
//! (`forward_rows`, the path of the scheduler), against one cache for the
//! batch, with the M-axis thresholds of `m_axis_rows`.

#![cfg(feature = "metal")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;
use fornace_engine::model::{Config, ModelForCausalLM, RowCache};
use fornace_engine::quantize::gguf_path;
use fornace_engine::{AttentionKind, Dtype, KvCacheKind};
use tokenizers::Tokenizer;

const WARMUP: usize = 8;
const TIMED: usize = 64;
const ROWS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn prompt(dir: &Path) -> Vec<u32> {
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let text = std::fs::read_to_string(repo().join("bench/prompts/alice-512.txt")).unwrap();
    tokenizer
        .encode(text.as_str(), true)
        .unwrap()
        .get_ids()
        .to_vec()
}

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

/// Prefills `rows` copies of `ids`, and returns the logits of the first
/// decode step and the mean time of the timed decode steps, in ms.
fn run(model: &mut ModelForCausalLM, ids: &[u32], rows: usize, device: &Device) -> (Tensor, f64) {
    model.reset_kv_cache(KvCacheKind::Prealloc, ids.len() + WARMUP + TIMED + 1);
    let input = Tensor::new(ids, device)
        .unwrap()
        .unsqueeze(0)
        .unwrap()
        .repeat((rows, 1))
        .unwrap();
    let logits = model.forward(&input, 0).unwrap();
    let mut next: Vec<u32> = logits
        .squeeze(1)
        .unwrap()
        .argmax(D::Minus1)
        .unwrap()
        .to_vec1()
        .unwrap();
    let mut first = None;
    let mut start = Instant::now();
    for step in 0..WARMUP + TIMED {
        if step == WARMUP {
            start = Instant::now();
        }
        let input = Tensor::from_vec(next.clone(), (rows, 1), device).unwrap();
        let logits = model
            .forward(&input, ids.len() + step)
            .unwrap()
            .squeeze(1)
            .unwrap();
        next = logits.argmax(D::Minus1).unwrap().to_vec1().unwrap();
        if step == 0 {
            first = Some(logits.to_dtype(DType::F32).unwrap());
        }
    }
    let ms = start.elapsed().as_secs_f64() * 1e3 / TIMED as f64;
    (first.unwrap(), ms)
}

/// As `run`, with one cache for each row (`forward_rows`): each row is
/// prefilled alone, and each step runs RoPE, the cache append, and the
/// attention once for each row. Returns the mean step time in ms.
fn run_rows(model: &mut ModelForCausalLM, ids: &[u32], rows: usize, device: &Device) -> f64 {
    let mut caches: Vec<RowCache> = (0..rows)
        .map(|_| model.new_row_cache(ids.len() + WARMUP + TIMED + 1))
        .collect();
    let prompt = Tensor::new(ids, device).unwrap().unsqueeze(0).unwrap();
    let mut next = Vec::new();
    for cache in caches.iter_mut() {
        let logits = model.forward_rows(&prompt, &mut [cache]).unwrap();
        next.push(
            logits
                .squeeze(1)
                .unwrap()
                .argmax(D::Minus1)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap()[0],
        );
    }
    let mut start = Instant::now();
    for step in 0..WARMUP + TIMED {
        if step == WARMUP {
            start = Instant::now();
        }
        let input = Tensor::from_vec(next.clone(), (rows, 1), device).unwrap();
        let mut refs: Vec<&mut RowCache> = caches.iter_mut().collect();
        let logits = model
            .forward_rows(&input, &mut refs)
            .unwrap()
            .squeeze(1)
            .unwrap();
        next = logits.argmax(D::Minus1).unwrap().to_vec1().unwrap();
    }
    start.elapsed().as_secs_f64() * 1e3 / TIMED as f64
}

/// The fewest rows that go on the M axis, from the first run of
/// `batch_step_cost` at `aa23bc8`: the crossover points.
fn m_axis_rows(format: Dtype) -> usize {
    match format {
        Dtype::Q8 => 5,
        Dtype::Q4 => 7,
        _ => usize::MAX,
    }
}

/// The step with one cache for each row, against one cache for the batch,
/// with the M-axis thresholds.
#[test]
#[ignore = "a measurement on Metal with the model"]
fn row_caches_step_cost() {
    let dir = repo().join("models/qwen2.5-0.5b-instruct");
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipped: {} is missing", dir.display());
        return;
    }
    let device = Device::new_metal(0).unwrap();
    let config: Config =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let ids = prompt(&dir);
    println!(
        "| Format | B | One batch cache, ms | One cache for each row, ms | Difference, ms | Total tok/s, own caches |"
    );
    println!("|---|---:|---:|---:|---:|---:|");
    for format in [Dtype::F16, Dtype::Q8, Dtype::Q4] {
        let Some(mut model) = load(&dir, &config, format, &device) else {
            continue;
        };
        model.set_quantized_m_axis_rows(m_axis_rows(format));
        let name = if format == Dtype::F16 {
            "bf16".to_string()
        } else {
            format.to_string()
        };
        for &rows in &ROWS {
            let (_, batch) = run(&mut model, &ids, rows, &device);
            let own = run_rows(&mut model, &ids, rows, &device);
            println!(
                "| {name} | {rows} | {batch:.3} | {own:.3} | {:+.3} | {:.1} |",
                own - batch,
                rows as f64 * 1e3 / own
            );
        }
    }
}

#[test]
#[ignore = "a measurement on Metal with the model"]
fn batch_step_cost() {
    let dir = repo().join("models/qwen2.5-0.5b-instruct");
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipped: {} is missing", dir.display());
        return;
    }
    let device = Device::new_metal(0).unwrap();
    let config: Config =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let ids = prompt(&dir);
    println!("prompt tokens: {}", ids.len());
    println!(
        "| Format | B | Layout | Step ms | Step / step at B = 1 | Total tok/s | Logits max diff | Same top tokens |"
    );
    println!("|---|---:|---|---:|---:|---:|---:|---|");
    for format in [Dtype::F16, Dtype::Q8, Dtype::Q4] {
        let Some(mut model) = load(&dir, &config, format, &device) else {
            eprintln!("skipped: no {format} file");
            continue;
        };
        let name = if format == Dtype::F16 {
            "bf16".to_string()
        } else {
            format.to_string()
        };
        let layouts: &[bool] = if format == Dtype::F16 {
            &[false]
        } else {
            &[false, true]
        };
        let mut base = None;
        for &rows in &ROWS {
            let mut batch_axis = None;
            for &on_m in layouts {
                model.set_quantized_rows_on_m(on_m);
                let (logits, ms) = run(&mut model, &ids, rows, &device);
                let base_ms = *base.get_or_insert(ms);
                let layout = match (format, on_m) {
                    (Dtype::F16, _) => "dense (M axis)",
                    (_, false) => "batch axis",
                    (_, true) => "M axis",
                };
                let (diff, same) = match &batch_axis {
                    None => (String::from("—"), String::from("—")),
                    Some(reference) => {
                        let d: f32 = (&logits - reference)
                            .unwrap()
                            .abs()
                            .unwrap()
                            .max_all()
                            .unwrap()
                            .to_scalar()
                            .unwrap();
                        let a: Vec<u32> = logits.argmax(D::Minus1).unwrap().to_vec1().unwrap();
                        let b: Vec<u32> = reference.argmax(D::Minus1).unwrap().to_vec1().unwrap();
                        (format!("{d:.2e}"), (a == b).to_string())
                    }
                };
                println!(
                    "| {name} | {rows} | {layout} | {ms:.3} | {:.2}× | {:.1} | {diff} | {same} |",
                    ms / base_ms,
                    rows as f64 * 1e3 / ms
                );
                if !on_m {
                    batch_axis = Some(logits);
                }
            }
        }
        model.set_quantized_rows_on_m(false);
    }
}

/// The attention of one decode step in 24 layers: one fused launch for B
/// rows against B launches of one row, each row with its own K and V.
#[test]
#[ignore = "a measurement on Metal"]
fn attention_once_for_each_row_cost() {
    let device = Device::new_metal(0).unwrap();
    let (layers, heads, kv_heads, head_dim, kv_len) = (24, 14, 2, 64, 513);
    let scale = 1.0 / (head_dim as f32).sqrt();
    println!("| dtype | B | One launch, ms | One launch for each row, ms | Difference, ms |");
    println!("|---|---:|---:|---:|---:|");
    for dtype in [DType::BF16, DType::F32] {
        for rows in ROWS {
            let rand = |shape: (usize, usize, usize, usize)| {
                Tensor::randn(0f32, 1.0, shape, &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap()
            };
            let q = rand((rows, heads, 1, head_dim));
            let k = rand((rows, kv_heads, kv_len, head_dim));
            let v = rand((rows, kv_heads, kv_len, head_dim));
            let q_rows: Vec<Tensor> = (0..rows).map(|_| rand((1, heads, 1, head_dim))).collect();
            let k_rows: Vec<Tensor> = (0..rows)
                .map(|_| rand((1, kv_heads, kv_len, head_dim)))
                .collect();
            let v_rows: Vec<Tensor> = (0..rows)
                .map(|_| rand((1, kv_heads, kv_len, head_dim)))
                .collect();
            let time = |f: &dyn Fn() -> Tensor| {
                for _ in 0..WARMUP {
                    f();
                }
                device.synchronize().unwrap();
                let start = Instant::now();
                for _ in 0..TIMED {
                    f();
                }
                device.synchronize().unwrap();
                start.elapsed().as_secs_f64() * 1e3 / TIMED as f64
            };
            let one = time(&|| {
                let mut out = q.clone();
                for _ in 0..layers {
                    out = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale, 1.0).unwrap();
                }
                out
            });
            let each = time(&|| {
                let mut out = q.clone();
                for _ in 0..layers {
                    let parts: Vec<Tensor> = (0..rows)
                        .map(|r| {
                            candle_nn::ops::sdpa(
                                &q_rows[r], &k_rows[r], &v_rows[r], None, false, scale, 1.0,
                            )
                            .unwrap()
                        })
                        .collect();
                    out = Tensor::cat(&parts, 0).unwrap();
                }
                out
            });
            println!(
                "| {dtype:?} | {rows} | {one:.3} | {each:.3} | {:.3} |",
                each - one
            );
        }
    }
}
