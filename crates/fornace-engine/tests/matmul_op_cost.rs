//! Times one Candle matmul at the decode shapes of Qwen2.5-0.5B and
//! Qwen2.5-1.5B, for the Metal gap work (step 1).
//!
//! This is a measurement, not a check, so it is ignored by default. Run it
//! on a quiet machine with:
//!
//! ```sh
//! FORNACE_OP_COST_OUT=bench/runs/<file>.csv \
//! cargo test --release --features metal -p fornace-engine \
//!     --test matmul_op_cost -- --ignored --nocapture --test-threads=1
//! ```
//!
//! For each shape (N, K) of a projection, at M = 1, 2, 4, 8, 16 rows:
//!
//! - `bf16`: `candle_nn::Linear` on a `[M, 1, K]` input, as in a decode
//!   step. Candle makes it one `[M, K]` matmul.
//! - Q8_0 and Q4_0: `QMatMul` on F32 input, in two layouts. `batch_axis`
//!   is `[M, 1, K]` (one matrix-vector launch for each row). `m_axis` is
//!   `[M, K]` (the matmul kernel for M > 1). At M = 1 the two are the same.
//!
//! The weights are random. Two modes:
//!
//! - `hot`: the same weights for each op, as `test-backend-ops perf` does.
//!   The weights of a small matrix can stay in the GPU cache.
//! - `cold`: a ring of copies of the weights, 256 MiB or more in total, so
//!   each op reads its weights from memory, as a decode step does.
//!
//! One sample queues 32 to 256 ops, enough to read 2 GiB of weights, and
//! then waits for the GPU. Both modes use the same number of ops, so the
//! cost of the wait is the same in both. The time of one op is the sample
//! time divided by the number of ops. So for a small matrix it can be the
//! time to encode the launch on the CPU, not the kernel time. The test
//! takes 10 warm-up samples, then reports the median of 100 samples.
//!
//! The roofline is the bytes of the weights, the input, and the output at
//! 273 GB/s (the memory bandwidth of the M4 Pro).
//!
//! Environment variables:
//!
//! - `FORNACE_OP_COST_OUT`: also write the CSV to this file.
//! - `FORNACE_OP_COST_SAMPLES`: the number of samples (default 100). Use a
//!   small number for a quick check of the test itself.
//! - `FORNACE_OP_COST_MODELS`: a comma list, for example `0.5b`.

#![cfg(feature = "metal")]

use std::io::Write;
use std::time::Instant;

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::Linear;

const WARMUP: usize = 10;
const MIN_OPS: usize = 32;
const MAX_OPS: usize = 256;
const SAMPLE_BYTES: usize = 2 << 30;
const COLD_BYTES: usize = 256 << 20;
const BANDWIDTH: f64 = 273e9;
const ROWS: [usize; 5] = [1, 2, 4, 8, 16];

struct Shape {
    op: &'static str,
    n: usize,
    k: usize,
    /// The number of times the op runs in one decode step.
    calls: usize,
}

/// The matmul shapes of one decode step, from the `config.json` of each
/// model: Q, K and V, O, gate and up, down, and `lm_head`.
fn shapes(model: &str) -> Vec<Shape> {
    let (hidden, kv, mlp, layers) = match model {
        "0.5b" => (896, 128, 4864, 24),
        "1.5b" => (1536, 256, 8960, 28),
        _ => unreachable!(),
    };
    let vocab = 151_936;
    vec![
        Shape {
            op: "q",
            n: hidden,
            k: hidden,
            calls: layers,
        },
        Shape {
            op: "k_v",
            n: kv,
            k: hidden,
            calls: 2 * layers,
        },
        Shape {
            op: "o",
            n: hidden,
            k: hidden,
            calls: layers,
        },
        Shape {
            op: "gate_up",
            n: mlp,
            k: hidden,
            calls: 2 * layers,
        },
        Shape {
            op: "down",
            n: hidden,
            k: mlp,
            calls: layers,
        },
        Shape {
            op: "lm_head",
            n: vocab,
            k: hidden,
            calls: 1,
        },
    ]
}

fn weight_bytes(format: &str, n: usize, k: usize) -> usize {
    match format {
        "bf16" => n * k * 2,
        "q8_0" => n * k / 32 * 34,
        "q4_0" => n * k / 32 * 18,
        _ => unreachable!(),
    }
}

/// One op: a matmul with its own weights.
enum Op {
    Dense(Linear),
    Quantized(QMatMul),
}

impl Op {
    fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            Op::Dense(l) => l.forward(x).unwrap(),
            Op::Quantized(q) => q.forward(x).unwrap(),
        }
    }
}

/// `copies` ops with the same values in separate buffers.
fn make_ops(format: &str, w: &Tensor, copies: usize, device: &Device) -> Vec<Op> {
    let (n, k) = w.dims2().unwrap();
    match format {
        "bf16" => {
            // `Tensor::copy` shares the Metal buffer, so make each copy
            // with a cast, which writes a new buffer.
            let w = w.to_device(device).unwrap();
            (0..copies)
                .map(|_| Op::Dense(Linear::new(w.to_dtype(DType::BF16).unwrap(), None)))
                .collect()
        }
        _ => {
            let dtype = if format == "q8_0" {
                GgmlDType::Q8_0
            } else {
                GgmlDType::Q4_0
            };
            let q = QTensor::quantize(w, dtype).unwrap();
            let data = q.data().unwrap();
            (0..copies)
                .map(|_| {
                    let storage = QStorage::from_data(data.clone(), device, dtype).unwrap();
                    let t = QTensor::new(storage, (n, k)).unwrap();
                    Op::Quantized(QMatMul::from_qtensor(t).unwrap())
                })
                .collect()
        }
    }
}

/// Returns the median, the 10th, and the 90th percentile of the op time,
/// in µs. Each sample runs `per_sample` ops, in turn from `ops`.
fn time(
    ops: &[Op],
    x: &Tensor,
    per_sample: usize,
    samples: usize,
    device: &Device,
) -> (f64, f64, f64) {
    let mut next = 0;
    let mut times = Vec::with_capacity(samples);
    for sample in 0..WARMUP + samples {
        device.synchronize().unwrap();
        let start = Instant::now();
        for _ in 0..per_sample {
            ops[next].forward(x);
            next = (next + 1) % ops.len();
        }
        device.synchronize().unwrap();
        if sample >= WARMUP {
            times.push(start.elapsed().as_secs_f64() * 1e6 / per_sample as f64);
        }
    }
    times.sort_by(f64::total_cmp);
    let at = |p: f64| times[((times.len() - 1) as f64 * p).round() as usize];
    (at(0.5), at(0.1), at(0.9))
}

#[test]
#[ignore = "a measurement: run it on a quiet machine"]
fn matmul_op_cost() {
    let device = Device::new_metal(0).unwrap();
    let samples: usize = std::env::var("FORNACE_OP_COST_SAMPLES")
        .map(|s| s.parse().unwrap())
        .unwrap_or(100);
    let models: Vec<String> = std::env::var("FORNACE_OP_COST_MODELS")
        .unwrap_or_else(|_| "0.5b,1.5b".into())
        .split(',')
        .map(str::to_owned)
        .collect();
    let mut out = std::env::var("FORNACE_OP_COST_OUT")
        .ok()
        .map(|p| std::fs::File::create(p).unwrap());
    let mut emit = |line: String| {
        println!("{line}");
        if let Some(f) = out.as_mut() {
            writeln!(f, "{line}").unwrap();
        }
    };
    emit(
        "model,op,n,k,calls,format,layout,mode,m,ops_per_sample,median_us,p10_us,p90_us,\
         bytes,roofline_us,roofline_pct"
            .into(),
    );

    for model in &models {
        for s in shapes(model) {
            let w = (Tensor::randn(0f32, 1.0, (s.n, s.k), &Device::Cpu).unwrap() * 0.02).unwrap();
            for format in ["bf16", "q8_0", "q4_0"] {
                let wb = weight_bytes(format, s.n, s.k);
                let copies = COLD_BYTES.div_ceil(wb).max(2);
                let cold = make_ops(format, &w, copies, &device);
                let per_sample = SAMPLE_BYTES.div_ceil(wb).clamp(MIN_OPS, MAX_OPS);
                let layouts: &[&str] = if format == "bf16" {
                    &["linear"]
                } else {
                    &["batch_axis", "m_axis"]
                };
                for &layout in layouts {
                    for m in ROWS {
                        if m == 1 && layout == "m_axis" {
                            continue;
                        }
                        let x = Tensor::randn(0f32, 1.0, (m, 1, s.k), &device).unwrap();
                        let x = match (format, layout) {
                            ("bf16", _) => x.to_dtype(DType::BF16).unwrap(),
                            (_, "m_axis") => x.reshape((m, s.k)).unwrap(),
                            _ => x,
                        };
                        let act = if format == "bf16" { 2 } else { 4 };
                        let bytes = wb + m * (s.k + s.n) * act;
                        let roof = bytes as f64 / BANDWIDTH * 1e6;
                        for (mode, ops) in [("hot", &cold[..1]), ("cold", &cold[..])] {
                            let (med, p10, p90) = time(ops, &x, per_sample, samples, &device);
                            emit(format!(
                                "{model},{},{},{},{},{format},{layout},{mode},{m},{per_sample},\
                                 {med:.2},{p10:.2},{p90:.2},{bytes},{roof:.2},{:.1}",
                                s.op,
                                s.n,
                                s.k,
                                s.calls,
                                roof / med * 100.0
                            ));
                        }
                    }
                }
            }
        }
    }
}
