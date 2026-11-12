//! Times Candle's Metal kernels that the paged KV cache uses, against a
//! contiguous element-wise kernel over the same data.
//!
//! This is a measurement, not a check, so it is ignored by default. Run it
//! with:
//!
//! ```sh
//! cargo test --release --features metal -p fornace-engine \
//!     --test metal_gather_kernels -- --ignored --nocapture
//! ```
//!
//! Each row is the time of 48 calls, as in one decode step of
//! Qwen2.5-0.5B (24 layers, one call for K and one for V), with a sync
//! after each set of 48. The tensors have the shape of one layer of the
//! paged pool: 2 KV heads, head dim 64.
//!
//! - `index_select`: the paged gather by position, as in
//!   `LayerCache::append_paged`.
//! - `index_select` by block: the same data, gathered by block of 16.
//! - strided copy: `narrow` of a larger buffer, then `contiguous`.
//! - element-wise: `affine(1, 0)` on a contiguous tensor of the same size.
//!   This reads and writes the same bytes with a simple index.
//! - `scatter_set`: the write of one new position.

#![cfg(feature = "metal")]

use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};

const CALLS: usize = 48;
const REPEATS: usize = 50;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const BLOCK_SIZE: usize = 16;

/// The mean time in milliseconds of one set of `CALLS` calls, after one
/// set to warm up.
fn time(device: &Device, mut calls: impl FnMut() -> Result<()>) -> Result<f64> {
    let mut set = || -> Result<()> {
        for _ in 0..CALLS {
            calls()?;
        }
        device.synchronize()
    };
    set()?;
    let start = Instant::now();
    for _ in 0..REPEATS {
        set()?;
    }
    Ok(start.elapsed().as_secs_f64() * 1e3 / REPEATS as f64)
}

#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn time_the_paged_gather_kernels_on_metal() -> Result<()> {
    let device = Device::new_metal(0)?;
    println!(
        "| dtype | positions | index_select, ms | index_select by block of {BLOCK_SIZE}, ms | \
         strided copy, ms | element-wise, ms | scatter_set 1 position, ms |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for dtype in [DType::BF16, DType::F32] {
        // 512 + 128 and 2,048 + 128 positions, as in the bench prompts.
        for seq in [640usize, 2176] {
            let slots = seq + BLOCK_SIZE;
            let pool =
                Tensor::randn(0f32, 1., (KV_HEADS, slots, HEAD_DIM), &device)?.to_dtype(dtype)?;
            let positions = Tensor::from_vec((0..seq as u32).collect(), seq, &device)?;
            let blocks = Tensor::from_vec(
                (0..(seq / BLOCK_SIZE) as u32).collect(),
                seq / BLOCK_SIZE,
                &device,
            )?;
            let by_block = pool.reshape((KV_HEADS, slots / BLOCK_SIZE, BLOCK_SIZE * HEAD_DIM))?;
            let larger = Tensor::randn(0f32, 1., (1, KV_HEADS, slots, HEAD_DIM), &device)?
                .to_dtype(dtype)?;
            let same_size =
                Tensor::randn(0f32, 1., (KV_HEADS, seq, HEAD_DIM), &device)?.to_dtype(dtype)?;
            let one = Tensor::randn(0f32, 1., (KV_HEADS, 1, HEAD_DIM), &device)?.to_dtype(dtype)?;
            let slot = Tensor::from_vec(vec![5u32], (1, 1, 1), &device)?
                .broadcast_as((KV_HEADS, 1, HEAD_DIM))?
                .contiguous()?;

            let gather = time(&device, || pool.index_select(&positions, 1).map(drop))?;
            let block_gather = time(&device, || by_block.index_select(&blocks, 1).map(drop))?;
            let strided = time(&device, || larger.narrow(2, 0, seq)?.contiguous().map(drop))?;
            let elementwise = time(&device, || same_size.affine(1.0, 0.0).map(drop))?;
            let scatter = time(&device, || pool.scatter_set(&slot, &one, 1))?;
            println!(
                "| {dtype:?} | {seq} | {gather:.2} | {block_gather:.2} | {strided:.2} | \
                 {elementwise:.2} | {scatter:.2} |"
            );
        }
    }
    Ok(())
}
