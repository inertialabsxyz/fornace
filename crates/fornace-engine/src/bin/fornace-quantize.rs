//! The `fornace-quantize` command: write Q8_0 or Q4_0 weights to a GGUF
//! file next to `model.safetensors`. The engine loads that file for
//! `--dtype q8` or `--dtype q4`.

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use fornace_engine::Dtype;
use fornace_engine::quantize::quantize_model;

#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// The directory that contains `model.safetensors`. The GGUF file is
    /// written to the same directory.
    #[arg(long, value_name = "DIR")]
    model: PathBuf,
    /// The format: q8 (Q8_0) or q4 (Q4_0).
    #[arg(long)]
    dtype: Dtype,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let start = Instant::now();
    let path = quantize_model(&args.model, args.dtype)?;
    let bytes = std::fs::metadata(&path)?.len();
    println!(
        "Wrote {} ({:.1} MB) in {:.1} s",
        path.display(),
        bytes as f64 / 1e6,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
