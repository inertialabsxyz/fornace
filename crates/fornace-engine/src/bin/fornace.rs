//! The `fornace` command: load a model and generate text from a prompt.

use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use fornace_engine::{DeviceKind, Dtype, Engine, LoadConfig};

#[cfg(target_os = "macos")]
const DEFAULT_DEVICE: &str = "metal";
#[cfg(not(target_os = "macos"))]
const DEFAULT_DEVICE: &str = "cpu";

#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// The directory that contains the model files.
    #[arg(long, value_name = "DIR")]
    model: PathBuf,
    /// The prompt to continue.
    #[arg(long, value_name = "TEXT")]
    prompt: String,
    /// The maximum number of tokens to generate.
    #[arg(long, value_name = "N", default_value_t = 128)]
    max_new_tokens: usize,
    /// The data type of the weights: f16, q8, or q4.
    #[arg(long, default_value = "f16")]
    dtype: Dtype,
    /// The device: cpu or metal.
    #[arg(long, default_value = DEFAULT_DEVICE)]
    device: DeviceKind,
    /// The number of sequences to generate at the same time.
    #[arg(long, value_name = "N", default_value_t = 1)]
    batch: usize,
    /// Print the report as JSON instead of text.
    #[arg(long)]
    json: bool,
    /// Out to file
    #[arg(long, value_name = "FILE")]
    out: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = LoadConfig {
        model_dir: args.model,
        dtype: args.dtype,
        device: args.device,
        max_new_tokens: args.max_new_tokens,
        batch: args.batch,
    };
    let mut engine = Engine::load(&config)?;
    let report = engine.generate(&args.prompt)?;
    if let Some(path) = &args.out {
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(path, json).with_context(|| format!("cannot write {}", path.display()))?;
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{}", report.text);
        println!();
        println!("prompt tokens:        {}", report.prompt_tokens);
        println!("generated tokens:     {}", report.generated_tokens);
        println!("prefill tokens/s:     {:.1}", report.prefill_tokens_per_s);
        println!("decode tokens/s:      {:.1}", report.decode_tokens_per_s);
        println!(
            "time to first token:  {:.1} ms",
            report.time_to_first_token_ms
        );
        println!("peak RSS:             {} bytes", report.peak_rss_bytes);
    }
    Ok(())
}
