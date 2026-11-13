//! The `fornace-server` command: serve one model over an OpenAI-compatible
//! HTTP API.

use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use fornace_engine::{AttentionKind, DeviceKind, Dtype, Engine, KvCacheKind, LoadConfig};
use fornace_server::Config;
use tokenizers::Tokenizer;

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
    /// The data type of the weights: f16 (dense, BF16 on Metal and F32 on
    /// the CPU), f32, q8 (Q8_0), or q4 (Q4_0).
    #[arg(long, default_value = "f16")]
    dtype: Dtype,
    /// The attention: naive or fused. Fused needs --device metal.
    #[arg(long, default_value = "fused")]
    attention: AttentionKind,
    /// The device: cpu or metal.
    #[arg(long, default_value = DEFAULT_DEVICE)]
    device: DeviceKind,
    /// The address to listen on.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// The port to listen on.
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// The most prompt and new tokens for one request.
    #[arg(long, value_name = "N", default_value_t = 4096)]
    max_context: usize,
    /// The repeat penalty of a request that does not give one. 1.0 turns it
    /// off.
    #[arg(long, value_name = "P", default_value_t = fornace_engine::DEFAULT_REPEAT_PENALTY)]
    repeat_penalty: f32,
    /// The most requests in one decode step. More requests wait in the
    /// queue. 1 runs one request at a time.
    #[arg(long, value_name = "N", default_value_t = 8)]
    max_batch: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let path = args.model.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("cannot load {}", path.display()))?;
    let model_name = args
        .model
        .file_name()
        .map_or_else(|| "model".into(), |n| n.to_string_lossy().into_owned());
    let load = LoadConfig {
        model_dir: args.model,
        dtype: args.dtype,
        device: args.device,
        // Each request gives its own count.
        max_new_tokens: args.max_context,
        batch: 1,
        kv_cache: KvCacheKind::Prealloc,
        block_size: fornace_engine::DEFAULT_BLOCK_SIZE,
        // Each request reserves its prompt and its new tokens, as in the
        // bench.
        kv_capacity: None,
        share_prefix: false,
        attention: args.attention,
        sync_spans: false,
        ignore_eos: false,
        repeat_penalty: args.repeat_penalty,
        draft: None,
    };
    let config = Config {
        model_name,
        max_context: args.max_context,
        repeat_penalty: args.repeat_penalty,
        max_batch: args.max_batch,
    };
    let app = fornace_server::app(config, tokenizer, move || Ok(Engine::load(&load)?));
    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("cannot listen on {addr}"))?;
    eprintln!("listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
