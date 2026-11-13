//! The `fornace-bench` command: run each engine and write one CSV row per run.
//!
//! The peak memory of a process cannot be reset (see `fornace_engine::metrics`).
//! So the command starts itself again as a worker for each (dtype, KV cache,
//! block size, shared prefix, batch, draft dtype, K) set. Each worker loads the engine once and writes its rows to
//! stdout as CSV. The parent appends the rows to the file and prints a summary.
//!
//! With `--transport http`, the parent starts the server of each engine
//! instead, sends the requests, and times them as a client.

use std::error::Error;
use std::fs::OpenOptions;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{ArgAction, Parser, ValueEnum};
use fornace_bench::agent::{self, AgentRequest};
use fornace_bench::http::{self, Timing};
use fornace_bench::servers::{self, Server, ServerEngine, ServerSpec};
use fornace_engine::{
    AttentionKind, DEFAULT_BLOCK_SIZE, DeviceKind, DraftConfig, Dtype, Engine, KvCacheKind,
    LoadConfig,
};
use serde::{Deserialize, Serialize};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[cfg(target_os = "macos")]
const DEFAULT_DEVICE: &str = "metal";
#[cfg(not(target_os = "macos"))]
const DEFAULT_DEVICE: &str = "cpu";

/// How the bench reaches the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Transport {
    /// The engine in a worker process, as a library.
    Process,
    /// The server of the engine, over HTTP.
    Http,
}

impl Transport {
    fn name(self) -> &'static str {
        match self {
            Transport::Process => "process",
            Transport::Http => "http",
        }
    }
}

/// The requests of an HTTP level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Load {
    /// Every request is the prompt file.
    Plain,
    /// The agent load: a shared system prompt, an own part of 64, 256, or
    /// 1,024 tokens, and 32, 128, or 256 new tokens.
    Agent,
}

#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// The directory that contains the model files.
    #[arg(long, value_name = "DIR")]
    model: PathBuf,
    /// The prompt files, separated by commas.
    #[arg(
        long,
        value_name = "FILES",
        value_delimiter = ',',
        default_value = "bench/prompts/alice-512.txt,bench/prompts/alice-2048.txt,bench/prompts/alice-4096.txt"
    )]
    prompts: Vec<PathBuf>,
    /// The data types of the weights, separated by commas: f16 (dense, BF16
    /// on Metal and F32 on the CPU), f32, q8 (Q8_0), or q4 (Q4_0).
    #[arg(long, value_delimiter = ',', default_value = "f16")]
    dtypes: Vec<Dtype>,
    /// The KV caches, separated by commas: cat, cat-contiguous, prealloc,
    /// or paged.
    #[arg(long, value_delimiter = ',', default_value = "prealloc")]
    kv_cache: Vec<KvCacheKind>,
    /// The block sizes of the paged cache, separated by commas. Only paged
    /// rows use them. A paged row has `block=<N>` in the tag column.
    #[arg(long, value_delimiter = ',', default_value = "16")]
    block_size: Vec<NonZeroUsize>,
    /// The positions that the cache reserves for each sequence. The default
    /// is the prompt and --max-new-tokens. A row with this flag has
    /// `capacity=<N>` in the tag column.
    #[arg(long, value_name = "N")]
    kv_capacity: Option<NonZeroUsize>,
    /// Run the batch with a shared prefix: true, false, or both, separated
    /// by commas. The engine prefills the tokens that all sequences have in
    /// common at the start once. Only paged and prealloc rows with a batch
    /// above 1 use true. They have `shared` in the tag column.
    #[arg(long, value_delimiter = ',', default_value = "false", action = ArgAction::Set)]
    share_prefix: Vec<bool>,
    /// A text file for a different end of each sequence. Sequence b is the
    /// prompt, then tokens b × N to (b + 1) × N of this file, with N from
    /// --suffix-tokens. So all sequences share the prompt (a "system
    /// prompt") and have their own "user message". A row with this flag has
    /// `suffix=<file>:<N>` in the tag column.
    #[arg(long, value_name = "FILE")]
    suffix: Option<PathBuf>,
    /// The tokens of each sequence from --suffix.
    #[arg(long, value_name = "N", default_value = "64")]
    suffix_tokens: NonZeroUsize,
    /// The batch sizes, separated by commas. A batch of B runs B copies of
    /// the prompt at the same time.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    batch: Vec<NonZeroUsize>,
    /// The attention: naive or fused. A row with fused attention has
    /// `attention=fused` in the tag column.
    #[arg(long, default_value = "naive")]
    attention: AttentionKind,
    /// The device: cpu or metal.
    #[arg(long, default_value = DEFAULT_DEVICE)]
    device: DeviceKind,
    /// The number of tokens to generate.
    #[arg(long, value_name = "N", default_value_t = 128)]
    max_new_tokens: usize,
    /// The number of recorded generations for each prompt.
    #[arg(long, value_name = "N", default_value_t = 3)]
    repeats: usize,
    /// The number of generations before the recorded ones. They are not
    /// recorded.
    #[arg(long, value_name = "N", default_value_t = 1)]
    warmup: usize,
    /// Do not stop at the end-of-sequence token, so that each run generates
    /// exactly --max-new-tokens tokens.
    #[arg(long, value_name = "BOOL", default_value_t = true, action = ArgAction::Set)]
    ignore_eos: bool,
    /// A draft model for speculative decoding: a model directory with the
    /// tokenizer of --model. Each K of --spec-k above 0 runs once with each
    /// draft dtype. It needs --batch 1 and --kv-cache prealloc.
    #[arg(long, value_name = "DIR")]
    draft_model: Option<PathBuf>,
    /// The data types of the draft weights, separated by commas, as for
    /// --dtypes.
    #[arg(long, value_delimiter = ',', default_value = "f16")]
    draft_dtypes: Vec<Dtype>,
    /// The guesses of the draft in each round, separated by commas. 0 runs
    /// the model without a draft, so that list "0,4" puts the target alone
    /// and the speculative rows in one command. Above 0 needs --draft-model.
    #[arg(long, value_delimiter = ',', default_value = "0")]
    spec_k: Vec<usize>,
    /// The engines to measure, separated by commas.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "fornace",
        value_parser = ["fornace", "mistralrs", "llamacpp"]
    )]
    engines: Vec<String>,
    /// The CSV file for the results. If it exists, the rows are appended.
    /// The default is bench/runs/<date>-<commit>.csv.
    #[arg(long, value_name = "CSV")]
    out: Option<PathBuf>,
    /// A note for the tag column.
    #[arg(long, value_name = "TEXT", default_value = "")]
    tag: String,
    /// The repeat penalty. 1.0 turns it off.
    #[arg(long, value_name = "P", default_value_t = fornace_engine::DEFAULT_REPEAT_PENALTY)]
    repeat_penalty: f32,
    /// The transports, separated by commas: process (the engine in a
    /// worker) or http (the server of the engine). Only fornace runs in a
    /// process.
    #[arg(long, value_delimiter = ',', default_value = "process")]
    transport: Vec<Transport>,
    /// HTTP only: the requests in flight at the same time, separated by
    /// commas. Each value is one level.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    concurrency: Vec<NonZeroUsize>,
    /// HTTP only: the recorded requests of each level. With --load agent, a
    /// level at concurrency C sends at least 4 × C and at least 16.
    #[arg(long, value_name = "N", default_value_t = 3)]
    requests: usize,
    /// HTTP only: plain (the prompt files) or agent (the agent load).
    #[arg(long, default_value = "plain")]
    load: Load,
    /// HTTP only: the port of the server.
    #[arg(long, default_value_t = 8090)]
    port: u16,
    /// HTTP only: the context positions of each server slot.
    #[arg(long, value_name = "N", default_value_t = 4096)]
    server_context: usize,
    /// HTTP only: the slots of llama.cpp (-np) and mistral.rs (--max-seqs).
    /// The default is the largest --concurrency.
    #[arg(long, value_name = "N")]
    parallel: Option<NonZeroUsize>,
    /// HTTP, llama.cpp only: keep its prompt caches on. The rows get
    /// `cache_prompt` in the tag column.
    #[arg(long)]
    cache_prompt: bool,
    /// A text file for the quality column. Each worker computes the
    /// perplexity of the model on it once, after its speed runs, and writes
    /// it to each of its rows. The tag column gets `quality=<file name>`.
    #[arg(long, value_name = "FILE")]
    quality: Option<PathBuf>,
    /// Run one (dtype, KV cache, batch) set and write the rows to stdout.
    #[arg(long, hide = true)]
    worker: bool,
    /// The commit for the rows. The parent gives it to each worker.
    #[arg(long, hide = true)]
    commit: Option<String>,
    /// The date for the rows. The parent gives it to each worker.
    #[arg(long, hide = true)]
    date: Option<String>,
}

/// One CSV row: one recorded generation. The field order is the column order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Row {
    commit: String,
    date: String,
    engine: String,
    model: String,
    device: String,
    dtype: String,
    kv_cache: String,
    batch: usize,
    prompt_file: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    run: usize,
    load_ms: f64,
    prefill_tok_s: f64,
    decode_tok_s: f64,
    ttft_ms: f64,
    decode_ms_mean_last64: Option<f64>,
    peak_mem_mb: f64,
    quality: String,
    tag: String,
    /// The draft model and the format of its weights, for example
    /// `qwen2.5-0.5b-instruct:q8`. Empty without a draft.
    draft: String,
    spec_k: Option<usize>,
    rounds: Option<usize>,
    accepted: Option<usize>,
    tested: Option<usize>,
    /// accepted / tested, with 3 decimals.
    alpha: Option<f64>,
    draft_ms: Option<f64>,
    verify_ms: Option<f64>,
    /// `fornace@<commit>`, `llama.cpp@b<build>`, or `mistral.rs@<tag>`.
    engine_version: String,
    /// `process` or `http`.
    transport: String,
    /// The requests in flight at the same time. 1 for process rows.
    concurrency: usize,
    /// The index of the request in its level, from 0. Empty for process
    /// rows.
    request: Option<usize>,
    /// 1.0 is off.
    repeat_penalty: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.worker {
        worker(&args)
    } else {
        parent(&args)
    }
}

/// Starts one worker for each (dtype, KV cache, batch) set, appends the
/// rows to the CSV file, and prints a summary table.
fn parent(args: &Args) -> Result<()> {
    if args.draft_model.is_none() && args.spec_k.iter().any(|&k| k > 0) {
        return Err("--spec-k above 0 needs --draft-model".into());
    }
    let commit = git_commit();
    let date = today();
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("bench/runs/{date}-{commit}.csv")));
    check_transport(args)?;
    let mut rows = Vec::new();
    let mut levels = Vec::new();
    if args.transport.contains(&Transport::Http) {
        let tokenizer = load_tokenizer(&args.model)?;
        let agent_list = match args.load {
            Load::Plain => None,
            Load::Agent => Some(agent_list(args, &tokenizer, &out)?),
        };
        for engine in &args.engines {
            for dtype in &args.dtypes {
                let (new, new_levels) = run_http(
                    args,
                    &commit,
                    &date,
                    engine,
                    *dtype,
                    &tokenizer,
                    agent_list.as_deref(),
                )?;
                append_rows(&out, &new)?;
                rows.extend(new);
                levels.extend(new_levels);
            }
        }
    }
    for engine in &args.engines {
        if engine != "fornace" || !args.transport.contains(&Transport::Process) {
            continue;
        }
        for dtype in &args.dtypes {
            for kv_cache in &args.kv_cache {
                for (block_size, share_prefix) in cache_variants(args, *kv_cache) {
                    for batch in &args.batch {
                        if share_prefix && batch.get() == 1 {
                            continue;
                        }
                        for (draft_dtype, spec_k) in spec_variants(args) {
                            let set = Set {
                                dtype: *dtype,
                                kv_cache: *kv_cache,
                                block_size,
                                share_prefix,
                                batch: *batch,
                                draft_dtype,
                                spec_k,
                            };
                            let new = run_worker(args, &commit, &date, &set)?;
                            append_rows(&out, &new)?;
                            rows.extend(new);
                        }
                    }
                }
            }
        }
    }
    eprintln!("Wrote {} rows to {}", rows.len(), out.display());
    let process: Vec<Row> = rows
        .iter()
        .filter(|r| r.transport == "process")
        .cloned()
        .collect();
    if !process.is_empty() {
        print!("{}", summary(&process));
    }
    if !levels.is_empty() {
        print!("{}", level_summary(&levels));
    }
    Ok(())
}

/// The flags that do not go together.
fn check_transport(args: &Args) -> Result<()> {
    let process = args.transport.contains(&Transport::Process);
    let http = args.transport.contains(&Transport::Http);
    if process && args.engines.iter().any(|e| e != "fornace") {
        return Err("only fornace runs with --transport process. \
                    The other engines run only over HTTP"
            .into());
    }
    if process && (args.concurrency.iter().any(|c| c.get() > 1) || args.load == Load::Agent) {
        return Err("--concurrency and --load agent need --transport http".into());
    }
    if http {
        let one = |v: &[NonZeroUsize]| v.iter().all(|b| b.get() == 1);
        if !one(&args.batch)
            || args.kv_cache.iter().any(|k| *k != KvCacheKind::Prealloc)
            || args.spec_k.iter().any(|&k| k > 0)
            || args.share_prefix.contains(&true)
            || args.suffix.is_some()
            || args.quality.is_some()
            || args.kv_capacity.is_some()
        {
            return Err("--transport http runs batch 1, prealloc, and no draft, \
                        suffix, quality, or capacity flags"
                .into());
        }
    }
    if args.cache_prompt && args.engines.iter().any(|e| e != "llamacpp") {
        return Err("--cache-prompt is for --engines llamacpp only".into());
    }
    Ok(())
}

fn load_tokenizer(model: &Path) -> Result<tokenizers::Tokenizer> {
    let path = model.join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&path)
        .map_err(|e| format!("cannot load {}: {e}", path.display()).into())
}

/// The agent load for the largest level. Writes the list next to the CSV
/// file, so that the load can be checked.
fn agent_list(
    args: &Args,
    tokenizer: &tokenizers::Tokenizer,
    out: &Path,
) -> Result<Vec<AgentRequest>> {
    let ids = |path: &str| -> Result<Vec<u32>> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        Ok(tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec())
    };
    let system = ids("bench/prompts/alice-2048.txt")?[..agent::SYSTEM_TOKENS].to_vec();
    let sources = [
        ids("bench/prompts/code-512.txt")?,
        ids("bench/prompts/alice-de-512.txt")?,
    ];
    let count = args
        .concurrency
        .iter()
        .map(|c| level_count(args, c.get()))
        .max()
        .unwrap_or(0);
    let lens: Vec<usize> = sources.iter().map(Vec::len).collect();
    let list = agent::build(&agent::plan(count, &lens), &system, &sources, tokenizer)?;
    let path = out.with_file_name(format!(
        "{}-agent-load.csv",
        out.file_stem().unwrap_or_default().to_string_lossy()
    ));
    let mut w = csv::Writer::from_path(&path)?;
    w.write_record([
        "index",
        "source",
        "offset",
        "system_tokens",
        "own_tokens",
        "prompt_tokens",
        "max_tokens",
    ])?;
    let names = ["code-512.txt", "alice-de-512.txt"];
    for r in &list {
        let p = &r.plan;
        w.write_record([
            p.index.to_string(),
            names[p.source].to_string(),
            p.offset.to_string(),
            agent::SYSTEM_TOKENS.to_string(),
            p.own_tokens.to_string(),
            r.prompt_tokens.to_string(),
            p.max_tokens.to_string(),
        ])?;
    }
    w.flush()?;
    eprintln!(
        "Wrote the agent load ({count} requests) to {}",
        path.display()
    );
    Ok(list)
}

/// The recorded requests of a level at concurrency `c`.
fn level_count(args: &Args, c: usize) -> usize {
    match args.load {
        Load::Plain => args.requests,
        Load::Agent => args.requests.max(4 * c).max(16),
    }
}

/// One request of a level.
#[derive(Clone)]
struct Request {
    prompt_file: String,
    prompt: String,
    /// The Fornace token count of the prompt.
    prompt_tokens: usize,
    max_tokens: usize,
}

/// The numbers of one (engine, dtype, concurrency, prompt) level.
struct Level {
    engine: String,
    dtype: String,
    concurrency: usize,
    prompt_file: String,
    requests: usize,
    ttft_p50: f64,
    ttft_max: f64,
    decode_p50: f64,
    total_tok_s: f64,
    peak_mem_mb: f64,
}

/// The `dtype` column of an HTTP row.
fn http_dtype(engine: ServerEngine, dtype: Dtype, device: DeviceKind) -> Dtype {
    match (engine, dtype, device) {
        (ServerEngine::Fornace, Dtype::F16, DeviceKind::Cpu) => Dtype::F32,
        (_, Dtype::F16, _) => Dtype::Bf16,
        _ => dtype,
    }
}

/// Starts the server of `engine` with `dtype`, runs each level, and stops
/// the server. One server for all levels, so the peak memory of a level
/// includes the levels before it.
fn run_http(
    args: &Args,
    commit: &str,
    date: &str,
    engine_name: &str,
    dtype: Dtype,
    tokenizer: &tokenizers::Tokenizer,
    agent_list: Option<&[AgentRequest]>,
) -> Result<(Vec<Row>, Vec<Level>)> {
    let engine = ServerEngine::parse(engine_name).ok_or("unknown engine")?;
    let largest = args.concurrency.iter().map(|c| c.get()).max().unwrap_or(1);
    let spec = ServerSpec {
        engine,
        dtype,
        model_dir: args.model.clone(),
        port: args.port,
        context: args.server_context,
        parallel: args.parallel.map_or(largest, NonZeroUsize::get),
        attention: args.attention,
        device: args.device,
        cache_prompt: args.cache_prompt,
    };
    // The prompts of the levels, before the server starts.
    let mut sets: Vec<(usize, Vec<Request>)> = Vec::new();
    for c in &args.concurrency {
        let c = c.get();
        let count = level_count(args, c);
        match agent_list {
            Some(list) => sets.push((
                c,
                list[..count]
                    .iter()
                    .map(|r| Request {
                        prompt_file: "agent".into(),
                        prompt: r.prompt.clone(),
                        prompt_tokens: r.prompt_tokens,
                        max_tokens: r.plan.max_tokens,
                    })
                    .collect(),
            )),
            None => {
                for path in &args.prompts {
                    let prompt = std::fs::read_to_string(path)
                        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
                    let prompt_tokens = tokenizer
                        .encode(prompt.as_str(), true)
                        .map_err(|e| e.to_string())?
                        .len();
                    let request = Request {
                        prompt_file: file_name(path),
                        prompt,
                        prompt_tokens,
                        max_tokens: args.max_new_tokens,
                    };
                    sets.push((c, vec![request; count]));
                }
            }
        }
    }
    for request in sets.iter().flat_map(|(_, r)| r) {
        if request.prompt_tokens + request.max_tokens > spec.context {
            return Err(format!(
                "{} needs {} positions, --server-context is {}",
                request.prompt_file,
                request.prompt_tokens + request.max_tokens,
                spec.context
            )
            .into());
        }
    }

    let log = std::env::temp_dir().join(format!("fornace-bench-{engine_name}-{dtype}.log"));
    let server = Server::start(&spec, &log)?;
    let version = servers::version(engine, commit);
    let url = format!("{}/v1/completions", server.base);
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(600)))
        .build()
        .into();
    let body = |r: &Request, stream: bool| {
        servers::request_body(
            engine,
            &r.prompt,
            r.max_tokens,
            args.repeat_penalty,
            args.ignore_eos,
            args.cache_prompt,
            stream,
        )
    };
    let column_dtype = http_dtype(engine, dtype, args.device).to_string();
    let mut tag: Vec<String> = Vec::new();
    if engine == ServerEngine::Fornace && args.attention != AttentionKind::default() {
        tag.push(format!("attention={}", args.attention));
    }
    if engine != ServerEngine::Fornace {
        tag.push(format!("np={}", spec.parallel));
    }
    if args.load == Load::Agent {
        tag.push("load=agent".into());
    }
    if args.cache_prompt {
        tag.push("cache_prompt".into());
    }

    let mut rows = Vec::new();
    let mut levels = Vec::new();
    for (concurrency, requests) in &sets {
        // One warm-up request without a stream. It also checks the counts
        // of the server, because not every server sends usage in a stream.
        let (_, usage) = http::complete(&agent, &url, &body(&requests[0], false))?;
        check_counts(engine_name, &requests[0], usage, args.ignore_eos)?;

        let start = std::time::Instant::now();
        let results = run_level(&agent, &url, requests, *concurrency, &body)?;
        let wall = start.elapsed().as_secs_f64();
        let peak_mem_mb = peak_mem_mb(server.peak_memory_bytes()?);

        let mut level_rows = Vec::new();
        for (i, timing) in results {
            let request = &requests[i];
            let (n, inferred) = match timing.usage {
                Some(u) => {
                    check_counts(engine_name, request, u, args.ignore_eos)?;
                    (u.completion_tokens, false)
                }
                // No usage in the stream (mistral.rs). With EOS ignored, a
                // stop at the length limit gives exactly max_tokens.
                None if args.ignore_eos && timing.finish.as_deref() == Some("length") => {
                    (request.max_tokens, true)
                }
                None => {
                    return Err(format!(
                        "{engine_name}: request {i} has no usage and finish {:?}",
                        timing.finish
                    )
                    .into());
                }
            };
            let ttft_ms = timing.ttft_ms().ok_or("a request gave no text")?;
            let mut row_tag = [Some(args.tag.clone()).filter(|t| !t.is_empty())]
                .into_iter()
                .flatten()
                .chain(tag.iter().cloned())
                .collect::<Vec<_>>();
            if inferred {
                row_tag.push("n=max_tokens".into());
            }
            level_rows.push(Row {
                commit: commit.to_string(),
                date: date.to_string(),
                engine: engine_name.to_string(),
                model: file_name(&args.model),
                device: args.device.to_string(),
                dtype: column_dtype.clone(),
                kv_cache: if engine == ServerEngine::Fornace {
                    KvCacheKind::Prealloc.to_string()
                } else {
                    String::new()
                },
                batch: 1,
                prompt_file: request.prompt_file.clone(),
                prompt_tokens: request.prompt_tokens,
                generated_tokens: n,
                run: i + 1,
                load_ms: server.load_ms,
                prefill_tok_s: request.prompt_tokens as f64 / (ttft_ms / 1e3),
                decode_tok_s: timing.decode_tok_s(n).unwrap_or(0.0),
                ttft_ms,
                decode_ms_mean_last64: timing.decode_ms_mean_last64(n),
                peak_mem_mb,
                quality: String::new(),
                tag: row_tag.join("; "),
                draft: String::new(),
                spec_k: None,
                rounds: None,
                accepted: None,
                tested: None,
                alpha: None,
                draft_ms: None,
                verify_ms: None,
                engine_version: version.clone(),
                transport: Transport::Http.name().into(),
                concurrency: *concurrency,
                request: Some(i),
                repeat_penalty: args.repeat_penalty,
            });
        }
        level_rows.sort_by_key(|r| r.request);
        let ttft: Vec<f64> = level_rows.iter().map(|r| r.ttft_ms).collect();
        let decode: Vec<f64> = level_rows.iter().map(|r| r.decode_tok_s).collect();
        let tokens: usize = level_rows.iter().map(|r| r.generated_tokens).sum();
        let level = Level {
            engine: engine_name.into(),
            dtype: column_dtype.clone(),
            concurrency: *concurrency,
            prompt_file: requests[0].prompt_file.clone(),
            requests: level_rows.len(),
            ttft_p50: http::percentile(&ttft, 50.0).unwrap_or(0.0),
            ttft_max: http::percentile(&ttft, 100.0).unwrap_or(0.0),
            decode_p50: http::percentile(&decode, 50.0).unwrap_or(0.0),
            total_tok_s: http::total_tok_s(tokens, wall),
            peak_mem_mb,
        };
        eprintln!(
            "{engine_name} {} C={} {}: {} requests, ttft p50 {:.1} ms, decode p50 {:.1} tok/s, total {:.1} tok/s",
            level.dtype,
            level.concurrency,
            level.prompt_file,
            level.requests,
            level.ttft_p50,
            level.decode_p50,
            level.total_tok_s
        );
        rows.extend(level_rows);
        levels.push(level);
    }
    drop(server);
    Ok((rows, levels))
}

/// The server must see the Fornace prompt tokens. With EOS ignored, it
/// must give `max_tokens`.
fn check_counts(
    engine: &str,
    request: &Request,
    usage: http::Usage,
    ignore_eos: bool,
) -> Result<()> {
    if usage.prompt_tokens != request.prompt_tokens {
        return Err(format!(
            "{engine} counts {} prompt tokens in {}, Fornace counts {}",
            usage.prompt_tokens, request.prompt_file, request.prompt_tokens
        )
        .into());
    }
    if ignore_eos && usage.completion_tokens != request.max_tokens {
        return Err(format!(
            "{engine} gave {} tokens, not {}",
            usage.completion_tokens, request.max_tokens
        )
        .into());
    }
    Ok(())
}

/// Sends the requests with `concurrency` threads. Each thread sends its
/// next request when its last one is done (a closed loop). Returns the
/// timing of each request with its index.
fn run_level(
    agent: &ureq::Agent,
    url: &str,
    requests: &[Request],
    concurrency: usize,
    body: &(dyn Fn(&Request, bool) -> serde_json::Value + Sync),
) -> Result<Vec<(usize, Timing)>> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let next = AtomicUsize::new(0);
    let results = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..concurrency)
            .map(|_| {
                scope.spawn(|| -> std::result::Result<Vec<(usize, Timing)>, String> {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::SeqCst);
                        if i >= requests.len() {
                            return Ok(done);
                        }
                        done.push((i, http::stream(agent, url, &body(&requests[i], true))?));
                    }
                })
            })
            .collect();
        threads
            .into_iter()
            .map(|t| t.join().expect("a client thread panicked"))
            .collect::<std::result::Result<Vec<_>, String>>()
    })?;
    Ok(results.into_iter().flatten().collect())
}

/// A Markdown table with one line for each HTTP level. With fewer than 100
/// requests, p99 is the largest value, so the table gives the maximum.
fn level_summary(levels: &[Level]) -> String {
    let mut out = String::from(
        "| engine | dtype | prompt | C | requests | TTFT p50 ms | TTFT max ms | decode p50 tok/s | total tok/s | peak MB |\n\
         |---|---|---|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    for l in levels {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |\n",
            l.engine,
            l.dtype,
            l.prompt_file,
            l.concurrency,
            l.requests,
            l.ttft_p50,
            l.ttft_max,
            l.decode_p50,
            l.total_tok_s,
            l.peak_mem_mb
        ));
    }
    out
}

/// The settings of one worker.
struct Set {
    dtype: Dtype,
    kv_cache: KvCacheKind,
    block_size: NonZeroUsize,
    share_prefix: bool,
    batch: NonZeroUsize,
    /// The draft dtype, or `None` without a draft (K = 0).
    draft_dtype: Option<Dtype>,
    spec_k: usize,
}

/// The (draft dtype, K) pairs: `(None, 0)` once if --spec-k has 0, then
/// each draft dtype with each K above 0.
fn spec_variants(args: &Args) -> Vec<(Option<Dtype>, usize)> {
    let mut variants = Vec::new();
    if args.spec_k.contains(&0) {
        variants.push((None, 0));
    }
    for &dtype in &args.draft_dtypes {
        for &k in args.spec_k.iter().filter(|&&k| k > 0) {
            variants.push((Some(dtype), k));
        }
    }
    variants
}

/// The (block size, shared prefix) pairs for one KV cache. Only the paged
/// cache has block sizes. Only the paged and prealloc caches share.
fn cache_variants(args: &Args, kv_cache: KvCacheKind) -> Vec<(NonZeroUsize, bool)> {
    let default = NonZeroUsize::new(DEFAULT_BLOCK_SIZE).unwrap();
    match kv_cache {
        KvCacheKind::Paged => args
            .block_size
            .iter()
            .flat_map(|&b| args.share_prefix.iter().map(move |&s| (b, s)))
            .collect(),
        KvCacheKind::Prealloc => args.share_prefix.iter().map(|&s| (default, s)).collect(),
        _ => vec![(default, false)],
    }
}

/// The token rows of a batch: `prompt` in each row, then a different
/// window of `n` tokens of `suffix` for each row.
fn suffix_rows(prompt: &[u32], suffix: &[u32], batch: usize, n: usize) -> Result<Vec<Vec<u32>>> {
    if suffix.len() < batch * n {
        return Err(format!(
            "--suffix has {} tokens, a batch of {batch} needs {}",
            suffix.len(),
            batch * n
        )
        .into());
    }
    Ok(suffix
        .chunks(n)
        .take(batch)
        .map(|window| [prompt, window].concat())
        .collect())
}

/// Starts this program again as a worker for one set and reads its rows.
fn run_worker(args: &Args, commit: &str, date: &str, set: &Set) -> Result<Vec<Row>> {
    let Set {
        dtype,
        kv_cache,
        block_size,
        share_prefix,
        batch,
        draft_dtype,
        spec_k,
    } = set;
    let prompts = args
        .prompts
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(",");
    let output = Command::new(std::env::current_exe()?)
        .arg("--worker")
        .arg("--model")
        .arg(&args.model)
        .args(["--prompts", &prompts])
        .args(["--dtypes", &dtype.to_string()])
        .args(["--kv-cache", &kv_cache.to_string()])
        .args(["--batch", &batch.to_string()])
        .args(["--block-size", &block_size.to_string()])
        .args(["--share-prefix", &share_prefix.to_string()])
        .args(["--suffix-tokens", &args.suffix_tokens.to_string()])
        .args(
            args.suffix
                .iter()
                .flat_map(|q| [std::ffi::OsStr::new("--suffix"), q.as_os_str()]),
        )
        .args(
            args.kv_capacity
                .iter()
                .flat_map(|c| ["--kv-capacity".to_string(), c.to_string()]),
        )
        .args(["--attention", &args.attention.to_string()])
        .args(["--device", &args.device.to_string()])
        .args(["--max-new-tokens", &args.max_new_tokens.to_string()])
        .args(["--repeats", &args.repeats.to_string()])
        .args(["--warmup", &args.warmup.to_string()])
        .args(["--ignore-eos", &args.ignore_eos.to_string()])
        .args(["--repeat-penalty", &args.repeat_penalty.to_string()])
        .args(["--tag", &args.tag])
        .args(
            args.quality
                .iter()
                .flat_map(|q| [std::ffi::OsStr::new("--quality"), q.as_os_str()]),
        )
        .args(["--spec-k", &spec_k.to_string()])
        .args(
            draft_dtype
                .iter()
                .map(|d| ["--draft-dtypes".to_string(), d.to_string()])
                .chain(
                    args.draft_model
                        .iter()
                        .map(|m| ["--draft-model".to_string(), m.display().to_string()]),
                )
                .flatten(),
        )
        .args(["--commit", commit])
        .args(["--date", date])
        .stderr(Stdio::inherit())
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "the worker for {dtype} {kv_cache} block {block_size} shared {share_prefix} batch {batch} draft {draft_dtype:?} K {spec_k} failed: {}",
            output.status
        )
        .into());
    }
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(output.stdout.as_slice());
    Ok(reader.deserialize().collect::<csv::Result<_>>()?)
}

/// Loads the engine once for the first dtype, KV cache, and batch size. Runs
/// each prompt as a batch of that many copies, and writes one CSV row for
/// each recorded generation to stdout.
fn worker(args: &Args) -> Result<()> {
    let config = LoadConfig {
        model_dir: args.model.clone(),
        dtype: args.dtypes[0],
        device: args.device,
        max_new_tokens: args.max_new_tokens,
        batch: args.batch[0].get(),
        kv_cache: args.kv_cache[0],
        block_size: args.block_size[0].get(),
        kv_capacity: args.kv_capacity.map(NonZeroUsize::get),
        share_prefix: args.share_prefix[0],
        attention: args.attention,
        sync_spans: false,
        ignore_eos: args.ignore_eos,
        repeat_penalty: args.repeat_penalty,
        draft: match (&args.draft_model, args.spec_k[0]) {
            (Some(dir), k) if k > 0 => Some(DraftConfig {
                model_dir: dir.clone(),
                dtype: args.draft_dtypes[0],
                k,
            }),
            _ => None,
        },
    };
    let mut engine = Engine::load(&config)?;
    let dtype = engine.effective_dtype();
    if dtype != config.dtype {
        eprintln!(
            "warning: requested dtype {}, but the engine uses {dtype}. The rows say {dtype}.",
            config.dtype
        );
    }
    let model = file_name(&args.model);
    let draft = match (&config.draft, engine.draft_effective_dtype()) {
        (Some(d), Some(dtype)) => format!("{}:{dtype}", file_name(&d.model_dir)),
        _ => String::new(),
    };
    let quality_file = args.quality.as_deref().map(file_name);
    let mut rows = Vec::new();
    for prompt_path in &args.prompts {
        let prompt = std::fs::read_to_string(prompt_path)
            .map_err(|e| format!("cannot read {}: {e}", prompt_path.display()))?;
        let prompt_file = file_name(prompt_path);
        let tokens = engine.tokenize(&prompt)?;
        let batch_rows = match &args.suffix {
            None => vec![tokens; config.batch],
            Some(path) => {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
                let suffix = engine.tokenize(&text)?;
                suffix_rows(&tokens, &suffix, config.batch, args.suffix_tokens.get())?
            }
        };
        for _ in 0..args.warmup {
            engine.generate_tokens(&batch_rows)?;
        }
        for run in 1..=args.repeats {
            let report = engine.generate_tokens(&batch_rows)?;
            let row = Row {
                commit: args.commit.clone().unwrap_or_else(git_commit),
                date: args.date.clone().unwrap_or_else(today),
                engine: "fornace".to_string(),
                model: model.clone(),
                device: config.device.to_string(),
                dtype: dtype.to_string(),
                kv_cache: config.kv_cache.to_string(),
                batch: report.batch,
                prompt_file: prompt_file.clone(),
                prompt_tokens: report.prompt_tokens,
                generated_tokens: report.generated_tokens,
                run,
                load_ms: engine.load_ms,
                prefill_tok_s: report.prefill_tokens_per_s,
                decode_tok_s: report.decode_tokens_per_s,
                ttft_ms: report.time_to_first_token_ms,
                decode_ms_mean_last64: decode_ms_mean_last64(&report.decode_token_ms),
                peak_mem_mb: peak_mem_mb(report.peak_memory_bytes),
                quality: String::new(),
                tag: row_tag(
                    &args.tag,
                    config.attention,
                    &[cache_tag(&config), suffix_tag(args)].concat(),
                    quality_file.as_deref(),
                ),
                draft: draft.clone(),
                spec_k: report.speculative.as_ref().map(|s| s.k),
                rounds: report.speculative.as_ref().map(|s| s.rounds),
                accepted: report.speculative.as_ref().map(|s| s.accepted),
                tested: report.speculative.as_ref().map(|s| s.tested),
                alpha: report
                    .speculative
                    .as_ref()
                    .map(|s| (s.alpha() * 1000.0).round() / 1000.0),
                draft_ms: report.speculative.as_ref().map(|s| s.draft_ms),
                verify_ms: report.speculative.as_ref().map(|s| s.verify_ms),
                engine_version: format!(
                    "fornace@{}",
                    args.commit.clone().unwrap_or_else(git_commit)
                ),
                transport: Transport::Process.name().into(),
                concurrency: 1,
                request: None,
                repeat_penalty: config.repeat_penalty,
            };
            let blocks = report.kv_blocks.map_or(String::new(), |b| {
                format!(
                    ", kv blocks {} of {} in use ({} positions each, {} stored, {:.2} % empty)",
                    b.in_use,
                    b.blocks,
                    b.block_size,
                    b.stored_positions,
                    b.waste() * 100.0
                )
            });
            let blocks = format!(
                "{blocks}, shared prefix {} tokens",
                report.shared_prefix_tokens
            );
            let blocks = match (&row.spec_k, &row.alpha) {
                (Some(k), Some(alpha)) => format!("{blocks}, draft {draft} K {k} alpha {alpha:.3}"),
                _ => blocks,
            };
            eprintln!(
                "{} {} batch {} {} run {run}/{}: {:.1} decode tok/s total, ttft {:.1} ms{blocks}",
                row.dtype,
                row.kv_cache,
                row.batch,
                row.prompt_file,
                args.repeats,
                row.decode_tok_s,
                row.ttft_ms
            );
            rows.push(row);
        }
    }
    // After the speed runs, so that their peak memory does not include it.
    if let Some(path) = &args.quality {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let perplexity = format!("{:.3}", engine.perplexity(&text)?);
        eprintln!("{dtype} perplexity on {}: {perplexity}", file_name(path));
        for row in &mut rows {
            row.quality = perplexity.clone();
        }
    }
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(std::io::stdout());
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

/// The cache settings for the tag column: `block=<N>` for the paged
/// cache, `shared` for a shared prefix, and `capacity=<N>` if
/// --kv-capacity is given.
fn cache_tag(config: &LoadConfig) -> Vec<String> {
    let mut tags = Vec::new();
    if config.kv_cache == KvCacheKind::Paged {
        tags.push(format!("block={}", config.block_size));
    }
    if config.share_prefix && config.batch > 1 {
        tags.push("shared".to_string());
    }
    if let Some(capacity) = config.kv_capacity {
        tags.push(format!("capacity={capacity}"));
    }
    tags
}

/// `suffix=<file>:<N>` if --suffix is given.
fn suffix_tag(args: &Args) -> Vec<String> {
    args.suffix
        .iter()
        .map(|path| format!("suffix={}:{}", file_name(path), args.suffix_tokens))
        .collect()
}

/// The tag column: the --tag text, plus `attention=fused` if the attention
/// is not the default, plus the cache tags, plus `quality=<file>` if there
/// is a quality text.
fn row_tag(tag: &str, attention: AttentionKind, cache: &[String], quality: Option<&str>) -> String {
    let attention =
        (attention != AttentionKind::default()).then(|| format!("attention={attention}"));
    let quality = quality.map(|q| format!("quality={q}"));
    [Some(tag.to_string()).filter(|t| !t.is_empty()), attention]
        .into_iter()
        .flatten()
        .chain(cache.iter().cloned())
        .chain(quality)
        .collect::<Vec<_>>()
        .join("; ")
}

/// Appends rows to the CSV file. A new or empty file gets the header first.
fn append_rows(path: &Path, rows: &[Row]) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let new = file.metadata()?.len() == 0;
    let mut writer = csv::WriterBuilder::new().has_headers(new).from_writer(file);
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

/// The mean of the last 64 decode step times, or of all if there are fewer.
/// `None` if there are no decode steps.
fn decode_ms_mean_last64(decode_token_ms: &[f64]) -> Option<f64> {
    let last = &decode_token_ms[decode_token_ms.len().saturating_sub(64)..];
    (!last.is_empty()).then(|| last.iter().sum::<f64>() / last.len() as f64)
}

/// Bytes to MiB, rounded to one decimal.
fn peak_mem_mb(bytes: u64) -> f64 {
    (bytes as f64 / 1_048_576.0 * 10.0).round() / 10.0
}

/// A Markdown table with one line for each (prompt, dtype, KV cache, batch,
/// draft, K): the mean total decode tok/s, the mean α, and the mean peak
/// memory over the repeats.
fn summary(rows: &[Row]) -> String {
    let mut groups: Vec<(&Row, Vec<&Row>)> = Vec::new();
    for row in rows {
        let key = |r: &Row| {
            (
                r.prompt_file.clone(),
                r.dtype.clone(),
                r.kv_cache.clone(),
                r.batch,
                r.draft.clone(),
                r.spec_k,
            )
        };
        match groups.iter_mut().find(|(first, _)| key(first) == key(row)) {
            Some((_, group)) => group.push(row),
            None => groups.push((row, vec![row])),
        }
    }
    let mut out = String::from(
        "| prompt | dtype | kv_cache | batch | draft | K | decode tok/s | alpha | peak MB |\n\
         |---|---|---|---:|---|---:|---:|---:|---:|\n",
    );
    for (first, group) in groups {
        let mean =
            |f: fn(&Row) -> f64| group.iter().map(|r| f(r)).sum::<f64>() / group.len() as f64;
        let alpha = first
            .alpha
            .map(|_| format!("{:.3}", mean(|r| r.alpha.unwrap_or(0.0))))
            .unwrap_or_default();
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.1} | {alpha} | {:.1} |\n",
            first.prompt_file,
            first.dtype,
            first.kv_cache,
            first.batch,
            first.draft,
            first.spec_k.map(|k| k.to_string()).unwrap_or_default(),
            mean(|r| r.decode_tok_s),
            mean(|r| r.peak_mem_mb),
        ));
    }
    out
}

/// The short hash of HEAD, with `-dirty` if tracked files have changes, or
/// `unknown` if git fails.
fn git_commit() -> String {
    let Ok(output) = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
    else {
        return "unknown".to_string();
    };
    if !output.status.success() {
        return "unknown".to_string();
    }
    let hash = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let clean = Command::new("git")
        .args(["diff", "--quiet", "HEAD"])
        .status()
        .is_ok_and(|s| s.success());
    if clean { hash } else { format!("{hash}-dirty") }
}

/// Today's date in UTC, as YYYY-MM-DD.
fn today() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("the clock is after 1970")
        .as_secs();
    iso_date(secs / 86_400)
}

/// Formats days since 1970-01-01 as YYYY-MM-DD. This is Howard Hinnant's
/// `civil_from_days` algorithm.
fn iso_date(days: u64) -> String {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z % 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row() -> Row {
        Row {
            commit: "abc1234-dirty".to_string(),
            date: "2026-11-02".to_string(),
            engine: "fornace".to_string(),
            model: "qwen2.5-0.5b-instruct".to_string(),
            device: "metal".to_string(),
            dtype: "bf16".to_string(),
            kv_cache: "prealloc".to_string(),
            batch: 1,
            prompt_file: "alice-512.txt".to_string(),
            prompt_tokens: 512,
            generated_tokens: 128,
            run: 2,
            load_ms: 250.5,
            prefill_tok_s: 4000.0,
            decode_tok_s: 115.25,
            ttft_ms: 128.0,
            decode_ms_mean_last64: Some(8.5),
            peak_mem_mb: 1234.5,
            quality: String::new(),
            tag: "after-4077".to_string(),
            draft: String::new(),
            spec_k: None,
            rounds: None,
            accepted: None,
            tested: None,
            alpha: None,
            draft_ms: None,
            verify_ms: None,
            engine_version: "fornace@abc1234-dirty".to_string(),
            transport: "process".to_string(),
            concurrency: 1,
            request: None,
            repeat_penalty: 1.1,
        }
    }

    #[test]
    fn batch_is_one_by_default_and_parses_a_list() {
        let base = ["fornace-bench", "--model", "m"];
        let args = Args::try_parse_from(base).unwrap();
        assert_eq!(args.batch, [NonZeroUsize::MIN]);
        let args = Args::try_parse_from(base.iter().chain(&["--batch", "1,2,4"])).unwrap();
        let sizes: Vec<usize> = args.batch.iter().map(|b| b.get()).collect();
        assert_eq!(sizes, [1, 2, 4]);
        assert!(Args::try_parse_from(base.iter().chain(&["--batch", "1,0"])).is_err());
    }

    #[test]
    fn row_tag_names_only_the_fused_attention() {
        assert_eq!(row_tag("", AttentionKind::Naive, &[], None), "");
        assert_eq!(row_tag("run 2", AttentionKind::Naive, &[], None), "run 2");
        assert_eq!(
            row_tag("", AttentionKind::Fused, &[], None),
            "attention=fused"
        );
        assert_eq!(
            row_tag("run 2", AttentionKind::Fused, &[], None),
            "run 2; attention=fused"
        );
    }

    #[test]
    fn row_tag_names_the_quality_text() {
        assert_eq!(
            row_tag("", AttentionKind::Naive, &[], Some("alice-2048.txt")),
            "quality=alice-2048.txt"
        );
        assert_eq!(
            row_tag("case Q", AttentionKind::Fused, &[], Some("alice-2048.txt")),
            "case Q; attention=fused; quality=alice-2048.txt"
        );
    }

    #[test]
    fn row_tag_names_the_cache_settings() {
        let cache = ["block=16".to_string(), "shared".to_string()];
        assert_eq!(
            row_tag("case C", AttentionKind::Fused, &cache, Some("a.txt")),
            "case C; attention=fused; block=16; shared; quality=a.txt"
        );
    }

    fn config(kv_cache: KvCacheKind, batch: usize, share_prefix: bool) -> LoadConfig {
        LoadConfig {
            model_dir: PathBuf::from("m"),
            dtype: Dtype::F16,
            device: DeviceKind::Cpu,
            max_new_tokens: 1,
            batch,
            kv_cache,
            block_size: 64,
            kv_capacity: None,
            share_prefix,
            attention: AttentionKind::Fused,
            sync_spans: false,
            ignore_eos: true,
            repeat_penalty: fornace_engine::DEFAULT_REPEAT_PENALTY,
            draft: None,
        }
    }

    #[test]
    fn cache_tag_names_block_size_sharing_and_capacity() {
        assert!(cache_tag(&config(KvCacheKind::Prealloc, 4, false)).is_empty());
        assert_eq!(
            cache_tag(&config(KvCacheKind::Paged, 1, true)),
            ["block=64"]
        );
        assert_eq!(
            cache_tag(&config(KvCacheKind::Paged, 4, true)),
            ["block=64", "shared"]
        );
        assert_eq!(
            cache_tag(&config(KvCacheKind::Prealloc, 4, true)),
            ["shared"]
        );
        let mut reserved = config(KvCacheKind::Prealloc, 1, false);
        reserved.kv_capacity = Some(32768);
        assert_eq!(cache_tag(&reserved), ["capacity=32768"]);
    }

    #[test]
    fn the_default_block_size_is_the_engine_default() {
        let args = Args::try_parse_from(["fornace-bench", "--model", "m"]).unwrap();
        let sizes: Vec<usize> = args.block_size.iter().map(|b| b.get()).collect();
        assert_eq!(sizes, [DEFAULT_BLOCK_SIZE]);
        assert_eq!(args.share_prefix, [false]);
    }

    #[test]
    fn only_paged_gets_block_sizes_and_only_paged_and_prealloc_share() {
        let base = ["fornace-bench", "--model", "m"];
        let flags = ["--block-size", "8,16", "--share-prefix", "false,true"];
        let args = Args::try_parse_from(base.iter().chain(&flags)).unwrap();
        let sizes = |kind| -> Vec<(usize, bool)> {
            cache_variants(&args, kind)
                .into_iter()
                .map(|(b, s)| (b.get(), s))
                .collect()
        };
        assert_eq!(sizes(KvCacheKind::Prealloc), [(16, false), (16, true)]);
        assert_eq!(sizes(KvCacheKind::Cat), [(16, false)]);
        assert_eq!(
            sizes(KvCacheKind::Paged),
            [(8, false), (8, true), (16, false), (16, true)]
        );
    }

    #[test]
    fn suffix_rows_share_the_prompt_and_differ_at_the_end() {
        let rows = suffix_rows(&[1, 2], &[10, 11, 12, 13, 14, 15, 16], 3, 2).unwrap();
        assert_eq!(
            rows,
            [vec![1, 2, 10, 11], vec![1, 2, 12, 13], vec![1, 2, 14, 15]]
        );
        assert!(suffix_rows(&[1], &[10, 11, 12], 2, 2).is_err());
    }

    #[test]
    fn quality_is_an_optional_file() {
        let base = ["fornace-bench", "--model", "m"];
        assert_eq!(Args::try_parse_from(base).unwrap().quality, None);
        let args =
            Args::try_parse_from(base.iter().chain(&["--quality", "bench/prompts/a.txt"])).unwrap();
        assert_eq!(args.quality, Some(PathBuf::from("bench/prompts/a.txt")));
    }

    #[test]
    fn row_serialises_to_the_columns_in_order() {
        let mut writer = csv::Writer::from_writer(Vec::new());
        writer.serialize(row()).unwrap();
        let text = String::from_utf8(writer.into_inner().unwrap()).unwrap();
        assert_eq!(
            text,
            "commit,date,engine,model,device,dtype,kv_cache,batch,prompt_file,\
             prompt_tokens,generated_tokens,run,load_ms,prefill_tok_s,decode_tok_s,\
             ttft_ms,decode_ms_mean_last64,peak_mem_mb,quality,tag,draft,spec_k,\
             rounds,accepted,tested,alpha,draft_ms,verify_ms,engine_version,\
             transport,concurrency,request,repeat_penalty\n\
             abc1234-dirty,2026-11-02,fornace,qwen2.5-0.5b-instruct,metal,bf16,\
             prealloc,1,alice-512.txt,512,128,2,250.5,4000.0,115.25,128.0,8.5,\
             1234.5,,after-4077,,,,,,,,,fornace@abc1234-dirty,process,1,,1.1\n"
        );
    }

    #[test]
    fn row_round_trips_through_csv_without_header() {
        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(Vec::new());
        let mut empty = row();
        empty.decode_ms_mean_last64 = None;
        let mut spec = row();
        spec.draft = "qwen2.5-0.5b-instruct:q8".to_string();
        spec.spec_k = Some(4);
        spec.rounds = Some(70);
        spec.accepted = Some(56);
        spec.tested = Some(127);
        spec.alpha = Some(0.441);
        spec.draft_ms = Some(1361.5);
        spec.verify_ms = Some(2699.25);
        writer.serialize(row()).unwrap();
        writer.serialize(&empty).unwrap();
        writer.serialize(&spec).unwrap();
        let bytes = writer.into_inner().unwrap();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(bytes.as_slice());
        let rows: Vec<Row> = reader.deserialize().map(|r| r.unwrap()).collect();
        assert_eq!(rows, vec![row(), empty, spec]);
    }

    #[test]
    fn decode_mean_uses_the_last_64_steps() {
        let mut steps = vec![100.0; 10];
        steps.extend(vec![2.0; 64]);
        assert_eq!(decode_ms_mean_last64(&steps), Some(2.0));
    }

    #[test]
    fn decode_mean_uses_all_steps_if_fewer_than_64() {
        assert_eq!(decode_ms_mean_last64(&[1.0, 2.0, 6.0]), Some(3.0));
        assert_eq!(decode_ms_mean_last64(&[]), None);
    }

    #[test]
    fn peak_mem_is_mib_with_one_decimal() {
        assert_eq!(peak_mem_mb(1_048_576), 1.0);
        assert_eq!(peak_mem_mb(1_677_722), 1.6);
    }

    #[test]
    fn iso_date_formats_days_since_epoch() {
        assert_eq!(iso_date(0), "1970-01-01");
        assert_eq!(iso_date(11_016), "2000-02-29");
        assert_eq!(iso_date(20_759), "2026-11-02");
    }

    #[test]
    fn spec_variants_run_the_target_alone_once_and_each_draft_with_each_k() {
        let base = ["fornace-bench", "--model", "m"];
        let args = Args::try_parse_from(base).unwrap();
        assert_eq!(spec_variants(&args), [(None, 0)]);
        let flags = [
            "--draft-model",
            "d",
            "--draft-dtypes",
            "f16,q8",
            "--spec-k",
            "0,2,4",
        ];
        let args = Args::try_parse_from(base.iter().chain(&flags)).unwrap();
        assert_eq!(
            spec_variants(&args),
            [
                (None, 0),
                (Some(Dtype::F16), 2),
                (Some(Dtype::F16), 4),
                (Some(Dtype::Q8), 2),
                (Some(Dtype::Q8), 4),
            ]
        );
        let flags = ["--draft-model", "d", "--spec-k", "4"];
        let args = Args::try_parse_from(base.iter().chain(&flags)).unwrap();
        assert_eq!(spec_variants(&args), [(Some(Dtype::F16), 4)]);
    }

    #[test]
    fn summary_has_one_line_per_group_with_means() {
        let mut second = row();
        second.decode_tok_s = 125.0;
        second.peak_mem_mb = 1236.5;
        let mut other = row();
        other.kv_cache = "cat".to_string();
        other.decode_tok_s = 58.0;
        let mut batched = row();
        batched.batch = 4;
        batched.decode_tok_s = 400.0;
        let mut spec = row();
        spec.draft = "d:q8".to_string();
        spec.spec_k = Some(4);
        spec.alpha = Some(0.4);
        spec.decode_tok_s = 30.0;
        let mut spec2 = spec.clone();
        spec2.alpha = Some(0.5);
        let table = summary(&[row(), second, other, batched, spec, spec2]);
        assert_eq!(
            table,
            "| prompt | dtype | kv_cache | batch | draft | K | decode tok/s | alpha | peak MB |\n\
             |---|---|---|---:|---|---:|---:|---:|---:|\n\
             | alice-512.txt | bf16 | prealloc | 1 |  |  | 120.1 |  | 1235.5 |\n\
             | alice-512.txt | bf16 | cat | 1 |  |  | 58.0 |  | 1234.5 |\n\
             | alice-512.txt | bf16 | prealloc | 4 |  |  | 400.0 |  | 1234.5 |\n\
             | alice-512.txt | bf16 | prealloc | 1 | d:q8 | 4 | 30.0 | 0.450 | 1234.5 |\n"
        );
    }
    #[test]
    fn other_engines_run_only_over_http() {
        let parse = |extra: &[&str]| {
            let base = ["fornace-bench", "--model", "m"];
            Args::try_parse_from(base.iter().chain(extra)).unwrap()
        };
        assert!(check_transport(&parse(&["--engines", "llamacpp"])).is_err());
        assert!(check_transport(&parse(&["--engines", "llamacpp", "--transport", "http"])).is_ok());
        assert!(check_transport(&parse(&["--concurrency", "4"])).is_err());
        assert!(check_transport(&parse(&["--transport", "http", "--batch", "4"])).is_err());
        assert!(check_transport(&parse(&["--transport", "http", "--cache-prompt"])).is_err());
    }

    #[test]
    fn an_agent_level_has_at_least_4_c_and_16_requests() {
        let base = ["fornace-bench", "--model", "m", "--load", "agent"];
        let args = Args::try_parse_from(base).unwrap();
        assert_eq!(level_count(&args, 1), 16);
        assert_eq!(level_count(&args, 8), 32);
        let args = Args::try_parse_from(base.iter().chain(&["--requests", "40"])).unwrap();
        assert_eq!(level_count(&args, 8), 40);
        let plain = Args::try_parse_from(["fornace-bench", "--model", "m"]).unwrap();
        assert_eq!(level_count(&plain, 8), 3);
    }
}
