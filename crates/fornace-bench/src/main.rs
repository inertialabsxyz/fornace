//! The `fornace-bench` command: run each engine and write one CSV row per run.
//!
//! The peak memory of a process cannot be reset (see `fornace_engine::metrics`).
//! So the command starts itself again as a worker for each (dtype, KV cache,
//! block size, shared prefix, batch) set. Each worker loads the engine once and writes its rows to
//! stdout as CSV. The parent appends the rows to the file and prints a summary.

use std::error::Error;
use std::fs::OpenOptions;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{ArgAction, Parser};
use fornace_engine::{
    AttentionKind, DEFAULT_BLOCK_SIZE, DeviceKind, Dtype, Engine, KvCacheKind, LoadConfig,
};
use serde::{Deserialize, Serialize};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

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
    let commit = git_commit();
    let date = today();
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("bench/runs/{date}-{commit}.csv")));
    let mut rows = Vec::new();
    for engine in &args.engines {
        if engine != "fornace" {
            eprintln!("{engine}: not yet, skipped");
            continue;
        }
        for dtype in &args.dtypes {
            for kv_cache in &args.kv_cache {
                for (block_size, share_prefix) in cache_variants(args, *kv_cache) {
                    for batch in &args.batch {
                        if share_prefix && batch.get() == 1 {
                            continue;
                        }
                        let set = Set {
                            dtype: *dtype,
                            kv_cache: *kv_cache,
                            block_size,
                            share_prefix,
                            batch: *batch,
                        };
                        let new = run_worker(args, &commit, &date, &set)?;
                        append_rows(&out, &new)?;
                        rows.extend(new);
                    }
                }
            }
        }
    }
    eprintln!("Wrote {} rows to {}", rows.len(), out.display());
    print!("{}", summary(&rows));
    Ok(())
}

/// The settings of one worker.
struct Set {
    dtype: Dtype,
    kv_cache: KvCacheKind,
    block_size: NonZeroUsize,
    share_prefix: bool,
    batch: NonZeroUsize,
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
        .args(["--tag", &args.tag])
        .args(
            args.quality
                .iter()
                .flat_map(|q| [std::ffi::OsStr::new("--quality"), q.as_os_str()]),
        )
        .args(["--commit", commit])
        .args(["--date", date])
        .stderr(Stdio::inherit())
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "the worker for {dtype} {kv_cache} block {block_size} shared {share_prefix} batch {batch} failed: {}",
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
        draft: None,
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

/// A Markdown table with one line for each (prompt, dtype, KV cache, batch):
/// the mean total decode tok/s and the mean peak memory over the repeats.
fn summary(rows: &[Row]) -> String {
    let mut groups: Vec<(&Row, Vec<&Row>)> = Vec::new();
    for row in rows {
        let key = |r: &Row| {
            (
                r.prompt_file.clone(),
                r.dtype.clone(),
                r.kv_cache.clone(),
                r.batch,
            )
        };
        match groups.iter_mut().find(|(first, _)| key(first) == key(row)) {
            Some((_, group)) => group.push(row),
            None => groups.push((row, vec![row])),
        }
    }
    let mut out = String::from(
        "| prompt | dtype | kv_cache | batch | decode tok/s | peak MB |\n|---|---|---|---:|---:|---:|\n",
    );
    for (first, group) in groups {
        let mean =
            |f: fn(&Row) -> f64| group.iter().map(|r| f(r)).sum::<f64>() / group.len() as f64;
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.1} | {:.1} |\n",
            first.prompt_file,
            first.dtype,
            first.kv_cache,
            first.batch,
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
             ttft_ms,decode_ms_mean_last64,peak_mem_mb,quality,tag\n\
             abc1234-dirty,2026-11-02,fornace,qwen2.5-0.5b-instruct,metal,bf16,\
             prealloc,1,alice-512.txt,512,128,2,250.5,4000.0,115.25,128.0,8.5,\
             1234.5,,after-4077\n"
        );
    }

    #[test]
    fn row_round_trips_through_csv_without_header() {
        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(Vec::new());
        let mut empty = row();
        empty.decode_ms_mean_last64 = None;
        writer.serialize(row()).unwrap();
        writer.serialize(&empty).unwrap();
        let bytes = writer.into_inner().unwrap();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(bytes.as_slice());
        let rows: Vec<Row> = reader.deserialize().map(|r| r.unwrap()).collect();
        assert_eq!(rows, vec![row(), empty]);
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
        let table = summary(&[row(), second, other, batched]);
        assert_eq!(
            table,
            "| prompt | dtype | kv_cache | batch | decode tok/s | peak MB |\n\
             |---|---|---|---:|---:|---:|\n\
             | alice-512.txt | bf16 | prealloc | 1 | 120.1 | 1235.5 |\n\
             | alice-512.txt | bf16 | cat | 1 | 58.0 | 1234.5 |\n\
             | alice-512.txt | bf16 | prealloc | 4 | 400.0 | 1234.5 |\n"
        );
    }
}
