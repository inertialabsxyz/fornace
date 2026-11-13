//! The servers of the three engines: the command of each, the start, the
//! wait until it is ready, the peak memory, and the stop.

use std::fs::File;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use fornace_engine::{AttentionKind, DeviceKind, Dtype};
use serde_json::{Value, json};

/// The time that a server has to load.
pub const READY_TIMEOUT: Duration = Duration::from_secs(120);

/// An engine behind an HTTP server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerEngine {
    Fornace,
    LlamaCpp,
    MistralRs,
}

impl ServerEngine {
    /// The name in `--engines` and in the `engine` column.
    pub fn parse(name: &str) -> Option<ServerEngine> {
        match name {
            "fornace" => Some(ServerEngine::Fornace),
            "llamacpp" => Some(ServerEngine::LlamaCpp),
            "mistralrs" => Some(ServerEngine::MistralRs),
            _ => None,
        }
    }
}

/// The settings of one server.
#[derive(Debug, Clone)]
pub struct ServerSpec {
    pub engine: ServerEngine,
    pub dtype: Dtype,
    pub model_dir: PathBuf,
    pub port: u16,
    /// The context positions of each slot.
    pub context: usize,
    /// The requests that the server runs at the same time (llama.cpp
    /// `-np`, mistral.rs `--max-seqs`). Fornace has a queue, so it ignores
    /// this.
    pub parallel: usize,
    /// Fornace only.
    pub attention: AttentionKind,
    /// Fornace only.
    pub device: DeviceKind,
    /// llama.cpp only: keep its prompt caches on.
    pub cache_prompt: bool,
}

/// The weight file of a reference engine for `dtype`, from
/// `scripts/make-reference-weights.sh`.
fn llama_file(dtype: Dtype) -> Result<&'static str, String> {
    match dtype {
        Dtype::F16 | Dtype::Bf16 => Ok("model-bf16.llama.gguf"),
        Dtype::Q8 => Ok("model-q8_0.llama.gguf"),
        Dtype::Q4 => Ok("model-q4_0.llama.gguf"),
        Dtype::F32 => Err("the reference engines run f16, q8, and q4 only".into()),
    }
}

/// The directory of the reference engines, as in
/// `scripts/get-reference-engines.sh`.
fn ref_dir() -> PathBuf {
    std::env::var_os("FORNACE_REF_DIR").map_or_else(
        || PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/fornace/ref"),
        PathBuf::from,
    )
}

fn mistralrs_bin() -> PathBuf {
    let tag = std::env::var("MISTRALRS_TAG").unwrap_or_else(|_| "v0.9.5".into());
    ref_dir().join(format!("mistralrs-{tag}/bin/mistralrs"))
}

/// `fornace-server` next to this program. Build it with the same profile
/// and features.
fn fornace_server_bin() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let bin = exe.with_file_name("fornace-server");
    if bin.exists() {
        Ok(bin)
    } else {
        Err(format!(
            "{} does not exist. Build it: cargo build --release -p fornace-server --features metal",
            bin.display()
        ))
    }
}

/// The command of each engine. This is the only place with the flags.
pub fn command(spec: &ServerSpec) -> Result<Command, String> {
    let port = spec.port.to_string();
    let model = &spec.model_dir;
    let mut cmd = match spec.engine {
        ServerEngine::Fornace => {
            let mut cmd = Command::new(fornace_server_bin()?);
            cmd.arg("--model").arg(model);
            cmd.args(["--dtype", &spec.dtype.to_string()]);
            cmd.args(["--attention", &spec.attention.to_string()]);
            cmd.args(["--device", &spec.device.to_string()]);
            cmd.args(["--max-context", &spec.context.to_string()]);
            cmd.args(["--port", &port]);
            cmd
        }
        ServerEngine::LlamaCpp => {
            let mut cmd = Command::new("llama-server");
            cmd.arg("-m").arg(model.join(llama_file(spec.dtype)?));
            // All layers on the GPU, flash attention on, and each slot gets
            // `context` positions.
            cmd.args(["-ngl", "99", "-fa", "on"]);
            // Read the weights into memory, not a map of the file. A mapped
            // file is clean file memory, and the physical footprint does not
            // count it. Fornace and mistral.rs hold their weights in the
            // footprint, so the memory column needs the same here.
            cmd.args(["-lm", "none"]);
            cmd.args(["-c", &(spec.context * spec.parallel).to_string()]);
            cmd.args(["-np", &spec.parallel.to_string()]);
            if !spec.cache_prompt {
                // Both prompt caches off: the reuse from the last request,
                // and the cache in host memory.
                cmd.args(["--no-cache-prompt", "-cram", "0"]);
            }
            cmd.args(["--host", "127.0.0.1", "--port", &port]);
            cmd
        }
        ServerEngine::MistralRs => {
            let mut cmd = Command::new(mistralrs_bin());
            cmd.arg("serve");
            cmd.args(["-p", &port, "--host", "127.0.0.1", "--no-ui"]);
            cmd.args(["--prefix-cache-n", "0"]);
            cmd.args(["--max-seqs", &spec.parallel.to_string()]);
            // The run-time compile of its paged attention kernels fails
            // without the Metal toolchain.
            cmd.args(["--paged-attn", "off"]);
            if spec.dtype != Dtype::F16 && spec.dtype != Dtype::Bf16 {
                // GGUF with BF16 activations fails. F32 activations, as in
                // Fornace with quantized weights.
                cmd.args(["--dtype", "f32"]);
                cmd.arg("-m").arg(model);
                cmd.args(["-f", llama_file(spec.dtype)?]);
            } else {
                cmd.arg("-m").arg(model);
            }
            cmd
        }
    };
    cmd.stdin(Stdio::null());
    Ok(cmd)
}

/// The `engine_version` column. `commit` is the Fornace commit.
pub fn version(engine: ServerEngine, commit: &str) -> String {
    let output = |mut cmd: Command| {
        cmd.output()
            .map(|o| {
                String::from_utf8_lossy(&o.stdout).into_owned()
                    + &String::from_utf8_lossy(&o.stderr)
            })
            .unwrap_or_default()
    };
    match engine {
        ServerEngine::Fornace => format!("fornace@{commit}"),
        ServerEngine::LlamaCpp => {
            let mut cmd = Command::new("llama-server");
            cmd.arg("--version");
            let text = output(cmd);
            let build = text
                .split("build ")
                .nth(1)
                .and_then(|s| s.split(|c: char| !c.is_ascii_digit()).next())
                .unwrap_or("unknown");
            format!("llama.cpp@b{build}")
        }
        ServerEngine::MistralRs => {
            let mut cmd = Command::new(mistralrs_bin());
            cmd.arg("--version");
            let text = output(cmd);
            let v = text.split_whitespace().nth(1).unwrap_or("unknown");
            format!("mistral.rs@v{v}")
        }
    }
}

/// The body of one completion request, with the field names of each
/// engine.
pub fn request_body(
    engine: ServerEngine,
    prompt: &str,
    max_tokens: usize,
    repeat_penalty: f32,
    ignore_eos: bool,
    cache_prompt: bool,
    stream: bool,
) -> Value {
    let mut body = json!({
        "model": if engine == ServerEngine::MistralRs { "default" } else { "fornace-bench" },
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0,
        "ignore_eos": ignore_eos,
    });
    match engine {
        ServerEngine::Fornace => body["repeat_penalty"] = json!(repeat_penalty),
        ServerEngine::LlamaCpp => {
            body["repeat_penalty"] = json!(repeat_penalty);
            body["cache_prompt"] = json!(cache_prompt);
        }
        ServerEngine::MistralRs => body["repetition_penalty"] = json!(repeat_penalty),
    }
    body
}

/// A running server. Dropping it stops the process.
pub struct Server {
    child: Child,
    pub base: String,
    /// From the start to the first 200 of the health route, in ms.
    pub load_ms: f64,
}

impl Server {
    /// Starts the server, writes its output to `log`, and waits until its
    /// health route gives 200.
    pub fn start(spec: &ServerSpec, log: &Path) -> Result<Server, String> {
        if TcpStream::connect(("127.0.0.1", spec.port)).is_ok() {
            return Err(format!(
                "port {} is in use. Stop that server first",
                spec.port
            ));
        }
        let mut cmd = command(spec)?;
        eprintln!("start: {cmd:?}");
        let file = File::create(log).map_err(|e| format!("create {}: {e}", log.display()))?;
        let err = file.try_clone().map_err(|e| e.to_string())?;
        cmd.stdout(file).stderr(err);
        let start = Instant::now();
        let child = cmd.spawn().map_err(|e| format!("start {cmd:?}: {e}"))?;
        let mut server = Server {
            child,
            base: format!("http://127.0.0.1:{}", spec.port),
            load_ms: 0.0,
        };
        let health = format!("{}/health", server.base);
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(2)))
            .build()
            .into();
        while start.elapsed() < READY_TIMEOUT {
            if let Ok(Some(status)) = server.child.try_wait() {
                return Err(format!(
                    "the server stopped ({status}). See {}",
                    log.display()
                ));
            }
            if agent.get(&health).call().is_ok() {
                server.load_ms = start.elapsed().as_secs_f64() * 1e3;
                return Ok(server);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        Err(format!(
            "the server is not ready after {} s. See {}",
            READY_TIMEOUT.as_secs(),
            log.display()
        ))
    }

    /// The peak memory of the server process since it started, in bytes.
    pub fn peak_memory_bytes(&self) -> Result<u64, String> {
        fornace_engine::metrics::peak_memory_bytes_of(self.child.id())
            .ok_or_else(|| format!("cannot read the memory of pid {}", self.child.id()))
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(engine: ServerEngine, dtype: Dtype) -> ServerSpec {
        ServerSpec {
            engine,
            dtype,
            model_dir: PathBuf::from("/m"),
            port: 8090,
            context: 4096,
            parallel: 8,
            attention: AttentionKind::Fused,
            device: DeviceKind::Metal,
            cache_prompt: false,
        }
    }

    fn args(cmd: &Command) -> Vec<String> {
        cmd.get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn llama_cpp_gets_all_slots_and_no_prompt_cache() {
        let cmd = command(&spec(ServerEngine::LlamaCpp, Dtype::Q4)).unwrap();
        assert_eq!(
            args(&cmd).join(" "),
            "-m /m/model-q4_0.llama.gguf -ngl 99 -fa on -lm none -c 32768 -np 8 \
             --no-cache-prompt -cram 0 --host 127.0.0.1 --port 8090"
        );
        let mut on = spec(ServerEngine::LlamaCpp, Dtype::F16);
        on.cache_prompt = true;
        let a = args(&command(&on).unwrap());
        assert!(a.contains(&"/m/model-bf16.llama.gguf".to_string()));
        assert!(!a.contains(&"--no-cache-prompt".to_string()));
    }

    #[test]
    fn mistral_rs_reads_gguf_with_f32_activations() {
        let a = args(&command(&spec(ServerEngine::MistralRs, Dtype::Q8)).unwrap()).join(" ");
        assert!(
            a.ends_with("--dtype f32 -m /m -f model-q8_0.llama.gguf"),
            "{a}"
        );
        assert!(
            a.contains("--prefix-cache-n 0 --max-seqs 8 --paged-attn off"),
            "{a}"
        );
        let a = args(&command(&spec(ServerEngine::MistralRs, Dtype::F16)).unwrap()).join(" ");
        assert!(a.ends_with("-m /m") && !a.contains("--dtype"), "{a}");
        assert!(command(&spec(ServerEngine::MistralRs, Dtype::F32)).is_err());
    }

    #[test]
    fn each_engine_gets_its_penalty_field() {
        let body = |e| request_body(e, "p", 8, 1.0, true, false, true);
        assert_eq!(body(ServerEngine::Fornace)["repeat_penalty"], 1.0);
        assert_eq!(body(ServerEngine::LlamaCpp)["cache_prompt"], false);
        let mistral = body(ServerEngine::MistralRs);
        assert_eq!(mistral["repetition_penalty"], 1.0);
        assert_eq!(mistral["model"], "default");
        assert!(mistral.get("repeat_penalty").is_none());
    }
}
