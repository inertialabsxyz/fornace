//! An OpenAI-compatible HTTP server for one Fornace engine.
//!
//! The HTTP side runs on tokio. The engine runs on its own thread, and
//! takes one request at a time from a queue (see [`engine`]). The server
//! holds no inference state.

pub mod api;
pub mod engine;

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use fornace_engine::{Engine, GenerateOptions};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokenizers::Tokenizer;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::api::{ApiError, Kind, Shape, Usage};
use crate::engine::{Job, Out};

/// The settings of the server.
#[derive(Debug, Clone)]
pub struct Config {
    /// The model id in the responses and in `/v1/models`.
    pub model_name: String,
    /// The most prompt and new tokens for one request.
    pub max_context: usize,
    /// The repeat penalty of a request that does not give one.
    pub repeat_penalty: f32,
}

struct Inner {
    config: Config,
    /// Counts the prompt tokens for the context check, before the queue.
    tokenizer: Tokenizer,
    jobs: mpsc::Sender<Job>,
    ready: Arc<AtomicBool>,
    queued: Arc<AtomicUsize>,
    next_id: AtomicU64,
}

type AppState = Arc<Inner>;

/// Starts the engine thread, which runs `load` and then the jobs, and
/// returns the routes. Requests that arrive before the load wait in the
/// queue. If `load` fails, the process stops, because the server cannot do
/// anything without the engine.
pub fn app(
    config: Config,
    tokenizer: Tokenizer,
    load: impl FnOnce() -> anyhow::Result<Engine> + Send + 'static,
) -> Router {
    let (jobs, rx) = mpsc::channel();
    let ready = Arc::new(AtomicBool::new(false));
    let queued = Arc::new(AtomicUsize::new(0));
    let (thread_ready, thread_queued) = (ready.clone(), queued.clone());
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || match load() {
            Ok(engine) => {
                eprintln!("engine loaded in {:.0} ms", engine.load_ms);
                thread_ready.store(true, Ordering::SeqCst);
                engine::run(engine, rx, thread_queued);
            }
            Err(e) => {
                eprintln!("error: cannot load the engine: {e:#}");
                std::process::exit(1);
            }
        })
        .expect("start the engine thread");
    let state = Arc::new(Inner {
        config,
        tokenizer,
        jobs,
        ready,
        queued,
        next_id: AtomicU64::new(0),
    });
    Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .with_state(state)
}

async fn health(State(s): State<AppState>) -> Response {
    if s.ready.load(Ordering::SeqCst) {
        Json(json!({"status": "ok"})).into_response()
    } else {
        let body = Json(json!({"status": "loading"}));
        (StatusCode::SERVICE_UNAVAILABLE, body).into_response()
    }
}

async fn models(State(s): State<AppState>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{
        "id": s.config.model_name, "object": "model", "created": 0, "owned_by": "fornace",
    }]}))
}

async fn completions(State(s): State<AppState>, Json(req): Json<api::Request>) -> Response {
    handle(s, Kind::Completion, req).await
}

async fn chat(State(s): State<AppState>, Json(req): Json<api::Request>) -> Response {
    handle(s, Kind::Chat, req).await
}

fn bad_request(e: ApiError) -> Response {
    (StatusCode::BAD_REQUEST, Json(e.body())).into_response()
}

fn server_error(message: &str) -> Response {
    let body = json!({"error": {"message": message, "type": "server_error"}});
    (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
}

async fn handle(s: AppState, kind: Kind, req: api::Request) -> Response {
    let params = match api::validate(&req, kind) {
        Ok(p) => p,
        Err(e) => return bad_request(e),
    };
    let prompt_tokens = match s.tokenizer.encode(params.prompt.as_str(), true) {
        Ok(enc) => enc.len(),
        Err(e) => return server_error(&format!("cannot tokenize the prompt: {e}")),
    };
    let max_new_tokens =
        match api::token_budget(prompt_tokens, params.max_tokens, s.config.max_context) {
            Ok(n) => n,
            Err(e) => return bad_request(e),
        };
    let (out, rx) = tokio::sync::mpsc::unbounded_channel();
    let job = Job {
        prompt: params.prompt,
        options: GenerateOptions {
            max_new_tokens,
            ignore_eos: params.ignore_eos,
            repeat_penalty: params.repeat_penalty.unwrap_or(s.config.repeat_penalty),
        },
        stop: params.stop,
        out,
    };
    let queued = s.queued.fetch_add(1, Ordering::SeqCst) + 1;
    eprintln!("new job: {queued} in the queue, {prompt_tokens} prompt tokens");
    if s.jobs.send(job).is_err() {
        return server_error("the engine thread stopped");
    }
    let shape = Shape {
        kind,
        id: format!("cmpl-{}", s.next_id.fetch_add(1, Ordering::SeqCst)),
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        model: s.config.model_name.clone(),
    };
    if params.stream {
        stream(shape, rx)
    } else {
        full(shape, rx).await
    }
}

fn data(v: &Value) -> Result<Event, Infallible> {
    Ok(Event::default().data(v.to_string()))
}

/// SSE: a role chunk for chat, one chunk for each new text, the last chunk
/// with the finish reason and the usage, then `[DONE]`. If the client
/// closes the stream, the receiver is dropped and the engine stops.
fn stream(shape: Shape, rx: tokio::sync::mpsc::UnboundedReceiver<Out>) -> Response {
    let head = (shape.kind == Kind::Chat).then(|| data(&shape.role_chunk()));
    let body = UnboundedReceiverStream::new(rx).flat_map(move |out| {
        let events = match out {
            Out::Text(text) => vec![data(&shape.chunk(&text))],
            Out::Done {
                prompt_tokens,
                completion_tokens,
                finish,
            } => {
                let usage = Usage {
                    prompt_tokens,
                    completion_tokens,
                };
                vec![
                    data(&shape.last_chunk(finish, usage)),
                    Ok(Event::default().data("[DONE]")),
                ]
            }
            Out::Error(message) => {
                vec![data(
                    &json!({"error": {"message": message, "type": "server_error"}}),
                )]
            }
        };
        futures_util::stream::iter(events)
    });
    Sse::new(futures_util::stream::iter(head).chain(body)).into_response()
}

async fn full(shape: Shape, mut rx: tokio::sync::mpsc::UnboundedReceiver<Out>) -> Response {
    let mut text = String::new();
    while let Some(out) = rx.recv().await {
        match out {
            Out::Text(t) => text.push_str(&t),
            Out::Done {
                prompt_tokens,
                completion_tokens,
                finish,
            } => {
                let usage = Usage {
                    prompt_tokens,
                    completion_tokens,
                };
                return Json(shape.full(&text, finish, usage)).into_response();
            }
            Out::Error(message) => return server_error(&message),
        }
    }
    server_error("the engine dropped the request")
}
