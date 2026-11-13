//! The engine thread. It owns the [`Engine`] and runs one job at a time, in
//! the order the jobs arrive. The channel of jobs is the queue.

use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;

use fornace_engine::{Engine, GenerateOptions};
use tokio::sync::mpsc::UnboundedSender;

use crate::api::StopFilter;

/// One request for the engine thread.
pub struct Job {
    pub prompt: String,
    pub options: GenerateOptions,
    pub stop: Vec<String>,
    pub out: UnboundedSender<Out>,
}

/// What the engine thread sends back for a job.
#[derive(Debug)]
pub enum Out {
    /// New text, up to the last complete character.
    Text(String),
    /// The end of the job.
    Done {
        prompt_tokens: usize,
        completion_tokens: usize,
        finish: &'static str,
    },
    /// The engine failed.
    Error(String),
}

/// Runs the jobs until all senders are dropped. `queued` counts the jobs
/// that wait.
pub fn run(mut engine: Engine, jobs: Receiver<Job>, queued: Arc<AtomicUsize>) {
    for job in jobs {
        queued.fetch_sub(1, Ordering::SeqCst);
        // The client closed the connection while the job waited.
        if job.out.is_closed() {
            continue;
        }
        run_job(&mut engine, job);
    }
}

fn run_job(engine: &mut Engine, job: Job) {
    let mut filter = StopFilter::new(job.stop);
    let mut events = String::new();
    let mut stopped = false;
    let out = &job.out;
    let mut on_token = |event: fornace_engine::TokenEvent| {
        events.push_str(&event.text);
        let (text, found) = filter.push(&event.text);
        if !text.is_empty() && out.send(Out::Text(text)).is_err() {
            return ControlFlow::Break(());
        }
        stopped = found;
        // A closed stream stops the engine at this token.
        if found || out.is_closed() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    };
    let result = engine.generate_stream(&job.prompt, &job.options, &mut on_token);
    let report = match result {
        Ok(report) => report,
        Err(e) => {
            let _ = out.send(Out::Error(format!("{:#}", anyhow::Error::from(e))));
            return;
        }
    };
    if !stopped {
        // A last token that ends inside a character is in the report as
        // U+FFFD, and in no event. Send it, so that a stream gives the same
        // text as a response without a stream.
        let rest = report.texts[0].strip_prefix(&events).unwrap_or("");
        let (mut text, found) = filter.push(rest);
        stopped = found;
        text.push_str(&filter.finish());
        if !text.is_empty() {
            let _ = out.send(Out::Text(text));
        }
    }
    let finish = if stopped || report.generated_tokens < job.options.max_new_tokens {
        "stop"
    } else {
        "length"
    };
    let _ = out.send(Out::Done {
        prompt_tokens: report.prompt_tokens,
        completion_tokens: report.generated_tokens,
        finish,
    });
}
