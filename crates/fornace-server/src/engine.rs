//! The engine thread: the scheduler. It owns the [`Engine`] and one
//! [`Batch`]. Before each decode step it moves at most one job from the
//! queue into the batch (a prefill alone), if a row is free. Then one step
//! runs for all rows. A row leaves when it ends, and the next job can take
//! its place. The channel of jobs is the queue.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};

use fornace_engine::batch::{Batch, Done, Finish, RowId};
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

/// The output side of one row: its channel, its stop filter, and the text
/// of its events so far.
struct RowOut {
    out: UnboundedSender<Out>,
    filter: StopFilter,
    events: String,
    /// A stop string was found.
    stopped: bool,
}

/// Runs the jobs until all senders are dropped, with at most `max_batch`
/// rows in one step. `queued` counts the jobs that wait.
pub fn run(mut engine: Engine, jobs: Receiver<Job>, queued: Arc<AtomicUsize>, max_batch: usize) {
    let mut batch = Batch::new();
    let mut rows: HashMap<RowId, Rc<RefCell<RowOut>>> = HashMap::new();
    loop {
        if batch.len() < max_batch {
            // With no row, wait for a job. With rows, take a job only if one
            // waits, so the rows do not stop.
            let job = if batch.is_empty() {
                match jobs.recv() {
                    Ok(job) => Some(job),
                    Err(_) => return,
                }
            } else {
                match jobs.try_recv() {
                    Ok(job) => Some(job),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) if batch.is_empty() => return,
                    Err(TryRecvError::Disconnected) => None,
                }
            };
            if let Some(job) = job {
                let waiting = queued.fetch_sub(1, Ordering::SeqCst) - 1;
                // The client closed the connection while the job waited.
                if !job.out.is_closed() {
                    join(&mut engine, &mut batch, &mut rows, job);
                    eprintln!("join: {} rows, {waiting} waiting", batch.len());
                }
            }
        }
        if batch.is_empty() {
            continue;
        }
        match batch.step(&mut engine) {
            Ok(done) => {
                for d in done {
                    finish(&mut rows, d);
                }
            }
            Err(e) => {
                // A failed step ends every row.
                let message = format!("{:#}", anyhow::Error::from(e));
                for (_, row) in rows.drain() {
                    let _ = row.borrow().out.send(Out::Error(message.clone()));
                }
                batch = Batch::new();
            }
        }
    }
}

fn join(
    engine: &mut Engine,
    batch: &mut Batch,
    rows: &mut HashMap<RowId, Rc<RefCell<RowOut>>>,
    job: Job,
) {
    let row = Rc::new(RefCell::new(RowOut {
        out: job.out.clone(),
        filter: StopFilter::new(job.stop),
        events: String::new(),
        stopped: false,
    }));
    let inner = row.clone();
    let on_token = Box::new(move |event: fornace_engine::TokenEvent| {
        let mut r = inner.borrow_mut();
        r.events.push_str(&event.text);
        let (text, found) = r.filter.push(&event.text);
        if !text.is_empty() && r.out.send(Out::Text(text)).is_err() {
            return ControlFlow::Break(());
        }
        r.stopped = found;
        // A closed stream stops its row at this token.
        if found || r.out.is_closed() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    match batch.join(engine, &job.prompt, &job.options, on_token) {
        Ok((id, left)) => {
            rows.insert(id, row);
            if let Some(d) = left {
                finish(rows, d);
            }
        }
        Err(e) => {
            let _ = job
                .out
                .send(Out::Error(format!("{:#}", anyhow::Error::from(e))));
        }
    }
}

/// Sends the rest of a row that left, and its end.
fn finish(rows: &mut HashMap<RowId, Rc<RefCell<RowOut>>>, d: Done) {
    let Some(row) = rows.remove(&d.id) else {
        return;
    };
    let mut r = row.borrow_mut();
    if !r.stopped {
        // A last token that ends inside a character is in the text as
        // U+FFFD, and in no event. Send it, so that a stream gives the same
        // text as a response without a stream.
        let rest = d
            .text
            .strip_prefix(r.events.as_str())
            .unwrap_or("")
            .to_string();
        let (mut text, found) = r.filter.push(&rest);
        r.stopped = found;
        text.push_str(&r.filter.finish());
        if !text.is_empty() {
            let _ = r.out.send(Out::Text(text));
        }
    }
    let finish = match d.finish {
        Finish::Length if !r.stopped => "length",
        _ => "stop",
    };
    let _ = r.out.send(Out::Done {
        prompt_tokens: d.prompt_tokens,
        completion_tokens: d.tokens.len(),
        finish,
    });
}
