//! The `--trace summary` table: time and memory for each span path.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tracing::Subscriber;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

/// The span names from the root span down to this span.
type Path = Vec<&'static str>;

/// A `tracing` layer that collects the time and the memory change of each
/// span, grouped by span path.
///
/// "Δmem" is the process memory from `fornace_engine::metrics`. On Apple
/// silicon it includes Metal buffers, so "of which metal" is a part of
/// "Δmem", not an addition to it.
#[derive(Clone, Default)]
pub struct Summary {
    rows: Arc<Mutex<HashMap<Path, Row>>>,
}

#[derive(Default)]
struct Row {
    /// The order of first appearance. Parents appear before their children.
    order: usize,
    busy_ms: Vec<f64>,
    gpu_delta_bytes: Option<i64>,
    memory_delta_bytes: i64,
}

/// The state of one span between its creation and its close.
#[derive(Default)]
struct Open {
    gpu_delta_bytes: Option<i64>,
}

/// One entry on the stack of entered spans.
struct Entered {
    path: Path,
    at: Instant,
    memory_bytes: i64,
}

thread_local! {
    /// The spans that are entered on this thread, innermost last. The path
    /// of a span is the place where it is entered, not where it was created.
    /// Candle, for example, creates its `linear` spans at load time and
    /// enters them in each forward pass.
    static STACK: RefCell<Vec<Entered>> = const { RefCell::new(Vec::new()) };
}

struct GpuDelta<'a>(&'a mut Option<i64>);

impl Visit for GpuDelta<'_> {
    fn record_i64(&mut self, field: &Field, value: i64) {
        if field.name() == "gpu_delta_bytes" {
            *self.0 = Some(value);
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

fn memory_bytes() -> i64 {
    fornace_engine::metrics::memory_bytes() as i64
}

impl<S> Layer<S> for Summary
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let mut open = Open::default();
        attrs.record(&mut GpuDelta(&mut open.gpu_delta_bytes));
        span.extensions_mut().insert(open);
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        if let Some(open) = span.extensions_mut().get_mut::<Open>() {
            values.record(&mut GpuDelta(&mut open.gpu_delta_bytes));
        }
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        STACK.with_borrow_mut(|stack| {
            let mut path = stack.last().map_or_else(Vec::new, |e| e.path.clone());
            path.push(span.name());
            let mut rows = self.rows.lock().unwrap();
            let order = rows.len();
            rows.entry(path.clone()).or_insert_with(|| Row {
                order,
                ..Row::default()
            });
            stack.push(Entered {
                path,
                at: Instant::now(),
                memory_bytes: memory_bytes(),
            });
        });
    }

    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let Some(entered) = STACK.with_borrow_mut(Vec::pop) else {
            return;
        };
        let busy_ms = entered.at.elapsed().as_secs_f64() * 1e3;
        let memory_delta = memory_bytes() - entered.memory_bytes;
        // DeviceSpan records the GPU change just before it exits.
        let gpu_delta = span
            .extensions_mut()
            .get_mut::<Open>()
            .and_then(|open| open.gpu_delta_bytes.take());
        let mut rows = self.rows.lock().unwrap();
        if let Some(row) = rows.get_mut(&entered.path) {
            row.busy_ms.push(busy_ms);
            row.memory_delta_bytes += memory_delta;
            if let Some(delta) = gpu_delta {
                *row.gpu_delta_bytes.get_or_insert(0) += delta;
            }
        }
    }
}

impl Summary {
    /// Returns the table: one row for each span path, in tree order.
    pub fn render(&self) -> String {
        let rows = self.rows.lock().unwrap();
        let mut paths: Vec<&Path> = rows.keys().collect();
        paths.sort_by_key(|p| rows[*p].order);
        let total = |p: &[&'static str]| rows.get(p).map(|r| r.busy_ms.iter().sum::<f64>());

        let mut out = String::new();
        let _ = writeln!(
            out,
            "{:<28} {:>7} {:>10} {:>9} {:>8} {:>8} {:>9} {:>8} {:>9}",
            "span",
            "count",
            "total ms",
            "mean ms",
            "p50 ms",
            "p95 ms",
            "% parent",
            "Δmem MB",
            "metal MB"
        );
        for path in paths {
            let row = &rows[path];
            let mut ms = row.busy_ms.clone();
            if ms.is_empty() {
                continue;
            }
            ms.sort_by(f64::total_cmp);
            let sum: f64 = ms.iter().sum();
            let parent = match total(&path[..path.len() - 1]) {
                Some(p) if p > 0.0 => format!("{:.1}", 100.0 * sum / p),
                _ => "-".to_string(),
            };
            let gpu = row
                .gpu_delta_bytes
                .map_or("-".to_string(), |b| format!("{:.1}", b as f64 / 1e6));
            let name = format!("{}{}", "  ".repeat(path.len() - 1), path[path.len() - 1]);
            let _ = writeln!(
                out,
                "{name:<28} {:>7} {sum:>10.1} {:>9.2} {:>8.2} {:>8.2} {parent:>9} {:>8.1} {gpu:>9}",
                ms.len(),
                sum / ms.len() as f64,
                percentile(&ms, 0.50),
                percentile(&ms, 0.95),
                row.memory_delta_bytes as f64 / 1e6,
            );
        }
        out
    }
}

/// Returns the nearest-rank percentile `p` (0 to 1) of `sorted`, which must
/// be sorted and not empty.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = (p * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::layer::SubscriberExt;

    #[test]
    fn summary_puts_a_span_under_the_span_where_it_is_entered() {
        let summary = Summary::default();
        let subscriber = tracing_subscriber::registry().with(summary.clone());
        tracing::subscriber::with_default(subscriber, || {
            let linear = {
                let _load = tracing::info_span!("load").entered();
                tracing::info_span!("linear")
            };
            let _forward = tracing::info_span!("forward").entered();
            for _ in 0..2 {
                let _l = linear.enter();
            }
        });

        let table = summary.render();
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines.len(), 4, "{table}");
        assert!(lines[2].starts_with("forward"), "{table}");
        assert!(lines[3].starts_with("  linear "), "{table}");
        assert_eq!(lines[3].split_whitespace().nth(1), Some("2"));
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&sorted, 0.5), 5.0);
        assert_eq!(percentile(&sorted, 0.95), 10.0);
        assert_eq!(percentile(&[7.0], 0.5), 7.0);
    }

    #[test]
    fn summary_groups_spans_by_path_in_tree_order() {
        let summary = Summary::default();
        let subscriber = tracing_subscriber::registry().with(summary.clone());
        tracing::subscriber::with_default(subscriber, || {
            let _outer = tracing::info_span!("outer").entered();
            for _ in 0..3 {
                let _inner = tracing::info_span!("inner").entered();
            }
            let gpu = tracing::info_span!("gpu", gpu_delta_bytes = tracing::field::Empty).entered();
            gpu.record("gpu_delta_bytes", 2_000_000_i64);
        });

        let table = summary.render();
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines.len(), 4, "{table}");
        let cells =
            |line: &str| -> Vec<String> { line.split_whitespace().map(str::to_string).collect() };
        assert_eq!(cells(lines[1])[..2], ["outer", "1"]);
        assert!(lines[2].starts_with("  inner"), "{table}");
        assert_eq!(cells(lines[2])[1], "3");
        assert!(lines[3].starts_with("  gpu"), "{table}");
        assert_eq!(cells(lines[3])[8], "2.0");
        assert_eq!(cells(lines[1])[8], "-");
    }
}
