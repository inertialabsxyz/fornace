//! Terminal charts of a `GenerationReport`.

use fornace_engine::GenerationReport;
use textplots::{Chart, Plot, Shape};

/// The maximum number of rows in the table under the chart.
const TABLE_ROWS: usize = 16;

/// Draws the decode time of each token against its position, then a table
/// of the mean time for each block of positions.
pub fn decode_times(report: &GenerationReport) {
    let times = &report.decode_token_ms;
    if times.is_empty() {
        println!("No decode steps to plot.");
        return;
    }
    // decode_token_ms[i] is the token at position prompt_tokens + 1 + i.
    let first_pos = report.prompt_tokens + 1;
    let points: Vec<(f32, f32)> = times
        .iter()
        .enumerate()
        .map(|(i, &ms)| ((first_pos + i) as f32, ms as f32))
        .collect();
    let last_pos = (first_pos + times.len() - 1) as f32;

    println!("Decode time per token (ms) by position");
    Chart::new(
        160,
        60,
        first_pos as f32,
        last_pos.max(first_pos as f32 + 1.0),
    )
    .lineplot(&Shape::Lines(&points))
    .display();

    println!();
    println!("{:>15}  {:>9}", "positions", "mean ms");
    for (start, end, mean) in block_means(times, first_pos, TABLE_ROWS) {
        println!("{:>15}  {mean:>9.2}", format!("{start}-{end}"));
    }
}

/// Splits `times` into at most `rows` equal blocks. Returns the first and
/// last position and the mean time of each block.
fn block_means(times: &[f64], first_pos: usize, rows: usize) -> Vec<(usize, usize, f64)> {
    let size = times.len().div_ceil(rows).max(1);
    times
        .chunks(size)
        .enumerate()
        .map(|(i, block)| {
            let start = first_pos + i * size;
            let mean = block.iter().sum::<f64>() / block.len() as f64;
            (start, start + block.len() - 1, mean)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_means_covers_every_position_once() {
        let times = [1.0, 3.0, 5.0, 7.0, 9.0];
        let blocks = block_means(&times, 10, 2);
        assert_eq!(blocks, vec![(10, 12, 3.0), (13, 14, 8.0)]);
    }

    #[test]
    fn block_means_uses_one_position_per_row_for_short_runs() {
        let blocks = block_means(&[2.0, 4.0], 6, 16);
        assert_eq!(blocks, vec![(6, 6, 2.0), (7, 7, 4.0)]);
    }
}
