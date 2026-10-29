//! The `fornace-bench` command: run each engine and write one CSV row per run.

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// The directory that contains the model files.
    #[arg(long, value_name = "DIR")]
    model: Option<PathBuf>,
    /// The engines to measure, separated by commas.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "fornace",
        value_parser = ["fornace", "mistralrs", "llamacpp"]
    )]
    engines: Vec<String>,
    /// The CSV file for the results.
    #[arg(long, value_name = "CSV", default_value = "bench/runs/latest.csv")]
    out: PathBuf,
}

fn main() {
    let _args = Args::parse();
    println!("fornace-bench: the harness lands in the next commit.");
}
