//! `fornace-receipt key`: prints the public key of the Secure Enclave key,
//! and makes the key first if it does not exist.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(about = "Signed receipts for Fornace generations")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print the public key of the Secure Enclave key (uncompressed SEC1,
    /// hex). Make the key first if the key file does not exist.
    Key {
        /// The key file. The default is
        /// `~/Library/Application Support/fornace/receipt-key.se`.
        #[arg(long)]
        key: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Key { key } => key_command(key.unwrap_or_else(fornace_receipt::default_key_path)),
    }
}

#[cfg(target_os = "macos")]
fn key_command(path: PathBuf) -> ExitCode {
    use fornace_receipt::{EnclaveSigner, Signer};
    let existed = path.exists();
    match EnclaveSigner::open_or_create_file(&path) {
        Ok(signer) => {
            let hex: String = signer
                .public_key()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            eprintln!(
                "{} key file {}",
                if existed { "opened" } else { "made" },
                path.display()
            );
            println!("{hex}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn key_command(_path: PathBuf) -> ExitCode {
    eprintln!("error: the Secure Enclave signer is only on macOS");
    ExitCode::FAILURE
}
