//! A signed receipt for one Fornace generation, and its verifier.
//!
//! A receipt binds a model, an engine binary, the settings, the prompt
//! tokens, and the output tokens to one P-256 device key. It is a
//! COSE_Sign1 message (RFC 9052) with ES256. Its payload is a CBOR map in
//! deterministic encoding. `FORMAT.md` gives each field.
//!
//! A valid signature shows that the holder of the device key signed the
//! claims. It does not show where the model ran.

mod cbor;
#[cfg(target_os = "macos")]
pub mod enclave;
pub mod manifest;
pub mod receipt;
pub mod signer;
pub mod verify;

#[cfg(target_os = "macos")]
pub use enclave::{EnclaveError, EnclaveSigner};
pub use manifest::{Manifest, ManifestFile};
pub use receipt::{NONCE_LEN, PROFILE, Receipt, Settings, request_hash, response_hash};
pub use signer::{SignError, Signature, Signer, SoftwareSigner, sign};
pub use verify::{Verified, VerifyError, verify};

/// The default key file of the Secure Enclave signer:
/// `~/Library/Application Support/fornace/receipt-key.se`. It holds an
/// encrypted key blob, not the key. Do not put it in a repository.
pub fn default_key_path() -> std::path::PathBuf {
    let home = std::env::var_os("HOME").unwrap_or_default();
    std::path::PathBuf::from(home).join("Library/Application Support/fornace/receipt-key.se")
}

/// A SHA-256 digest.
pub type Hash = [u8; 32];

/// A P-256 public key as an uncompressed SEC1 point: `0x04`, then `x` and
/// `y`, 32 bytes each.
pub type PublicKey = [u8; 65];
