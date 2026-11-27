//! A signer with a P-256 key in the Secure Enclave of a Mac, through
//! CryptoKit (`swift/enclave.swift`).
//!
//! The key file holds an encrypted blob, not the key. Only the Secure
//! Enclave of the Mac that made the blob can use it. But any process of the
//! user who can read the file can ask the Secure Enclave to sign with it.
//! So the key names the Mac, not the program.

use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use crate::PublicKey;
use crate::signer::{SignError, Signature, Signer};

const BLOB_CAP: usize = 1024;
const ERR_CAP: usize = 512;

unsafe extern "C" {
    fn fornace_se_create(
        blob: *mut u8,
        blob_cap: usize,
        blob_len: *mut usize,
        public_key: *mut u8,
        err: *mut u8,
        err_cap: usize,
        err_len: *mut usize,
    ) -> i32;
    fn fornace_se_public_key(
        blob: *const u8,
        blob_len: usize,
        public_key: *mut u8,
        err: *mut u8,
        err_cap: usize,
        err_len: *mut usize,
    ) -> i32;
    fn fornace_se_sign(
        blob: *const u8,
        blob_len: usize,
        message: *const u8,
        message_len: usize,
        signature: *mut u8,
        err: *mut u8,
        err_cap: usize,
        err_len: *mut usize,
    ) -> i32;
}

#[derive(Debug, thiserror::Error)]
pub enum EnclaveError {
    #[error("Secure Enclave error {code}: {message}")]
    Enclave { code: i32, message: String },
    #[error("key file {path}: {source}")]
    File {
        path: PathBuf,
        source: std::io::Error,
    },
}

/// Calls `f` with an error buffer. Returns the error if `f` gives a code
/// other than 0.
fn call(f: impl FnOnce(*mut u8, usize, *mut usize) -> i32) -> Result<(), EnclaveError> {
    let mut err = [0u8; ERR_CAP];
    let mut err_len = 0usize;
    match f(err.as_mut_ptr(), ERR_CAP, &mut err_len) {
        0 => Ok(()),
        code => Err(EnclaveError::Enclave {
            code,
            message: String::from_utf8_lossy(&err[..err_len]).into_owned(),
        }),
    }
}

/// A key in the Secure Enclave.
pub struct EnclaveSigner {
    blob: Vec<u8>,
    public_key: PublicKey,
}

impl EnclaveSigner {
    /// Makes a new key in the Secure Enclave. The key exists only while the
    /// blob exists: see [`EnclaveSigner::create_file`].
    pub fn create() -> Result<Self, EnclaveError> {
        let mut blob = vec![0u8; BLOB_CAP];
        let mut blob_len = 0usize;
        let mut public_key = [0u8; 65];
        // SAFETY: each buffer is valid for the length that is given with it.
        call(|err, cap, len| unsafe {
            fornace_se_create(
                blob.as_mut_ptr(),
                BLOB_CAP,
                &mut blob_len,
                public_key.as_mut_ptr(),
                err,
                cap,
                len,
            )
        })?;
        blob.truncate(blob_len);
        Ok(EnclaveSigner { blob, public_key })
    }

    /// The key of a blob from [`EnclaveSigner::blob`].
    pub fn from_blob(blob: Vec<u8>) -> Result<Self, EnclaveError> {
        let mut public_key = [0u8; 65];
        // SAFETY: each buffer is valid for the length that is given with it.
        call(|err, cap, len| unsafe {
            fornace_se_public_key(
                blob.as_ptr(),
                blob.len(),
                public_key.as_mut_ptr(),
                err,
                cap,
                len,
            )
        })?;
        Ok(EnclaveSigner { blob, public_key })
    }

    /// The encrypted key blob.
    pub fn blob(&self) -> &[u8] {
        &self.blob
    }

    /// Makes a new key and writes its blob to `path`, readable only by the
    /// user. Fails if `path` exists, so that it cannot replace a key.
    pub fn create_file(path: &Path) -> Result<Self, EnclaveError> {
        let file_err = |source| EnclaveError::File {
            path: path.to_path_buf(),
            source,
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(file_err)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .map_err(file_err)?;
        let signer = EnclaveSigner::create()?;
        file.write_all(&signer.blob).map_err(file_err)?;
        Ok(signer)
    }

    /// The key whose blob is in `path`.
    pub fn open_file(path: &Path) -> Result<Self, EnclaveError> {
        let blob = std::fs::read(path).map_err(|source| EnclaveError::File {
            path: path.to_path_buf(),
            source,
        })?;
        EnclaveSigner::from_blob(blob)
    }

    /// The key in `path`, or a new key in `path` if the file does not exist.
    pub fn open_or_create_file(path: &Path) -> Result<Self, EnclaveError> {
        if path.exists() {
            EnclaveSigner::open_file(path)
        } else {
            EnclaveSigner::create_file(path)
        }
    }
}

impl Signer for EnclaveSigner {
    fn public_key(&self) -> PublicKey {
        self.public_key
    }

    fn sign(&self, message: &[u8]) -> Result<Signature, SignError> {
        let mut signature = [0u8; 64];
        // SAFETY: each buffer is valid for the length that is given with it.
        call(|err, cap, len| unsafe {
            fornace_se_sign(
                self.blob.as_ptr(),
                self.blob.len(),
                message.as_ptr(),
                message.len(),
                signature.as_mut_ptr(),
                err,
                cap,
                len,
            )
        })
        .map_err(|e| SignError(e.to_string()))?;
        Ok(signature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blob_that_is_not_a_key_gives_an_error_with_a_message() {
        let Err(EnclaveError::Enclave { code, message }) = EnclaveSigner::from_blob(vec![0; 16])
        else {
            panic!("a 16-byte blob of zeros is not a key");
        };
        assert_eq!(code, 3);
        assert!(!message.is_empty());
    }
}
