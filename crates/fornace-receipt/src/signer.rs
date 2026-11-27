//! The signers, and the COSE_Sign1 message that holds a signed receipt.

use coset::{CoseSign1Builder, HeaderBuilder, TaggedCborSerializable, iana};
use p256::ecdsa::SigningKey;
use p256::ecdsa::signature::Signer as _;

use crate::{PublicKey, Receipt};

/// An ES256 signature: ECDSA on P-256 over SHA-256 of the message, as the
/// 32 bytes of `r` and the 32 bytes of `s`.
pub type Signature = [u8; 64];

#[derive(Debug, thiserror::Error)]
#[error("signing failed: {0}")]
pub struct SignError(pub String);

/// A P-256 key that signs with ES256.
pub trait Signer {
    /// The public key, as an uncompressed SEC1 point (65 bytes).
    fn public_key(&self) -> PublicKey;
    /// Signs SHA-256 of `message`.
    fn sign(&self, message: &[u8]) -> Result<Signature, SignError>;
}

/// A P-256 key in process memory. The tests use it, and so does Linux.
pub struct SoftwareSigner {
    key: SigningKey,
}

impl SoftwareSigner {
    /// A new key from the random source of the operating system.
    pub fn generate() -> Self {
        loop {
            let mut secret = [0u8; 32];
            getrandom::fill(&mut secret).expect("the OS random source is available");
            // A value of 0, or of the group order or more, is not a key.
            // The chance is below 2^-32.
            if let Ok(key) = SigningKey::from_slice(&secret) {
                return SoftwareSigner { key };
            }
        }
    }

    /// The key with the 32-byte secret scalar `secret`.
    pub fn from_secret(secret: &[u8; 32]) -> Option<Self> {
        SigningKey::from_slice(secret)
            .ok()
            .map(|key| SoftwareSigner { key })
    }

    /// The 32-byte secret scalar.
    pub fn secret(&self) -> [u8; 32] {
        self.key.to_bytes().into()
    }
}

impl Signer for SoftwareSigner {
    fn public_key(&self) -> PublicKey {
        let point = self.key.verifying_key().to_sec1_bytes();
        (*point)
            .try_into()
            .expect("an uncompressed P-256 point is 65 bytes")
    }

    fn sign(&self, message: &[u8]) -> Result<Signature, SignError> {
        let signature: p256::ecdsa::Signature = self.key.sign(message);
        Ok(signature.to_bytes().into())
    }
}

/// The protected header of every receipt: `alg` ES256 and nothing else.
pub(crate) fn protected_header() -> coset::Header {
    HeaderBuilder::new()
        .algorithm(iana::Algorithm::ES256)
        .build()
}

/// Signs `receipt` with `signer`. Returns a tagged COSE_Sign1 message
/// (RFC 9052) with the receipt as its payload.
pub fn sign(receipt: &Receipt, signer: &dyn Signer) -> Result<Vec<u8>, SignError> {
    sign_payload(receipt.encode(&signer.public_key()), signer)
}

pub(crate) fn sign_payload(payload: Vec<u8>, signer: &dyn Signer) -> Result<Vec<u8>, SignError> {
    let message = CoseSign1Builder::new()
        .protected(protected_header())
        .payload(payload)
        .try_create_signature(b"", |data| signer.sign(data).map(|s| s.to_vec()))?
        .build();
    message
        .to_tagged_vec()
        .map_err(|e| SignError(e.to_string()))
}
