//! The first checks of a receipt: the encoding and the signature.

use coset::{CoseSign1, TaggedCborSerializable};
use p256::ecdsa::signature::Verifier as _;
use p256::ecdsa::{Signature, VerifyingKey};

use crate::manifest::MODEL_HASH_SCHEME;
use crate::receipt::PROFILE;
use crate::signer::protected_header;
use crate::{PublicKey, Receipt};

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum VerifyError {
    #[error("not a tagged COSE_Sign1 message")]
    Cose,
    #[error(
        "the protected header is not `alg` ES256 alone, or the unprotected header is not empty"
    )]
    Header,
    #[error("the message has no payload")]
    NoPayload,
    #[error("claim `{0}` is missing or has the wrong type")]
    Claim(&'static str),
    #[error("the profile is `{0}`, not the profile of this format")]
    Profile(String),
    #[error("the model hash scheme is `{0}`, not `{MODEL_HASH_SCHEME}`")]
    ModelHashScheme(String),
    #[error("the payload is not the deterministic encoding of its claims")]
    NotDeterministic,
    #[error("model_hash is not the hash of model_manifest")]
    ModelHash,
    #[error("device_key is not a P-256 public key")]
    DeviceKey,
    #[error("the signature is not valid for device_key")]
    Signature,
}

/// A receipt with a valid signature by `device_key`.
#[derive(Debug, Clone, PartialEq)]
pub struct Verified {
    pub receipt: Receipt,
    pub device_key: PublicKey,
}

/// Checks the encoding of a signed receipt and its signature.
///
/// The signature shows only that the holder of `device_key` signed these
/// claims. It does not show that the key is in a Secure Enclave, or that
/// the named model made the output.
pub fn verify(message: &[u8]) -> Result<Verified, VerifyError> {
    let sign1 = CoseSign1::from_tagged_slice(message).map_err(|_| VerifyError::Cose)?;
    if sign1.protected.header != protected_header() || !sign1.unprotected.is_empty() {
        return Err(VerifyError::Header);
    }
    let payload = sign1.payload.as_deref().ok_or(VerifyError::NoPayload)?;

    let decoded = Receipt::decode(payload).map_err(VerifyError::Claim)?;
    if decoded.profile != PROFILE {
        return Err(VerifyError::Profile(decoded.profile));
    }
    if decoded.model_hash_scheme != MODEL_HASH_SCHEME {
        return Err(VerifyError::ModelHashScheme(decoded.model_hash_scheme));
    }
    if decoded.model_hash != decoded.receipt.model.hash() {
        return Err(VerifyError::ModelHash);
    }
    // This also rejects unknown claims, duplicate keys, and keys out of order.
    if decoded.receipt.encode(&decoded.device_key) != payload {
        return Err(VerifyError::NotDeterministic);
    }

    let key =
        VerifyingKey::from_sec1_bytes(&decoded.device_key).map_err(|_| VerifyError::DeviceKey)?;
    sign1.verify_signature(b"", |signature, data| {
        let signature = Signature::from_slice(signature).map_err(|_| VerifyError::Signature)?;
        key.verify(data, &signature)
            .map_err(|_| VerifyError::Signature)
    })?;

    Ok(Verified {
        receipt: decoded.receipt,
        device_key: decoded.device_key,
    })
}

#[cfg(test)]
mod tests {
    use ciborium::Value;

    use super::*;
    use crate::receipt::tests::receipt;
    use crate::signer::{SoftwareSigner, sign, sign_payload};
    use crate::{Signer, cbor};

    fn signer() -> SoftwareSigner {
        SoftwareSigner::from_secret(&[0x42; 32]).unwrap()
    }

    /// The payload of `receipt()` as a CBOR value, for tests that sign a
    /// changed payload.
    fn payload_value() -> Value {
        ciborium::from_reader(receipt().encode(&signer().public_key()).as_slice()).unwrap()
    }

    fn set(value: &mut Value, key: &str, new: Value) {
        let entry = value
            .as_map_mut()
            .unwrap()
            .iter_mut()
            .find(|(k, _)| k.as_text() == Some(key))
            .unwrap();
        entry.1 = new;
    }

    #[test]
    fn a_signed_receipt_verifies_and_gives_back_its_claims() {
        let signer = signer();
        let verified = verify(&sign(&receipt(), &signer).unwrap()).unwrap();
        assert_eq!(verified.receipt, receipt());
        assert_eq!(verified.device_key, signer.public_key());
    }

    #[test]
    fn a_change_to_any_byte_of_the_message_makes_the_check_fail() {
        let message = sign(&receipt(), &signer()).unwrap();
        for i in 0..message.len() {
            for flip in [0x01, 0x80] {
                let mut changed = message.clone();
                changed[i] ^= flip;
                assert!(verify(&changed).is_err(), "byte {i}, flip {flip:#04x}");
            }
        }
    }

    #[test]
    fn a_signature_by_another_key_fails() {
        // The payload names the first key, and the second key signs it.
        let payload = receipt().encode(&signer().public_key());
        let other = SoftwareSigner::from_secret(&[0x43; 32]).unwrap();
        let message = sign_payload(payload, &other).unwrap();
        assert_eq!(verify(&message), Err(VerifyError::Signature));
    }

    #[test]
    fn a_payload_that_is_not_deterministic_fails_even_with_a_valid_signature() {
        let mut value = payload_value();
        value.as_map_mut().unwrap().reverse();
        let mut payload = Vec::new();
        ciborium::into_writer(&value, &mut payload).unwrap();
        let message = sign_payload(payload, &signer()).unwrap();
        assert_eq!(verify(&message), Err(VerifyError::NotDeterministic));
    }

    #[test]
    fn an_unknown_claim_fails_even_with_a_valid_signature() {
        let mut value = payload_value();
        value
            .as_map_mut()
            .unwrap()
            .push((Value::Text("extra".into()), Value::Bool(true)));
        let message = sign_payload(cbor::encode(&value), &signer()).unwrap();
        assert_eq!(verify(&message), Err(VerifyError::NotDeterministic));
    }

    #[test]
    fn a_model_hash_that_does_not_match_the_manifest_fails() {
        let mut value = payload_value();
        set(&mut value, "model_hash", Value::Bytes(vec![0; 32]));
        let message = sign_payload(cbor::encode(&value), &signer()).unwrap();
        assert_eq!(verify(&message), Err(VerifyError::ModelHash));
    }

    #[test]
    fn a_missing_claim_is_named() {
        let mut value = payload_value();
        value
            .as_map_mut()
            .unwrap()
            .retain(|(k, _)| k.as_text() != Some("response_hash"));
        let message = sign_payload(cbor::encode(&value), &signer()).unwrap();
        assert_eq!(verify(&message), Err(VerifyError::Claim("response_hash")));
    }

    #[test]
    fn an_unknown_profile_fails() {
        let mut value = payload_value();
        let entry = value
            .as_map_mut()
            .unwrap()
            .iter_mut()
            .find(|(k, _)| k == &Value::Integer(265.into()))
            .unwrap();
        entry.1 = Value::Text("tag:example.com,2026:other".into());
        let message = sign_payload(cbor::encode(&value), &signer()).unwrap();
        assert!(matches!(verify(&message), Err(VerifyError::Profile(_))));
    }

    #[test]
    fn a_software_key_round_trips_through_its_secret() {
        let a = SoftwareSigner::generate();
        let b = SoftwareSigner::from_secret(&a.secret()).unwrap();
        assert_eq!(a.public_key(), b.public_key());
        assert_ne!(a.public_key(), SoftwareSigner::generate().public_key());
    }
}
