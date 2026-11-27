//! The Secure Enclave signer on a real Mac. These tests make keys in the
//! Secure Enclave, so they are ignored by default. Run them with
//! `cargo test -p fornace-receipt --test enclave -- --ignored --nocapture`.
#![cfg(target_os = "macos")]

use std::time::Instant;

use fornace_receipt::*;

fn temp_key_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("fornace-receipt-{name}-{}.se", std::process::id()))
}

fn receipt() -> Receipt {
    let settings = Settings {
        dtype: "q8".into(),
        device: "metal".into(),
        attention: "fused".into(),
        kv_cache: "prealloc".into(),
        block_size: 16,
        kv_capacity: None,
        share_prefix: false,
        batch: 1,
        temperature: None,
        seed: 42,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
        max_new_tokens: 256,
        ignore_eos: false,
        chat_template: None,
        candle: "e8540dbb51bdbc3e25e47d21aba5977eaf4cb864".into(),
    };
    Receipt {
        issued_at: 1_795_000_000,
        nonce: [9; NONCE_LEN],
        model: Manifest::new(vec![ManifestFile {
            name: "config.json".into(),
            size: 659,
            sha256: [2; 32],
        }]),
        engine_hash: [4; 32],
        request_hash: request_hash(&[1, 2, 3], &settings),
        response_hash: response_hash(&[4, 5, 6]),
        settings,
    }
}

#[test]
#[ignore = "makes a key in the Secure Enclave"]
fn an_enclave_receipt_verifies_with_the_software_verifier() {
    let path = temp_key_path("verify");
    let signer = EnclaveSigner::create_file(&path).unwrap();
    let message = sign(&receipt(), &signer).unwrap();

    let verified = verify(&message).unwrap();
    assert_eq!(verified.receipt, receipt());
    assert_eq!(verified.device_key, signer.public_key());

    let mut changed = message.clone();
    let last = changed.len() - 1;
    changed[last] ^= 1;
    assert_eq!(verify(&changed), Err(VerifyError::Signature));
    std::fs::remove_file(&path).unwrap();
}

#[test]
#[ignore = "makes a key in the Secure Enclave"]
fn the_key_file_gives_the_same_key_again() {
    let path = temp_key_path("reopen");
    let made = EnclaveSigner::create_file(&path).unwrap();
    let opened = EnclaveSigner::open_file(&path).unwrap();
    assert_eq!(made.public_key(), opened.public_key());
    // A second create must not replace the key.
    assert!(EnclaveSigner::create_file(&path).is_err());
    assert_eq!(
        EnclaveSigner::open_file(&path).unwrap().public_key(),
        made.public_key()
    );
    std::fs::remove_file(&path).unwrap();
}

#[test]
#[ignore = "makes a key in the Secure Enclave, and prints the time of 100 signatures"]
fn time_100_signatures() {
    let path = temp_key_path("time");
    let signer = EnclaveSigner::create_file(&path).unwrap();
    let payload = receipt().encode(&signer.public_key());
    let mut raw = Vec::new();
    let mut full = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        signer.sign(&[7u8; 32]).unwrap();
        raw.push(start.elapsed().as_secs_f64() * 1e3);
        let start = Instant::now();
        sign(&receipt(), &signer).unwrap();
        full.push(start.elapsed().as_secs_f64() * 1e3);
    }
    for (name, mut t) in [("sign 32 bytes", raw), ("sign a receipt", full)] {
        t.sort_by(f64::total_cmp);
        println!(
            "{name}: n 100, min {:.2} ms, p25 {:.2}, median {:.2}, p75 {:.2}, p95 {:.2}, max {:.2}",
            t[0], t[25], t[50], t[75], t[95], t[99]
        );
    }
    println!("payload {} bytes", payload.len());
    std::fs::remove_file(&path).unwrap();
}
