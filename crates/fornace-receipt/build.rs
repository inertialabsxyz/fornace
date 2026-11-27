//! On macOS, compiles the Secure Enclave signer (`swift/enclave.swift`) into
//! a static library and links it with CryptoKit and the Swift runtime.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=swift/enclave.swift");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let arch = match env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
        "aarch64" => "arm64",
        other => other,
    }
    .to_string();
    // Build the Swift code for the deployment target of the Rust code, so
    // that the linker sees one minimum macOS version.
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    let min = env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "11.0".into());
    let lib = out.join("libfornace_enclave.a");
    let status = Command::new("xcrun")
        .args([
            "swiftc",
            "-O",
            "-parse-as-library",
            "-emit-library",
            "-static",
        ])
        .args(["-module-name", "FornaceEnclave"])
        .args(["-target", &format!("{arch}-apple-macosx{min}")])
        .arg("swift/enclave.swift")
        .arg("-o")
        .arg(&lib)
        .status()
        .expect("xcrun swiftc runs (install the Xcode command line tools)");
    assert!(status.success(), "swiftc failed for swift/enclave.swift");

    let sdk = Command::new("xcrun")
        .args(["--show-sdk-path"])
        .output()
        .expect("xcrun --show-sdk-path runs");
    let sdk = String::from_utf8(sdk.stdout).unwrap();
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=fornace_enclave");
    println!(
        "cargo:rustc-link-search=native={}/usr/lib/swift",
        sdk.trim()
    );
    println!("cargo:rustc-link-search=native=/usr/lib/swift");
    // The back-deployment libraries (`swiftCompatibility*`) for a
    // deployment target before the Swift runtime in the OS.
    let swiftc = Command::new("xcrun")
        .args(["--find", "swiftc"])
        .output()
        .expect("xcrun --find swiftc runs");
    let swiftc = PathBuf::from(String::from_utf8(swiftc.stdout).unwrap().trim());
    let toolchain = swiftc.parent().unwrap().join("../lib/swift/macosx");
    println!("cargo:rustc-link-search=native={}", toolchain.display());
    println!("cargo:rustc-link-lib=framework=CryptoKit");
    println!("cargo:rustc-link-lib=framework=Foundation");
}
