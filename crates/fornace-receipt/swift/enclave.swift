// The Secure Enclave signer for fornace-receipt, through CryptoKit.
//
// The key is a P-256 key in the Secure Enclave. CryptoKit gives it as an
// encrypted blob that only this Secure Enclave can use. The caller keeps the
// blob in a file, so the key needs no keychain and no entitlement.
//
// Each function returns 0 on success. On failure it returns a code and
// writes a description of the error to `err` (at most `errCap` bytes, not
// terminated).

import CryptoKit
import Foundation

private func fail(_ code: Int32, _ error: Any, _ err: UnsafeMutablePointer<UInt8>, _ errCap: Int, _ errLen: UnsafeMutablePointer<Int>) -> Int32 {
    let text = Array(String(describing: error).utf8.prefix(errCap))
    text.withUnsafeBufferPointer { err.update(from: $0.baseAddress!, count: text.count) }
    errLen.pointee = text.count
    return code
}

private func copyPublicKey(_ key: SecureEnclave.P256.Signing.PrivateKey, _ pub: UnsafeMutablePointer<UInt8>) {
    key.publicKey.x963Representation.copyBytes(to: pub, count: 65)
}

/// Makes a new key. Writes its blob to `blob` and its public key (65 bytes,
/// uncompressed SEC1) to `pub`.
@_cdecl("fornace_se_create")
public func fornace_se_create(
    _ blob: UnsafeMutablePointer<UInt8>, _ blobCap: Int, _ blobLen: UnsafeMutablePointer<Int>,
    _ pub: UnsafeMutablePointer<UInt8>,
    _ err: UnsafeMutablePointer<UInt8>, _ errCap: Int, _ errLen: UnsafeMutablePointer<Int>
) -> Int32 {
    guard SecureEnclave.isAvailable else { return fail(1, "the Secure Enclave is not available", err, errCap, errLen) }
    do {
        let key = try SecureEnclave.P256.Signing.PrivateKey()
        let data = key.dataRepresentation
        guard data.count <= blobCap else { return fail(4, "the key blob is \(data.count) bytes", err, errCap, errLen) }
        data.copyBytes(to: blob, count: data.count)
        blobLen.pointee = data.count
        copyPublicKey(key, pub)
        return 0
    } catch {
        return fail(2, error, err, errCap, errLen)
    }
}

/// Writes the public key of the key in `blob` to `pub`.
@_cdecl("fornace_se_public_key")
public func fornace_se_public_key(
    _ blob: UnsafePointer<UInt8>, _ blobLen: Int,
    _ pub: UnsafeMutablePointer<UInt8>,
    _ err: UnsafeMutablePointer<UInt8>, _ errCap: Int, _ errLen: UnsafeMutablePointer<Int>
) -> Int32 {
    do {
        let key = try SecureEnclave.P256.Signing.PrivateKey(dataRepresentation: Data(bytes: blob, count: blobLen))
        copyPublicKey(key, pub)
        return 0
    } catch {
        return fail(3, error, err, errCap, errLen)
    }
}

/// Signs SHA-256 of `msg` with the key in `blob`. Writes the signature to
/// `sig` as 64 bytes: r, then s.
@_cdecl("fornace_se_sign")
public func fornace_se_sign(
    _ blob: UnsafePointer<UInt8>, _ blobLen: Int,
    _ msg: UnsafePointer<UInt8>, _ msgLen: Int,
    _ sig: UnsafeMutablePointer<UInt8>,
    _ err: UnsafeMutablePointer<UInt8>, _ errCap: Int, _ errLen: UnsafeMutablePointer<Int>
) -> Int32 {
    let key: SecureEnclave.P256.Signing.PrivateKey
    do {
        key = try SecureEnclave.P256.Signing.PrivateKey(dataRepresentation: Data(bytes: blob, count: blobLen))
    } catch {
        return fail(3, error, err, errCap, errLen)
    }
    do {
        let signature = try key.signature(for: Data(bytes: msg, count: msgLen))
        signature.rawRepresentation.copyBytes(to: sig, count: 64)
        return 0
    } catch {
        return fail(5, error, err, errCap, errLen)
    }
}
