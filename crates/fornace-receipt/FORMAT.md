# Fornace receipt format, version 1

A receipt is a signed statement about one generation. It names the model
files, the engine binary, the settings, the prompt tokens, and the output
tokens. A P-256 device key signs it.

A valid receipt shows that the holder of the device key signed these claims.
It does not show that the named engine ran, that the model ran on this
device, or that the device key is in secure hardware.

## The message

The receipt is a tagged `COSE_Sign1` message (RFC 9052, CBOR tag 18):

| Part | Value |
|---|---|
| Protected header | `{1: -7}`: `alg` is ES256. No other parameter |
| Unprotected header | Empty |
| Payload | The claims (below), as a byte string |
| Signature | ES256: ECDSA on P-256 over SHA-256 of the `Sig_structure` (RFC 9052, section 4.4), with empty external data. 64 bytes: `r`, then `s` |

ES256 is the only algorithm, because the Apple Secure Enclave signs only with
P-256. The verifier accepts a high `s` value.

## The claims

The payload is a CBOR map in deterministic encoding (RFC 8949, section
4.2.1): the shortest form of each integer, float, and length; definite
lengths; and the keys of each map in the bytewise order of their encodings.
The verifier decodes the claims, encodes them again, and rejects the receipt
if the bytes differ. So a receipt with an unknown claim, a duplicate key, or
keys out of order fails.

The keys 6, 10, and 265 are registered CWT and EAT claims (RFC 8392, RFC
9711). The text keys come from the AIR draft
(`draft-tsyrulnikov-rats-attested-inference-receipt-02`) where it has one.
The other text keys are Fornace claims.

| Key | Name | Type | Source | Meaning |
|---|---|---|---|---|
| 6 | `iat` | uint | CWT | Seconds since the Unix epoch, from the clock of the device |
| 10 | `eat_nonce` | bstr, 16 bytes | EAT | A fresh random value |
| 265 | `eat_profile` | tstr | EAT | `tag:inertialabs.xyz,2026:fornace-receipt-1` |
| `"model_hash"` | | bstr, 32 bytes | AIR | SHA-256 of the deterministic CBOR of `model_manifest` |
| `"model_hash_scheme"` | | tstr | AIR | `fornace-manifest-1` |
| `"model_manifest"` | | array | Fornace | One entry for each model file (below) |
| `"engine_hash"` | | bstr, 32 bytes | Fornace | SHA-256 of the engine binary |
| `"settings"` | | map | Fornace | Each setting that can change the tokens (below) |
| `"request_hash"` | | bstr, 32 bytes | AIR | SHA-256 of the prompt tokens and the settings (below) |
| `"response_hash"` | | bstr, 32 bytes | AIR | SHA-256 of the output tokens (below) |
| `"device_key"` | | bstr, 65 bytes | Fornace | The public key: an uncompressed SEC1 point (`0x04`, `x`, `y`) |

### `model_manifest`

An array of `[name, size, sha256]`, in the order of `name`:

| Field | Type | Meaning |
|---|---|---|
| `name` | tstr | The file name, relative to the model directory |
| `size` | uint | The size in bytes |
| `sha256` | bstr, 32 bytes | SHA-256 of the file |

The manifest holds the weight files that the engine loads, `config.json`,
and `tokenizer.json`. The tokenizer and the configuration change the output
also, so the weights alone do not name the model.

### `settings`

A map with text keys. Every key is present. A setting that has no value is
`null`.

| Key | Type | Meaning |
|---|---|---|
| `dtype` | tstr | The weight type, for example `q8` |
| `device` | tstr | `cpu` or `metal` |
| `attention` | tstr | The attention kernel, for example `fused` |
| `kv_cache` | tstr | The KV cache, for example `prealloc` |
| `block_size` | uint | The positions in a block of the paged cache |
| `kv_capacity` | uint or null | The positions that the cache reserves for each row |
| `share_prefix` | bool | The batch prefills the shared prefix once |
| `batch` | uint | The rows in the batch |
| `temperature` | float or null | `null` is greedy sampling |
| `seed` | uint | The seed of the sampler |
| `repeat_penalty` | float | A 32-bit float. 1.0 turns it off |
| `repeat_last_n` | uint | The earlier generated tokens that the repeat penalty sees |
| `max_new_tokens` | uint | The limit on the output tokens |
| `ignore_eos` | bool | The generation does not stop at the end-of-sequence token |
| `chat_template` | tstr or null | The template that made the prompt tokens. `null` is a raw prompt |
| `candle` | tstr | The Candle revision of the engine build |

### The token hashes

A token sequence is encoded as the bytes of each token ID, as a 32-bit
unsigned integer, little-endian, in order.

- `response_hash` is SHA-256 of the output token IDs. It covers the
  generated tokens only, not the prompt.
- `request_hash` is SHA-256 of the deterministic CBOR of the array
  `[prompt, settings]`. `prompt` is a byte string of the prompt token IDs,
  and `settings` is the map above.

The receipt does not hold the prompt. A verifier needs the prompt tokens to
check `request_hash`.

## What the verifier checks

In this order:

1. The message is a tagged `COSE_Sign1`, the protected header is `alg`
   ES256 alone, and the unprotected header is empty.
2. Each claim is present with its type.
3. `eat_profile` and `model_hash_scheme` are the values of this version.
4. `model_hash` is the hash of `model_manifest`.
5. The payload is the deterministic encoding of its claims.
6. The signature is valid for `device_key`.

These checks do not compare `device_key` with a list of known keys, the
manifest with a model directory, or `engine_hash` with a binary. They do
not check that the model makes the output tokens.

## Field names in other formats

| Field | `draft-mw-spice-inference-chain-00` (expired) | AIR (`-02`) | EAT (RFC 9711) |
|---|---|---|---|
| Model hash | `model_fingerprint` | `model_hash` | `measurements` |
| Engine hash | None | None (`enclave_measurements` is for a TEE) | `swname`, `measurements` |
| Input hash | `input_hash` | `request_hash` | None |
| Output hash | `output_hash` | `response_hash` | None |
| Device key | None | None | None (`ueid` is an identifier, not a key) |
| Time | `iat` | `iat` | `iat` |
| Nonce | None | `eat_nonce` | `eat_nonce` |
| Encoding | JWT (JSON) | `COSE_Sign1`, CBOR | JWT or CWT |
| Algorithm | Not given | Ed25519 only | Any |

AIR permits only Ed25519. The Secure Enclave cannot sign with Ed25519, so
this format uses AIR's claim names and ES256.
