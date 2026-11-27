//! The claims of a receipt, and their deterministic CBOR encoding.

use ciborium::Value;
use sha2::{Digest, Sha256};

use crate::manifest::{MODEL_HASH_SCHEME, Manifest};
use crate::{Hash, PublicKey, cbor};

/// The `eat_profile` of this format (EAT, RFC 9711). A new version of
/// `FORMAT.md` gets a new profile.
pub const PROFILE: &str = "tag:inertialabs.xyz,2026:fornace-receipt-1";

/// The length of `eat_nonce` in bytes. EAT permits 8 to 64.
pub const NONCE_LEN: usize = 16;

// CWT and EAT claim keys (RFC 8392, RFC 9711).
const KEY_IAT: i64 = 6;
const KEY_NONCE: i64 = 10;
const KEY_PROFILE: i64 = 265;

/// Each setting that can change the generated tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub dtype: String,
    pub device: String,
    pub attention: String,
    pub kv_cache: String,
    pub block_size: u64,
    pub kv_capacity: Option<u64>,
    pub share_prefix: bool,
    pub batch: u64,
    /// `None` is greedy sampling.
    pub temperature: Option<f64>,
    pub seed: u64,
    pub repeat_penalty: f32,
    /// The number of earlier generated tokens that the repeat penalty sees.
    pub repeat_last_n: u64,
    pub max_new_tokens: u64,
    pub ignore_eos: bool,
    /// The chat template that made the prompt tokens, or `None` for a raw
    /// prompt.
    pub chat_template: Option<String>,
    /// The Candle revision of the engine build.
    pub candle: String,
}

/// The claims of one receipt, without the device key.
#[derive(Debug, Clone, PartialEq)]
pub struct Receipt {
    /// Seconds since the Unix epoch (`iat`).
    pub issued_at: u64,
    /// A fresh random value (`eat_nonce`).
    pub nonce: [u8; NONCE_LEN],
    pub model: Manifest,
    /// SHA-256 of the engine binary.
    pub engine_hash: Hash,
    pub settings: Settings,
    /// See [`request_hash`].
    pub request_hash: Hash,
    /// See [`response_hash`].
    pub response_hash: Hash,
}

/// SHA-256 of the token IDs, each as 4 bytes little-endian, in order.
pub fn response_hash(tokens: &[u32]) -> Hash {
    Sha256::digest(token_bytes(tokens)).into()
}

/// SHA-256 of the deterministic CBOR of `[prompt, settings]`. `prompt` is
/// a byte string of the prompt token IDs, each as 4 bytes little-endian.
pub fn request_hash(prompt: &[u32], settings: &Settings) -> Hash {
    let value = Value::Array(vec![Value::Bytes(token_bytes(prompt)), settings.to_value()]);
    Sha256::digest(cbor::encode(&value)).into()
}

fn token_bytes(tokens: &[u32]) -> Vec<u8> {
    tokens.iter().flat_map(|t| t.to_le_bytes()).collect()
}

fn text(s: &str) -> Value {
    Value::Text(s.to_string())
}

fn uint(n: u64) -> Value {
    Value::Integer(n.into())
}

fn bytes(b: &[u8]) -> Value {
    Value::Bytes(b.to_vec())
}

fn int_key(k: i64) -> Value {
    Value::Integer(k.into())
}

impl Settings {
    fn to_value(&self) -> Value {
        let opt = |v: Option<Value>| v.unwrap_or(Value::Null);
        Value::Map(vec![
            (text("dtype"), text(&self.dtype)),
            (text("device"), text(&self.device)),
            (text("attention"), text(&self.attention)),
            (text("kv_cache"), text(&self.kv_cache)),
            (text("block_size"), uint(self.block_size)),
            (text("kv_capacity"), opt(self.kv_capacity.map(uint))),
            (text("share_prefix"), Value::Bool(self.share_prefix)),
            (text("batch"), uint(self.batch)),
            (text("temperature"), opt(self.temperature.map(Value::Float))),
            (text("seed"), uint(self.seed)),
            (
                text("repeat_penalty"),
                Value::Float(f64::from(self.repeat_penalty)),
            ),
            (text("repeat_last_n"), uint(self.repeat_last_n)),
            (text("max_new_tokens"), uint(self.max_new_tokens)),
            (text("ignore_eos"), Value::Bool(self.ignore_eos)),
            (
                text("chat_template"),
                opt(self.chat_template.as_deref().map(text)),
            ),
            (text("candle"), text(&self.candle)),
        ])
    }

    fn from_value(value: &Value) -> Result<Self, &'static str> {
        let m = Fields::new(value, "settings")?;
        Ok(Settings {
            dtype: m.text("dtype")?,
            device: m.text("device")?,
            attention: m.text("attention")?,
            kv_cache: m.text("kv_cache")?,
            block_size: m.uint("block_size")?,
            kv_capacity: m.or_null("kv_capacity", |v| u64::try_from(v.as_integer()?).ok())?,
            share_prefix: m.bool("share_prefix")?,
            batch: m.uint("batch")?,
            temperature: m.or_null("temperature", Value::as_float)?,
            seed: m.uint("seed")?,
            // An f32 widened to f64 converts back exactly. A value that is
            // not an f32 does not re-encode to the same bytes, so the
            // deterministic check rejects it.
            repeat_penalty: m
                .get("repeat_penalty")?
                .as_float()
                .ok_or("repeat_penalty")? as f32,
            repeat_last_n: m.uint("repeat_last_n")?,
            max_new_tokens: m.uint("max_new_tokens")?,
            ignore_eos: m.bool("ignore_eos")?,
            chat_template: m.or_null("chat_template", |v| v.as_text().map(String::from))?,
            candle: m.text("candle")?,
        })
    }
}

impl Receipt {
    /// The deterministic CBOR of the claims with `device_key`: the payload
    /// that the key signs.
    pub fn encode(&self, device_key: &PublicKey) -> Vec<u8> {
        cbor::encode(&Value::Map(vec![
            (int_key(KEY_IAT), uint(self.issued_at)),
            (int_key(KEY_NONCE), bytes(&self.nonce)),
            (int_key(KEY_PROFILE), text(PROFILE)),
            (text("model_hash"), bytes(&self.model.hash())),
            (text("model_hash_scheme"), text(MODEL_HASH_SCHEME)),
            (text("model_manifest"), self.model.to_value()),
            (text("engine_hash"), bytes(&self.engine_hash)),
            (text("settings"), self.settings.to_value()),
            (text("request_hash"), bytes(&self.request_hash)),
            (text("response_hash"), bytes(&self.response_hash)),
            (text("device_key"), bytes(device_key)),
        ]))
    }

    /// Decodes a payload. It does not check `model_hash`, the profile, or
    /// that the payload is deterministic: [`crate::verify`] does.
    pub(crate) fn decode(payload: &[u8]) -> Result<Decoded, &'static str> {
        let value: Value = ciborium::from_reader(payload).map_err(|_| "payload")?;
        let m = Fields::new(&value, "payload")?;
        let receipt = Receipt {
            issued_at: u64::try_from(m.get_int(KEY_IAT)?.as_integer().ok_or("iat")?)
                .map_err(|_| "iat")?,
            nonce: m
                .get_int(KEY_NONCE)?
                .as_bytes()
                .and_then(|b| b.as_slice().try_into().ok())
                .ok_or("eat_nonce")?,
            model: Manifest::from_value(m.get("model_manifest")?).ok_or("model_manifest")?,
            engine_hash: m.hash("engine_hash")?,
            settings: Settings::from_value(m.get("settings")?)?,
            request_hash: m.hash("request_hash")?,
            response_hash: m.hash("response_hash")?,
        };
        Ok(Decoded {
            receipt,
            profile: m
                .get_int(KEY_PROFILE)?
                .as_text()
                .ok_or("eat_profile")?
                .to_string(),
            model_hash: m.hash("model_hash")?,
            model_hash_scheme: m.text("model_hash_scheme")?,
            device_key: m
                .get("device_key")?
                .as_bytes()
                .and_then(|b| b.as_slice().try_into().ok())
                .ok_or("device_key")?,
        })
    }
}

pub(crate) struct Decoded {
    pub receipt: Receipt,
    pub profile: String,
    pub model_hash: Hash,
    pub model_hash_scheme: String,
    pub device_key: PublicKey,
}

/// The entries of a CBOR map, looked up by key. Each error names the claim.
struct Fields<'a>(&'a [(Value, Value)]);

impl<'a> Fields<'a> {
    fn new(value: &'a Value, name: &'static str) -> Result<Self, &'static str> {
        value.as_map().map(|m| Fields(m.as_slice())).ok_or(name)
    }

    fn find(&self, key: &Value, name: &'static str) -> Result<&'a Value, &'static str> {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
            .ok_or(name)
    }

    fn get(&self, name: &'static str) -> Result<&'a Value, &'static str> {
        self.find(&text(name), name)
    }

    fn get_int(&self, key: i64) -> Result<&'a Value, &'static str> {
        let name = match key {
            KEY_IAT => "iat",
            KEY_NONCE => "eat_nonce",
            _ => "eat_profile",
        };
        self.find(&int_key(key), name)
    }

    fn text(&self, name: &'static str) -> Result<String, &'static str> {
        self.get(name)?.as_text().map(String::from).ok_or(name)
    }

    fn uint(&self, name: &'static str) -> Result<u64, &'static str> {
        u64::try_from(self.get(name)?.as_integer().ok_or(name)?).map_err(|_| name)
    }

    fn bool(&self, name: &'static str) -> Result<bool, &'static str> {
        self.get(name)?.as_bool().ok_or(name)
    }

    fn hash(&self, name: &'static str) -> Result<Hash, &'static str> {
        self.get(name)?
            .as_bytes()
            .and_then(|b| b.as_slice().try_into().ok())
            .ok_or(name)
    }

    fn or_null<T>(
        &self,
        name: &'static str,
        parse: impl Fn(&Value) -> Option<T>,
    ) -> Result<Option<T>, &'static str> {
        match self.get(name)? {
            Value::Null => Ok(None),
            v => parse(v).map(Some).ok_or(name),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::manifest::ManifestFile;

    pub(crate) fn settings() -> Settings {
        Settings {
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
            chat_template: Some("qwen2.5".into()),
            candle: "e8540dbb51bdbc3e25e47d21aba5977eaf4cb864".into(),
        }
    }

    pub(crate) fn receipt() -> Receipt {
        let settings = settings();
        Receipt {
            issued_at: 1_795_000_000,
            nonce: [9; NONCE_LEN],
            model: Manifest::new(vec![
                ManifestFile {
                    name: "model-q8_0.gguf".into(),
                    size: 531_000_000,
                    sha256: [1; 32],
                },
                ManifestFile {
                    name: "config.json".into(),
                    size: 659,
                    sha256: [2; 32],
                },
                ManifestFile {
                    name: "tokenizer.json".into(),
                    size: 7_031_645,
                    sha256: [3; 32],
                },
            ]),
            engine_hash: [4; 32],
            request_hash: request_hash(&[151644, 872, 198], &settings),
            response_hash: response_hash(&[785, 6722, 315, 17689, 374, 24081, 13]),
            settings,
        }
    }

    #[test]
    fn the_same_receipt_encodes_to_the_same_bytes() {
        let key = [5u8; 65];
        assert_eq!(receipt().encode(&key), receipt().encode(&key));
    }

    #[test]
    fn a_receipt_round_trips_through_its_encoding() {
        let key = [5u8; 65];
        let decoded = Receipt::decode(&receipt().encode(&key)).unwrap();
        assert_eq!(decoded.receipt, receipt());
        assert_eq!(decoded.device_key, key);
        assert_eq!(decoded.profile, PROFILE);
        assert_eq!(decoded.model_hash, receipt().model.hash());
    }

    #[test]
    fn optional_settings_round_trip() {
        let mut r = receipt();
        r.settings.temperature = Some(0.7);
        r.settings.kv_capacity = Some(4096);
        r.settings.chat_template = None;
        let decoded = Receipt::decode(&r.encode(&[5; 65])).unwrap();
        assert_eq!(decoded.receipt, r);
    }

    #[test]
    fn each_setting_changes_the_request_hash() {
        let base = request_hash(&[1, 2], &settings());
        let mut changed: Vec<Settings> = Vec::new();
        let mut push = |f: &dyn Fn(&mut Settings)| {
            let mut s = settings();
            f(&mut s);
            changed.push(s);
        };
        push(&|s| s.dtype = "q4".into());
        push(&|s| s.device = "cpu".into());
        push(&|s| s.attention = "naive".into());
        push(&|s| s.kv_cache = "paged".into());
        push(&|s| s.block_size = 32);
        push(&|s| s.kv_capacity = Some(1024));
        push(&|s| s.share_prefix = true);
        push(&|s| s.batch = 4);
        push(&|s| s.temperature = Some(0.7));
        push(&|s| s.seed = 43);
        push(&|s| s.repeat_penalty = 1.0);
        push(&|s| s.repeat_last_n = 32);
        push(&|s| s.max_new_tokens = 128);
        push(&|s| s.ignore_eos = true);
        push(&|s| s.chat_template = None);
        push(&|s| s.candle = "0".into());
        for s in &changed {
            assert_ne!(request_hash(&[1, 2], s), base, "{s:?}");
        }
        assert_ne!(request_hash(&[1, 3], &settings()), base);
    }

    #[test]
    fn response_hash_is_sha256_of_little_endian_token_ids() {
        // SHA-256 of the 8 bytes 01 00 00 00 02 00 00 00.
        let mut h = Sha256::new();
        h.update([1, 0, 0, 0, 2, 0, 0, 0]);
        let expected: Hash = h.finalize().into();
        assert_eq!(response_hash(&[1, 2]), expected);
        assert_ne!(response_hash(&[2, 1]), expected);
    }
}
