//! The model manifest: the name, size, and SHA-256 of each model file.

use std::io::Read;
use std::path::Path;

use ciborium::Value;
use sha2::{Digest, Sha256};

use crate::{Hash, cbor};

/// The name of the model hash scheme in the receipt (`model_hash_scheme`).
pub const MODEL_HASH_SCHEME: &str = "fornace-manifest-1";

/// One model file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestFile {
    /// The file name, relative to the model directory.
    pub name: String,
    /// The size in bytes.
    pub size: u64,
    pub sha256: Hash,
}

/// The files of one model, in the order of their names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    files: Vec<ManifestFile>,
}

impl Manifest {
    /// Puts the files in the order of their names.
    pub fn new(mut files: Vec<ManifestFile>) -> Self {
        files.sort_by(|a, b| a.name.cmp(&b.name));
        Manifest { files }
    }

    /// Reads and hashes each file in `names`, relative to `dir`.
    pub fn hash_files(dir: &Path, names: &[&str]) -> std::io::Result<Self> {
        let files = names
            .iter()
            .map(|name| {
                let (size, sha256) = hash_file(&dir.join(name))?;
                Ok(ManifestFile {
                    name: (*name).to_string(),
                    size,
                    sha256,
                })
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        Ok(Manifest::new(files))
    }

    pub fn files(&self) -> &[ManifestFile] {
        &self.files
    }

    /// The model hash: SHA-256 of the deterministic CBOR of the manifest.
    pub fn hash(&self) -> Hash {
        Sha256::digest(cbor::encode(&self.to_value())).into()
    }

    pub(crate) fn to_value(&self) -> Value {
        Value::Array(
            self.files
                .iter()
                .map(|f| {
                    Value::Array(vec![
                        Value::Text(f.name.clone()),
                        Value::Integer(f.size.into()),
                        Value::Bytes(f.sha256.to_vec()),
                    ])
                })
                .collect(),
        )
    }

    pub(crate) fn from_value(value: &Value) -> Option<Self> {
        let files = value
            .as_array()?
            .iter()
            .map(|entry| match entry.as_array()?.as_slice() {
                [
                    Value::Text(name),
                    Value::Integer(size),
                    Value::Bytes(sha256),
                ] => Some(ManifestFile {
                    name: name.clone(),
                    size: u64::try_from(*size).ok()?,
                    sha256: sha256.as_slice().try_into().ok()?,
                }),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Manifest::new(files))
    }
}

/// The size and SHA-256 of the file at `path`, read in blocks of 1 MiB.
pub fn hash_file(path: &Path) -> std::io::Result<(u64, Hash)> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut size = 0u64;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        size += n as u64;
    }
    Ok((size, hasher.finalize().into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA256_ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn hash_files_gives_each_file_in_name_order() {
        let dir =
            std::env::temp_dir().join(format!("fornace-receipt-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), b"abc").unwrap();
        std::fs::write(dir.join("config.json"), b"").unwrap();
        let manifest = Manifest::hash_files(&dir, &["tokenizer.json", "config.json"]).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        let files = manifest.files();
        assert_eq!(files[0].name, "config.json");
        assert_eq!(files[0].size, 0);
        assert_eq!(files[1].name, "tokenizer.json");
        assert_eq!(files[1].size, 3);
        assert_eq!(hex(&files[1].sha256), SHA256_ABC);
    }

    #[test]
    fn the_order_of_the_input_does_not_change_the_model_hash() {
        let a = ManifestFile {
            name: "a".into(),
            size: 1,
            sha256: [1; 32],
        };
        let b = ManifestFile {
            name: "b".into(),
            size: 2,
            sha256: [2; 32],
        };
        assert_eq!(
            Manifest::new(vec![a.clone(), b.clone()]).hash(),
            Manifest::new(vec![b, a]).hash()
        );
    }

    #[test]
    fn a_manifest_round_trips_through_cbor() {
        let m = Manifest::new(vec![ManifestFile {
            name: "model.safetensors".into(),
            size: 988_097_824,
            sha256: [7; 32],
        }]);
        assert_eq!(Manifest::from_value(&m.to_value()), Some(m));
    }
}
