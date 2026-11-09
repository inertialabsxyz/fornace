//! Quantize the safetensors weights of a model to a GGUF file.
//!
//! `fornace-quantize` calls [`quantize_model`]. Each matmul weight and the
//! token embedding (every rank-2 `*.weight` tensor) becomes Q8_0 or Q4_0.
//! Biases and norm weights stay F32. The tensors keep their safetensors
//! names, so the model code uses one set of names for both formats.
//!
//! The GGUF file holds no config. The engine reads `config.json` and
//! `tokenizer.json` from the same directory for both formats, so one file
//! stays the only source of the model shape.

use std::path::{Path, PathBuf};

use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
use candle_core::{Device, Tensor};

use crate::{Dtype, Error, Result};

/// The GGUF type for a quantized `Dtype`, or `None` for a dense one.
pub fn ggml_dtype(dtype: Dtype) -> Option<GgmlDType> {
    match dtype {
        Dtype::Q8 => Some(GgmlDType::Q8_0),
        Dtype::Q4 => Some(GgmlDType::Q4_0),
        Dtype::F16 | Dtype::Bf16 | Dtype::F32 => None,
    }
}

/// The GGUF file for a quantized `Dtype` in `model_dir`, for example
/// `model-q8_0.gguf`. `None` for a dense `Dtype`.
pub fn gguf_path(model_dir: &Path, dtype: Dtype) -> Option<PathBuf> {
    let name = match ggml_dtype(dtype)? {
        GgmlDType::Q8_0 => "model-q8_0.gguf",
        GgmlDType::Q4_0 => "model-q4_0.gguf",
        _ => unreachable!("ggml_dtype gives only Q8_0 and Q4_0"),
    };
    Some(model_dir.join(name))
}

/// Quantizes each tensor: a rank-2 `*.weight` tensor to `dtype`, any other
/// tensor to F32. Returns an error that names the tensor if its last
/// dimension is not a multiple of the block size.
pub fn quantize_tensors(
    tensors: &[(String, Tensor)],
    dtype: GgmlDType,
) -> Result<Vec<(String, QTensor)>> {
    tensors
        .iter()
        .map(|(name, tensor)| {
            let target = if tensor.rank() == 2 && name.ends_with(".weight") {
                dtype
            } else {
                GgmlDType::F32
            };
            let qtensor = QTensor::quantize(tensor, target).map_err(|source| Error::Quantize {
                name: name.clone(),
                source,
            })?;
            Ok((name.clone(), qtensor))
        })
        .collect()
}

/// Writes the tensors to a GGUF file, with no metadata.
pub fn write_gguf<W: std::io::Seek + std::io::Write>(
    w: &mut W,
    tensors: &[(String, QTensor)],
) -> Result<()> {
    let refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    gguf_file::write(w, &[], &refs)?;
    Ok(())
}

/// Reads `model.safetensors` in `model_dir`, quantizes it to `dtype` (`Q8`
/// or `Q4`), and writes the file of [`gguf_path`]. Returns the path.
pub fn quantize_model(model_dir: &Path, dtype: Dtype) -> Result<PathBuf> {
    let (Some(ggml), Some(out)) = (ggml_dtype(dtype), gguf_path(model_dir, dtype)) else {
        return Err(Error::NotQuantized(dtype));
    };
    let path = model_dir.join("model.safetensors");
    let mut tensors: Vec<(String, Tensor)> = candle_core::safetensors::load(&path, &Device::Cpu)?
        .into_iter()
        .collect();
    // A fixed order gives the same file on each run.
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let quantized = quantize_tensors(&tensors, ggml)?;
    let file = std::fs::File::create(&out).map_err(|source| Error::Write {
        path: out.clone(),
        source,
    })?;
    let mut w = std::io::BufWriter::new(file);
    write_gguf(&mut w, &quantized)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A fixed tensor with values between -1 and 1. No random generator, so
    /// the test is the same on each run.
    fn fixed(rows: usize, cols: usize) -> Tensor {
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.37).sin())
            .collect();
        Tensor::from_vec(values, (rows, cols), &Device::Cpu).unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap()
    }

    #[test]
    fn quantized_tensors_round_trip_through_a_gguf_file() {
        for (dtype, tolerance) in [(GgmlDType::Q8_0, 0.01), (GgmlDType::Q4_0, 0.15)] {
            let tensors = vec![
                ("layer.proj.weight".to_string(), fixed(16, 64)),
                (
                    "layer.proj.bias".to_string(),
                    fixed(1, 16).flatten_all().unwrap(),
                ),
            ];
            let quantized = quantize_tensors(&tensors, dtype).unwrap();
            assert_eq!(quantized[0].1.dtype(), dtype);
            assert_eq!(quantized[1].1.dtype(), GgmlDType::F32);

            let mut file = Cursor::new(Vec::new());
            write_gguf(&mut file, &quantized).unwrap();
            file.set_position(0);
            let content = gguf_file::Content::read(&mut file).unwrap();
            for ((name, original), (_, written)) in tensors.iter().zip(&quantized) {
                let read = content.tensor(&mut file, name, &Device::Cpu).unwrap();
                assert_eq!(read.dtype(), written.dtype(), "{name}");
                assert_eq!(read.shape(), original.shape(), "{name}");
                let read = read.dequantize(&Device::Cpu).unwrap();
                // The file holds exactly the quantized values.
                let written = written.dequantize(&Device::Cpu).unwrap();
                assert_eq!(max_abs_diff(&read, &written), 0.0, "{name}");
                // The quantized values are near the original values.
                let diff = max_abs_diff(&read, original);
                assert!(diff <= tolerance, "{name} {dtype:?}: diff {diff}");
            }
        }
    }

    #[test]
    fn a_shape_that_is_not_a_multiple_of_the_block_is_an_error() {
        let tensors = vec![("bad.weight".to_string(), fixed(4, 33))];
        match quantize_tensors(&tensors, GgmlDType::Q8_0) {
            Err(Error::Quantize { name, .. }) => assert_eq!(name, "bad.weight"),
            other => panic!("expected a quantize error, got {other:?}"),
        }
    }

    #[test]
    fn gguf_path_names_the_format() {
        let dir = Path::new("m");
        assert_eq!(gguf_path(dir, Dtype::Q8), Some(dir.join("model-q8_0.gguf")));
        assert_eq!(gguf_path(dir, Dtype::Q4), Some(dir.join("model-q4_0.gguf")));
        assert_eq!(gguf_path(dir, Dtype::F16), None);
    }
}
