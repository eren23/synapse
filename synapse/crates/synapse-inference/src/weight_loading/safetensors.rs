use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use super::{RawTensor, WeightError};
use super::converter::{bf16_to_f32, f16_to_f32};

/// Load tensors from a safetensors file, returning raw tensors ready for model loading.
///
/// Memory-maps the file for zero-copy access to tensor data.
pub fn load_safetensors(path: &Path) -> Result<HashMap<String, RawTensor>, WeightError> {
    let file = File::open(path).map_err(WeightError::Io)?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(WeightError::Io)?;
    parse_safetensors(&mmap)
}

/// Parse safetensors from a byte slice into raw f32 data.
///
/// Format: `[u64 header_size][JSON header][tensor data...]`
pub fn parse_safetensors(data: &[u8]) -> Result<HashMap<String, RawTensor>, WeightError> {
    if data.len() < 8 {
        return Err(WeightError::InvalidFormat("File too small".into()));
    }

    let header_size = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    if 8 + header_size > data.len() {
        return Err(WeightError::InvalidFormat(
            "Header size exceeds file length".into(),
        ));
    }

    let header_json = std::str::from_utf8(&data[8..8 + header_size])
        .map_err(|e| WeightError::InvalidFormat(format!("Invalid UTF-8 in header: {e}")))?;

    let header: HashMap<String, serde_json::Value> = serde_json::from_str(header_json)
        .map_err(|e| WeightError::InvalidFormat(format!("Invalid JSON header: {e}")))?;

    let data_start = 8 + header_size;
    let mut result = HashMap::new();

    for (name, info) in &header {
        if name == "__metadata__" {
            continue;
        }

        let dtype = info
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| WeightError::InvalidFormat(format!("Missing dtype for {name}")))?;

        let shape: Vec<usize> = info
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| WeightError::InvalidFormat(format!("Missing shape for {name}")))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();

        let offsets = info
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                WeightError::InvalidFormat(format!("Missing data_offsets for {name}"))
            })?;

        let start = offsets[0].as_u64().unwrap_or(0) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;

        if data_start + end > data.len() {
            return Err(WeightError::InvalidFormat(format!(
                "Tensor {name} data exceeds file bounds"
            )));
        }

        let tensor_bytes = &data[data_start + start..data_start + end];

        let f32_data = match dtype {
            "F32" => bytes_to_f32(tensor_bytes),
            "F16" => {
                let u16_data = bytes_to_u16(tensor_bytes);
                f16_to_f32(&u16_data)
            }
            "BF16" => {
                let u16_data = bytes_to_u16(tensor_bytes);
                bf16_to_f32(&u16_data)
            }
            other => return Err(WeightError::UnsupportedDtype(other.to_string())),
        };

        result.insert(name.clone(), RawTensor { data: f32_data, shape });
    }

    Ok(result)
}
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal safetensors binary with given tensors.
    fn make_safetensors(tensors: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
        // Collect tensor data and build header entries
        let mut data_section = Vec::new();
        let mut header_map = serde_json::Map::new();

        for &(name, dtype, shape, raw_bytes) in tensors {
            let offset_start = data_section.len();
            data_section.extend_from_slice(raw_bytes);
            let offset_end = data_section.len();

            let shape_json: Vec<serde_json::Value> =
                shape.iter().map(|&d| serde_json::json!(d)).collect();

            header_map.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape_json,
                    "data_offsets": [offset_start, offset_end],
                }),
            );
        }

        let header_str = serde_json::to_string(&serde_json::Value::Object(header_map)).unwrap();
        let header_bytes = header_str.as_bytes();

        let mut file_data = Vec::new();
        file_data.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        file_data.extend_from_slice(header_bytes);
        file_data.extend_from_slice(&data_section);
        file_data
    }

    fn f32_to_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    fn f16_to_bytes(vals: &[u16]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn load_two_f32_tensors_bit_exact() {
        let weight_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight_b: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0];

        let bytes_a = f32_to_bytes(&weight_a);
        let bytes_b = f32_to_bytes(&weight_b);

        let file_data = make_safetensors(&[
            ("weight_a", "F32", &[2, 3], &bytes_a),
            ("weight_b", "F32", &[4], &bytes_b),
        ]);

        let tensors = parse_safetensors(&file_data).unwrap();
        assert_eq!(tensors.len(), 2);

        let a = &tensors["weight_a"];
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.data, weight_a, "weight_a must be bit-exact");

        let b = &tensors["weight_b"];
        assert_eq!(b.shape, vec![4]);
        assert_eq!(b.data, weight_b, "weight_b must be bit-exact");
    }

    #[test]
    fn load_f16_tensor_converts_correctly() {
        // f16 bits: 1.0=0x3C00, 2.0=0x4000, 0.5=0x3800
        let f16_bits: Vec<u16> = vec![0x3C00, 0x4000, 0x3800];
        let raw = f16_to_bytes(&f16_bits);

        let file_data = make_safetensors(&[("w", "F16", &[3], &raw)]);
        let tensors = parse_safetensors(&file_data).unwrap();

        let w = &tensors["w"];
        assert_eq!(w.shape, vec![3]);
        assert_eq!(w.data, vec![1.0, 2.0, 0.5]);
    }

    #[test]
    fn load_bf16_tensor_converts_within_tolerance() {
        // bf16 bits: 1.0=0x3F80, 2.0=0x4000, -1.0=0xBF80
        let bf16_bits: Vec<u16> = vec![0x3F80, 0x4000, 0xBF80];
        let raw = f16_to_bytes(&bf16_bits);

        let file_data = make_safetensors(&[("w", "BF16", &[3], &raw)]);
        let tensors = parse_safetensors(&file_data).unwrap();

        let w = &tensors["w"];
        assert_eq!(w.data, vec![1.0, 2.0, -1.0]);
    }

    #[test]
    fn unsupported_dtype_returns_error() {
        // Craft a header with an unsupported dtype
        let header = r#"{"t":{"dtype":"INT4","shape":[2],"data_offsets":[0,1]}}"#;
        let header_bytes = header.as_bytes();
        let mut data = Vec::new();
        data.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(header_bytes);
        data.push(0); // 1 byte of tensor data

        let result = parse_safetensors(&data);
        assert!(matches!(result, Err(WeightError::UnsupportedDtype(_))));
    }

    #[test]
    fn truncated_file_returns_error() {
        // Only 4 bytes — can't even read header size
        let result = parse_safetensors(&[0u8; 4]);
        assert!(matches!(result, Err(WeightError::InvalidFormat(_))));
    }

    #[test]
    fn metadata_key_is_skipped() {
        let header = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
        });
        let header_str = serde_json::to_string(&header).unwrap();
        let header_bytes = header_str.as_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(header_bytes);
        data.extend_from_slice(&1.0f32.to_le_bytes());

        let tensors = parse_safetensors(&data).unwrap();
        assert_eq!(tensors.len(), 1);
        assert!(tensors.contains_key("w"));
    }
}
