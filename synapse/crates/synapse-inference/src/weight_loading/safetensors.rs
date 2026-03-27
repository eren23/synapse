use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use super::converter::{bf16_bits_to_f32, f16_bits_to_f32};
use super::{AlignedBuffer, RawTensor, WeightError};
use memmap2::Mmap;
use serde::Deserialize;

/// Represents the `model.safetensors.index.json` file used by sharded checkpoints.
#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Load tensors from a sharded safetensors checkpoint.
///
/// Reads `model.safetensors.index.json` from `model_dir`, then loads each
/// unique shard file sequentially, extracting only the tensors listed in the
/// index's `weight_map`. Shards are dropped after extraction to limit memory.
pub fn load_safetensors_sharded(
    model_dir: &Path,
) -> Result<HashMap<String, RawTensor>, WeightError> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let index_data = std::fs::read_to_string(&index_path).map_err(WeightError::Io)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_data)
        .map_err(|e| WeightError::InvalidFormat(format!("Invalid index JSON: {e}")))?;

    // Group tensor names by shard filename
    let mut shard_to_tensors: HashMap<String, Vec<String>> = HashMap::new();
    for (tensor_name, shard_filename) in &index.weight_map {
        shard_to_tensors
            .entry(shard_filename.clone())
            .or_default()
            .push(tensor_name.clone());
    }

    let mut result = HashMap::new();

    // Load shards sequentially to avoid memory spikes
    for (shard_filename, tensor_names) in &shard_to_tensors {
        let shard_path = model_dir.join(shard_filename);
        let shard_tensors = load_safetensors(&shard_path).map_err(|e| {
            WeightError::InvalidFormat(format!("Failed to load shard '{}': {}", shard_filename, e))
        })?;

        for tensor_name in tensor_names {
            match shard_tensors.get(tensor_name) {
                Some(tensor) => {
                    result.insert(tensor_name.clone(), tensor.clone());
                }
                None => {
                    return Err(WeightError::InvalidFormat(format!(
                        "Tensor '{}' listed in index but not found in shard '{}'",
                        tensor_name, shard_filename
                    )));
                }
            }
        }
        // shard_tensors is dropped here, freeing memory from unused tensors
    }

    Ok(result)
}

/// Load tensors from a safetensors file, returning raw tensors ready for model loading.
///
/// Memory-maps the file for zero-copy access to tensor data.
/// Tensor data is loaded directly into 64-byte-aligned buffers in a single pass.
pub fn load_safetensors(path: &Path) -> Result<HashMap<String, RawTensor>, WeightError> {
    let file = File::open(path).map_err(WeightError::Io)?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(WeightError::Io)?;
    parse_safetensors(&mmap)
}

/// Parse safetensors from a byte slice into aligned f32 buffers.
///
/// Format: `[u64 header_size][JSON header][tensor data...]`
///
/// For F32 tensors: raw bytes are copied directly into a 64-byte-aligned buffer
/// (single allocation, single memcpy — no intermediate Vec).
///
/// For F16/BF16: elements are converted directly into the aligned buffer in one pass.
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

        let aligned_data = match dtype {
            // Fast path: direct memcpy from mmap into aligned buffer (no intermediate alloc)
            "F32" => AlignedBuffer::from_f32_bytes(tensor_bytes),
            // Single-pass conversion: bytes → aligned f32 buffer (no intermediate Vec<u16>)
            "F16" => {
                let count = tensor_bytes.len() / 2;
                let mut buf = AlignedBuffer::new_zeroed(count);
                for (i, chunk) in tensor_bytes.chunks_exact(2).enumerate() {
                    let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                    buf[i] = f16_bits_to_f32(bits);
                }
                buf
            }
            "BF16" => {
                let count = tensor_bytes.len() / 2;
                let mut buf = AlignedBuffer::new_zeroed(count);
                for (i, chunk) in tensor_bytes.chunks_exact(2).enumerate() {
                    let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                    buf[i] = bf16_bits_to_f32(bits);
                }
                buf
            }
            other => return Err(WeightError::UnsupportedDtype(other.to_string())),
        };

        result.insert(
            name.clone(),
            RawTensor {
                data: aligned_data,
                shape,
            },
        );
    }

    Ok(result)
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

    #[test]
    fn all_buffers_are_64_byte_aligned() {
        let f32_vals: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let f32_bytes = f32_to_bytes(&f32_vals);

        let f16_bits: Vec<u16> = vec![0x3C00, 0x4000, 0x3800, 0xBC00];
        let f16_raw = f16_to_bytes(&f16_bits);

        let bf16_bits: Vec<u16> = vec![0x3F80, 0x4000, 0xBF80];
        let bf16_raw = f16_to_bytes(&bf16_bits);

        let file_data = make_safetensors(&[
            ("f32_tensor", "F32", &[256], &f32_bytes),
            ("f16_tensor", "F16", &[4], &f16_raw),
            ("bf16_tensor", "BF16", &[3], &bf16_raw),
        ]);

        let tensors = parse_safetensors(&file_data).unwrap();

        for (name, tensor) in &tensors {
            assert!(
                tensor.data.is_aligned(),
                "Tensor '{name}' buffer is not 64-byte aligned"
            );
        }
    }

    #[test]
    fn f32_loading_identical_within_tolerance() {
        // Verify the aligned path produces values identical to the old bytes_to_f32 path
        let original: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001 - 0.5).collect();
        let bytes = f32_to_bytes(&original);

        let file_data = make_safetensors(&[("w", "F32", &[1024], &bytes)]);
        let tensors = parse_safetensors(&file_data).unwrap();
        let loaded = &tensors["w"];

        for (i, (&got, &want)) in loaded.data.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-7,
                "Mismatch at index {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn parse_safetensors_index_json() {
        let index_json = r#"{
            "metadata": { "total_size": 12345 },
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00002.safetensors"
            }
        }"#;

        let index: SafetensorsIndex = serde_json::from_str(index_json).unwrap();
        assert_eq!(index.weight_map.len(), 3);
        assert_eq!(
            index.weight_map["model.embed_tokens.weight"],
            "model-00001-of-00002.safetensors"
        );
        assert_eq!(
            index.weight_map["model.layers.1.self_attn.q_proj.weight"],
            "model-00002-of-00002.safetensors"
        );
    }

    #[test]
    fn load_sharded_safetensors_from_directory() {
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();

        // Create shard 1 with two tensors
        let w1: Vec<f32> = vec![1.0, 2.0, 3.0];
        let w2: Vec<f32> = vec![4.0, 5.0];
        let shard1 = make_safetensors(&[
            ("tensor_a", "F32", &[3], &f32_to_bytes(&w1)),
            ("tensor_b", "F32", &[2], &f32_to_bytes(&w2)),
        ]);
        std::fs::write(dir.path().join("model-00001-of-00002.safetensors"), &shard1).unwrap();

        // Create shard 2 with one tensor (plus an extra tensor NOT in the index)
        let w3: Vec<f32> = vec![6.0, 7.0, 8.0, 9.0];
        let w_extra: Vec<f32> = vec![99.0];
        let shard2 = make_safetensors(&[
            ("tensor_c", "F32", &[4], &f32_to_bytes(&w3)),
            ("extra_tensor", "F32", &[1], &f32_to_bytes(&w_extra)),
        ]);
        std::fs::write(dir.path().join("model-00002-of-00002.safetensors"), &shard2).unwrap();

        // Create index file — only references tensor_a, tensor_b, tensor_c (not extra_tensor)
        let index = serde_json::json!({
            "metadata": { "total_size": 100 },
            "weight_map": {
                "tensor_a": "model-00001-of-00002.safetensors",
                "tensor_b": "model-00001-of-00002.safetensors",
                "tensor_c": "model-00002-of-00002.safetensors"
            }
        });
        let mut f = File::create(dir.path().join("model.safetensors.index.json")).unwrap();
        write!(f, "{}", serde_json::to_string(&index).unwrap()).unwrap();

        let tensors = load_safetensors_sharded(dir.path()).unwrap();

        // Should contain exactly 3 tensors (extra_tensor excluded)
        assert_eq!(tensors.len(), 3);
        assert_eq!(tensors["tensor_a"].data, w1);
        assert_eq!(tensors["tensor_a"].shape, vec![3]);
        assert_eq!(tensors["tensor_b"].data, w2);
        assert_eq!(tensors["tensor_b"].shape, vec![2]);
        assert_eq!(tensors["tensor_c"].data, w3);
        assert_eq!(tensors["tensor_c"].shape, vec![4]);
        assert!(!tensors.contains_key("extra_tensor"));
    }

    #[test]
    fn sharded_loading_missing_shard_returns_error() {
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();

        // Create index that references a shard that does not exist
        let index = serde_json::json!({
            "metadata": { "total_size": 100 },
            "weight_map": {
                "tensor_a": "missing-shard.safetensors"
            }
        });
        let mut f = File::create(dir.path().join("model.safetensors.index.json")).unwrap();
        write!(f, "{}", serde_json::to_string(&index).unwrap()).unwrap();

        let result = load_safetensors_sharded(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn sharded_loading_missing_tensor_in_shard_returns_error() {
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();

        // Create a shard that contains "actual_tensor" but not "claimed_tensor"
        let w: Vec<f32> = vec![1.0];
        let shard = make_safetensors(&[("actual_tensor", "F32", &[1], &f32_to_bytes(&w))]);
        std::fs::write(dir.path().join("shard.safetensors"), &shard).unwrap();

        // Index claims "claimed_tensor" is in the shard
        let index = serde_json::json!({
            "weight_map": {
                "claimed_tensor": "shard.safetensors"
            }
        });
        let mut f = File::create(dir.path().join("model.safetensors.index.json")).unwrap();
        write!(f, "{}", serde_json::to_string(&index).unwrap()).unwrap();

        let result = load_safetensors_sharded(dir.path());
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("claimed_tensor"));
    }
}
