use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use super::converter::f16_to_f32;
use super::{AlignedBuffer, RawTensor, WeightError};
use memmap2::Mmap;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

// GGML tensor types
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q4_1: u32 = 3;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q6_K: u32 = 14;

// Block sizes
const QK8_0: usize = 32; // Q8_0: 32 elements per block
const QK4_0: usize = 32; // Q4_0: 32 elements per block
const QK4_1: usize = 32; // Q4_1: 32 elements per block
const QK_K: usize = 256; // K-quant super-block: 256 elements

// GGUF metadata value types
const GGUF_META_UINT8: u32 = 0;
const GGUF_META_INT8: u32 = 1;
const GGUF_META_UINT16: u32 = 2;
const GGUF_META_INT16: u32 = 3;
const GGUF_META_UINT32: u32 = 4;
const GGUF_META_INT32: u32 = 5;
const GGUF_META_FLOAT32: u32 = 6;
const GGUF_META_BOOL: u32 = 7;
const GGUF_META_STRING: u32 = 8;
const GGUF_META_ARRAY: u32 = 9;
const GGUF_META_UINT64: u32 = 10;
const GGUF_META_INT64: u32 = 11;
const GGUF_META_FLOAT64: u32 = 12;

/// Load tensors from a GGUF file, returning raw tensors ready for model loading.
pub fn load_gguf(path: &Path) -> Result<HashMap<String, RawTensor>, WeightError> {
    let file = File::open(path).map_err(WeightError::Io)?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(WeightError::Io)?;
    parse_gguf(&mmap)
}

/// Parse GGUF from a byte slice into raw f32 data.
///
/// Supports GGUF v3. Tensor types: F32, F16, Q8_0.
pub fn parse_gguf(data: &[u8]) -> Result<HashMap<String, RawTensor>, WeightError> {
    let mut cur = Cursor::new(data);

    // Header
    let magic = cur.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(WeightError::InvalidFormat(format!(
            "Bad GGUF magic: 0x{magic:08X}"
        )));
    }

    let version = cur.read_u32()?;
    if version < 2 || version > 3 {
        return Err(WeightError::InvalidFormat(format!(
            "Unsupported GGUF version: {version}"
        )));
    }

    let tensor_count = cur.read_u64()? as usize;
    let metadata_kv_count = cur.read_u64()? as usize;

    // Skip metadata KV pairs
    for _ in 0..metadata_kv_count {
        skip_metadata_kv(&mut cur)?;
    }

    // Read tensor info entries
    struct TensorInfo {
        name: String,
        shape: Vec<usize>,
        dtype: u32,
        offset: usize,
    }

    let mut tensor_infos = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = cur.read_string()?;
        let n_dims = cur.read_u32()? as usize;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(cur.read_u64()? as usize);
        }
        let dtype = cur.read_u32()?;
        let offset = cur.read_u64()? as usize;
        tensor_infos.push(TensorInfo {
            name,
            shape,
            dtype,
            offset,
        });
    }

    // Data section starts at next alignment boundary
    let data_start = align_up(cur.pos, GGUF_DEFAULT_ALIGNMENT);

    // Parse tensor data
    let mut result = HashMap::new();
    for info in tensor_infos {
        let numel: usize = info.shape.iter().product();
        let tensor_data_offset = data_start + info.offset;

        let f32_data = match info.dtype {
            GGML_TYPE_F32 => {
                let byte_len = numel * 4;
                let bytes = &data[tensor_data_offset..tensor_data_offset + byte_len];
                bytes_to_f32(bytes)
            }
            GGML_TYPE_F16 => {
                let byte_len = numel * 2;
                let bytes = &data[tensor_data_offset..tensor_data_offset + byte_len];
                let u16_data = bytes_to_u16(bytes);
                f16_to_f32(&u16_data)
            }
            GGML_TYPE_Q4_0 => dequantize_q4_0(&data[tensor_data_offset..], numel)?,
            GGML_TYPE_Q4_1 => dequantize_q4_1(&data[tensor_data_offset..], numel)?,
            GGML_TYPE_Q8_0 => dequantize_q8_0(&data[tensor_data_offset..], numel)?,
            GGML_TYPE_Q4_K => dequantize_q4_k(&data[tensor_data_offset..], numel)?,
            GGML_TYPE_Q6_K => dequantize_q6_k(&data[tensor_data_offset..], numel)?,
            other => {
                return Err(WeightError::UnsupportedDtype(format!("GGML type {other}")));
            }
        };

        result.insert(
            info.name,
            RawTensor {
                data: AlignedBuffer::from_slice(&f32_data),
                shape: info.shape,
            },
        );
    }

    Ok(result)
}

/// Dequantize Q8_0 data to f32.
///
/// Each block: `[f16 scale (2 bytes)][32 × i8 quants (32 bytes)]` = 34 bytes.
/// Output: `scale * quant` for each element.
fn dequantize_q8_0(data: &[u8], numel: usize) -> Result<Vec<f32>, WeightError> {
    if numel % QK8_0 != 0 {
        return Err(WeightError::InvalidFormat(format!(
            "Q8_0 element count {numel} not a multiple of block size {QK8_0}"
        )));
    }

    let n_blocks = numel / QK8_0;
    let block_size = 2 + QK8_0; // f16 scale + 32 i8 quants
    let required = n_blocks * block_size;

    if data.len() < required {
        return Err(WeightError::InvalidFormat("Q8_0 data too short".into()));
    }

    let mut output = Vec::with_capacity(numel);

    for block_idx in 0..n_blocks {
        let block_start = block_idx * block_size;

        // Read f16 scale
        let scale_bits = u16::from_le_bytes(data[block_start..block_start + 2].try_into().unwrap());
        let scale = super::converter::f16_to_f32(&[scale_bits])[0];

        // Read and dequantize i8 quants
        let quant_start = block_start + 2;
        for i in 0..QK8_0 {
            let q = data[quant_start + i] as i8;
            output.push(scale * q as f32);
        }
    }

    Ok(output)
}

/// Dequantize Q4_0 data to f32.
///
/// Each block (32 elements): `[f16 delta (2 bytes)][16 bytes packed nibbles]` = 18 bytes.
/// Each byte holds 2 nibbles (low nibble first). Value = (nibble - 8) * delta.
fn dequantize_q4_0(data: &[u8], numel: usize) -> Result<Vec<f32>, WeightError> {
    if numel % QK4_0 != 0 {
        return Err(WeightError::InvalidFormat(format!(
            "Q4_0 element count {numel} not a multiple of {QK4_0}"
        )));
    }

    let n_blocks = numel / QK4_0;
    let block_size = 2 + QK4_0 / 2; // f16 delta + 16 bytes nibbles
    let required = n_blocks * block_size;
    if data.len() < required {
        return Err(WeightError::InvalidFormat("Q4_0 data too short".into()));
    }

    let mut output = Vec::with_capacity(numel);

    for block_idx in 0..n_blocks {
        let bs = block_idx * block_size;
        let delta_bits = u16::from_le_bytes(data[bs..bs + 2].try_into().unwrap());
        let delta = super::converter::f16_to_f32(&[delta_bits])[0];
        let nibble_start = bs + 2;

        for i in 0..QK4_0 / 2 {
            let byte = data[nibble_start + i];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            output.push(delta * lo as f32);
            output.push(delta * hi as f32);
        }
    }

    Ok(output)
}

/// Dequantize Q4_1 data to f32.
///
/// Each block (32 elements): `[f16 delta][f16 min][16 bytes packed nibbles]` = 20 bytes.
/// Value = nibble * delta + min.
fn dequantize_q4_1(data: &[u8], numel: usize) -> Result<Vec<f32>, WeightError> {
    if numel % QK4_1 != 0 {
        return Err(WeightError::InvalidFormat(format!(
            "Q4_1 element count {numel} not a multiple of {QK4_1}"
        )));
    }

    let n_blocks = numel / QK4_1;
    let block_size = 4 + QK4_1 / 2; // f16 delta + f16 min + 16 bytes nibbles
    let required = n_blocks * block_size;
    if data.len() < required {
        return Err(WeightError::InvalidFormat("Q4_1 data too short".into()));
    }

    let mut output = Vec::with_capacity(numel);

    for block_idx in 0..n_blocks {
        let bs = block_idx * block_size;
        let delta_bits = u16::from_le_bytes(data[bs..bs + 2].try_into().unwrap());
        let min_bits = u16::from_le_bytes(data[bs + 2..bs + 4].try_into().unwrap());
        let delta = super::converter::f16_to_f32(&[delta_bits])[0];
        let min = super::converter::f16_to_f32(&[min_bits])[0];
        let nibble_start = bs + 4;

        for i in 0..QK4_1 / 2 {
            let byte = data[nibble_start + i];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            output.push(lo * delta + min);
            output.push(hi * delta + min);
        }
    }

    Ok(output)
}

/// Dequantize Q4_K data to f32 (k-quant, 256-element super-blocks).
///
/// Super-block layout (144 bytes for 256 elements):
/// - f16 d (2 bytes): super-block scale
/// - f16 dmin (2 bytes): super-block minimum
/// - 12 bytes: packed 6-bit scales and mins for 8 sub-blocks
/// - 128 bytes: packed 4-bit nibbles (2 per byte)
fn dequantize_q4_k(data: &[u8], numel: usize) -> Result<Vec<f32>, WeightError> {
    if numel % QK_K != 0 {
        return Err(WeightError::InvalidFormat(format!(
            "Q4_K element count {numel} not a multiple of {QK_K}"
        )));
    }

    let n_blocks = numel / QK_K;
    let block_size = 144; // 2 + 2 + 12 + 128
    let required = n_blocks * block_size;
    if data.len() < required {
        return Err(WeightError::InvalidFormat("Q4_K data too short".into()));
    }

    let mut output = Vec::with_capacity(numel);

    for block_idx in 0..n_blocks {
        let bs = block_idx * block_size;

        // Super-block scale and min
        let d_bits = u16::from_le_bytes(data[bs..bs + 2].try_into().unwrap());
        let dmin_bits = u16::from_le_bytes(data[bs + 2..bs + 4].try_into().unwrap());
        let d = super::converter::f16_to_f32(&[d_bits])[0];
        let dmin = super::converter::f16_to_f32(&[dmin_bits])[0];

        // Unpack 6-bit scales and mins for 8 sub-blocks from 12 bytes
        let scales_bytes = &data[bs + 4..bs + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        for i in 0..4 {
            scales[i] = scales_bytes[i] & 0x3F;
            scales[i + 4] = (scales_bytes[i + 4] & 0x0F) | ((scales_bytes[i] >> 6) << 4);
            mins[i] = scales_bytes[i + 4] >> 4 | ((scales_bytes[i + 8] & 0x0F) << 4);
            mins[i + 4] = scales_bytes[i + 8] >> 4 | ((scales_bytes[i] >> 4 & 0x0C) << 4);
        }

        // Nibble data: 128 bytes for 256 elements (2 per byte)
        let nibble_start = bs + 16;

        for sub_block in 0..8 {
            let sc = d * scales[sub_block] as f32;
            let m = dmin * mins[sub_block] as f32;
            let nibble_off = nibble_start + sub_block * 16;

            for i in 0..16 {
                let byte = data[nibble_off + i];
                let lo = (byte & 0x0F) as f32;
                let hi = ((byte >> 4) & 0x0F) as f32;
                output.push(lo * sc - m);
                output.push(hi * sc - m);
            }
        }
    }

    Ok(output)
}

/// Dequantize Q6_K data to f32 (k-quant 6-bit, 256-element super-blocks).
///
/// Super-block layout (210 bytes for 256 elements):
/// - 128 bytes: low 4 bits of quantized values (packed 2 per byte)
/// - 64 bytes: high 2 bits of quantized values (packed 4 per byte)
/// - 16 bytes: int8 scales per 16-element sub-block
/// - f16 d (2 bytes): super-block scale
fn dequantize_q6_k(data: &[u8], numel: usize) -> Result<Vec<f32>, WeightError> {
    if numel % QK_K != 0 {
        return Err(WeightError::InvalidFormat(format!(
            "Q6_K element count {numel} not a multiple of {QK_K}"
        )));
    }

    let n_blocks = numel / QK_K;
    let block_size = 210; // 128 + 64 + 16 + 2
    let required = n_blocks * block_size;
    if data.len() < required {
        return Err(WeightError::InvalidFormat("Q6_K data too short".into()));
    }

    let mut output = Vec::with_capacity(numel);

    for block_idx in 0..n_blocks {
        let bs = block_idx * block_size;

        let ql = &data[bs..bs + 128]; // low 4 bits
        let qh = &data[bs + 128..bs + 192]; // high 2 bits
        let sc = &data[bs + 192..bs + 208]; // int8 scales
        let d_bits = u16::from_le_bytes(data[bs + 208..bs + 210].try_into().unwrap());
        let d = super::converter::f16_to_f32(&[d_bits])[0];

        for sub in 0..16 {
            let scale = sc[sub] as i8 as f32 * d;
            for i in 0..16 {
                let idx = sub * 16 + i;
                // Low 4 bits: packed 2 per byte in ql
                let ql_byte = ql[idx / 2];
                let q_lo = if idx % 2 == 0 {
                    ql_byte & 0x0F
                } else {
                    ql_byte >> 4
                };
                // High 2 bits: packed 4 per byte in qh
                let qh_byte = qh[idx / 4];
                let q_hi = (qh_byte >> ((idx % 4) * 2)) & 0x03;
                // Combine: 6-bit value = lo | (hi << 4), centered at 32
                let q = ((q_lo | (q_hi << 4)) as i32 - 32) as f32;
                output.push(q * scale);
            }
        }
    }

    Ok(output)
}

fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
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

// ── Cursor for reading GGUF binary data ──────────────────────────

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], WeightError> {
        if self.remaining() < n {
            return Err(WeightError::InvalidFormat("Unexpected EOF".into()));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u32(&mut self) -> Result<u32, WeightError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, WeightError> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_string(&mut self) -> Result<String, WeightError> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| WeightError::InvalidFormat(format!("Invalid UTF-8: {e}")))
    }
}

/// Skip a single metadata KV pair.
fn skip_metadata_kv(cur: &mut Cursor) -> Result<(), WeightError> {
    // Skip key (string)
    let key_len = cur.read_u64()? as usize;
    cur.read_bytes(key_len)?;
    // Skip value
    let value_type = cur.read_u32()?;
    skip_metadata_value(cur, value_type)
}

/// Skip a metadata value based on its type.
fn skip_metadata_value(cur: &mut Cursor, vtype: u32) -> Result<(), WeightError> {
    match vtype {
        GGUF_META_UINT8 | GGUF_META_INT8 | GGUF_META_BOOL => {
            cur.read_bytes(1)?;
        }
        GGUF_META_UINT16 | GGUF_META_INT16 => {
            cur.read_bytes(2)?;
        }
        GGUF_META_UINT32 | GGUF_META_INT32 | GGUF_META_FLOAT32 => {
            cur.read_bytes(4)?;
        }
        GGUF_META_UINT64 | GGUF_META_INT64 | GGUF_META_FLOAT64 => {
            cur.read_bytes(8)?;
        }
        GGUF_META_STRING => {
            let len = cur.read_u64()? as usize;
            cur.read_bytes(len)?;
        }
        GGUF_META_ARRAY => {
            let elem_type = cur.read_u32()?;
            let count = cur.read_u64()? as usize;
            for _ in 0..count {
                skip_metadata_value(cur, elem_type)?;
            }
        }
        other => {
            return Err(WeightError::InvalidFormat(format!(
                "Unknown GGUF metadata type: {other}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: write a GGUF string (u64 length + bytes).
    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// Build a minimal GGUF v3 binary with the given F32 tensors.
    fn make_gguf_f32(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); // tensor_count
        buf.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

        // Tensor info entries — compute offsets
        let mut current_data_offset: usize = 0;
        let mut tensor_data_parts: Vec<Vec<u8>> = Vec::new();

        for &(name, shape, values) in tensors {
            write_gguf_string(&mut buf, name);
            buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for &dim in shape {
                buf.extend_from_slice(&(dim as u64).to_le_bytes());
            }
            buf.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
            buf.extend_from_slice(&(current_data_offset as u64).to_le_bytes());

            let data_bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
            current_data_offset += data_bytes.len();
            tensor_data_parts.push(data_bytes);
        }

        // Pad to alignment
        let aligned = align_up(buf.len(), GGUF_DEFAULT_ALIGNMENT);
        buf.resize(aligned, 0);

        // Tensor data
        for part in &tensor_data_parts {
            buf.extend_from_slice(part);
        }

        buf
    }

    /// Build a GGUF v3 binary with a Q8_0 tensor.
    fn make_gguf_q8_0(name: &str, shape: &[usize], scale_bits: u16, quants: &[i8]) -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

        // Tensor info
        write_gguf_string(&mut buf, name);
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &dim in shape {
            buf.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        buf.extend_from_slice(&GGML_TYPE_Q8_0.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset 0

        // Pad to alignment
        let aligned = align_up(buf.len(), GGUF_DEFAULT_ALIGNMENT);
        buf.resize(aligned, 0);

        // Q8_0 block data: [f16 scale][32 × i8 quants]
        let n_blocks = quants.len() / QK8_0;
        for block in 0..n_blocks {
            buf.extend_from_slice(&scale_bits.to_le_bytes());
            for i in 0..QK8_0 {
                buf.push(quants[block * QK8_0 + i] as u8);
            }
        }

        buf
    }

    #[test]
    fn load_two_f32_tensors_bit_exact() {
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0];

        let gguf_bytes =
            make_gguf_f32(&[("tensor_a", &[2, 3], &a_data), ("tensor_b", &[4], &b_data)]);

        let tensors = parse_gguf(&gguf_bytes).unwrap();
        assert_eq!(tensors.len(), 2);

        let a = &tensors["tensor_a"];
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.data, a_data, "tensor_a must be bit-exact");

        let b = &tensors["tensor_b"];
        assert_eq!(b.shape, vec![4]);
        assert_eq!(b.data, b_data, "tensor_b must be bit-exact");
    }

    #[test]
    fn load_q8_0_tensor_within_tolerance() {
        // 32 elements = 1 Q8_0 block
        // scale = 0.5 (f16 0x3800)
        let scale_bits: u16 = 0x3800;
        let quants: Vec<i8> = (1..=32).map(|i| i as i8).collect();
        let expected: Vec<f32> = quants.iter().map(|&q| 0.5 * q as f32).collect();

        let gguf_bytes = make_gguf_q8_0("q8_tensor", &[32], scale_bits, &quants);
        let tensors = parse_gguf(&gguf_bytes).unwrap();

        let t = &tensors["q8_tensor"];
        assert_eq!(t.shape, vec![32]);
        assert_eq!(t.data.len(), 32);

        for (i, (&got, &want)) in t.data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "Q8_0 mismatch at [{i}]: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn bad_magic_returns_error() {
        let mut data = vec![0u8; 32];
        // Write wrong magic
        data[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        let result = parse_gguf(&data);
        assert!(matches!(result, Err(WeightError::InvalidFormat(_))));
    }

    #[test]
    fn gguf_with_metadata_skipped() {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        buf.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata KV

        // Metadata: key="general.name", value_type=STRING, value="test"
        write_gguf_string(&mut buf, "general.name");
        buf.extend_from_slice(&GGUF_META_STRING.to_le_bytes());
        write_gguf_string(&mut buf, "test");

        // Tensor info: "w" shape [2] F32
        write_gguf_string(&mut buf, "w");
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&2u64.to_le_bytes()); // dim 0
        buf.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Pad to alignment
        let aligned = align_up(buf.len(), GGUF_DEFAULT_ALIGNMENT);
        buf.resize(aligned, 0);

        // Tensor data
        buf.extend_from_slice(&42.0f32.to_le_bytes());
        buf.extend_from_slice(&43.0f32.to_le_bytes());

        let tensors = parse_gguf(&buf).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors["w"].data, vec![42.0, 43.0]);
    }

    /// Build a GGUF with a single tensor of the given type and raw block data.
    fn make_gguf_raw(name: &str, shape: &[usize], dtype: u32, block_data: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        write_gguf_string(&mut buf, name);
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &dim in shape {
            buf.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        buf.extend_from_slice(&dtype.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let aligned = align_up(buf.len(), GGUF_DEFAULT_ALIGNMENT);
        buf.resize(aligned, 0);
        buf.extend_from_slice(block_data);
        buf
    }

    #[test]
    fn load_q4_0_tensor() {
        // 32 elements = 1 Q4_0 block: f16 delta + 16 bytes nibbles
        // delta = 1.0 (f16 0x3C00)
        // nibbles: each byte has lo=0, hi=1 → values (0-8)*1.0=-8, (1-8)*1.0=-7
        let mut block_data = Vec::new();
        block_data.extend_from_slice(&0x3C00u16.to_le_bytes()); // delta = 1.0
        for _ in 0..16 {
            block_data.push(0x10); // lo=0, hi=1
        }

        let gguf = make_gguf_raw("q4_tensor", &[32], GGML_TYPE_Q4_0, &block_data);
        let tensors = parse_gguf(&gguf).unwrap();
        let t = &tensors["q4_tensor"];
        assert_eq!(t.shape, vec![32]);
        assert_eq!(t.data.len(), 32);

        // lo nibble = 0: (0-8)*1.0 = -8.0
        // hi nibble = 1: (1-8)*1.0 = -7.0
        for i in 0..16 {
            assert!(
                (t.data[i * 2] - (-8.0)).abs() < 1e-3,
                "Q4_0 lo [{i}]: got {}, want -8.0",
                t.data[i * 2]
            );
            assert!(
                (t.data[i * 2 + 1] - (-7.0)).abs() < 1e-3,
                "Q4_0 hi [{i}]: got {}, want -7.0",
                t.data[i * 2 + 1]
            );
        }
    }

    #[test]
    fn unsupported_ggml_type_returns_error() {
        // Type 99 doesn't exist
        let block_data = vec![0u8; 64];
        let gguf = make_gguf_raw("bad", &[32], 99, &block_data);
        let result = parse_gguf(&gguf);
        assert!(matches!(result, Err(WeightError::UnsupportedDtype(_))));
    }
}
