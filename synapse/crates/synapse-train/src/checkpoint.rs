use std::collections::BTreeMap;
use std::io::{self, Cursor, Read, Write};

const MAGIC: &[u8; 4] = b"SYNP";
const VERSION: u32 = 1;

/// Model state dictionary: parameter name -> (shape, data).
pub type StateDict = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

/// Serialize a state dict to a writer in a binary format.
///
/// Format:
/// - 4 bytes: magic "SYNP"
/// - 4 bytes: version (u32 LE)
/// - 4 bytes: number of parameters (u32 LE)
/// - For each parameter (sorted by name):
///   - 4 bytes: name length (u32 LE)
///   - N bytes: name (UTF-8)
///   - 4 bytes: number of dimensions (u32 LE)
///   - 4*ndim bytes: shape (u32 LE each)
///   - 4*numel bytes: data (f32 LE each)
pub fn save_checkpoint(state: &StateDict, writer: &mut dyn Write) -> io::Result<()> {
    writer.write_all(MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    writer.write_all(&(state.len() as u32).to_le_bytes())?;

    for (name, (shape, data)) in state {
        let name_bytes = name.as_bytes();
        writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(name_bytes)?;

        writer.write_all(&(shape.len() as u32).to_le_bytes())?;
        for &dim in shape {
            writer.write_all(&(dim as u32).to_le_bytes())?;
        }

        for &val in data {
            writer.write_all(&val.to_le_bytes())?;
        }
    }

    Ok(())
}

/// Deserialize a state dict from a reader.
pub fn load_checkpoint(reader: &mut dyn Read) -> io::Result<StateDict> {
    let mut buf4 = [0u8; 4];

    reader.read_exact(&mut buf4)?;
    if &buf4 != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid checkpoint magic number",
        ));
    }

    reader.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported checkpoint version: {}", version),
        ));
    }

    reader.read_exact(&mut buf4)?;
    let num_params = u32::from_le_bytes(buf4) as usize;

    let mut state = BTreeMap::new();

    for _ in 0..num_params {
        reader.read_exact(&mut buf4)?;
        let name_len = u32::from_le_bytes(buf4) as usize;
        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        reader.read_exact(&mut buf4)?;
        let ndim = u32::from_le_bytes(buf4) as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            reader.read_exact(&mut buf4)?;
            shape.push(u32::from_le_bytes(buf4) as usize);
        }

        let numel: usize = shape.iter().product();
        let mut data = Vec::with_capacity(numel);
        for _ in 0..numel {
            reader.read_exact(&mut buf4)?;
            data.push(f32::from_le_bytes(buf4));
        }

        state.insert(name, (shape, data));
    }

    Ok(state)
}

/// Convenience: serialize to a byte vector.
pub fn save_to_bytes(state: &StateDict) -> Vec<u8> {
    let mut buf = Vec::new();
    save_checkpoint(state, &mut buf).expect("writing to Vec should not fail");
    buf
}

/// Convenience: deserialize from a byte slice.
pub fn load_from_bytes(bytes: &[u8]) -> io::Result<StateDict> {
    load_checkpoint(&mut Cursor::new(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_roundtrip_bit_exact() {
        let mut state = StateDict::new();
        state.insert(
            "layer1.weight".to_string(),
            (vec![3, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        );
        state.insert(
            "layer1.bias".to_string(),
            (vec![3], vec![0.1, 0.2, 0.3]),
        );
        state.insert(
            "layer2.weight".to_string(),
            (vec![2, 3], vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]),
        );

        let bytes = save_to_bytes(&state);
        let loaded = load_from_bytes(&bytes).unwrap();

        assert_eq!(state.len(), loaded.len());
        for (name, (shape, data)) in &state {
            let (loaded_shape, loaded_data) = loaded.get(name).unwrap();
            assert_eq!(shape, loaded_shape, "shape mismatch for {}", name);
            assert_eq!(data.len(), loaded_data.len(), "data len mismatch for {}", name);
            for (i, (&a, &b)) in data.iter().zip(loaded_data.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "bit mismatch at {}[{}]: {} vs {}",
                    name,
                    i,
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn checkpoint_special_values() {
        let mut state = StateDict::new();
        state.insert(
            "special".to_string(),
            (
                vec![5],
                vec![0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, 1.23456789e-30],
            ),
        );

        let bytes = save_to_bytes(&state);
        let loaded = load_from_bytes(&bytes).unwrap();
        let (_, loaded_data) = loaded.get("special").unwrap();

        for (i, (&a, &b)) in state["special"]
            .1
            .iter()
            .zip(loaded_data.iter())
            .enumerate()
        {
            assert_eq!(a.to_bits(), b.to_bits(), "bit mismatch at index {}", i);
        }
    }

    #[test]
    fn checkpoint_invalid_magic() {
        let bytes = b"BADM\x01\x00\x00\x00\x00\x00\x00\x00";
        let result = load_from_bytes(bytes);
        assert!(result.is_err());
    }
}
