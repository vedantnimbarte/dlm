//! Minimal, zero-copy parser for the `safetensors` on-disk format.
//!
//! Layout:
//! ```text
//!  ┌────────────┬──────────────────────────┬───────────────────────┐
//!  │ u64 LE (8) │ JSON header (header_len)  │ tensor data (rest)    │
//!  └────────────┴──────────────────────────┴───────────────────────┘
//! ```
//! The JSON header maps each tensor name to its dtype, shape, and a
//! `[begin, end)` byte range **relative to the start of the data section**.
//! We parse only the header here; the actual bytes are served as slices
//! straight out of the memory map (see [`crate::storage::MmapStore`]).

use crate::error::{FlipError, Result};
use serde::Deserialize;
use std::collections::BTreeMap;

/// The 8-byte little-endian length prefix that precedes the JSON header.
pub const HEADER_LEN_PREFIX: usize = 8;

/// Element data types defined by the safetensors spec (subset in common use).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    U8,
    I8,
    F8E4M3,
    F8E5M2,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    I64,
    U64,
    F64,
}

impl Dtype {
    /// Bytes occupied by a single element of this dtype.
    pub fn size_in_bytes(self) -> usize {
        match self {
            Dtype::Bool | Dtype::U8 | Dtype::I8 | Dtype::F8E4M3 | Dtype::F8E5M2 => 1,
            Dtype::I16 | Dtype::U16 | Dtype::F16 | Dtype::BF16 => 2,
            Dtype::I32 | Dtype::U32 | Dtype::F32 => 4,
            Dtype::I64 | Dtype::U64 | Dtype::F64 => 8,
        }
    }

    fn from_tag(tag: &str) -> Result<Self> {
        Ok(match tag {
            "BOOL" => Dtype::Bool,
            "U8" => Dtype::U8,
            "I8" => Dtype::I8,
            "F8_E4M3" => Dtype::F8E4M3,
            "F8_E5M2" => Dtype::F8E5M2,
            "I16" => Dtype::I16,
            "U16" => Dtype::U16,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "I32" => Dtype::I32,
            "U32" => Dtype::U32,
            "F32" => Dtype::F32,
            "I64" => Dtype::I64,
            "U64" => Dtype::U64,
            "F64" => Dtype::F64,
            other => {
                return Err(FlipError::SafetensorsHeader(format!(
                    "unsupported dtype tag {other:?}"
                )))
            }
        })
    }
}

/// Raw per-tensor record as it appears in the JSON header.
#[derive(Debug, Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// Convert a half-precision (IEEE 754 binary16) bit pattern to `f32`.
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val = if exp == 0 {
        // Zero or subnormal.
        (mant as f32) * 2f32.powi(-24)
    } else if exp == 0x1f {
        if mant == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + mant as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

/// Decode a tensor's raw little-endian bytes into `f32` elements.
///
/// Supports the float dtypes small checkpoints ship in — F32, F16, BF16. Integer
/// and quantized dtypes are rejected (those go through the dequant path).
pub fn bytes_to_f32(bytes: &[u8], dtype: Dtype) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            if bytes.len() % 4 != 0 {
                return Err(FlipError::SafetensorsHeader(format!(
                    "F32 tensor byte length {} is not a multiple of 4",
                    bytes.len()
                )));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        Dtype::F16 => {
            if bytes.len() % 2 != 0 {
                return Err(FlipError::SafetensorsHeader(format!(
                    "F16 tensor byte length {} is not a multiple of 2",
                    bytes.len()
                )));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        Dtype::BF16 => {
            if bytes.len() % 2 != 0 {
                return Err(FlipError::SafetensorsHeader(format!(
                    "BF16 tensor byte length {} is not a multiple of 2",
                    bytes.len()
                )));
            }
            // bfloat16 is the high 16 bits of an f32.
            Ok(bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect())
        }
        other => Err(FlipError::InvalidConfig(format!(
            "cannot convert dtype {other:?} to f32 (use the dequant path)"
        ))),
    }
}

/// Parsed metadata for a single tensor. Byte offsets are relative to the start
/// of the data section (i.e. after the 8-byte prefix and JSON header).
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Inclusive start offset into the data section.
    pub begin: usize,
    /// Exclusive end offset into the data section.
    pub end: usize,
}

impl TensorInfo {
    /// Number of bytes this tensor occupies on disk.
    pub fn byte_len(&self) -> usize {
        self.end - self.begin
    }

    /// Total element count (product of the shape dims).
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// The fully parsed safetensors header: an index of tensors plus the offset at
/// which the data section begins in the underlying file/map.
#[derive(Debug, Clone)]
pub struct SafetensorsHeader {
    /// Absolute byte offset where tensor data starts in the file.
    pub data_offset: usize,
    /// Tensors keyed by name, sorted for stable iteration.
    pub tensors: BTreeMap<String, TensorInfo>,
    /// Free-form `__metadata__` string map, if present.
    pub metadata: BTreeMap<String, String>,
}

impl SafetensorsHeader {
    /// Parse the header from the leading bytes of a safetensors file.
    ///
    /// `file_len` is the total file size, used to validate that declared tensor
    /// ranges actually fit within the data section.
    pub fn parse(bytes: &[u8], file_len: usize) -> Result<Self> {
        if bytes.len() < HEADER_LEN_PREFIX {
            return Err(FlipError::SafetensorsHeader(format!(
                "file too small ({} bytes) to contain an 8-byte header prefix",
                bytes.len()
            )));
        }

        let header_len =
            u64::from_le_bytes(bytes[..HEADER_LEN_PREFIX].try_into().unwrap()) as usize;

        let data_offset = HEADER_LEN_PREFIX
            .checked_add(header_len)
            .ok_or_else(|| FlipError::SafetensorsHeader("header length overflow".into()))?;

        if data_offset > file_len {
            return Err(FlipError::SafetensorsHeader(format!(
                "declared header length {header_len} runs past end of file ({file_len} bytes)"
            )));
        }
        if bytes.len() < data_offset {
            return Err(FlipError::SafetensorsHeader(
                "provided prefix is shorter than the declared header".into(),
            ));
        }

        let json = &bytes[HEADER_LEN_PREFIX..data_offset];
        let raw: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(json).map_err(|source| FlipError::Json {
                context: "safetensors header".to_string(),
                source,
            })?;

        let data_section_len = file_len - data_offset;
        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();

        for (name, value) in raw {
            if name == "__metadata__" {
                metadata = serde_json::from_value(value).map_err(|source| FlipError::Json {
                    context: "safetensors __metadata__".to_string(),
                    source,
                })?;
                continue;
            }

            let raw_tensor: RawTensor =
                serde_json::from_value(value).map_err(|source| FlipError::Json {
                    context: format!("safetensors tensor {name:?}"),
                    source,
                })?;

            let dtype = Dtype::from_tag(&raw_tensor.dtype)?;
            let [begin, end] = raw_tensor.data_offsets;

            if begin > end || end > data_section_len {
                return Err(FlipError::TensorOutOfBounds {
                    name,
                    start: begin,
                    end,
                    len: data_section_len,
                });
            }

            // Cross-check the declared byte range against dtype * shape so a
            // corrupt header can't hand out a mis-sized slice downstream.
            let expected = raw_tensor.shape.iter().product::<usize>() * dtype.size_in_bytes();
            if expected != end - begin {
                return Err(FlipError::SafetensorsHeader(format!(
                    "tensor {name:?}: shape/dtype implies {expected} bytes but range spans {}",
                    end - begin
                )));
            }

            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dtype,
                    shape: raw_tensor.shape,
                    begin,
                    end,
                },
            );
        }

        Ok(SafetensorsHeader {
            data_offset,
            tensors,
            metadata,
        })
    }
}
