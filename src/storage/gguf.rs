//! GGUF: the container llama.cpp models ship in.
//!
//! A GGUF file is a header of typed key/value metadata, then a table of tensors,
//! then the tensor data, aligned. Everything dlm reads from a `config.json` and
//! a set of safetensors shards is in that one file instead — which is the point:
//! it is what a user downloads when they go looking for a model to run locally.
//!
//! This module reads the header into the same [`SafetensorsHeader`] the rest of
//! the loader already consumes, translating llama.cpp's tensor names into the
//! Hugging Face ones dlm's loader expects, so a GGUF checkpoint travels every
//! path a safetensors one does — including layer streaming, since a layer is
//! still just a byte range in a mapped file.
//!
//! The weights themselves are block-quantized; [`crate::storage::ggml_quant`]
//! decodes those blocks.

use crate::error::{DlmError, Result};
use crate::storage::safetensors::{Dtype, SafetensorsHeader, TensorInfo};
use std::collections::BTreeMap;

/// `GGUF` little-endian, the first four bytes of the file.
pub const MAGIC: [u8; 4] = *b"GGUF";

/// Metadata values, in the subset of GGUF's type system that model metadata
/// actually uses. Arrays keep only what dlm reads: strings (the vocabulary and
/// merges), and numbers (per-layer lists).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Strings(Vec<String>),
    Numbers(Vec<f64>),
}

impl Value {
    /// This value as an integer, when it is one.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U64(v) => Some(*v),
            Value::I64(v) => u64::try_from(*v).ok(),
            Value::F64(v) if v.fract() == 0.0 && *v >= 0.0 => Some(*v as u64),
            Value::Bool(b) => Some(*b as u64),
            _ => None,
        }
    }

    /// This value as a float, when it is a number.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F64(v) => Some(*v),
            Value::U64(v) => Some(*v as f64),
            Value::I64(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_strings(&self) -> Option<&[String]> {
        match self {
            Value::Strings(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_numbers(&self) -> Option<&[f64]> {
        match self {
            Value::Numbers(v) => Some(v),
            _ => None,
        }
    }
}

/// A GGUF file's header: its metadata, and its tensors under dlm's names.
#[derive(Debug)]
pub struct GgufHeader {
    /// The `general.*` and `{architecture}.*` keys, plus `tokenizer.ggml.*`.
    pub metadata: BTreeMap<String, Value>,
    /// Tensors, renamed from llama.cpp's scheme to the Hugging Face names the
    /// loader looks up.
    pub tensors: BTreeMap<String, TensorInfo>,
    /// Where tensor data starts, after the header is padded to the alignment.
    pub data_offset: usize,
}

/// A reader over the header bytes that refuses to run off the end.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .ok_or_else(|| bad("length overflow"))?;
        let slice = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| bad("header ends mid-value"))?;
        self.at = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// A GGUF string: a 64-bit byte length, then UTF-8.
    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| bad(&format!("metadata string is not UTF-8: {e}")))
    }

    /// One value of the given GGUF type id.
    fn value(&mut self, kind: u32) -> Result<Value> {
        Ok(match kind {
            0 => Value::U64(self.take(1)?[0] as u64),
            1 => Value::I64(self.take(1)?[0] as i8 as i64),
            2 => Value::U64(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => Value::I64(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::U64(self.u32()? as u64),
            5 => Value::I64(self.u32()? as i32 as i64),
            6 => Value::F64(f32::from_bits(self.u32()?) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::String(self.string()?),
            9 => self.array()?,
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => return Err(bad(&format!("unknown metadata value type {other}"))),
        })
    }

    /// An array value. Strings and numbers are kept; nested arrays are not, since
    /// no model metadata uses them and keeping them would mean a recursive value.
    fn array(&mut self) -> Result<Value> {
        let kind = self.u32()?;
        let len = self.u64()? as usize;
        if kind == 8 {
            let mut out = Vec::with_capacity(len.min(1 << 20));
            for _ in 0..len {
                out.push(self.string()?);
            }
            return Ok(Value::Strings(out));
        }
        if kind == 9 {
            return Err(bad("arrays of arrays in metadata are not supported"));
        }
        let mut out = Vec::with_capacity(len.min(1 << 20));
        for _ in 0..len {
            out.push(
                self.value(kind)?
                    .as_f64()
                    .ok_or_else(|| bad("array element is neither a number nor a string"))?,
            );
        }
        Ok(Value::Numbers(out))
    }
}

fn bad(what: &str) -> DlmError {
    DlmError::SafetensorsHeader(format!("gguf: {what}"))
}

/// Whether these leading bytes are a GGUF file.
pub fn is_gguf(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && bytes[..4] == MAGIC
}

/// Parse a GGUF header from the start of a mapped file.
pub fn parse(bytes: &[u8], file_len: usize) -> Result<GgufHeader> {
    let mut r = Reader::new(bytes);
    if r.take(4)? != MAGIC {
        return Err(bad("not a GGUF file"));
    }
    let version = r.u32()?;
    // v1 used 32-bit lengths throughout and is long gone from the hub; v2 and v3
    // share this layout (v3 added big-endian files, which are still rare enough
    // that reading one as little-endian would be a silent wrong answer -- so the
    // file's own byte order is checked by the magic above, which would not match).
    if version != 2 && version != 3 {
        return Err(bad(&format!(
            "unsupported GGUF version {version}; dlm reads versions 2 and 3"
        )));
    }
    let tensor_count = r.u64()? as usize;
    let kv_count = r.u64()? as usize;

    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = r.string()?;
        let kind = r.u32()?;
        metadata.insert(key, r.value(kind)?);
    }

    // Tensor table: name, dims, type, offset into the data section.
    let mut raw = Vec::with_capacity(tensor_count.min(1 << 16));
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dims = r.u32()? as usize;
        if n_dims > 4 {
            return Err(bad(&format!("tensor {name:?} has {n_dims} dimensions")));
        }
        // GGUF lists dimensions fastest-varying first, the reverse of the row-major
        // shape the rest of dlm uses: a linear's `ne` is [in_features,
        // out_features] where safetensors says [out_features, in_features]. The
        // bytes are laid out the same way; only the shape is written backwards.
        let mut dims: Vec<usize> = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(r.u64()? as usize);
        }
        dims.reverse();
        let kind = r.u32()?;
        let offset = r.u64()? as usize;
        raw.push((name, dims, kind, offset));
    }

    // Data starts after the header, padded to `general.alignment` (default 32).
    let alignment = metadata
        .get("general.alignment")
        .and_then(Value::as_u64)
        .unwrap_or(32)
        .max(1) as usize;
    if !alignment.is_power_of_two() {
        return Err(bad(&format!("alignment {alignment} is not a power of two")));
    }
    let data_offset = r.at.div_ceil(alignment) * alignment;
    if data_offset > file_len {
        return Err(bad("header runs past the end of the file"));
    }
    let data_len = file_len - data_offset;

    let mut tensors = BTreeMap::new();
    for (gguf_name, shape, kind, offset) in raw {
        let dtype = dtype_of(kind, &gguf_name)?;
        let elements: usize = shape.iter().product();
        let block = dtype.block_elems();
        if elements % block != 0 {
            return Err(bad(&format!(
                "tensor {gguf_name:?} has {elements} elements, not a multiple of its \
                 {block}-element block"
            )));
        }
        let byte_len = elements / block * dtype.block_bytes();
        let end = offset
            .checked_add(byte_len)
            .ok_or_else(|| bad("tensor size overflow"))?;
        if end > data_len {
            return Err(bad(&format!(
                "tensor {gguf_name:?} runs past the end of the data section"
            )));
        }
        let name = match hf_name(&gguf_name) {
            Some(name) => name,
            // Tensors dlm has no use for (llama.cpp's rope frequency tables, for
            // instance) keep their own name rather than being dropped: the loader
            // looks up what it needs by name and never iterates blindly.
            None => gguf_name.clone(),
        };
        tensors.insert(
            name.clone(),
            TensorInfo {
                name,
                dtype,
                shape,
                begin: offset,
                end,
            },
        );
    }

    Ok(GgufHeader {
        metadata,
        tensors,
        data_offset,
    })
}

impl GgufHeader {
    /// This header as the one the rest of the loader consumes. The metadata is
    /// dropped: it is read separately, into a [`crate::model::ModelConfig`].
    pub fn into_safetensors_header(self) -> SafetensorsHeader {
        SafetensorsHeader {
            data_offset: self.data_offset,
            tensors: self.tensors,
            metadata: BTreeMap::new(),
        }
    }
}

/// The name a GGUF file's `general.file_type` stands for — "Q4_K_M" and the
/// like, which is what the file was downloaded as and what its user calls it.
/// `None` for a value llama.cpp has since added or dlm does not recognize.
pub fn file_type_name(metadata: &BTreeMap<String, Value>) -> Option<&'static str> {
    Some(match metadata.get("general.file_type")?.as_u64()? {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        32 => "BF16",
        _ => return None,
    })
}

/// The ggml type id as a [`Dtype`].
fn dtype_of(kind: u32, name: &str) -> Result<Dtype> {
    Ok(match kind {
        0 => Dtype::F32,
        1 => Dtype::F16,
        2 => Dtype::Q4_0,
        3 => Dtype::Q4_1,
        6 => Dtype::Q5_0,
        7 => Dtype::Q5_1,
        8 => Dtype::Q8_0,
        12 => Dtype::Q4K,
        13 => Dtype::Q5K,
        14 => Dtype::Q6K,
        30 => Dtype::BF16,
        // Named rather than numbered, because "ggml type 19" tells a user nothing
        // about what to download instead.
        other => {
            let known = match other {
                10 => "Q2_K",
                11 => "Q3_K",
                16..=23 | 29 => "an IQ (importance-matrix) type",
                34 | 35 => "a ternary (TQ) type",
                _ => "an unrecognized type",
            };
            return Err(DlmError::UnsupportedQuant(format!(
                "tensor {name:?} is {known} (ggml type {other}), which dlm does not decode. \
                 It reads F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K and \
                 Q6_K -- a Q4_K_M, Q5_K_M, Q6_K or Q8_0 download is the usual choice. \
                 (Q3_K and Q2_K files often store their embedding as an IQ type, so they \
                 are refused here too.)"
            )));
        }
    })
}

/// llama.cpp's tensor name as the Hugging Face name dlm's loader looks up, or
/// `None` for one it has no equivalent for.
///
/// This is the whole of GGUF's "different architecture" for a dense model: the
/// same tensors under different names. Keeping the translation here means every
/// family the loader already supports arrives at it unchanged.
fn hf_name(gguf: &str) -> Option<String> {
    // Model-level tensors.
    match gguf {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output_norm.bias" => return Some("model.norm.bias".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }
    let rest = gguf.strip_prefix("blk.")?;
    let (layer, rest) = rest.split_once('.')?;
    layer.parse::<u32>().ok()?;
    let (part, suffix) = match rest.rsplit_once('.') {
        Some((part, s @ ("weight" | "bias"))) => (part, s),
        _ => return None,
    };
    let hf = match part {
        "attn_norm" => "input_layernorm",
        "attn_q" => "self_attn.q_proj",
        "attn_k" => "self_attn.k_proj",
        "attn_v" => "self_attn.v_proj",
        "attn_output" => "self_attn.o_proj",
        "attn_q_norm" => "self_attn.q_norm",
        "attn_k_norm" => "self_attn.k_norm",
        "ffn_norm" => "post_attention_layernorm",
        "ffn_gate" => "mlp.gate_proj",
        "ffn_up" => "mlp.up_proj",
        "ffn_down" => "mlp.down_proj",
        // Gemma 2/3's extra norm pair.
        "post_attention_norm" => "post_attention_layernorm",
        "post_ffw_norm" => "post_feedforward_layernorm",
        "attn_post_norm" => "post_attention_layernorm",
        "ffn_post_norm" => "post_feedforward_layernorm",
        _ => return None,
    };
    // Gemma writes `attn_norm` for the pre-attention norm and `post_attention_norm`
    // for the one after it; both map above, and the loader tells them apart by
    // which names are present, exactly as it does for a safetensors Gemma.
    Some(format!("model.layers.{layer}.{hf}.{suffix}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal GGUF: one f32 tensor and a couple of metadata keys, built by
    /// hand so the parser is checked against the format rather than against
    /// whatever produced a fixture.
    fn tiny_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&2u64.to_le_bytes()); // kv count

        let kv_string = |key: &str, value: &str, b: &mut Vec<u8>| {
            b.extend_from_slice(&(key.len() as u64).to_le_bytes());
            b.extend_from_slice(key.as_bytes());
            b.extend_from_slice(&8u32.to_le_bytes());
            b.extend_from_slice(&(value.len() as u64).to_le_bytes());
            b.extend_from_slice(value.as_bytes());
        };
        kv_string("general.architecture", "llama", &mut b);
        // llama.block_count = 2, as a u32.
        let key = "llama.block_count";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&2u32.to_le_bytes());

        // One tensor: blk.0.attn_q.weight, f32, ne = [4, 2] (in, out).
        let name = "blk.0.attn_q.weight";
        b.extend_from_slice(&(name.len() as u64).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&4u64.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset
        b
    }

    #[test]
    fn parses_metadata_tensors_and_names() {
        let header = tiny_gguf();
        // 8 elements of f32 in the data section, after alignment padding.
        let file_len = header.len().div_ceil(32) * 32 + 32;
        let g = parse(&header, file_len).unwrap();

        assert_eq!(
            g.metadata.get("general.architecture").unwrap().as_str(),
            Some("llama")
        );
        assert_eq!(
            g.metadata.get("llama.block_count").unwrap().as_u64(),
            Some(2)
        );
        assert_eq!(g.data_offset % 32, 0);

        // Renamed, and the shape read back into row-major [out, in].
        let t = g
            .tensors
            .get("model.layers.0.self_attn.q_proj.weight")
            .expect("gguf name translated to the HF one");
        assert_eq!(t.shape, vec![2, 4]);
        assert_eq!(t.dtype, Dtype::F32);
        assert_eq!(t.byte_len(), 32);
    }

    /// A tensor whose data runs past the end must be refused, not read: the
    /// offsets come from the file, and a truncated download is the common case.
    #[test]
    fn refuses_a_tensor_that_runs_past_the_file() {
        let header = tiny_gguf();
        let err = parse(&header, header.len() + 4).expect_err("must refuse");
        assert!(format!("{err}").contains("past the end"), "{err}");
    }

    /// The quant types dlm cannot decode are refused by name, so the message says
    /// what to download instead.
    #[test]
    fn unsupported_quant_types_are_named() {
        let err = dtype_of(19, "blk.0.ffn_down.weight").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("IQ"), "{msg}");
        assert!(msg.contains("Q4_K_M"), "{msg}");
    }

    #[test]
    fn translates_the_names_a_dense_model_uses() {
        let cases = [
            ("token_embd.weight", "model.embed_tokens.weight"),
            ("output.weight", "lm_head.weight"),
            ("output_norm.weight", "model.norm.weight"),
            (
                "blk.7.ffn_down.weight",
                "model.layers.7.mlp.down_proj.weight",
            ),
            ("blk.3.attn_k.bias", "model.layers.3.self_attn.k_proj.bias"),
            (
                "blk.0.attn_k_norm.weight",
                "model.layers.0.self_attn.k_norm.weight",
            ),
        ];
        for (gguf, hf) in cases {
            assert_eq!(hf_name(gguf).as_deref(), Some(hf), "{gguf}");
        }
        // Not a tensor dlm knows: kept under its own name rather than guessed at.
        assert_eq!(hf_name("rope_freqs.weight"), None);
    }
}
