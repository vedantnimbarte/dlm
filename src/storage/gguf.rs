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
            // Booleans count as 0/1: Gemma 4 marks its windowed layers with an
            // array of them.
            let v = match self.value(kind)? {
                Value::Bool(b) => Some(b as u8 as f64),
                other => other.as_f64(),
            };
            out.push(v.ok_or_else(|| bad("array element is neither a number nor a string"))?);
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

    // Gemma 2, 3 and 4 norm the FFN input with their own norm and keep
    // `post_attention_norm` for the attention output, so their `ffn_norm` is
    // `pre_feedforward_layernorm`; everywhere else `ffn_norm` is the one norm
    // between attention and FFN, `post_attention_layernorm`.
    let arch = metadata.get("general.architecture").and_then(Value::as_str);
    let sandwich = matches!(arch, Some("gemma2" | "gemma3" | "gemma4"));
    let gemma4 = arch == Some("gemma4");
    // llama.cpp numbers a model's MTP layers after its own; they are read as
    // `mtp.layers.{i}`, as transformers names them.
    let arch_int = |key: &str| {
        arch.and_then(|a| metadata.get(&format!("{a}.{key}")))
            .and_then(Value::as_u64)
    };
    let mtp_from = match (arch_int("block_count"), arch_int("nextn_predict_layers")) {
        (Some(blocks), Some(mtp)) if mtp > 0 && mtp <= blocks => Some(blocks - mtp),
        _ => None,
    };
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
        let qwen35 = (arch == Some("qwen35moe"))
            .then(|| qwen35_name(&gguf_name))
            .flatten();
        let name = match qwen35
            .or_else(|| gemma4_name(gemma4, &gguf_name))
            .or_else(|| hf_name(&gguf_name, sandwich))
        {
            Some(name) => mtp_layer_name(name, mtp_from),
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
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
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
        10 => Dtype::Q2K,
        11 => Dtype::Q3K,
        16 => Dtype::IQ2XXS,
        17 => Dtype::IQ2XS,
        18 => Dtype::IQ3XXS,
        20 => Dtype::IQ4NL,
        21 => Dtype::IQ3S,
        22 => Dtype::IQ2S,
        23 => Dtype::IQ4XS,
        30 => Dtype::BF16,
        // Named rather than numbered, because "ggml type 19" tells a user nothing
        // about what to download instead.
        other => {
            let known = match other {
                19 | 29 => "an IQ1 type",
                34 | 35 => "a ternary (TQ) type",
                _ => "an unrecognized type",
            };
            return Err(DlmError::UnsupportedQuant(format!(
                "tensor {name:?} is {known} (ggml type {other}), which dlm does not decode. \
                 It reads F32, F16, BF16, Q4_0 to Q8_0, Q2_K to Q6_K, IQ2_XXS, IQ2_XS, \
                 IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS: any download from Q2_K or \
                 IQ2_XXS up, such as Q4_K_M or IQ3_M."
            )));
        }
    })
}

/// The Hugging Face name of a Gemma 4 tensor `hf_name` does not cover, when
/// `gemma4`: its per-layer-input tensors and its per-layer output scale.
fn gemma4_name(gemma4: bool, gguf: &str) -> Option<String> {
    if !gemma4 {
        return None;
    }
    match gguf {
        "per_layer_token_embd.weight" => return Some("model.embed_tokens_per_layer.weight".into()),
        "per_layer_model_proj.weight" => {
            return Some("model.per_layer_model_projection.weight".into())
        }
        "per_layer_proj_norm.weight" => {
            return Some("model.per_layer_projection_norm.weight".into())
        }
        _ => {}
    }
    let (layer, rest) = gguf.strip_prefix("blk.")?.split_once('.')?;
    layer.parse::<u32>().ok()?;
    let hf = match rest {
        "inp_gate.weight" => "per_layer_input_gate.weight",
        "proj.weight" => "per_layer_projection.weight",
        "post_norm.weight" => "post_per_layer_input_norm.weight",
        "layer_output_scale.weight" => "layer_scalar",
        _ => return None,
    };
    Some(format!("model.layers.{layer}.{hf}"))
}

/// llama.cpp's tensor name as the Hugging Face name dlm's loader looks up, or
/// `None` for one it has no equivalent for.
///
/// This is the whole of GGUF's "different architecture" for a dense model: the
/// same tensors under different names. Keeping the translation here means every
/// family the loader already supports arrives at it unchanged.
fn hf_name(gguf: &str, sandwich: bool) -> Option<String> {
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
        "ffn_norm" if sandwich => "pre_feedforward_layernorm",
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
    Some(format!("model.layers.{layer}.{hf}.{suffix}"))
}

/// `model.layers.{n}.x` as `mtp.layers.{n - first}.x` when layer `n` is one of
/// the MTP layers llama.cpp counts after the model's (from `first` on).
fn mtp_layer_name(name: String, first: Option<u64>) -> String {
    let Some(first) = first else { return name };
    let Some((idx, rest)) = name
        .strip_prefix("model.layers.")
        .and_then(|r| r.split_once('.'))
    else {
        return name;
    };
    match idx.parse::<u64>() {
        Ok(i) if i >= first => format!("mtp.layers.{}.{rest}", i - first),
        _ => name,
    }
}

/// The names llama.cpp gives Qwen3.5/3.6's own tensors (`qwen35moe`), as the
/// loader reads them, or `None` for the ones [`hf_name`] covers. Kept apart
/// because some of these names mean something else in other architectures
/// (`attn_qkv` is Phi-3's fused attention).
///
/// Two are stored transformed and get names of their own: `ssm_a` holds
/// `-exp(A_log)` (read as `A_neg`), and the stacked experts are
/// `experts.{gate,up,down}_exps`. The MTP head's `nextn.*` tensors become
/// `mtp.*`.
fn qwen35_name(gguf: &str) -> Option<String> {
    let rest = gguf.strip_prefix("blk.")?;
    let (layer, rest) = rest.split_once('.')?;
    layer.parse::<u32>().ok()?;
    let mtp = match rest {
        "nextn.eh_proj.weight" => Some("fc"),
        "nextn.enorm.weight" => Some("pre_fc_norm_embedding"),
        "nextn.hnorm.weight" => Some("pre_fc_norm_hidden"),
        "nextn.shared_head_norm.weight" => Some("norm"),
        _ => None,
    };
    if let Some(m) = mtp {
        return Some(format!("mtp.{m}.weight"));
    }
    let hf = match rest {
        "attn_qkv.weight" => "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight" => "linear_attn.in_proj_z.weight",
        "ssm_beta.weight" => "linear_attn.in_proj_b.weight",
        "ssm_alpha.weight" => "linear_attn.in_proj_a.weight",
        "ssm_out.weight" => "linear_attn.out_proj.weight",
        "ssm_conv1d.weight" => "linear_attn.conv1d.weight",
        "ssm_norm.weight" => "linear_attn.norm.weight",
        "ssm_dt.bias" | "ssm_dt" => "linear_attn.dt_bias",
        "ssm_a" | "ssm_a.weight" => "linear_attn.A_neg",
        "ffn_gate_inp.weight" => "mlp.gate.weight",
        "ffn_gate_inp_shexp.weight" => "mlp.shared_expert_gate.weight",
        "ffn_gate_shexp.weight" => "mlp.shared_expert.gate_proj.weight",
        "ffn_up_shexp.weight" => "mlp.shared_expert.up_proj.weight",
        "ffn_down_shexp.weight" => "mlp.shared_expert.down_proj.weight",
        "ffn_gate_exps.weight" => "mlp.experts.gate_exps",
        "ffn_up_exps.weight" => "mlp.experts.up_exps",
        "ffn_down_exps.weight" => "mlp.experts.down_exps",
        _ => return None,
    };
    Some(format!("model.layers.{layer}.{hf}"))
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
        assert!(msg.contains("IQ1"), "{msg}");
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
            assert_eq!(hf_name(gguf, false).as_deref(), Some(hf), "{gguf}");
        }
        // Not a tensor dlm knows: kept under its own name rather than guessed at.
        assert_eq!(hf_name("rope_freqs.weight", false), None);
        // Gemma 2/3 carry four norms per layer; `ffn_norm` is the pre-FFN one,
        // and must not land on top of `post_attention_norm`.
        let gemma = [
            (
                "blk.2.attn_norm.weight",
                "model.layers.2.input_layernorm.weight",
            ),
            (
                "blk.2.post_attention_norm.weight",
                "model.layers.2.post_attention_layernorm.weight",
            ),
            (
                "blk.2.ffn_norm.weight",
                "model.layers.2.pre_feedforward_layernorm.weight",
            ),
            (
                "blk.2.post_ffw_norm.weight",
                "model.layers.2.post_feedforward_layernorm.weight",
            ),
        ];
        for (gguf, hf) in gemma {
            assert_eq!(hf_name(gguf, true).as_deref(), Some(hf), "{gguf}");
        }
        assert_eq!(
            hf_name("blk.2.ffn_norm.weight", false).as_deref(),
            Some("model.layers.2.post_attention_layernorm.weight")
        );
    }

    /// Qwen3.5/3.6's linear attention, stacked experts and MTP head arrive under
    /// the names the hybrid loader reads; a dense name still falls through.
    #[test]
    fn translates_the_names_qwen35moe_uses() {
        let cases = [
            (
                "blk.0.attn_qkv.weight",
                "model.layers.0.linear_attn.in_proj_qkv.weight",
            ),
            ("blk.2.ssm_a", "model.layers.2.linear_attn.A_neg"),
            ("blk.2.ssm_dt.bias", "model.layers.2.linear_attn.dt_bias"),
            (
                "blk.5.ffn_down_exps.weight",
                "model.layers.5.mlp.experts.down_exps",
            ),
            (
                "blk.5.ffn_gate_inp_shexp.weight",
                "model.layers.5.mlp.shared_expert_gate.weight",
            ),
            ("blk.40.nextn.eh_proj.weight", "mtp.fc.weight"),
        ];
        for (gguf, hf) in cases {
            assert_eq!(qwen35_name(gguf).as_deref(), Some(hf), "{gguf}");
        }
        assert_eq!(qwen35_name("blk.3.attn_q.weight"), None);
    }
}
