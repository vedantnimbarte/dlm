//! Weight quantization kernels — CPU dequantization of streamed 4-bit weights.

pub mod dequant;
pub mod packed;

pub use dequant::{pack_codes, quantize_affine, quantize_affine_int8, Quant4Tensor};
pub use packed::{
    dequantize_awq_4bit, dequantize_gptq_4bit, pack_awq_4bit, pack_gptq_4bit, unpack_gptq_4bit,
    PackedQuantConfig,
};
