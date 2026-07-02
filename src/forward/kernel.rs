//! Compute-kernel abstraction for the forward pass.
//!
//! The actual transformer math (matmuls, attention, activations) is the one
//! piece that must run on the GPU. Everything else in `flip` — streaming,
//! caching, residual pooling, KV paging — is orchestration around it. To keep
//! that orchestration testable off-GPU, the math sits behind the
//! [`ComputeKernel`] trait: a real CUDA/HIP kernel implements it for production,
//! and [`StubKernel`] implements it deterministically for tests.

use crate::error::Result;
use crate::quant::Quant4Tensor;

/// Materialized (dequantized) weights for one transformer block.
///
/// A real block has many matrices (q/k/v/o projections, gate/up/down MLP); the
/// skeleton keeps a single flat `values` array so the orchestration flow can be
/// exercised end-to-end. The GPU implementation replaces this with per-matrix
/// device tensors.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub layer: u32,
    pub values: Vec<f32>,
}

impl LayerWeights {
    /// Build from an already-materialized f32 array.
    pub fn new(layer: u32, values: Vec<f32>) -> Self {
        Self { layer, values }
    }

    /// Materialize a layer's weights by dequantizing a 4-bit tensor — the tie
    /// between the streaming/dequant path and the compute path.
    pub fn from_quant(layer: u32, tensor: &Quant4Tensor) -> Self {
        Self {
            layer,
            values: tensor.dequantize(),
        }
    }

    /// A scalar summary of the weights, used by the stub kernel to make each
    /// layer's contribution distinct and deterministic.
    pub fn mean(&self) -> f32 {
        if self.values.is_empty() {
            0.0
        } else {
            self.values.iter().sum::<f32>() / self.values.len() as f32
        }
    }
}

/// The pluggable transformer math backend.
///
/// Each method computes a sublayer's **residual delta** for the current hidden
/// state: the orchestrator adds it back (`h = h + f(h)`). Splitting attention
/// and MLP mirrors the two residual connections in a decoder block.
pub trait ComputeKernel {
    /// Attention sublayer: write the residual delta for `hidden` into `delta`.
    fn attention(&self, weights: &LayerWeights, hidden: &[f32], delta: &mut [f32]) -> Result<()>;

    /// MLP sublayer: write the residual delta for `hidden` into `delta`.
    fn mlp(&self, weights: &LayerWeights, hidden: &[f32], delta: &mut [f32]) -> Result<()>;
}

/// A deterministic CPU stand-in for a real GPU kernel.
///
/// It performs no real attention or matmul — each sublayer contributes a delta
/// equal to the layer's weight mean, so the orchestration (residual
/// accumulation, buffer reuse, KV growth) is exactly verifiable while remaining
/// numerically trivial. It is **not** an inference implementation.
#[derive(Debug, Default, Clone, Copy)]
pub struct StubKernel;

impl ComputeKernel for StubKernel {
    fn attention(&self, weights: &LayerWeights, _hidden: &[f32], delta: &mut [f32]) -> Result<()> {
        let s = weights.mean();
        delta.iter_mut().for_each(|d| *d = s);
        Ok(())
    }

    fn mlp(&self, weights: &LayerWeights, _hidden: &[f32], delta: &mut [f32]) -> Result<()> {
        let s = weights.mean();
        delta.iter_mut().for_each(|d| *d = s);
        Ok(())
    }
}
