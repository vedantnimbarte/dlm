//! Multi-device pipeline parallelism (`specs.md` §3.3).
//!
//! Splits a model's transformer layers into contiguous per-GPU stages (via
//! [`partition_layers`]) and, per token, flows the hidden state stage by stage:
//! before running a layer it makes that layer's assigned GPU the current device
//! ([`gpu::set_device`]), so each block executes on the GPU that owns it and only
//! the `hidden_size`-float residual crosses the boundary between stages — exactly
//! the ring-pipeline data path of `specs.md` §3.3, at single-token decode
//! granularity.
//!
//! It wraps **any** [`ComputeKernel`], so the whole generation / server /
//! speculative stack drives a multi-GPU model with no other changes. On a host
//! build (no GPU backend) `set_device` is a no-op, so the wrapped kernel runs
//! every layer on the CPU and the output is **bit-for-bit identical** to the same
//! kernel without the wrapper — the pipeline-parallel path is therefore testable
//! off-GPU (see the parity test below), like the distributed coordinator.
//!
//! [`partition_layers`]: crate::distributed::partition_layers
//! [`gpu::set_device`]: crate::gpu::set_device

use crate::distributed::partition_layers;
use crate::error::{FlipError, Result};
use crate::forward::cpu::KvLayerCache;
use crate::forward::kernel::ComputeKernel;

/// A [`ComputeKernel`] that runs each layer on its assigned local GPU, forming a
/// pipeline across `gpu_ids`.
pub struct PipelineParallelKernel<K> {
    inner: K,
    /// GPU id that owns each layer: `layer_device[l]` is the device for layer `l`.
    layer_device: Vec<u32>,
    /// The GPU ids in pipeline-stage order (as given).
    devices: Vec<u32>,
}

impl<K: ComputeKernel> PipelineParallelKernel<K> {
    /// Split `inner`'s layers across `gpu_ids` into contiguous pipeline stages
    /// (earlier stages absorb the remainder, matching [`partition_layers`]).
    pub fn new(inner: K, gpu_ids: &[u32]) -> Result<Self> {
        if gpu_ids.is_empty() {
            return Err(FlipError::InvalidConfig(
                "multi-gpu pipeline needs at least one gpu id".into(),
            ));
        }
        let num_layers = inner.num_layers() as usize;
        let shards = partition_layers(num_layers, gpu_ids.len());
        let mut layer_device = vec![0u32; num_layers];
        for (stage, shard) in shards.iter().enumerate() {
            layer_device[shard.start..shard.end].fill(gpu_ids[stage]);
        }
        Ok(Self {
            inner,
            layer_device,
            devices: gpu_ids.to_vec(),
        })
    }

    /// Layer count assigned to each GPU stage, in `gpu_ids` order — for the
    /// startup banner (e.g. `[(0, 40), (1, 40)]`).
    pub fn layers_per_device(&self) -> Vec<(u32, usize)> {
        self.devices
            .iter()
            .map(|&d| (d, self.layer_device.iter().filter(|&&x| x == d).count()))
            .collect()
    }
}

impl<K: ComputeKernel> ComputeKernel for PipelineParallelKernel<K> {
    fn num_layers(&self) -> u32 {
        self.inner.num_layers()
    }

    fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }

    fn kv_dim(&self) -> usize {
        self.inner.kv_dim()
    }

    fn run_block(
        &self,
        layer: u32,
        hidden: &mut [f32],
        kv: &mut KvLayerCache,
        position: usize,
    ) -> Result<()> {
        // Cross into the stage that owns this layer, then run it there.
        crate::gpu::set_device(self.layer_device[layer as usize])?;
        self.inner.run_block(layer, hidden, kv, position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::kernel::StubKernel;

    /// Layers split into contiguous, balanced stages (remainder to earlier GPUs).
    #[test]
    fn partitions_layers_across_gpus() {
        let k = PipelineParallelKernel::new(StubKernel::new(10, 4, 2), &[0, 1, 2]).unwrap();
        assert_eq!(k.layers_per_device(), vec![(0, 4), (1, 3), (2, 3)]);
        // Layers 0-3 → gpu 0, 4-6 → gpu 1, 7-9 → gpu 2.
        assert_eq!(k.layer_device, vec![0, 0, 0, 0, 1, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn rejects_empty_gpu_list() {
        assert!(PipelineParallelKernel::new(StubKernel::new(4, 4, 2), &[]).is_err());
    }

    /// The wrapper only chooses the device (a no-op off-GPU); the math is the
    /// inner kernel's, so wrapped output is identical to running it unwrapped.
    #[test]
    fn wrapped_output_matches_unwrapped() {
        let bare = StubKernel::new(6, 8, 2);
        let wrapped = PipelineParallelKernel::new(StubKernel::new(6, 8, 2), &[0, 1]).unwrap();

        let mut h_bare = vec![1.0f32; 8];
        let mut h_wrapped = vec![1.0f32; 8];
        let mut kv_bare = KvLayerCache::new(2);
        let mut kv_wrapped = KvLayerCache::new(2);

        for position in 0..3 {
            for layer in 0..6 {
                bare.run_block(layer, &mut h_bare, &mut kv_bare, position).unwrap();
                wrapped
                    .run_block(layer, &mut h_wrapped, &mut kv_wrapped, position)
                    .unwrap();
            }
        }
        assert_eq!(h_bare, h_wrapped);
        assert_eq!(kv_bare.len(), kv_wrapped.len());
    }
}
