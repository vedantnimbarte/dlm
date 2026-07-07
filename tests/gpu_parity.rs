//! GPU↔CPU parity: the CUDA `run_block` must match the CPU oracle.
//!
//! Compiled only under `--features cuda-kernels` (needs nvcc); runs only on a
//! CUDA GPU. On any other build this file is empty. The CPU kernel in
//! `src/forward/cpu.rs` is purpose-built to be the reference these device
//! kernels are checked against — and the `.cu` kernels use sequential reductions
//! in the same order as the CPU code, so the two should agree to near-f32
//! precision (the tolerance only absorbs libm-vs-device differences in
//! `exp`/`rsqrt`/`pow`).
#![cfg(feature = "cuda-kernels")]

use dlm::cache::{KvCacheConfig, PagedKvCache};
use dlm::forward::{
    BlockConfig, ComputeKernel, CpuKernel, ForwardOrchestrator, GpuKernel, KvLayerCache,
    LayerSource, LayerTensors, StreamingGpuKernel,
};

/// An in-memory layer source for the streaming GPU kernel.
struct VecSource(Vec<LayerTensors>);
impl LayerSource for VecSource {
    fn num_layers(&self) -> u32 {
        self.0.len() as u32
    }
    fn load_layer(&self, layer: u32) -> dlm::Result<LayerTensors> {
        Ok(self.0[layer as usize].clone())
    }
}

/// Deterministic SplitMix64 for reproducible synthetic weights.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
                (u * 2.0 - 1.0) * scale
            })
            .collect()
    }
}

fn random_layers(cfg: &BlockConfig, num_layers: u32, seed: u64) -> Vec<LayerTensors> {
    let mut rng = Rng::new(seed);
    let s = 0.05;
    (0..num_layers)
        .map(|_| LayerTensors {
            q_proj: rng.vec(cfg.q_dim() * cfg.hidden_size, s),
            k_proj: rng.vec(cfg.kv_dim() * cfg.hidden_size, s),
            v_proj: rng.vec(cfg.kv_dim() * cfg.hidden_size, s),
            o_proj: rng.vec(cfg.hidden_size * cfg.q_dim(), s),
            gate_proj: rng.vec(cfg.intermediate_size * cfg.hidden_size, s),
            up_proj: rng.vec(cfg.intermediate_size * cfg.hidden_size, s),
            down_proj: rng.vec(cfg.hidden_size * cfg.intermediate_size, s),
            input_layernorm: vec![1.0; cfg.hidden_size],
            post_attention_layernorm: vec![1.0; cfg.hidden_size],
        })
        .collect()
}

#[test]
fn gpu_run_block_matches_cpu() {
    let cfg = BlockConfig {
        hidden_size: 32,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        intermediate_size: 64,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
    };
    let num_layers = 2u32;

    // Identical weights to both kernels.
    let layers = random_layers(&cfg, num_layers, 0xC0FFEE);
    let cpu = CpuKernel::new(cfg, layers.clone()).unwrap();
    // KV capacity ≥ the tokens we decode below.
    let gpu = GpuKernel::new(cfg, layers, 64).unwrap();

    let kv_cfg = KvCacheConfig {
        num_layers,
        num_kv_heads: cfg.num_kv_heads as u32,
        head_dim: cfg.head_dim as u32,
        block_size: 16,
    };
    let mut orch_cpu = ForwardOrchestrator::new(cpu, PagedKvCache::new(kv_cfg, 16), dlm::forward::KvQuant::None);
    let mut orch_gpu = ForwardOrchestrator::new(gpu, PagedKvCache::new(kv_cfg, 16), dlm::forward::KvQuant::None);

    // Same starting hidden state, decoded autoregressively on both.
    let mut hidden_cpu: Vec<f32> = (0..cfg.hidden_size)
        .map(|i| (i as f32) * 0.03 - 0.5)
        .collect();
    let mut hidden_gpu = hidden_cpu.clone();

    for step in 0..4 {
        orch_cpu.decode_token(&mut hidden_cpu).unwrap();
        orch_gpu.decode_token(&mut hidden_gpu).unwrap();

        let max_diff = hidden_cpu
            .iter()
            .zip(&hidden_gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "step {step}: GPU diverged from CPU by {max_diff}"
        );

        // KV histories must also stay in lockstep.
        for l in 0..num_layers {
            assert_eq!(orch_cpu.layer_kv_len(l), orch_gpu.layer_kv_len(l));
        }
    }
}

fn small_cfg() -> BlockConfig {
    BlockConfig {
        hidden_size: 32,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        intermediate_size: 64,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
    }
}

#[test]
fn streaming_gpu_matches_resident() {
    let cfg = small_cfg();
    let num_layers = 6u32;
    let layers = random_layers(&cfg, num_layers, 0xBEEF);

    // Resident GPU (all layers) vs streaming GPU (window 2 of 6, prefetch on).
    let resident = GpuKernel::new(cfg, layers.clone(), 64).unwrap();
    let streaming = StreamingGpuKernel::new(cfg, VecSource(layers), 64, 2).unwrap();

    let kv_cfg = KvCacheConfig {
        num_layers,
        num_kv_heads: cfg.num_kv_heads as u32,
        head_dim: cfg.head_dim as u32,
        block_size: 16,
    };
    let mut orch_res = ForwardOrchestrator::new(resident, PagedKvCache::new(kv_cfg, 32), dlm::forward::KvQuant::None);
    let mut orch_str = ForwardOrchestrator::new(streaming, PagedKvCache::new(kv_cfg, 32), dlm::forward::KvQuant::None);

    let mut h_res: Vec<f32> = (0..cfg.hidden_size).map(|i| i as f32 * 0.02 - 0.3).collect();
    let mut h_str = h_res.clone();
    for step in 0..4 {
        orch_res.decode_token(&mut h_res).unwrap();
        orch_str.decode_token(&mut h_str).unwrap();
        let max_diff = h_res.iter().zip(&h_str).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // Same kernel + weights, only the *timing* of the upload differs.
        assert!(max_diff < 1e-4, "step {step}: streaming GPU diverged by {max_diff}");
    }
}

#[test]
fn streaming_gpu_worker_prefetches() {
    let cfg = small_cfg();
    let k = StreamingGpuKernel::new(cfg, VecSource(random_layers(&cfg, 4, 3)), 64, 4).unwrap();
    let mut h: Vec<f32> = (0..cfg.hidden_size).map(|i| i as f32 * 0.01).collect();
    let mut kv = KvLayerCache::new(cfg.kv_dim());

    // Compute layer 0 → requests a prefetch of layer 1; nothing else competes, so
    // the worker uploads it uncontended.
    k.run_block(0, &mut h, &mut kv, 0).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(80));
    assert!(k.stats().prefetched >= 1, "GPU worker should prefetch a layer: {:?}", k.stats());
}
