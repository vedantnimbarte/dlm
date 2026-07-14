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
            post_attention_layernorm: vec![1.0; cfg.hidden_size], ..Default::default()
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
        rms_eps: 1e-5, rope_scaling: None,
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

/// Decode a few tokens on both kernels from the same start state and assert the
/// GPU tracks the CPU oracle.
fn assert_gpu_matches_cpu(cfg: BlockConfig, layers: Vec<LayerTensors>, tol: f32, what: &str) {
    let num_layers = layers.len() as u32;
    let cpu = CpuKernel::new(cfg, layers.clone()).unwrap();
    let gpu = GpuKernel::new(cfg, layers, 64).unwrap();

    let kv_cfg = KvCacheConfig {
        num_layers,
        num_kv_heads: cfg.num_kv_heads as u32,
        head_dim: cfg.head_dim as u32,
        block_size: 16,
    };
    let mut orch_cpu =
        ForwardOrchestrator::new(cpu, PagedKvCache::new(kv_cfg, 16), dlm::forward::KvQuant::None);
    let mut orch_gpu =
        ForwardOrchestrator::new(gpu, PagedKvCache::new(kv_cfg, 16), dlm::forward::KvQuant::None);

    let mut h_cpu: Vec<f32> = (0..cfg.hidden_size)
        .map(|i| ((i % 17) as f32) * 0.03 - 0.25)
        .collect();
    let mut h_gpu = h_cpu.clone();

    for step in 0..3 {
        orch_cpu.decode_token(&mut h_cpu).unwrap();
        orch_gpu.decode_token(&mut h_gpu).unwrap();
        let max_diff = h_cpu
            .iter()
            .zip(&h_gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < tol,
            "{what}: step {step}: GPU diverged from CPU by {max_diff}"
        );
    }
}

/// Every real model has `hidden_size >= 2048`. The RMSNorm kernel used to launch
/// `<<<1, hidden_size>>>`, which exceeds CUDA's 1024 threads-per-block cap and
/// fails to launch — so the GPU path could never have run an actual checkpoint.
/// This pins the strided/block-reduction launch at a realistic width.
#[test]
fn gpu_matches_cpu_at_realistic_hidden_size() {
    let cfg = BlockConfig {
        hidden_size: 2048, // > 1024: the old kernel could not launch at all
        num_heads: 16,
        num_kv_heads: 4,
        head_dim: 128,
        intermediate_size: 512,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        rope_scaling: None,
    };
    let layers = random_layers(&cfg, 2, 0xA11CE);
    assert_gpu_matches_cpu(cfg, layers, 2e-3, "hidden_size=2048");
}

/// Qwen2-style Q/K/V biases must be applied on device exactly as on the CPU.
#[test]
fn gpu_matches_cpu_with_qkv_biases() {
    let cfg = BlockConfig {
        hidden_size: 64,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        intermediate_size: 128,
        rope_theta: 1_000_000.0,
        rms_eps: 1e-6,
        rope_scaling: None,
    };
    let mut rng = Rng::new(0xB1A5);
    let mut layers = random_layers(&cfg, 2, 0xB1A5);
    for l in layers.iter_mut() {
        l.q_bias = Some(rng.vec(cfg.q_dim(), 0.1));
        l.k_bias = Some(rng.vec(cfg.kv_dim(), 0.1));
        l.v_bias = Some(rng.vec(cfg.kv_dim(), 0.1));
    }
    assert_gpu_matches_cpu(cfg, layers, 1e-3, "qkv biases");
}

/// Llama-3 RoPE scaling: the device reads the host-computed `inv_freq`, so the
/// scaled frequencies must reach the GPU rather than the plain theta powers.
#[test]
fn gpu_matches_cpu_with_llama3_rope_scaling() {
    let cfg = BlockConfig {
        hidden_size: 64,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        intermediate_size: 128,
        rope_theta: 500_000.0,
        rms_eps: 1e-5,
        rope_scaling: Some(dlm::forward::RopeScaling::Llama3 {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position: 8192.0,
        }),
    };
    let layers = random_layers(&cfg, 2, 0x5CA1E);
    assert_gpu_matches_cpu(cfg, layers, 1e-3, "llama3 rope scaling");
}

fn small_cfg() -> BlockConfig {
    BlockConfig {
        hidden_size: 32,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        intermediate_size: 64,
        rope_theta: 10000.0,
        rms_eps: 1e-5, rope_scaling: None,
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
