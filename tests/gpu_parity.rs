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

use dlm::forward::Weights;
use dlm::cache::{KvCacheConfig, PagedKvCache};
use dlm::forward::{
    BlockConfig, ComputeKernel, CpuKernel, ExpertFfn, Ffn, ForwardOrchestrator, GpuKernel,
    KvLayerCache, LayerSource, LayerTensors, StreamingGpuKernel,
};

/// An in-memory layer source for the streaming GPU kernel.
struct VecSource(Vec<LayerTensors>);
impl LayerSource for VecSource {
    fn num_layers(&self) -> u32 {
        self.0.len() as u32
    }
    fn load_layer(&self, layer: u32) -> dlm::Result<std::sync::Arc<LayerTensors>> {
        Ok(std::sync::Arc::new(self.0[layer as usize].clone()))
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
            q_proj: Weights::from_f32(rng.vec(cfg.q_dim() * cfg.hidden_size, s)),
            k_proj: Weights::from_f32(rng.vec(cfg.kv_dim() * cfg.hidden_size, s)),
            v_proj: Weights::from_f32(rng.vec(cfg.kv_dim() * cfg.hidden_size, s)),
            o_proj: Weights::from_f32(rng.vec(cfg.hidden_size * cfg.q_dim(), s)),
            ffn: Ffn::Dense(ExpertFfn { gate: Weights::from_f32(rng.vec(cfg.intermediate_size * cfg.hidden_size, s)), up: Weights::from_f32(rng.vec(cfg.intermediate_size * cfg.hidden_size, s)), down: Weights::from_f32(rng.vec(cfg.hidden_size * cfg.intermediate_size, s)) }),
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
        rms_eps: 1e-5, rope_scaling: None, moe: None, sliding_window: None, activation: Default::default(), mla: None,
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

/// Two sessions sharing one GPU kernel — exactly what the batching server does —
/// must keep **independent** KV: interleaving their decode steps yields the same
/// trajectory as running each alone. This is the regression guard for
/// per-session GPU KV (`KvLayerCache::gpu_kv`). With the old kernel-owned
/// per-layer KV, the second session's writes clobbered the first's history and
/// this diverged.
#[test]
fn gpu_batched_sessions_keep_independent_kv() {
    let cfg = BlockConfig {
        hidden_size: 32,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        intermediate_size: 64,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        rope_scaling: None,
        moe: None, sliding_window: None, activation: Default::default(), mla: None,
    };
    let num_layers = 3u32;
    let layers = random_layers(&cfg, num_layers, 0xB0A7);
    // One kernel, shared by both sessions via `&gpu` (as the server shares it).
    let gpu = GpuKernel::new(cfg, layers, 64).unwrap();
    let kv_cfg = KvCacheConfig {
        num_layers,
        num_kv_heads: cfg.num_kv_heads as u32,
        head_dim: cfg.head_dim as u32,
        block_size: 16,
    };

    // Two distinct start states → distinct trajectories that would visibly
    // cross-contaminate if KV were shared.
    let start_a: Vec<f32> = (0..cfg.hidden_size).map(|i| (i as f32) * 0.03 - 0.5).collect();
    let start_b: Vec<f32> =
        (0..cfg.hidden_size).map(|i| ((i * 3 % 29) as f32) * 0.02 - 0.3).collect();

    let solo = |start: &[f32]| -> Vec<Vec<f32>> {
        let mut orch = ForwardOrchestrator::new(
            &gpu,
            PagedKvCache::new(kv_cfg, 16),
            dlm::forward::KvQuant::None,
        );
        let mut h = start.to_vec();
        (0..5)
            .map(|_| {
                orch.decode_token(&mut h).unwrap();
                h.clone()
            })
            .collect()
    };
    let solo_a = solo(&start_a);
    let solo_b = solo(&start_b);

    // Interleave A and B step-by-step on the SAME kernel.
    let mut orch_a =
        ForwardOrchestrator::new(&gpu, PagedKvCache::new(kv_cfg, 16), dlm::forward::KvQuant::None);
    let mut orch_b =
        ForwardOrchestrator::new(&gpu, PagedKvCache::new(kv_cfg, 16), dlm::forward::KvQuant::None);
    let mut ha = start_a.clone();
    let mut hb = start_b.clone();
    for step in 0..5 {
        orch_a.decode_token(&mut ha).unwrap();
        orch_b.decode_token(&mut hb).unwrap();
        assert_eq!(ha, solo_a[step], "session A diverged when interleaved with B at step {step}");
        assert_eq!(hb, solo_b[step], "session B diverged when interleaved with A at step {step}");
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

/// Sliding-window attention (Mistral) must match the CPU oracle op-for-op: the
/// device `attention_kernel` and CPU `attention()` both bound the read to the last
/// `window` positions. With window 2 over 3+ decoded tokens the window is active.
#[test]
fn gpu_sliding_window_matches_cpu() {
    let cfg = BlockConfig {
        hidden_size: 32,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        intermediate_size: 64,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        rope_scaling: None,
        moe: None,
        sliding_window: Some(2),
        activation: dlm::forward::Activation::Silu, mla: None,
    };
    let layers = random_layers(&cfg, 2, 0x5117);
    assert_gpu_matches_cpu(cfg, layers, 1e-3, "sliding-window");
}

/// Splitting a model across two GPUs ([`MultiGpuKernel`]) must produce the same
/// output as running it on one — the device split only moves where each layer
/// computes. Needs a second GPU; skips cleanly otherwise.
#[test]
fn gpu_multi_gpu_matches_single_device() {
    if dlm::gpu::set_device(1).is_err() {
        return; // no second GPU
    }
    let _ = dlm::gpu::set_device(0);
    let cfg = BlockConfig {
        hidden_size: 32,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        intermediate_size: 64,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        rope_scaling: None,
        moe: None,
        sliding_window: None,
        activation: dlm::forward::Activation::Silu,
        mla: None,
    };
    let num_layers = 4u32;
    let layers = random_layers(&cfg, num_layers, 0x11CE);
    let kv_cfg = KvCacheConfig {
        num_layers,
        num_kv_heads: cfg.num_kv_heads as u32,
        head_dim: cfg.head_dim as u32,
        block_size: 16,
    };
    let start: Vec<f32> = (0..cfg.hidden_size).map(|i| (i as f32) * 0.03 - 0.5).collect();

    let single = dlm::forward::GpuKernel::new(cfg, layers.clone(), 64).unwrap();
    let mut orch = ForwardOrchestrator::new(single, PagedKvCache::new(kv_cfg, 16), dlm::forward::KvQuant::None);
    let mut h_single = start.clone();
    for _ in 0..3 {
        orch.decode_token(&mut h_single).unwrap();
    }

    let multi = dlm::forward::MultiGpuKernel::new(cfg, layers, &[0, 1], 64).unwrap();
    let mut orch = ForwardOrchestrator::new(multi, PagedKvCache::new(kv_cfg, 16), dlm::forward::KvQuant::None);
    let mut h_multi = start;
    for _ in 0..3 {
        orch.decode_token(&mut h_multi).unwrap();
    }

    let max_diff = h_single.iter().zip(&h_multi).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 1e-4, "multi-gpu diverged from single-gpu by {max_diff}");
}

/// Multi-head Latent Attention (DeepSeek, dense FFN) must match the CPU oracle:
/// the device `dlm_mla_attn` + `dlm_dense_ffn` reproduce `mla_attention_sublayer`
/// (compressed-latent KV, on-the-fly reconstruction, decoupled RoPE).
#[test]
fn gpu_mla_matches_cpu() {
    use dlm::forward::MlaWeights;
    use dlm::model::MlaConfig;
    let mla = MlaConfig {
        q_lora_rank: Some(12),
        kv_lora_rank: 8,
        qk_nope_head_dim: 4,
        qk_rope_head_dim: 4,
        v_head_dim: 4,
    };
    let cfg = BlockConfig {
        hidden_size: 16,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 8,
        intermediate_size: 16,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        rope_scaling: None,
        moe: None,
        sliding_window: None,
        activation: dlm::forward::Activation::Silu,
        mla: Some(mla),
    };
    let (nh, qk, latent, rope, vdim, ql) = (2usize, 8usize, 8usize, 4usize, 4usize, 12usize);
    let nope = 4usize;
    let mut rng = Rng::new(0xDEEB);
    let s = 0.05;
    let layers: Vec<LayerTensors> = (0..2)
        .map(|_| LayerTensors {
            o_proj: Weights::from_f32(rng.vec(cfg.hidden_size * nh * vdim, s)),
            ffn: Ffn::Dense(ExpertFfn {
                gate: Weights::from_f32(rng.vec(cfg.intermediate_size * cfg.hidden_size, s)),
                up: Weights::from_f32(rng.vec(cfg.intermediate_size * cfg.hidden_size, s)),
                down: Weights::from_f32(rng.vec(cfg.hidden_size * cfg.intermediate_size, s)),
            }),
            input_layernorm: vec![1.0; cfg.hidden_size],
            post_attention_layernorm: vec![1.0; cfg.hidden_size],
            mla: Some(MlaWeights {
                q_a_proj: Some(Weights::from_f32(rng.vec(ql * cfg.hidden_size, s))),
                q_a_layernorm: Some(vec![1.0; ql]),
                q_b_proj: Weights::from_f32(rng.vec(nh * qk * ql, s)),
                kv_a_proj: Weights::from_f32(rng.vec((latent + rope) * cfg.hidden_size, s)),
                kv_a_layernorm: vec![1.0; latent],
                kv_b_proj: Weights::from_f32(rng.vec(nh * (nope + vdim) * latent, s)),
            }),
            ..Default::default()
        })
        .collect();
    assert_gpu_matches_cpu(cfg, layers, 1e-3, "mla");
}

/// YaRN RoPE (frequency blend + mscale folded into cos/sin) must match the CPU
/// oracle: the device `rope_kernel` and CPU `rope_inplace` apply the same scaled
/// rotation over the shared `rope_inv_freqs`.
#[test]
fn gpu_yarn_rope_matches_cpu() {
    let cfg = BlockConfig {
        hidden_size: 32,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        intermediate_size: 64,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        rope_scaling: Some(dlm::forward::RopeScaling::Yarn {
            factor: 4.0,
            original_max_position: 4096.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 0.1 * 4.0f32.ln() + 1.0,
        }),
        moe: None,
        sliding_window: None,
        activation: dlm::forward::Activation::Silu, mla: None,
    };
    let layers = random_layers(&cfg, 2, 0x7A54);
    assert_gpu_matches_cpu(cfg, layers, 1e-3, "yarn-rope");
}

/// Qwen3 per-head Q/K RMSNorm must match the CPU oracle: the device
/// `head_rmsnorm_kernel` and CPU `head_rmsnorm` normalize each head identically
/// before RoPE.
#[test]
fn gpu_qk_norm_matches_cpu() {
    let cfg = BlockConfig {
        hidden_size: 32,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        intermediate_size: 64,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        rope_scaling: None,
        moe: None,
        sliding_window: None,
        activation: dlm::forward::Activation::Silu, mla: None,
    };
    let mut layers = random_layers(&cfg, 2, 0x9317);
    for l in &mut layers {
        l.q_norm = Some((0..cfg.head_dim).map(|i| 0.9 + 0.02 * i as f32).collect());
        l.k_norm = Some((0..cfg.head_dim).map(|i| 1.1 - 0.02 * i as f32).collect());
    }
    assert_gpu_matches_cpu(cfg, layers, 1e-3, "qk-norm");
}

/// Gemma's GeGLU (tanh-GELU gate) must match the CPU oracle: the device
/// `swiglu_kernel` and CPU `activate()` compute the same tanh-approximate GELU.
#[test]
fn gpu_gelu_activation_matches_cpu() {
    let cfg = BlockConfig {
        hidden_size: 32,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        intermediate_size: 64,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        rope_scaling: None,
        moe: None,
        sliding_window: None,
        activation: dlm::forward::Activation::GeluTanh, mla: None,
    };
    let layers = random_layers(&cfg, 2, 0x6E10);
    assert_gpu_matches_cpu(cfg, layers, 1e-3, "gelu-activation");
}

/// `--quant int4` weights must decode identically on device and on the CPU
/// oracle.
///
/// Quantization loses accuracy against the *original* floats — that is the deal —
/// but both kernels read the same codes and the same per-group scales, so they
/// must agree with each other as tightly as the float paths do. A mismatch here
/// means the device's blob layout (`load_w<DLM_W_INT4>`) has drifted from
/// `Int4Layout`/`int4_get` on the host, which would silently corrupt weights
/// rather than merely round them.
#[test]
fn gpu_matches_cpu_with_int4_weights() {
    let cfg = BlockConfig {
        hidden_size: 256,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        intermediate_size: 512,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        rope_scaling: None, moe: None, sliding_window: None, activation: Default::default(), mla: None,
    };
    // Quantize the same random weights both kernels would otherwise share.
    let quantized: Vec<LayerTensors> = random_layers(&cfg, 2, 0x1174)
        .into_iter()
        .map(|mut l| {
            let q = |w: &dlm::forward::Weights| {
                let floats: Vec<f32> = (0..w.len()).map(|i| w.get(i)).collect();
                dlm::forward::Weights::quantize_int4(&floats, dlm::forward::QUANT_GROUP_SIZE).unwrap()
            };
            l.q_proj = q(&l.q_proj);
            l.k_proj = q(&l.k_proj);
            l.v_proj = q(&l.v_proj);
            l.o_proj = q(&l.o_proj);
            if let Ffn::Dense(f) = &mut l.ffn {
                f.gate = q(&f.gate);
                f.up = q(&f.up);
                f.down = q(&f.down);
            }
            l
        })
        .collect();
    assert_gpu_matches_cpu(cfg, quantized, 2e-3, "int4 weights");
}

/// Same contract as the int4 case, one bit-width up: int8 codes must decode
/// identically on device and host. Its blob shares the layout but packs one code
/// per byte, so the offsets differ — a drift here reads scales from the wrong
/// place and corrupts every weight.
#[test]
fn gpu_matches_cpu_with_int8_weights() {
    let cfg = BlockConfig {
        hidden_size: 256,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        intermediate_size: 512,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        rope_scaling: None, moe: None, sliding_window: None, activation: Default::default(), mla: None,
    };
    let quantized: Vec<LayerTensors> = random_layers(&cfg, 2, 0x8817)
        .into_iter()
        .map(|mut l| {
            let q = |w: &dlm::forward::Weights| {
                let floats: Vec<f32> = (0..w.len()).map(|i| w.get(i)).collect();
                dlm::forward::Weights::quantize_int8(&floats, dlm::forward::QUANT_GROUP_SIZE)
                    .unwrap()
            };
            l.q_proj = q(&l.q_proj);
            l.k_proj = q(&l.k_proj);
            l.v_proj = q(&l.v_proj);
            l.o_proj = q(&l.o_proj);
            if let Ffn::Dense(f) = &mut l.ffn {
                f.gate = q(&f.gate);
                f.up = q(&f.up);
                f.down = q(&f.down);
            }
            l
        })
        .collect();
    assert_gpu_matches_cpu(cfg, quantized, 2e-3, "int8 weights");
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
        rope_scaling: None, moe: None, sliding_window: None, activation: Default::default(), mla: None,
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
        rope_scaling: None, moe: None, sliding_window: None, activation: Default::default(), mla: None,
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
        moe: None, sliding_window: None, activation: Default::default(), mla: None,
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
        rms_eps: 1e-5, rope_scaling: None, moe: None, sliding_window: None, activation: Default::default(), mla: None,
    }
}

#[test]
fn streaming_gpu_matches_resident() {
    let cfg = small_cfg();
    let num_layers = 6u32;
    let layers = random_layers(&cfg, num_layers, 0xBEEF);

    // Resident GPU (all layers) vs streaming GPU (window 2 of 6, prefetch on).
    let resident = GpuKernel::new(cfg, layers.clone(), 64).unwrap();
    let streaming = StreamingGpuKernel::new(cfg, VecSource(layers), 64, 2, None).unwrap();

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
    let k = StreamingGpuKernel::new(cfg, VecSource(random_layers(&cfg, 4, 3)), 64, 4, None).unwrap();
    let mut h: Vec<f32> = (0..cfg.hidden_size).map(|i| i as f32 * 0.01).collect();
    let mut kv = KvLayerCache::new(cfg.kv_dim());

    // Compute layer 0 → requests a prefetch of layer 1; nothing else competes, so
    // the worker uploads it uncontended.
    k.run_block(0, &mut h, &mut kv, 0).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(80));
    assert!(k.stats().prefetched >= 1, "GPU worker should prefetch a layer: {:?}", k.stats());
}

// ── Mixture-of-Experts GPU parity ──
//
// The streaming GPU MoE path (dlm_moe_attn + router + per-expert dlm_apply_expert,
// experts streamed per (layer, expert)) must match the CPU oracle's `moe_ffn`.

use dlm::model::{MoeConfig, MoeNaming};

/// In-memory MoE source: holds full layers (experts inline) and serves the GPU
/// path a core (experts emptied) plus individual experts on demand.
struct MoeVecSource(Vec<LayerTensors>);
impl LayerSource for MoeVecSource {
    fn num_layers(&self) -> u32 {
        self.0.len() as u32
    }
    fn load_layer(&self, layer: u32) -> dlm::Result<std::sync::Arc<LayerTensors>> {
        Ok(std::sync::Arc::new(self.0[layer as usize].clone()))
    }
    fn load_layer_core(&self, layer: u32) -> dlm::Result<std::sync::Arc<LayerTensors>> {
        let mut t = self.0[layer as usize].clone();
        if let Ffn::Moe { experts, .. } = &mut t.ffn {
            experts.clear();
        }
        Ok(std::sync::Arc::new(t))
    }
    fn load_expert(&self, layer: u32, expert: u32) -> dlm::Result<std::sync::Arc<ExpertFfn>> {
        match &self.0[layer as usize].ffn {
            Ffn::Moe { experts, .. } => Ok(std::sync::Arc::new(experts[expert as usize].clone())),
            Ffn::Dense(_) => panic!("load_expert on dense layer"),
        }
    }
}

fn random_moe_layers(cfg: &BlockConfig, n: u32, seed: u64) -> Vec<LayerTensors> {
    let mut rng = Rng::new(seed);
    let m = cfg.moe.expect("moe cfg");
    let h = cfg.hidden_size;
    let inter = m.moe_intermediate_size as usize;
    let s = 0.05;
    let expert = |rng: &mut Rng| ExpertFfn {
        gate: Weights::from_f32(rng.vec(inter * h, s)),
        up: Weights::from_f32(rng.vec(inter * h, s)),
        down: Weights::from_f32(rng.vec(h * inter, s)),
    };
    (0..n)
        .map(|_| {
            let shared_inter = m.shared_intermediate_size.map(|si| si as usize);
            let ffn = Ffn::Moe {
                router: Weights::from_f32(rng.vec(m.num_experts as usize * h, s)),
                experts: (0..m.num_experts).map(|_| expert(&mut rng)).collect(),
                shared: shared_inter.map(|si| ExpertFfn {
                    gate: Weights::from_f32(rng.vec(si * h, s)),
                    up: Weights::from_f32(rng.vec(si * h, s)),
                    down: Weights::from_f32(rng.vec(h * si, s)),
                }),
                shared_gate: shared_inter.map(|_| Weights::from_f32(rng.vec(h, s))),
            };
            LayerTensors {
                q_proj: Weights::from_f32(rng.vec(cfg.q_dim() * h, s)),
                k_proj: Weights::from_f32(rng.vec(cfg.kv_dim() * h, s)),
                v_proj: Weights::from_f32(rng.vec(cfg.kv_dim() * h, s)),
                o_proj: Weights::from_f32(rng.vec(h * cfg.q_dim(), s)),
                ffn,
                input_layernorm: vec![1.0; h],
                post_attention_layernorm: vec![1.0; h],
                ..Default::default()
            }
        })
        .collect()
}

fn assert_moe_gpu_matches_cpu(m: MoeConfig, what: &str) {
    let cfg = BlockConfig {
        hidden_size: 32,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        intermediate_size: 64,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        rope_scaling: None,
        moe: Some(m), sliding_window: None, activation: Default::default(), mla: None,
    };
    let num_layers = 6u32;
    let layers = random_moe_layers(&cfg, num_layers, 0x50FA);

    // Resident CPU (all experts inline) vs streaming GPU (core resident window of
    // 2, experts streamed per (layer, expert)).
    let cpu = CpuKernel::new(cfg, layers.clone()).unwrap();
    let gpu = StreamingGpuKernel::new(cfg, MoeVecSource(layers), 64, 2, None).unwrap();

    let kv_cfg = KvCacheConfig {
        num_layers,
        num_kv_heads: cfg.num_kv_heads as u32,
        head_dim: cfg.head_dim as u32,
        block_size: 16,
    };
    let mut orch_cpu =
        ForwardOrchestrator::new(cpu, PagedKvCache::new(kv_cfg, 32), dlm::forward::KvQuant::None);
    let mut orch_gpu =
        ForwardOrchestrator::new(gpu, PagedKvCache::new(kv_cfg, 32), dlm::forward::KvQuant::None);

    let mut h_cpu: Vec<f32> = (0..cfg.hidden_size).map(|i| i as f32 * 0.02 - 0.3).collect();
    let mut h_gpu = h_cpu.clone();
    for step in 0..4 {
        orch_cpu.decode_token(&mut h_cpu).unwrap();
        orch_gpu.decode_token(&mut h_gpu).unwrap();
        let max_diff =
            h_cpu.iter().zip(&h_gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 2e-3, "{what}: step {step}: GPU MoE diverged by {max_diff}");
    }
}

/// Mixtral-style: routed experts, no shared expert.
#[test]
fn gpu_moe_matches_cpu_routed_only() {
    assert_moe_gpu_matches_cpu(
        MoeConfig {
            num_experts: 4,
            experts_per_tok: 2,
            moe_intermediate_size: 64,
            shared_intermediate_size: None,
            norm_topk_prob: true,
            naming: MoeNaming::Mixtral,
        },
        "routed-only",
    );
}

/// Qwen-style: routed experts plus a sigmoid-gated shared expert.
#[test]
fn gpu_moe_matches_cpu_with_shared_expert() {
    assert_moe_gpu_matches_cpu(
        MoeConfig {
            num_experts: 4,
            experts_per_tok: 2,
            moe_intermediate_size: 64,
            shared_intermediate_size: Some(64),
            norm_topk_prob: true,
            naming: MoeNaming::Qwen,
        },
        "shared-expert",
    );
}
