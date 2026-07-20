//! Multi-GPU pipeline parallelism (`specs.md` §3.3): splitting a model's layers
//! across several GPUs must produce exactly the same tokens as running it on one
//! device. Off-GPU, `gpu::set_device` is a no-op, so this parity check exercises
//! the real partitioning + per-layer dispatch on the CPU kernel — the same way
//! `distributed.rs` validates the cross-node path over localhost.

use dlm::forward::Weights;
use dlm::cache::KvCacheConfig;
use dlm::generate::{GenerationConfig, Generator, Sampler};
use dlm::loader::ModelParts;
use dlm::forward::{BlockConfig, ExpertFfn, Ffn, LayerTensors};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn vec(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n)
            .map(|_| ((self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * s)
            .collect()
    }
}

/// Build fresh synthetic parts (the loader consumes them by value, so each
/// generator needs its own copy).
fn build_parts() -> ModelParts {
    let (vocab, hidden, num_layers) = (32usize, 16usize, 5usize);
    let cfg = BlockConfig {
        hidden_size: hidden,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 32,
        rope_theta: 10000.0,
        rms_eps: 1e-5, rope_scaling: None, moe: None, sliding_window: None, activation: Default::default(), mla: None,
    };
    let mut rng = Rng::new(7);
    let s = 0.05;
    let layers = (0..num_layers)
        .map(|_| LayerTensors {
            q_proj: Weights::from_f32(rng.vec(cfg.q_dim() * hidden, s)),
            k_proj: Weights::from_f32(rng.vec(cfg.kv_dim() * hidden, s)),
            v_proj: Weights::from_f32(rng.vec(cfg.kv_dim() * hidden, s)),
            o_proj: Weights::from_f32(rng.vec(hidden * cfg.q_dim(), s)),
            ffn: Ffn::Dense(ExpertFfn { gate: Weights::from_f32(rng.vec(cfg.intermediate_size * hidden, s)), up: Weights::from_f32(rng.vec(cfg.intermediate_size * hidden, s)), down: Weights::from_f32(rng.vec(hidden * cfg.intermediate_size, s)) }),
            input_layernorm: vec![1.0; hidden],
            post_attention_layernorm: vec![1.0; hidden], ..Default::default()
        })
        .collect();
    ModelParts {
        cfg,
        layers,
        embedding: rng.vec(vocab * hidden, s),
        final_norm: vec![1.0; hidden],
        lm_head: rng.vec(vocab * hidden, s),
        vocab_size: vocab,
        rms_eps: cfg.rms_eps,
        kv_config: KvCacheConfig {
            num_layers: num_layers as u32,
            num_kv_heads: cfg.num_kv_heads as u32,
            head_dim: cfg.head_dim as u32,
            block_size: 16,
        },
        kv_blocks: 64,
        embed_scale: None,
    }
}

// build_parts uses the same seed each call, so two builds are identical models.
fn greedy<K: dlm::forward::ComputeKernel>(gen: &Generator<K>, prompt: &[u32], n: usize) -> Vec<u32> {
    gen.generate(
        prompt,
        &GenerationConfig { max_new_tokens: n, eos_token: None, sampler: Sampler::Greedy },
    )
    .unwrap()
}

/// Multi-GPU parity needs the requested device ids to actually exist. Off-GPU
/// `set_device` is a no-op so this always passes; under a real CUDA/HIP backend
/// it confirms the hardware is present, letting a runner with too few GPUs skip
/// rather than fail on `cudaSetDevice`.
fn devices_available(ids: &[u32]) -> bool {
    let ok = ids.iter().all(|&id| dlm::gpu::set_device(id).is_ok());
    let _ = dlm::gpu::set_device(0);
    ok
}

#[test]
fn pipeline_parallel_matches_single_device() {
    if !devices_available(&[0, 1, 2]) {
        eprintln!("skipping: needs 3 GPUs, not present on this runner");
        return;
    }
    let single = build_parts().into_cpu_generator().unwrap();

    // Split 5 layers across 3 GPUs (2/2/1). Output must be identical.
    let multi = build_parts()
        .into_pipeline_parallel_generator(&[0, 1, 2])
        .unwrap();

    let prompt = [4u32, 8, 15, 16];
    assert_eq!(
        greedy(&multi, &prompt, 8),
        greedy(&single, &prompt, 8),
        "multi-GPU pipeline diverged from single-device output",
    );
}

#[test]
fn single_gpu_id_is_whole_model_on_one_device() {
    let single = build_parts().into_cpu_generator().unwrap();
    // Device 0 (not an arbitrary id like 3) so this runs for real on any GPU
    // box; a single-element list still maps the whole model to one device.
    let one_gpu = build_parts().into_pipeline_parallel_generator(&[0]).unwrap();
    assert_eq!(greedy(&one_gpu, &[1, 2, 3], 6), greedy(&single, &[1, 2, 3], 6));
}
