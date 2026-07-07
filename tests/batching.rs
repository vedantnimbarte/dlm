//! Continuous batching must produce, per request, exactly the same tokens as
//! running that request in isolation — regardless of interleaving or batch size.

use dlm::batching::BatchScheduler;
use dlm::cache::KvCacheConfig;
use dlm::forward::{BlockConfig, CpuKernel, LayerTensors};
use dlm::generate::{GenerationConfig, Generator, Sampler};

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

fn build_generator() -> Generator<CpuKernel> {
    let (vocab, hidden) = (32usize, 16usize);
    let cfg = BlockConfig {
        hidden_size: hidden,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 32,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
    };
    let mut rng = Rng::new(7);
    let s = 0.05;
    let layers = vec![LayerTensors {
        q_proj: rng.vec(cfg.q_dim() * hidden, s),
        k_proj: rng.vec(cfg.kv_dim() * hidden, s),
        v_proj: rng.vec(cfg.kv_dim() * hidden, s),
        o_proj: rng.vec(hidden * cfg.q_dim(), s),
        gate_proj: rng.vec(cfg.intermediate_size * hidden, s),
        up_proj: rng.vec(cfg.intermediate_size * hidden, s),
        down_proj: rng.vec(hidden * cfg.intermediate_size, s),
        input_layernorm: vec![1.0; hidden],
        post_attention_layernorm: vec![1.0; hidden],
    }];
    let kernel = CpuKernel::new(cfg, layers).unwrap();
    Generator::new(
        kernel,
        rng.vec(vocab * hidden, s),
        vec![1.0; hidden],
        rng.vec(vocab * hidden, s),
        vocab,
        1e-5,
        KvCacheConfig { num_layers: 1, num_kv_heads: 2, head_dim: 4, block_size: 16 },
        64,
    )
    .unwrap()
}

fn greedy(gen: &Generator<CpuKernel>, prompt: &[u32], n: usize) -> Vec<u32> {
    gen.generate(
        prompt,
        &GenerationConfig { max_new_tokens: n, eos_token: None, sampler: Sampler::Greedy },
    )
    .unwrap()
}

#[test]
fn batched_requests_match_isolated_generation() {
    let generator = build_generator();
    // Requests with different prompts and lengths.
    let requests: Vec<(u64, Vec<u32>, usize)> = vec![
        (1, vec![1, 2, 3], 5),
        (2, vec![7], 8),
        (3, vec![4, 5], 3),
        (4, vec![9, 1, 2], 6),
    ];

    // max_batch = 2 forces staggered admission (4 requests, 2 slots).
    let mut sched = BatchScheduler::new(&generator, 2);
    for (id, prompt, n) in &requests {
        sched.submit(*id, prompt.clone(), *n, None).unwrap();
        assert!(sched.active_len() <= 2);
    }
    let mut results = sched.run().unwrap();
    results.sort_by_key(|f| f.id);

    assert_eq!(results.len(), requests.len());
    for (f, (id, prompt, n)) in results.iter().zip(&requests) {
        assert_eq!(f.id, *id);
        assert_eq!(f.tokens, greedy(&generator, prompt, *n), "request {id} diverged");
        assert_eq!(f.tokens.len(), *n);
    }
}

#[test]
fn batch_size_one_is_sequential_but_correct() {
    let generator = build_generator();
    let mut sched = BatchScheduler::new(&generator, 1);
    sched.submit(10, vec![1, 2], 4, None).unwrap();
    sched.submit(20, vec![3], 4, None).unwrap();

    let mut results = sched.run().unwrap();
    results.sort_by_key(|f| f.id);
    assert_eq!(results[0].tokens, greedy(&generator, &[1, 2], 4));
    assert_eq!(results[1].tokens, greedy(&generator, &[3], 4));
}

#[test]
fn eos_stops_a_request_early() {
    let generator = build_generator();
    // Find the first token greedily produced for a prompt, use it as EOS.
    let first = greedy(&generator, &[1], 1)[0];

    let mut sched = BatchScheduler::new(&generator, 4);
    sched.submit(1, vec![1], 10, Some(first)).unwrap();
    let results = sched.run().unwrap();

    assert_eq!(results.len(), 1);
    // Stops at the EOS token (included), well before the 10-token limit.
    assert_eq!(*results[0].tokens.last().unwrap(), first);
    assert!(results[0].tokens.len() < 10);
}
