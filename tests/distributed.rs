//! Distributed pipeline over localhost: a sharded forward pass must equal a
//! local one, fault-tolerant fallback must too, and heartbeats must reflect
//! worker liveness.

use dlm::forward::Weights;
use dlm::cache::KvCacheConfig;
use dlm::distributed::{partition_layers, Coordinator, ShardRoute, Worker};
use dlm::forward::{BlockConfig, CpuKernel, ExpertFfn, Ffn, LayerTensors};
use dlm::generate::{GenerationConfig, Generator, Sampler};
use std::net::TcpListener;

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

struct Model {
    cfg: BlockConfig,
    layers: Vec<LayerTensors>,
    embedding: Vec<f32>,
    final_norm: Vec<f32>,
    lm_head: Vec<f32>,
    vocab: usize,
}

fn build_model() -> Model {
    let (vocab, hidden, num_layers) = (32usize, 16usize, 4usize);
    let cfg = BlockConfig {
        hidden_size: hidden,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 32,
        rope_theta: 10000.0,
        rms_eps: 1e-5, rope_scaling: None, moe: None, sliding_window: None, activation: Default::default(),
    };
    let mut rng = Rng::new(99);
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
    Model {
        cfg,
        layers,
        embedding: rng.vec(vocab * hidden, s),
        final_norm: vec![1.0; hidden],
        lm_head: rng.vec(vocab * hidden, s),
        vocab,
    }
}

fn reference(m: &Model) -> Generator<CpuKernel> {
    let kernel = CpuKernel::new(m.cfg, m.layers.clone()).unwrap();
    Generator::new(
        kernel,
        m.embedding.clone(),
        m.final_norm.clone(),
        m.lm_head.clone(),
        m.vocab,
        m.cfg.rms_eps,
        KvCacheConfig {
            num_layers: m.layers.len() as u32,
            num_kv_heads: m.cfg.num_kv_heads as u32,
            head_dim: m.cfg.head_dim as u32,
            block_size: 16,
        },
        64,
    )
    .unwrap()
}

fn start_worker(cfg: BlockConfig, layers: Vec<LayerTensors>) -> String {
    start_worker_auth(cfg, layers, None)
}

fn start_worker_auth(cfg: BlockConfig, layers: Vec<LayerTensors>, secret: Option<&str>) -> String {
    let listener: TcpListener = dlm::distributed::worker::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let worker = Worker::new(cfg, layers).unwrap().with_auth(secret.map(String::from));
    std::thread::spawn(move || {
        let _ = worker.serve(listener);
    });
    addr
}

fn greedy(gen: &Generator<CpuKernel>, prompt: &[u32], n: usize) -> Vec<u32> {
    gen.generate(
        prompt,
        &GenerationConfig { max_new_tokens: n, eos_token: None, sampler: Sampler::Greedy },
    )
    .unwrap()
}

#[test]
fn distributed_forward_matches_local() {
    let m = build_model();
    let shards = partition_layers(m.layers.len(), 2);
    let a0 = start_worker(m.cfg, m.layers[shards[0].start..shards[0].end].to_vec());
    let a1 = start_worker(m.cfg, m.layers[shards[1].start..shards[1].end].to_vec());

    let routes = vec![
        ShardRoute { shard: shards[0], worker_addr: Some(a0) },
        ShardRoute { shard: shards[1], worker_addr: Some(a1) },
    ];
    let mut coord = Coordinator::new(
        m.cfg,
        m.layers.clone(),
        m.embedding.clone(),
        m.final_norm.clone(),
        m.lm_head.clone(),
        m.vocab,
        routes,
    )
    .unwrap();

    let dist = coord.generate(&[1, 2, 3], 6).unwrap();
    let local = greedy(&reference(&m), &[1, 2, 3], 6);
    assert_eq!(dist, local, "distributed diverged from local");
    assert!(coord.alive().iter().all(|&a| a));
}

#[test]
fn unreachable_worker_falls_back_to_local() {
    let m = build_model();
    let shards = partition_layers(m.layers.len(), 2);
    // Shard 0 has a live worker; shard 1 points at a dead port.
    let a0 = start_worker(m.cfg, m.layers[shards[0].start..shards[0].end].to_vec());
    let routes = vec![
        ShardRoute { shard: shards[0], worker_addr: Some(a0) },
        ShardRoute { shard: shards[1], worker_addr: Some("127.0.0.1:1".to_string()) },
    ];
    let mut coord = Coordinator::new(
        m.cfg,
        m.layers.clone(),
        m.embedding.clone(),
        m.final_norm.clone(),
        m.lm_head.clone(),
        m.vocab,
        routes,
    )
    .unwrap();

    // Still correct: the dead shard runs from the coordinator's local weights.
    let dist = coord.generate(&[1, 2, 3], 5).unwrap();
    assert_eq!(dist, greedy(&reference(&m), &[1, 2, 3], 5));
    assert_eq!(coord.alive(), &[true, false]);
}

#[test]
fn authenticated_worker_matches_local_and_rejects_wrong_secret() {
    let m = build_model();
    let shards = partition_layers(m.layers.len(), 2);
    let a0 = start_worker_auth(m.cfg, m.layers[shards[0].start..shards[0].end].to_vec(), Some("s3cret"));
    let a1 = start_worker_auth(m.cfg, m.layers[shards[1].start..shards[1].end].to_vec(), Some("s3cret"));
    let routes = vec![
        ShardRoute { shard: shards[0], worker_addr: Some(a0) },
        ShardRoute { shard: shards[1], worker_addr: Some(a1) },
    ];

    // Right secret → distributed output equals a local run, workers stay alive.
    let mut coord = Coordinator::new(
        m.cfg, m.layers.clone(), m.embedding.clone(), m.final_norm.clone(),
        m.lm_head.clone(), m.vocab, routes.clone(),
    )
    .unwrap()
    .with_auth(Some("s3cret".into()));
    assert_eq!(coord.generate(&[1, 2, 3], 6).unwrap(), greedy(&reference(&m), &[1, 2, 3], 6));
    assert!(coord.alive().iter().all(|&a| a));

    // Wrong secret → workers reject; coordinator falls back to local weights.
    // Output is still correct (fallback from position 0), but no worker is alive.
    let mut bad = Coordinator::new(
        m.cfg, m.layers.clone(), m.embedding.clone(), m.final_norm.clone(),
        m.lm_head.clone(), m.vocab, routes,
    )
    .unwrap()
    .with_auth(Some("wrong".into()));
    assert_eq!(bad.generate(&[1, 2, 3], 6).unwrap(), greedy(&reference(&m), &[1, 2, 3], 6));
    assert!(bad.alive().iter().all(|&a| !a), "wrong secret should mark all workers dead");
}

#[test]
fn heartbeat_reflects_worker_liveness() {
    let m = build_model();
    let shards = partition_layers(m.layers.len(), 2);
    let a0 = start_worker(m.cfg, m.layers[shards[0].start..shards[0].end].to_vec());
    let routes = vec![
        ShardRoute { shard: shards[0], worker_addr: Some(a0) },
        ShardRoute { shard: shards[1], worker_addr: Some("127.0.0.1:1".to_string()) },
    ];
    let mut coord = Coordinator::new(
        m.cfg,
        m.layers.clone(),
        m.embedding.clone(),
        m.final_norm.clone(),
        m.lm_head.clone(),
        m.vocab,
        routes,
    )
    .unwrap();

    assert_eq!(coord.heartbeat(), vec![true, false]);
}
