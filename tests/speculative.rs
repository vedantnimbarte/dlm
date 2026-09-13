//! Speculative decoding must produce exactly the target-greedy sequence.

use dlm::batching::BatchScheduler;
use dlm::cache::KvCacheConfig;
use dlm::forward::Weights;
use dlm::forward::{BlockConfig, CpuKernel, ExpertFfn, Ffn, LayerTensors};
use dlm::generate::{GenerationConfig, Generator, Sampler};
use dlm::speculative::{SpeculativeDecoder, SpeculativeSession};

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

/// Deterministic small generator keyed by `seed` (same seed → identical model).
fn build_generator(seed: u64) -> Generator<CpuKernel> {
    let vocab = 32usize;
    let hidden = 16usize;
    let cfg = BlockConfig {
        hidden_size: hidden,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 32,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        rope_scaling: None,
        moe: None,
        sliding_window: None,
        activation: Default::default(),
        mla: None,
        ..Default::default()
    };
    let mut rng = Rng::new(seed);
    let s = 0.05;
    let layers = vec![LayerTensors {
        q_proj: Weights::from_f32(rng.vec(cfg.q_dim() * hidden, s)),
        k_proj: Weights::from_f32(rng.vec(cfg.kv_dim() * hidden, s)),
        v_proj: Weights::from_f32(rng.vec(cfg.kv_dim() * hidden, s)),
        o_proj: Weights::from_f32(rng.vec(hidden * cfg.q_dim(), s)),
        ffn: Ffn::Dense(ExpertFfn {
            gate: Weights::from_f32(rng.vec(cfg.intermediate_size * hidden, s)),
            up: Weights::from_f32(rng.vec(cfg.intermediate_size * hidden, s)),
            down: Weights::from_f32(rng.vec(hidden * cfg.intermediate_size, s)),
            up_bias: None,
            down_bias: None,
        }),
        input_layernorm: vec![1.0; hidden],
        post_attention_layernorm: vec![1.0; hidden],
        ..Default::default()
    }];
    let kernel = CpuKernel::new(cfg, layers).unwrap();
    let embedding = rng.vec(vocab * hidden, s);
    let lm_head = rng.vec(vocab * hidden, s);
    Generator::new(
        kernel,
        embedding,
        vec![1.0; hidden],
        lm_head,
        vocab,
        1e-5,
        KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 4,
            block_size: 16,
        },
        64,
    )
    .unwrap()
}

fn greedy(gen: &Generator<CpuKernel>, prompt: &[u32], n: usize) -> Vec<u32> {
    gen.generate(
        prompt,
        &GenerationConfig {
            max_new_tokens: n,
            eos_token: None,
            sampler: Sampler::Greedy,
        },
    )
    .unwrap()
}

#[test]
fn speculative_output_equals_target_greedy() {
    let prompt = [1u32, 2, 3];
    let n = 8;

    // Different draft (seed 2) vs. target (seed 1): output must still be exact.
    // Every gamma, because a rejection at the *last* proposal is its own case
    // (it must not also draw a bonus token), and gamma 1 makes it the only case.
    let reference = greedy(&build_generator(1), &prompt, n);
    for gamma in [1, 2, 3, 4] {
        let decoder = SpeculativeDecoder::new(build_generator(1), build_generator(2), gamma);
        let spec = decoder.generate(&prompt, n).unwrap();
        assert_eq!(
            spec.tokens, reference,
            "gamma {gamma}: speculative diverged from target-greedy"
        );
        assert!(
            spec.accepted < spec.proposed,
            "gamma {gamma}: no rejection exercised"
        );
    }
}

#[test]
fn identical_draft_is_fully_accepted() {
    let prompt = [5u32, 6];
    let n = 6;

    // Draft == target → every proposal matches, 100% acceptance.
    let decoder = SpeculativeDecoder::new(build_generator(1), build_generator(1), 4);
    let spec = decoder.generate(&prompt, n).unwrap();

    assert_eq!(spec.tokens, greedy(&build_generator(1), &prompt, n));
    assert!(spec.proposed > 0);
    assert_eq!(spec.accepted, spec.proposed);
    assert!((spec.acceptance_rate() - 1.0).abs() < 1e-9);
}

#[test]
fn session_round_by_round_equals_target_greedy() {
    let prompt = [1u32, 2, 3];
    let n = 8;
    let (target, draft) = (build_generator(1), build_generator(2));

    // Drive the resumable session one round at a time (as the scheduler does),
    // capping each round to the tokens still wanted.
    let mut session =
        SpeculativeSession::new(&target, &draft, 4, &prompt, Sampler::Greedy).unwrap();
    let mut out = Vec::new();
    while out.len() < n {
        let emitted = session.step(n - out.len()).unwrap();
        assert!(!emitted.is_empty(), "a round must make progress");
        assert!(out.len() + emitted.len() <= n, "round overshot the budget");
        out.extend(emitted);
    }

    assert_eq!(out, greedy(&build_generator(1), &prompt, n));
    assert!(session.accepted() <= session.proposed());
}

#[test]
fn speculative_scheduler_matches_greedy_across_requests() {
    let target = build_generator(1);
    let draft = build_generator(2); // different draft: exactness must still hold
    let requests: Vec<(u64, Vec<u32>, usize)> =
        vec![(1, vec![1, 2, 3], 7), (2, vec![7], 5), (3, vec![4, 5], 6)];

    // max_batch = 2 forces staggered admission while speculating.
    let mut sched = BatchScheduler::with_speculative(&target, &draft, 2, 4);
    for (id, prompt, n) in &requests {
        sched.submit(*id, prompt.clone(), *n, vec![]).unwrap();
    }
    let mut results = sched.run().unwrap();
    results.sort_by_key(|f| f.id);

    assert_eq!(results.len(), requests.len());
    for (f, (id, prompt, n)) in results.iter().zip(&requests) {
        assert_eq!(
            f.tokens,
            greedy(&target, prompt, *n),
            "request {id} diverged"
        );
        assert_eq!(f.tokens.len(), *n);
    }
    // Every retired slot contributed to the acceptance stats.
    let (proposed, accepted) = sched.speculative_stats();
    assert!(proposed > 0 && accepted <= proposed);
}

#[test]
fn speculative_scheduler_respects_eos() {
    let target = build_generator(1);
    let draft = build_generator(1); // identical draft → full acceptance
                                    // The token greedy decoding would emit second, used as EOS.
    let two = greedy(&target, &[1, 2], 2);
    let eos = two[1];

    let mut sched = BatchScheduler::with_speculative(&target, &draft, 4, 4);
    sched.submit(1, vec![1, 2], 10, vec![eos]).unwrap();
    let results = sched.run().unwrap();

    assert_eq!(results.len(), 1);
    // Stops at EOS (included), before the 10-token cap, matching plain decoding.
    assert_eq!(*results[0].tokens.last().unwrap(), eos);
    assert_eq!(results[0].tokens, two);
}

/// Sampled speculation must follow the **target's** distribution, not the
/// draft's. Over many seeds, the first two emitted tokens' marginals under
/// speculation must match plain target sampling, while plain draft sampling
/// visibly does not — so a session that passed draft tokens through unchecked,
/// or resampled rejections from the wrong distribution, fails.
///
/// A low temperature sharpens both models' distributions, so target and draft
/// disagree enough for the test to have teeth. The first token exercises
/// accept/reject against the draft; the second also exercises the bonus draw
/// and the KV rollback between rounds (gamma 1 makes every token its own round).
#[test]
fn sampled_speculation_follows_the_target_distribution() {
    let (target, draft) = (build_generator(1), build_generator(2));
    let prompt = [3u32, 9];
    let runs = 8000;
    let sampler = |seed| Sampler::TopPK {
        temperature: 0.08,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        seed,
    };
    let vocab = 32;
    let mut spec = vec![vec![0f64; vocab]; 2];
    let mut plain_target = vec![vec![0f64; vocab]; 2];
    let mut plain_draft = vec![vec![0f64; vocab]; 2];
    for seed in 0..runs as u64 {
        let mut s = SpeculativeSession::new(&target, &draft, 1, &prompt, sampler(seed)).unwrap();
        let mut out = Vec::new();
        while out.len() < 2 {
            out.extend(s.step(2 - out.len()).unwrap());
        }
        let cfg = GenerationConfig {
            max_new_tokens: 2,
            eos_token: None,
            sampler: sampler(seed + 1_000_000),
        };
        let t = target.generate(&prompt, &cfg).unwrap();
        let d = draft.generate(&prompt, &cfg).unwrap();
        for i in 0..2 {
            spec[i][out[i] as usize] += 1.0;
            plain_target[i][t[i] as usize] += 1.0;
            plain_draft[i][d[i] as usize] += 1.0;
        }
    }
    let tv = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / runs as f64)
            .sum::<f64>()
            / 2.0
    };
    for i in 0..2 {
        let separation = tv(&plain_draft[i], &plain_target[i]);
        assert!(
            separation > 0.3,
            "token {i}: draft and target too alike ({separation:.3}) for this test to mean anything"
        );
        let err = tv(&spec[i], &plain_target[i]);
        assert!(
            err < 0.06,
            "token {i}: speculative marginal is {err:.3} from the target's (draft is {separation:.3})"
        );
    }
}
