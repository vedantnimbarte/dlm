//! Master/coordinator node (`specs.md` §3.4).
//!
//! Holds the **full** model (so it can serve any shard from local CPU RAM) plus
//! a routing table mapping each layer shard to a worker address (or local). A
//! forward pass streams the hidden state through the shards in pipeline order;
//! for each remote shard it sends [`Message::RunShard`] and awaits the result.
//!
//! **Fault tolerance** (`specs.md` §3.4 heartbeats): if a worker is unreachable,
//! the coordinator transparently runs that shard from its local weights instead
//! — the "CPU RAM fallback" — so a forward pass still completes. Because the
//! coordinator keeps its own KV for every layer, a shard that falls back *from
//! the start of a sequence* produces exactly the same output as a fully local
//! run. (Mid-sequence failover would need KV migration and is not attempted.)

use crate::distributed::protocol::{read_message, write_message, Message};
use crate::distributed::shard::LayerShard;
use crate::error::{DlmError, Result};
use crate::forward::cpu::{decode_block, matvec, rmsnorm, BlockConfig, KvLayerCache, LayerTensors};
use crate::generate::argmax;
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// Connect with a bounded timeout so an unreachable worker fails fast (rather
/// than blocking on the OS default connect timeout).
fn connect_timeout(addr: &str, timeout: Duration) -> std::io::Result<TcpStream> {
    let sock = addr
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "bad address"))?;
    TcpStream::connect_timeout(&sock, timeout)
}

/// A shard assignment: a layer range and the worker responsible for it (or
/// `None` for always-local).
#[derive(Debug, Clone)]
pub struct ShardRoute {
    pub shard: LayerShard,
    pub worker_addr: Option<String>,
}

/// The coordinator: full model weights + shard routing + local KV for fallback.
pub struct Coordinator {
    cfg: BlockConfig,
    layers: Vec<LayerTensors>,
    embedding: Vec<f32>,
    final_norm: Vec<f32>,
    lm_head: Vec<f32>,
    vocab_size: usize,
    kv: Vec<KvLayerCache>,
    routes: Vec<ShardRoute>,
    connections: Vec<Option<TcpStream>>,
    alive: Vec<bool>,
}

impl Coordinator {
    /// Build a coordinator over the full model and a routing table.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: BlockConfig,
        layers: Vec<LayerTensors>,
        embedding: Vec<f32>,
        final_norm: Vec<f32>,
        lm_head: Vec<f32>,
        vocab_size: usize,
        routes: Vec<ShardRoute>,
    ) -> Result<Self> {
        for layer in &layers {
            layer.validate(&cfg)?;
        }
        let kv = (0..layers.len())
            .map(|_| KvLayerCache::new(cfg.kv_dim()))
            .collect();
        let n = routes.len();
        Ok(Self {
            cfg,
            layers,
            embedding,
            final_norm,
            lm_head,
            vocab_size,
            kv,
            routes,
            connections: (0..n).map(|_| None).collect(),
            alive: vec![true; n],
        })
    }

    /// Latest per-shard liveness (updated by forward passes and heartbeats).
    pub fn alive(&self) -> &[bool] {
        &self.alive
    }

    fn embed(&self, token: u32) -> Result<Vec<f32>> {
        let idx = token as usize;
        if idx >= self.vocab_size {
            return Err(DlmError::InvalidConfig(format!(
                "token {token} out of vocab range {}",
                self.vocab_size
            )));
        }
        let h = self.cfg.hidden_size;
        Ok(self.embedding[idx * h..(idx + 1) * h].to_vec())
    }

    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let normed = rmsnorm(hidden, &self.final_norm, self.cfg.rms_eps);
        matvec(&self.lm_head, &normed, self.vocab_size, self.cfg.hidden_size)
    }

    fn run_local_shard(&mut self, shard: LayerShard, hidden: &mut [f32], position: usize) -> Result<()> {
        for l in shard.start..shard.end {
            let out = decode_block(&self.cfg, &self.layers[l], hidden, &mut self.kv[l], position)?;
            hidden.copy_from_slice(&out);
        }
        Ok(())
    }

    /// Send a shard's work to its worker, returning the updated hidden state.
    fn run_remote(&mut self, i: usize, hidden: &[f32], position: usize) -> Result<Vec<f32>> {
        if self.connections[i].is_none() {
            let addr = self.routes[i].worker_addr.as_ref().unwrap();
            let stream = connect_timeout(addr, Duration::from_millis(300))
                .map_err(|e| DlmError::Network(format!("connect {addr}: {e}")))?;
            self.connections[i] = Some(stream);
        }
        let stream = self.connections[i].as_mut().unwrap();
        write_message(
            stream,
            &Message::RunShard {
                position: position as u64,
                hidden: hidden.to_vec(),
            },
        )
        .map_err(|e| DlmError::Network(e.to_string()))?;
        match read_message(stream) {
            Ok(Message::ShardResult { hidden }) => Ok(hidden),
            Ok(Message::Error(e)) => Err(DlmError::Network(e)),
            Ok(_) => Err(DlmError::Network("unexpected reply".into())),
            Err(e) => Err(DlmError::Network(e.to_string())),
        }
    }

    /// One token through the whole pipeline; returns the final hidden state.
    fn forward_token(&mut self, token: u32, position: usize) -> Result<Vec<f32>> {
        let mut hidden = self.embed(token)?;
        for i in 0..self.routes.len() {
            let shard = self.routes[i].shard;
            if self.routes[i].worker_addr.is_some() {
                match self.run_remote(i, &hidden, position) {
                    Ok(h) => {
                        hidden = h;
                        self.alive[i] = true;
                    }
                    Err(_) => {
                        // Worker unreachable → CPU RAM fallback for this shard.
                        self.alive[i] = false;
                        self.connections[i] = None;
                        self.run_local_shard(shard, &mut hidden, position)?;
                    }
                }
            } else {
                self.run_local_shard(shard, &mut hidden, position)?;
            }
        }
        Ok(hidden)
    }

    /// Reset all local KV (start of a new sequence).
    fn reset_kv(&mut self) {
        for kv in &mut self.kv {
            *kv = KvLayerCache::new(self.cfg.kv_dim());
        }
    }

    /// Greedily generate `max_new_tokens` for `prompt`, routing every token
    /// through the pipeline. Output matches a local run of the same weights.
    pub fn generate(&mut self, prompt: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(DlmError::InvalidConfig("prompt must be non-empty".into()));
        }
        self.reset_kv();

        let mut position = 0usize;
        let mut hidden = vec![0.0f32; self.cfg.hidden_size];
        for &token in prompt {
            hidden = self.forward_token(token, position)?;
            position += 1;
        }

        let mut out = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let next = argmax(&self.logits(&hidden));
            out.push(next);
            hidden = self.forward_token(next, position)?;
            position += 1;
        }
        Ok(out)
    }

    /// Ping every remote worker and update liveness. Returns the per-shard state.
    pub fn heartbeat(&mut self) -> Vec<bool> {
        for i in 0..self.routes.len() {
            self.alive[i] = match self.routes[i].worker_addr.clone() {
                Some(addr) => ping(&addr),
                None => true,
            };
        }
        self.alive.clone()
    }
}

/// Connect, Ping, expect Pong — a low-overhead liveness check.
fn ping(addr: &str) -> bool {
    let Ok(mut stream) = connect_timeout(addr, Duration::from_millis(300)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    write_message(&mut stream, &Message::Ping).is_ok()
        && matches!(read_message(&mut stream), Ok(Message::Pong))
}
