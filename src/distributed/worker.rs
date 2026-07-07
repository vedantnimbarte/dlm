//! Worker node: owns a layer shard and serves forward-pass requests
//! (`specs.md` §3.4 Master-Worker topology).
//!
//! A worker holds its shard's weights and per-layer KV history and answers
//! [`Message::RunShard`] over TCP: it runs its transformer blocks for one token
//! and returns the updated hidden state. It resets its KV when it sees position
//! 0 (a new sequence). Each connection is handled on its own thread (state behind
//! a mutex), so heartbeat pings are answered even while a compute connection is
//! open.

use crate::distributed::protocol::{read_message, write_message, Message};
use crate::error::{DlmError, Result};
use crate::forward::cpu::{decode_block, BlockConfig, KvLayerCache, LayerTensors};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

/// Mutable worker state: its shard weights and KV history.
struct WorkerState {
    cfg: BlockConfig,
    layers: Vec<LayerTensors>,
    kv: Vec<KvLayerCache>,
}

impl WorkerState {
    fn run_shard(&mut self, hidden: &mut [f32], position: usize) -> Result<()> {
        if position == 0 {
            for kv in &mut self.kv {
                *kv = KvLayerCache::new(self.cfg.kv_dim());
            }
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let out = decode_block(&self.cfg, layer, hidden, &mut self.kv[i], position)?;
            hidden.copy_from_slice(&out);
        }
        Ok(())
    }
}

/// A worker holding one shard of the model.
pub struct Worker {
    state: Arc<Mutex<WorkerState>>,
    hidden_size: usize,
}

impl Worker {
    /// Create a worker for `layers` (its shard), validating dimensions.
    pub fn new(cfg: BlockConfig, layers: Vec<LayerTensors>) -> Result<Self> {
        for layer in &layers {
            layer.validate(&cfg)?;
        }
        let kv = (0..layers.len())
            .map(|_| KvLayerCache::new(cfg.kv_dim()))
            .collect();
        let hidden_size = cfg.hidden_size;
        Ok(Self {
            state: Arc::new(Mutex::new(WorkerState { cfg, layers, kv })),
            hidden_size,
        })
    }

    /// Serve requests on `listener` forever, one thread per connection.
    pub fn serve(self, listener: TcpListener) -> Result<()> {
        for conn in listener.incoming() {
            let Ok(stream) = conn else { continue };
            let state = Arc::clone(&self.state);
            let hidden_size = self.hidden_size;
            std::thread::spawn(move || handle_connection(stream, state, hidden_size));
        }
        Ok(())
    }
}

fn handle_connection(mut stream: TcpStream, state: Arc<Mutex<WorkerState>>, hidden_size: usize) {
    loop {
        match read_message(&mut stream) {
            Ok(Message::RunShard { position, mut hidden }) => {
                if hidden.len() != hidden_size {
                    let _ = write_message(&mut stream, &Message::Error("hidden size mismatch".into()));
                    break;
                }
                let result = {
                    let mut w = state.lock().unwrap();
                    w.run_shard(&mut hidden, position as usize)
                };
                let reply = match result {
                    Ok(()) => Message::ShardResult { hidden },
                    Err(e) => Message::Error(e.to_string()),
                };
                if write_message(&mut stream, &reply).is_err() {
                    break;
                }
            }
            Ok(Message::Ping) => {
                if write_message(&mut stream, &Message::Pong).is_err() {
                    break;
                }
            }
            Ok(_) => {}
            Err(_) => break, // connection closed or bad frame
        }
    }
}

/// Bind a worker listener to `addr` (e.g. `"127.0.0.1:0"`).
pub fn bind(addr: &str) -> Result<TcpListener> {
    TcpListener::bind(addr).map_err(|e| DlmError::Network(format!("bind {addr}: {e}")))
}
