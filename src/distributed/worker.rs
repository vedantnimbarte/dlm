//! Worker node: owns a layer shard and serves forward-pass requests
//! (`specs.md` §3.4 Master-Worker topology).
//!
//! A worker holds its shard's weights and per-layer KV history and answers
//! [`Message::RunShard`] over TCP: it runs its transformer blocks for one token
//! and returns the updated hidden state. It resets its KV when it sees position
//! 0 (a new sequence). Each connection is handled on its own thread (state behind
//! a mutex), so heartbeat pings are answered even while a compute connection is
//! open.

use crate::distributed::protocol::{read_message, secret_eq, write_message, Message};
use crate::error::{DlmError, Result};
use crate::forward::cpu::{decode_block, BlockConfig, KvLayerCache, LayerTensors};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Max concurrent worker connections. Beyond this, new sockets are dropped
/// rather than spawning an unbounded number of threads (a hostile peer could
/// otherwise open connections until the process runs out of threads/memory).
const MAX_CONNECTIONS: usize = 64;

/// Default per-connection read/write timeout. A shard's forward pass for one
/// token is sub-second even on CPU, so a stalled or slow-loris peer that goes
/// quiet for this long is dropped rather than pinning a thread forever.
/// Override with `--worker-timeout-secs` (see [`Worker::with_io_timeout`]) when a
/// deployment's links are slower than that.
const IO_TIMEOUT: Duration = Duration::from_secs(30);

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
    secret: Option<Arc<String>>,
    /// Per-connection read/write timeout; defaults to [`IO_TIMEOUT`].
    io_timeout: Duration,
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
            secret: None,
            io_timeout: IO_TIMEOUT,
        })
    }

    /// Require `secret` as the first frame on every connection. With `None`
    /// (the default) the worker accepts any peer — only safe on a trusted
    /// network or localhost.
    pub fn with_auth(mut self, secret: Option<String>) -> Self {
        self.secret = secret.map(Arc::new);
        self
    }

    /// Override the per-connection read/write timeout (`--worker-timeout-secs`).
    /// `None`, or a zero duration, keeps the [`IO_TIMEOUT`] default — a worker
    /// with no timeout at all would let one silent peer pin a thread forever, so
    /// it is not an option.
    pub fn with_io_timeout(mut self, timeout: Option<Duration>) -> Self {
        if let Some(t) = timeout {
            if !t.is_zero() {
                self.io_timeout = t;
            }
        }
        self
    }

    /// Serve requests on `listener` forever, one thread per connection, up to
    /// [`MAX_CONNECTIONS`] concurrently.
    pub fn serve(self, listener: TcpListener) -> Result<()> {
        let live = Arc::new(AtomicUsize::new(0));
        for conn in listener.incoming() {
            let Ok(stream) = conn else { continue };
            if live.load(Ordering::Acquire) >= MAX_CONNECTIONS {
                // At capacity: drop the socket (closes it) rather than spawning.
                continue;
            }
            let state = Arc::clone(&self.state);
            let hidden_size = self.hidden_size;
            let secret = self.secret.clone();
            let live = Arc::clone(&live);
            let io_timeout = self.io_timeout;
            live.fetch_add(1, Ordering::AcqRel);
            std::thread::spawn(move || {
                handle_connection(stream, state, hidden_size, secret.as_deref(), io_timeout);
                live.fetch_sub(1, Ordering::AcqRel);
            });
        }
        Ok(())
    }
}

fn handle_connection(
    mut stream: TcpStream,
    state: Arc<Mutex<WorkerState>>,
    hidden_size: usize,
    secret: Option<&String>,
    io_timeout: Duration,
) {
    let _ = stream.set_read_timeout(Some(io_timeout));
    let _ = stream.set_write_timeout(Some(io_timeout));

    // When auth is required the first frame must be a matching Auth; anything
    // else closes the connection before any compute is done.
    if let Some(secret) = secret {
        match read_message(&mut stream) {
            Ok(Message::Auth(token)) if secret_eq(&token, secret) => {}
            _ => {
                let _ = write_message(&mut stream, &Message::Error("unauthorized".into()));
                return;
            }
        }
    }

    loop {
        match read_message(&mut stream) {
            // A stray Auth after the handshake is a no-op (idempotent).
            Ok(Message::Auth(_)) => {}
            Ok(Message::RunShard { position, mut hidden }) => {
                if hidden.len() != hidden_size {
                    let _ = write_message(&mut stream, &Message::Error("hidden size mismatch".into()));
                    break;
                }
                let result = {
                    // Recover from a poisoned lock (a prior panic) rather than
                    // letting one fault wedge every future connection.
                    let mut w = state.lock().unwrap_or_else(|e| e.into_inner());
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
