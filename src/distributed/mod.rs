//! Distributed multi-server topology (`specs.md` §3.4): a master coordinator
//! streaming a pipelined forward pass across worker nodes, with heartbeats and a
//! local CPU-RAM fallback for fault tolerance.
//!
//! Messages are **Protobuf** (`prost`) sent as length-prefixed frames over plain
//! TCP ([`protocol`]) — tensors ride in a packed `repeated float` field so they
//! round-trip bit-for-bit. We keep the synchronous TCP framing rather than the
//! full gRPC/tonic/HTTP-2 stack, so the layer stays sync, thread-per-connection,
//! and testable over localhost.

pub mod coordinator;
pub mod protocol;
pub mod shard;
pub mod worker;

pub use coordinator::{Coordinator, ShardRoute};
pub use protocol::{read_message, write_message, Message};
pub use shard::{partition_layers, LayerShard};
pub use worker::Worker;
