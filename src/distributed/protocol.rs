//! Wire protocol for the distributed pipeline (`specs.md` §3.4).
//!
//! Messages are **Protocol Buffers** (encoded with `prost`), sent as a
//! length-prefixed frame over any `Read`/`Write` (TCP in practice). Per
//! `specs.md` §3.4, multi-dimensional tensors ride in a flat `repeated float`
//! field — proto3 packs these as raw little-endian `f32`, so a value computed on
//! a worker round-trips **bit-for-bit** and a distributed forward pass matches a
//! local one exactly.
//!
//! We use Protobuf for the *serialization* but keep the plain synchronous TCP
//! framing below (`[u32 payload_len][protobuf payload]`) rather than the full
//! gRPC/tonic/HTTP-2 stack — the distributed layer stays synchronous, thread-per-
//! connection, and dependency-light, and is testable over localhost.

use prost::Message as _;
use std::io::{self, Read, Write};

/// A protocol message (the public, hand-written API; the Protobuf types below
/// are an encoding detail).
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// Master → worker: run your layer shard for one token.
    RunShard { position: u64, hidden: Vec<f32> },
    /// Worker → master: the shard's output hidden state.
    ShardResult { hidden: Vec<f32> },
    /// Heartbeat request.
    Ping,
    /// Heartbeat reply.
    Pong,
    /// An error string.
    Error(String),
}

// ── Protobuf schema (equivalent to a hand-written `.proto`, so no `protoc`/
// build-time codegen is needed). Tensors are `repeated float`, packed by proto3
// into raw little-endian f32 → bit-exact. ──────────────────────────────────────

#[derive(Clone, PartialEq, ::prost::Message)]
struct RunShardPb {
    #[prost(uint64, tag = "1")]
    position: u64,
    #[prost(float, repeated, tag = "2")]
    hidden: Vec<f32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
struct HiddenPb {
    #[prost(float, repeated, tag = "1")]
    hidden: Vec<f32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
struct ErrorPb {
    #[prost(string, tag = "1")]
    message: String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
struct EmptyPb {}

#[derive(Clone, PartialEq, ::prost::Oneof)]
enum Body {
    #[prost(message, tag = "1")]
    RunShard(RunShardPb),
    #[prost(message, tag = "2")]
    ShardResult(HiddenPb),
    #[prost(message, tag = "3")]
    Ping(EmptyPb),
    #[prost(message, tag = "4")]
    Pong(EmptyPb),
    #[prost(message, tag = "5")]
    Error(ErrorPb),
}

#[derive(Clone, PartialEq, ::prost::Message)]
struct Envelope {
    #[prost(oneof = "Body", tags = "1, 2, 3, 4, 5")]
    body: Option<Body>,
}

impl From<&Message> for Envelope {
    fn from(msg: &Message) -> Self {
        let body = match msg {
            Message::RunShard { position, hidden } => Body::RunShard(RunShardPb {
                position: *position,
                hidden: hidden.clone(),
            }),
            Message::ShardResult { hidden } => Body::ShardResult(HiddenPb {
                hidden: hidden.clone(),
            }),
            Message::Ping => Body::Ping(EmptyPb {}),
            Message::Pong => Body::Pong(EmptyPb {}),
            Message::Error(s) => Body::Error(ErrorPb { message: s.clone() }),
        };
        Envelope { body: Some(body) }
    }
}

/// Encode a message into its Protobuf payload bytes (without the length prefix).
pub fn encode(msg: &Message) -> Vec<u8> {
    Envelope::from(msg).encode_to_vec()
}

/// Decode a message from its Protobuf payload bytes.
pub fn decode(payload: &[u8]) -> io::Result<Message> {
    let env = Envelope::decode(payload)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    match env.body {
        Some(Body::RunShard(r)) => Ok(Message::RunShard {
            position: r.position,
            hidden: r.hidden,
        }),
        Some(Body::ShardResult(s)) => Ok(Message::ShardResult { hidden: s.hidden }),
        Some(Body::Ping(_)) => Ok(Message::Ping),
        Some(Body::Pong(_)) => Ok(Message::Pong),
        Some(Body::Error(e)) => Ok(Message::Error(e.message)),
        None => Err(io::Error::new(io::ErrorKind::InvalidData, "empty envelope")),
    }
}

/// Write a length-prefixed message to `w`.
pub fn write_message(w: &mut impl Write, msg: &Message) -> io::Result<()> {
    let payload = encode(msg);
    w.write_all(&(payload.len() as u32).to_le_bytes())?;
    w.write_all(&payload)?;
    w.flush()
}

/// Read a length-prefixed message from `r`.
pub fn read_message(r: &mut impl Read) -> io::Result<Message> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload)?;
    decode(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(msg: Message) {
        let mut buf = Vec::new();
        write_message(&mut buf, &msg).unwrap();
        let decoded = read_message(&mut &buf[..]).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn messages_round_trip_exactly() {
        round_trip(Message::Ping);
        round_trip(Message::Pong);
        round_trip(Message::Error("boom".into()));
        round_trip(Message::RunShard {
            position: 7,
            hidden: vec![1.5, -2.25, 0.0, f32::MIN_POSITIVE, 12345.678],
        });
        round_trip(Message::ShardResult {
            hidden: vec![-0.001, 42.0],
        });
    }

    #[test]
    fn floats_are_bit_exact() {
        // Tricky values that a text encoding could perturb; proto3 packs
        // `repeated float` as raw LE f32, so equality is bit-for-bit.
        let hidden = vec![0.1f32, 0.2, 0.3, 1.0 / 3.0, std::f32::consts::PI];
        let mut buf = Vec::new();
        write_message(&mut buf, &Message::ShardResult { hidden: hidden.clone() }).unwrap();
        let Message::ShardResult { hidden: got } = read_message(&mut &buf[..]).unwrap() else {
            panic!("wrong message");
        };
        assert_eq!(got, hidden); // exact bit equality
    }

    #[test]
    fn payload_is_valid_protobuf() {
        // The frame payload is now the Protobuf encoding of the Envelope, not the
        // old hand-rolled tag framing; an empty tensor still decodes cleanly.
        let payload = encode(&Message::RunShard { position: 3, hidden: vec![] });
        assert_eq!(
            decode(&payload).unwrap(),
            Message::RunShard { position: 3, hidden: vec![] }
        );
    }
}
