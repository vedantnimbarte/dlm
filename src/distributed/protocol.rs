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

/// Hard cap on an inbound frame's declared payload length. A worker or
/// coordinator reads a `u32` length prefix and then allocates that many bytes,
/// so without a cap a 4-byte header could force a ~4 GiB allocation (remote
/// OOM). A single hidden state is `hidden_size * 4` bytes — a few KiB even for
/// 70B-class models — so 64 MiB is orders of magnitude more headroom than any
/// legitimate frame needs while still refusing an abusive one.
pub const MAX_FRAME_LEN: usize = 64 * 1024 * 1024;

/// A protocol message (the public, hand-written API; the Protobuf types below
/// are an encoding detail).
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// Master → worker: authenticate this connection with the shared cluster
    /// secret. Must be the first frame when the worker requires auth.
    Auth(String),
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

/// Constant-time byte comparison, so a wrong cluster secret can't be recovered
/// by timing how long the check takes to fail.
pub fn secret_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
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
struct AuthPb {
    #[prost(string, tag = "1")]
    token: String,
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
    #[prost(message, tag = "6")]
    Auth(AuthPb),
}

#[derive(Clone, PartialEq, ::prost::Message)]
struct Envelope {
    #[prost(oneof = "Body", tags = "1, 2, 3, 4, 5, 6")]
    body: Option<Body>,
}

impl From<&Message> for Envelope {
    fn from(msg: &Message) -> Self {
        let body = match msg {
            Message::Auth(token) => Body::Auth(AuthPb {
                token: token.clone(),
            }),
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
        Some(Body::Auth(a)) => Ok(Message::Auth(a.token)),
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
    if len > MAX_FRAME_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds cap {MAX_FRAME_LEN}"),
        ));
    }
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
        write_message(
            &mut buf,
            &Message::ShardResult {
                hidden: hidden.clone(),
            },
        )
        .unwrap();
        let Message::ShardResult { hidden: got } = read_message(&mut &buf[..]).unwrap() else {
            panic!("wrong message");
        };
        assert_eq!(got, hidden); // exact bit equality
    }

    #[test]
    fn oversized_frame_is_rejected_without_allocating() {
        // A hostile 4-byte header claiming ~4 GiB must be refused at the length
        // check, before the payload vec is allocated.
        let mut frame = (u32::MAX).to_le_bytes().to_vec();
        frame.extend_from_slice(b"not really that long");
        let err = read_message(&mut &frame[..]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn auth_round_trips_and_secret_eq_is_exact() {
        round_trip(Message::Auth("s3cret".into()));
        assert!(secret_eq("abc", "abc"));
        assert!(!secret_eq("abc", "abd"));
        assert!(!secret_eq("abc", "abcd"));
    }

    #[test]
    fn payload_is_valid_protobuf() {
        // The frame payload is now the Protobuf encoding of the Envelope, not the
        // old hand-rolled tag framing; an empty tensor still decodes cleanly.
        let payload = encode(&Message::RunShard {
            position: 3,
            hidden: vec![],
        });
        assert_eq!(
            decode(&payload).unwrap(),
            Message::RunShard {
                position: 3,
                hidden: vec![]
            }
        );
    }
}
