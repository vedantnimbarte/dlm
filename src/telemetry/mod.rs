//! Per-layer flow telemetry: what a transformer layer did, and how long each
//! stage of it took.
//!
//! `/metrics` already reports *aggregate* streaming counters — hits, misses,
//! evictions, prefetches. Those answer "is the window working?" but not "where
//! did the time go?", which is the question that matters when a model streams:
//! the cost of streaming is bandwidth, and bandwidth is spent in specific
//! stages of a specific data path.
//!
//! ```text
//! mmap(NVMe) ──► host RAM cache ──► pinned staging ──► VRAM ──► compute
//! ```
//!
//! This module carries one event per stage transition, with measured bytes and
//! duration, so a consumer can say "62% of this token went to H2D copies"
//! rather than guessing from counters.
//!
//! # Cost when disabled
//!
//! Everything is gated on one relaxed atomic subscriber count. With no
//! subscriber, [`is_enabled`] is a single atomic load and [`Timer::start`] does not call
//! `Instant::now()` at all — which matters, because this crate's entire premise
//! is that bandwidth and per-layer time are scarce. A telemetry system that
//! taxed the streaming path would be self-defeating.
//!
//! # What is deliberately absent
//!
//! No prompt text, no completion text, no token ids. Timings, byte counts and
//! layer indices only. That is what makes it safe to expose the stream over a
//! network to a remote UI.

use serde::Serialize;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Events retained between drains. At ~10 events per layer per token and 30
/// layers, this is several seconds of a slow streamed run — enough that a UI
/// polling at 20 Hz never misses anything, and bounded so a subscriber that
/// stalls cannot grow memory without limit.
const RING_CAPACITY: usize = 8192;

/// Number of attached subscribers. Collection runs while this is non-zero.
///
/// A plain on/off flag looked sufficient and was not: a subscriber that
/// disconnects notices only on its next write, which can happen *after* a new
/// subscriber has attached. With a boolean, the departing connection's cleanup
/// then switched collection off underneath the arriving one, and the new
/// subscriber silently received nothing. Counting makes the last one out turn
/// off the lights.
static SUBSCRIBERS: AtomicUsize = AtomicUsize::new(0);

/// Monotonic sequence, so a consumer can detect a gap even across drains.
static SEQ: AtomicU64 = AtomicU64::new(0);

/// The decode step events are currently attributed to.
static TOKEN: AtomicU64 = AtomicU64::new(0);

/// The bounded buffer itself, kept separate from the global so its overflow
/// behaviour can be tested deterministically. Testing it through the global
/// would be racy the moment any other test emits an event — which, now that the
/// streaming path is instrumented, they do.
struct Ring {
    events: VecDeque<FlowEvent>,
    capacity: usize,
    /// Events dropped because the ring was full. Reported to the consumer so a
    /// UI can say "sampled at N%" rather than silently drawing an incomplete
    /// picture.
    dropped: u64,
}

impl Ring {
    fn new(capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity),
            capacity,
            dropped: 0,
        }
    }

    /// Drop-oldest on overflow: the most recent activity is what a UI is
    /// showing, and blocking the compute thread to make room is never
    /// acceptable here.
    fn push(&mut self, event: FlowEvent) {
        if self.events.len() >= self.capacity {
            self.events.pop_front();
            self.dropped += 1;
        }
        self.events.push_back(event);
    }

    fn take(&mut self, max: usize) -> Vec<FlowEvent> {
        let n = max.min(self.events.len());
        self.events.drain(..n).collect()
    }

    fn clear(&mut self) {
        self.events.clear();
    }
}

fn ring() -> &'static Mutex<Ring> {
    static RING: OnceLock<Mutex<Ring>> = OnceLock::new();
    RING.get_or_init(|| Mutex::new(Ring::new(RING_CAPACITY)))
}

fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// A stage of the layer data path, or a window event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Stage {
    /// Layer bytes read (or faulted in) from the mmap'd checkpoint.
    MmapRead,
    /// Weights decoded/quantized into their in-memory form.
    Dequant,
    /// Served by the host RAM cache — no disk read.
    RamHit,
    /// Missed the host RAM cache.
    RamMiss,
    /// Copied into the page-locked staging buffer.
    PinStage,
    /// Host-to-device transfer across PCIe.
    H2d,
    /// The decode block itself.
    Compute,
    /// A layer left the resident window.
    Evict,
    /// The background worker materialized a layer ahead of demand.
    Prefetch,
}

/// One measured step.
///
/// Optional fields rather than a nested enum: this is serialized to JSON on a
/// hot path and consumed by a UI, and a flat object is cheaper to write and
/// simpler to read than a tagged union.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FlowEvent {
    /// Microseconds since the first telemetry call in this process.
    pub t_us: u64,
    pub seq: u64,
    /// Decode step this belongs to.
    pub token: u64,
    pub layer: u32,
    pub stage: Stage,
    /// Bytes moved by this stage; 0 where the stage moves nothing.
    pub bytes: u64,
    pub dur_us: u32,
    /// The duration is wall-clock around an async operation rather than a
    /// device measurement. The UI must render these differently — an estimate
    /// presented as a measurement is the one thing this system must not do.
    #[serde(skip_serializing_if = "is_false")]
    pub estimated: bool,
    /// [`Stage::Evict`]: the layer that was evicted to make room.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub victim: Option<u32>,
    /// [`Stage::Prefetch`]: the prefetch depth in effect.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<u32>,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl FlowEvent {
    /// A stage event with no extra detail.
    pub fn new(layer: u32, stage: Stage, bytes: u64, dur_us: u32) -> Self {
        Self {
            t_us: 0, // stamped by `emit`
            seq: 0,  // stamped by `emit`
            token: 0,
            layer,
            stage,
            bytes,
            dur_us,
            estimated: false,
            victim: None,
            depth: None,
        }
    }

    /// Mark this duration as wall-clock rather than device-measured.
    pub fn estimated(mut self) -> Self {
        self.estimated = true;
        self
    }

    pub fn with_victim(mut self, victim: u32) -> Self {
        self.victim = Some(victim);
        self
    }

    pub fn with_depth(mut self, depth: u32) -> Self {
        self.depth = Some(depth);
        self
    }
}

/// Whether `--telemetry` was passed. Distinct from the subscriber count, which
/// tracks whether anything is currently attached.
static AVAILABLE: AtomicBool = AtomicBool::new(false);

/// Allow subscriptions (set from `--telemetry`).
pub fn set_available(available: bool) {
    AVAILABLE.store(available, Ordering::Relaxed);
}

/// Whether the telemetry route should exist at all.
///
/// When false the server answers `/v1/telemetry` as it would any unknown path,
/// so a build with the flag off is indistinguishable from one without the
/// feature — which is what a client probing for support should see.
pub fn is_available() -> bool {
    AVAILABLE.load(Ordering::Relaxed)
}

/// Is anything listening? A single relaxed atomic load.
#[inline(always)]
pub fn is_enabled() -> bool {
    SUBSCRIBERS.load(Ordering::Relaxed) > 0
}

/// Attach a subscriber and start collecting if this is the first.
pub fn enable() {
    epoch(); // fix the time origin before the first event
    SUBSCRIBERS.fetch_add(1, Ordering::Relaxed);
}

/// Detach a subscriber. Collection stops, and the buffer is discarded, only
/// when the last one leaves.
pub fn disable() {
    let prev = SUBSCRIBERS.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
        Some(n.saturating_sub(1))
    });
    // Only the transition to zero clears the ring; clearing on every detach
    // would throw away a live subscriber's unread events.
    if matches!(prev, Ok(1)) {
        if let Ok(mut ring) = ring().lock() {
            ring.clear();
        }
    }
}

/// Force collection off regardless of subscriber count. Tests only.
#[cfg(test)]
fn reset_for_test() {
    SUBSCRIBERS.store(0, Ordering::Relaxed);
    if let Ok(mut ring) = ring().lock() {
        ring.clear();
    }
}

/// Attribute subsequent events to decode step `token`.
#[inline]
pub fn set_token(token: u64) {
    if is_enabled() {
        TOKEN.store(token, Ordering::Relaxed);
    }
}

/// Record an event. Cheap no-op when disabled.
///
/// Drop-oldest on a full ring: a UI showing the most recent activity is more
/// useful than one showing the oldest, and blocking the compute thread to make
/// room is never acceptable here.
#[inline]
pub fn emit(mut event: FlowEvent) {
    if !is_enabled() {
        return;
    }
    event.t_us = epoch().elapsed().as_micros() as u64;
    event.seq = SEQ.fetch_add(1, Ordering::Relaxed);
    event.token = TOKEN.load(Ordering::Relaxed);

    // A poisoned lock must not take down a generation; telemetry is strictly
    // observational.
    if let Ok(mut ring) = ring().lock() {
        ring.push(event);
    }
}

/// A batch of events plus how many were lost before it.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Batch {
    pub events: Vec<FlowEvent>,
    /// Cumulative drops since the process started. A consumer differences
    /// consecutive values to know what it missed.
    pub dropped: u64,
}

/// Take up to `max` buffered events.
pub fn drain(max: usize) -> Batch {
    match ring().lock() {
        Ok(mut ring) => Batch {
            events: ring.take(max),
            dropped: ring.dropped,
        },
        Err(_) => Batch::default(),
    }
}

/// Times a stage, but only when telemetry is on.
///
/// `Instant::now()` is a `vDSO`/`QueryPerformanceCounter` call — small, but not
/// free, and this would sit inside the per-layer loop. When disabled this holds
/// `None` and `elapsed_us` returns 0 without touching the clock.
pub struct Timer(Option<Instant>);

impl Timer {
    #[inline(always)]
    pub fn start() -> Self {
        Self(if is_enabled() {
            Some(Instant::now())
        } else {
            None
        })
    }

    #[inline(always)]
    pub fn elapsed_us(&self) -> u32 {
        match self.0 {
            Some(t) => t.elapsed().as_micros() as u32,
            None => 0,
        }
    }
}

/// Static context a late-joining subscriber needs to render anything: the plan
/// the engine is running under, and how big each layer is.
///
/// Sent once when a subscriber attaches, then periodically, so a UI opened
/// mid-generation does not have to wait for a full pass to learn the shape of
/// the model.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    pub num_layers: u32,
    /// Resident window size in layers.
    pub resident_layers: u32,
    /// Measured bytes of each layer, indexed by layer number.
    pub layer_bytes: Vec<u64>,
    /// Bytes held permanently (embedding, LM head, norms).
    pub pinned_bytes: u64,
    /// Whether weights are streaming at all.
    pub streaming: bool,
    /// Compute device: `"cpu"` or `"gpu"`. The data path differs — a CPU run
    /// has no PCIe stage, and drawing one would be a lie.
    pub device: String,
    /// True when H2D durations are wall-clock rather than device-measured
    /// (a build without CUDA events).
    pub estimated_h2d: bool,
}

fn snapshot_slot() -> &'static Mutex<Option<Snapshot>> {
    static SNAPSHOT: OnceLock<Mutex<Option<Snapshot>>> = OnceLock::new();
    SNAPSHOT.get_or_init(|| Mutex::new(None))
}

/// Publish the static context. Called once when the engine finishes setting up.
pub fn set_snapshot(snapshot: Snapshot) {
    if let Ok(mut slot) = snapshot_slot().lock() {
        *slot = Some(snapshot);
    }
}

pub fn snapshot() -> Option<Snapshot> {
    snapshot_slot().lock().ok().and_then(|s| s.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the tests that touch the process-global bus.
    ///
    /// This is *not* enough on its own: the streaming path is instrumented, so
    /// any other test in this binary that runs a kernel while telemetry is
    /// enabled will also emit into the shared ring. Tests that inspect events
    /// therefore tag them with a distinctive layer number and filter, and the
    /// overflow behaviour is tested against a local [`Ring`] instead.
    fn guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let g = LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        reset_for_test();
        g
    }

    /// Layer number no real model has, used to pick this test's own events out
    /// of anything a concurrently-running test emitted.
    const MARKER: u32 = 0xBEEF;

    fn mine(batch: &Batch) -> Vec<FlowEvent> {
        batch
            .events
            .iter()
            .copied()
            .filter(|e| e.layer == MARKER)
            .collect()
    }

    // --- ring behaviour (deterministic: no globals involved) -----------------

    #[test]
    fn ring_drops_oldest_when_full_and_counts_the_loss() {
        let mut ring = Ring::new(4);
        for i in 0..6 {
            ring.push(FlowEvent::new(i, Stage::Compute, 0, 1));
        }
        assert_eq!(ring.events.len(), 4, "capacity is a hard bound");
        assert_eq!(ring.dropped, 2);
        // The survivors are the most recent ones.
        assert_eq!(ring.events.front().unwrap().layer, 2);
        assert_eq!(ring.events.back().unwrap().layer, 5);
    }

    #[test]
    fn ring_take_respects_its_limit_and_leaves_the_rest() {
        let mut ring = Ring::new(16);
        for i in 0..10 {
            ring.push(FlowEvent::new(i, Stage::Compute, 0, 1));
        }
        assert_eq!(ring.take(4).len(), 4);
        assert_eq!(ring.take(100).len(), 6);
        assert_eq!(ring.take(100).len(), 0, "a taken event must not repeat");
    }

    #[test]
    fn ring_take_of_an_empty_ring_is_empty_not_a_panic() {
        let mut ring = Ring::new(4);
        assert!(ring.take(usize::MAX).is_empty());
    }

    /// The drop counter is cumulative: a consumer differences successive values
    /// to learn what it missed, so draining must not reset it.
    #[test]
    fn ring_drop_count_survives_a_take() {
        let mut ring = Ring::new(2);
        for i in 0..5 {
            ring.push(FlowEvent::new(i, Stage::Compute, 0, 1));
        }
        assert_eq!(ring.dropped, 3);
        let _ = ring.take(usize::MAX);
        assert_eq!(ring.dropped, 3);
    }

    // --- global bus ---------------------------------------------------------

    #[test]
    fn emitting_while_disabled_records_nothing() {
        let _g = guard();
        emit(FlowEvent::new(MARKER, Stage::Compute, 0, 100));
        assert!(mine(&drain(usize::MAX)).is_empty());
    }

    /// The whole cost argument rests on this: no clock read when off.
    #[test]
    fn timer_does_not_read_the_clock_when_disabled() {
        let _g = guard();
        let t = Timer::start();
        assert!(t.0.is_none());
        assert_eq!(t.elapsed_us(), 0);

        enable();
        let t = Timer::start();
        assert!(t.0.is_some());
        reset_for_test();
    }

    #[test]
    fn enabled_events_round_trip_with_stamps() {
        let _g = guard();
        enable();
        set_token(7);
        emit(FlowEvent::new(MARKER, Stage::H2d, 58_720_256, 5140));

        let events = mine(&drain(usize::MAX));
        assert_eq!(events.len(), 1);
        let e = events[0];
        assert_eq!(e.stage, Stage::H2d);
        assert_eq!(e.bytes, 58_720_256);
        assert_eq!(e.dur_us, 5140);
        assert_eq!(e.token, 7, "events carry the decode step");
        reset_for_test();
    }

    #[test]
    fn draining_takes_events_only_once() {
        let _g = guard();
        enable();
        emit(FlowEvent::new(MARKER, Stage::Compute, 0, 1));
        assert_eq!(mine(&drain(usize::MAX)).len(), 1);
        assert_eq!(
            mine(&drain(usize::MAX)).len(),
            0,
            "a drained event must not repeat"
        );
        reset_for_test();
    }

    #[test]
    fn sequence_numbers_are_monotonic_so_gaps_are_detectable() {
        let _g = guard();
        enable();
        for _ in 0..5 {
            emit(FlowEvent::new(MARKER, Stage::MmapRead, 10, 1));
        }
        let events = mine(&drain(usize::MAX));
        assert_eq!(events.len(), 5);
        for pair in events.windows(2) {
            assert!(pair[1].seq > pair[0].seq, "seq must strictly increase");
        }
        reset_for_test();
    }

    #[test]
    fn disable_clears_anything_buffered() {
        let _g = guard();
        enable();
        emit(FlowEvent::new(MARKER, Stage::Compute, 0, 1));
        reset_for_test();
        assert!(mine(&drain(usize::MAX)).is_empty());
    }

    /// The route is gated separately from collection: `--telemetry` decides
    /// whether the endpoint exists at all, subscribing decides whether events
    /// are collected.
    #[test]
    fn availability_is_independent_of_collection() {
        let _g = guard();
        set_available(false);
        assert!(!is_available());
        enable();
        assert!(is_enabled(), "a subscriber can be attached...");
        assert!(!is_available(), "...without the route being advertised");
        reset_for_test();
    }

    /// The JSON a subscriber receives. `estimated` and the detail fields are
    /// skipped when unset to keep the per-event payload small.
    #[test]
    fn json_shape_is_flat_and_omits_unset_detail() {
        let e = FlowEvent::new(4, Stage::RamHit, 1024, 12);
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains(r#""stage":"ramHit""#), "{json}");
        assert!(json.contains(r#""layer":4"#), "{json}");
        assert!(!json.contains("estimated"), "{json}");
        assert!(!json.contains("victim"), "{json}");

        let e = FlowEvent::new(4, Stage::Evict, 0, 0)
            .with_victim(9)
            .estimated();
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains(r#""victim":9"#), "{json}");
        assert!(json.contains(r#""estimated":true"#), "{json}");
    }

    #[test]
    fn snapshot_round_trips() {
        let _g = guard();
        set_snapshot(Snapshot {
            num_layers: 28,
            resident_layers: 2,
            layer_bytes: vec![1024; 28],
            pinned_bytes: 4096,
            streaming: true,
            device: "cpu".into(),
            estimated_h2d: true,
        });
        let s = snapshot().expect("snapshot present");
        assert_eq!(s.num_layers, 28);
        assert_eq!(s.resident_layers, 2);
        assert!(s.streaming);
        assert_eq!(s.device, "cpu");
    }
}
