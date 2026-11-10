//! A persistent worker pool for the CPU kernels' large GEMVs.
//!
//! `std::thread::scope` spawned a fresh OS thread per chunk on every call, which
//! on a small model's CPU decode meant hundreds of thread creations per token.
//! These workers are started once and reused. std only: the project keeps its
//! dependency list short, and this is all a scoped `parallel_for` needs.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

type Job = &'static (dyn Fn(usize) + Sync);

/// One [`parallel_for`] call, shared with the workers it woke.
///
/// Chunks are claimed from `next` by whichever threads are running, rather than
/// dealt one per thread: on a loaded machine a woken worker can wait a whole
/// scheduler quantum before it runs, and a call that waited for every worker
/// took that long however little work it had. Now the threads that are running
/// take the chunks, and a worker that shows up after the last one is claimed
/// leaves without touching the job.
struct Run {
    job: Job,
    chunks: usize,
    next: AtomicUsize,
    /// Workers inside `job`, and whether the call has stopped admitting them.
    state: Mutex<(usize, bool)>,
    idle: Condvar,
    panicked: AtomicBool,
}

impl Run {
    /// Claim and run chunks until none are left.
    fn drain(&self) {
        loop {
            let chunk = self.next.fetch_add(1, Ordering::Relaxed);
            if chunk >= self.chunks {
                return;
            }
            if catch_unwind(AssertUnwindSafe(|| (self.job)(chunk))).is_err() {
                self.panicked.store(true, Ordering::Relaxed);
            }
        }
    }
}

struct Pool {
    workers: Vec<Mutex<Sender<Arc<Run>>>>,
    /// One `parallel_for` at a time owns the workers.
    busy: Mutex<()>,
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism().map_or(1, |n| n.get());
        let workers = (1..n)
            .map(|i| {
                let (tx, rx) = channel::<Arc<Run>>();
                std::thread::Builder::new()
                    .name(format!("dlm-cpu-{i}"))
                    .spawn(move || {
                        for run in rx {
                            {
                                let mut state = run.state.lock().unwrap_or_else(|e| e.into_inner());
                                if state.1 {
                                    continue;
                                }
                                state.0 += 1;
                            }
                            run.drain();
                            let mut state = run.state.lock().unwrap_or_else(|e| e.into_inner());
                            state.0 -= 1;
                            run.idle.notify_all();
                        }
                    })
                    .expect("spawn CPU worker thread");
                Mutex::new(tx)
            })
            .collect();
        Pool {
            workers,
            busy: Mutex::new(()),
        }
    })
}

/// Threads [`parallel_for`] can run on at once: the calling thread plus the
/// workers.
pub(crate) fn threads() -> usize {
    pool().workers.len() + 1
}

/// Run `f(0..chunks)` across the pool and return once every call has finished.
/// Each chunk runs exactly once, on whichever thread claims it first; the
/// calling thread claims chunks too.
///
/// Runs serially when another caller holds the pool (concurrent tests, a second
/// generation thread). A serial run computes the same values, so contention
/// costs speed, never correctness.
pub(crate) fn parallel_for(chunks: usize, f: &(dyn Fn(usize) + Sync)) {
    let p = pool();
    let guard = match p.busy.try_lock() {
        Ok(g) if chunks > 1 && !p.workers.is_empty() => g,
        _ => {
            (0..chunks).for_each(f);
            return;
        }
    };
    // SAFETY: the reference outlives every use. A worker calls `job` only while
    // counted in `state.0`, and only if it was counted before `state.1` was set;
    // this function sets `state.1` and then does not return, or unwind, until
    // the count is back to zero.
    let job: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(f) };
    let run = Arc::new(Run {
        job,
        chunks,
        next: AtomicUsize::new(0),
        state: Mutex::new((0, false)),
        idle: Condvar::new(),
        panicked: AtomicBool::new(false),
    });
    for tx in p.workers.iter().take(chunks - 1) {
        // A worker whose thread died just never helps; the chunks still run.
        let _ = tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .send(run.clone());
    }
    run.drain();
    let mut state = run.state.lock().unwrap_or_else(|e| e.into_inner());
    state.1 = true;
    while state.0 > 0 {
        state = run.idle.wait(state).unwrap_or_else(|e| e.into_inner());
    }
    drop(state);
    drop(guard);
    if run.panicked.load(Ordering::Relaxed) {
        // Re-raise on the caller, as a serial loop would.
        panic!("a CPU chunk panicked during parallel_for");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_chunk_runs_exactly_once_and_calls_nest_serially() {
        for chunks in [1, 2, threads(), threads() + 3] {
            let hits: Vec<AtomicUsize> = (0..chunks).map(|_| AtomicUsize::new(0)).collect();
            parallel_for(chunks, &|i| {
                // A nested call finds the pool busy and runs inline.
                parallel_for(2, &|_| {});
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
    }

    /// Calls short enough that most workers wake after the last chunk is
    /// claimed: they must skip the call, not run its job after it has returned
    /// (the job borrows this stack frame, which the next call reuses).
    #[test]
    fn workers_that_arrive_late_skip_the_call() {
        for round in 0..2000 {
            let hits = AtomicUsize::new(0);
            parallel_for(threads() * 2, &|_| {
                hits.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(hits.into_inner(), threads() * 2, "round {round}");
        }
    }

    #[test]
    fn a_panicking_chunk_panics_the_caller_and_the_pool_survives() {
        let r = catch_unwind(|| parallel_for(threads().max(2), &|i| assert_ne!(i, 1)));
        assert!(r.is_err());
        let sum = AtomicUsize::new(0);
        parallel_for(threads(), &|i| {
            sum.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(sum.into_inner(), (0..threads()).sum::<usize>());
    }
}
