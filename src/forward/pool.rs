//! A persistent worker pool for the CPU kernels' large GEMVs.
//!
//! `std::thread::scope` spawned a fresh OS thread per chunk on every call, which
//! on a small model's CPU decode meant hundreds of thread creations per token.
//! These workers are started once and reused. std only: the project keeps its
//! dependency list short, and this is all a scoped `parallel_for` needs.

use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

type Job = &'static (dyn Fn(usize) + Sync);
/// A chunk to run: the job, its index, and the latch counting it.
type Task = (Job, usize, Arc<Latch>);

struct Latch {
    left: Mutex<usize>,
    done: Condvar,
    panicked: AtomicBool,
}

struct Pool {
    workers: Vec<Mutex<Sender<Task>>>,
    /// One `parallel_for` at a time owns the workers.
    busy: Mutex<()>,
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism().map_or(1, |n| n.get());
        let workers = (1..n)
            .map(|i| {
                let (tx, rx) = channel::<Task>();
                std::thread::Builder::new()
                    .name(format!("dlm-cpu-{i}"))
                    .spawn(move || {
                        for (job, chunk, latch) in rx {
                            if catch_unwind(AssertUnwindSafe(|| job(chunk))).is_err() {
                                latch.panicked.store(true, Ordering::Relaxed);
                            }
                            let mut left = latch.left.lock().unwrap_or_else(|e| e.into_inner());
                            *left -= 1;
                            latch.done.notify_all();
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

/// Chunks [`parallel_for`] can run at once: the calling thread plus the workers.
pub(crate) fn threads() -> usize {
    pool().workers.len() + 1
}

/// Run `f(0..chunks)` across the pool and return once every call has finished.
/// Chunk 0 runs on the calling thread.
///
/// Runs serially when `chunks` exceeds [`threads`], or when another caller holds
/// the pool (concurrent tests, a second generation thread). A serial run
/// computes the same values, so contention costs speed, never correctness.
pub(crate) fn parallel_for(chunks: usize, f: &(dyn Fn(usize) + Sync)) {
    let p = pool();
    let guard = match p.busy.try_lock() {
        Ok(g) if chunks > 1 && chunks <= threads() => g,
        _ => {
            (0..chunks).for_each(f);
            return;
        }
    };
    let latch = Arc::new(Latch {
        left: Mutex::new(chunks - 1),
        done: Condvar::new(),
        panicked: AtomicBool::new(false),
    });
    // SAFETY: the reference outlives every use. Each chunk sent below is
    // counted in `latch`, and this function does not return, or unwind, until
    // the count reaches zero, so no worker can still hold `job` afterwards.
    let job: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(f) };
    for chunk in 1..chunks {
        let tx = p.workers[chunk - 1]
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if tx.send((job, chunk, latch.clone())).is_err() {
            // The worker is gone (it can only exit if its thread died): do its
            // chunk here instead.
            drop(tx);
            f(chunk);
            *latch.left.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
        }
    }
    let own = catch_unwind(AssertUnwindSafe(|| f(0)));
    let mut left = latch.left.lock().unwrap_or_else(|e| e.into_inner());
    while *left > 0 {
        left = latch.done.wait(left).unwrap_or_else(|e| e.into_inner());
    }
    drop(left);
    drop(guard);
    if let Err(payload) = own {
        resume_unwind(payload);
    }
    if latch.panicked.load(Ordering::Relaxed) {
        panic!("a CPU worker panicked during parallel_for");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

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
