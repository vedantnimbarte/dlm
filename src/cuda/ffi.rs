//! Raw CUDA runtime FFI bindings, compiled only under the `cuda` feature.
//!
//! These map directly onto the CUDA Runtime API symbols in `libcudart`. Higher
//! layers never touch these declarations — they go through the safe wrappers in
//! [`crate::cuda`] and [`crate::memory::pinned`]. Keeping the `unsafe extern`
//! surface isolated to one file is what lets the rest of the crate stay
//! `#![forbid(unsafe_code)]`-clean in spirit.

#![allow(non_camel_case_types)]

use std::os::raw::{c_int, c_uint, c_void};

/// `cudaError_t`. `0` == `cudaSuccess`.
pub type cudaError_t = c_int;

/// `cudaSuccess`.
pub const CUDA_SUCCESS: cudaError_t = 0;

/// Opaque `cudaStream_t` handle.
pub type cudaStream_t = *mut c_void;
/// Opaque `cudaEvent_t` handle.
pub type cudaEvent_t = *mut c_void;

/// `cudaMemcpyKind` — host→device transfer direction used by the copy stream.
pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
/// `cudaMemcpyKind` — device→host.
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

/// Flag for [`cudaHostAlloc`]: default page-locked, non-portable allocation.
pub const CUDA_HOST_ALLOC_DEFAULT: c_uint = 0x00;
/// Flag: memory is considered pinned by all CUDA contexts.
pub const CUDA_HOST_ALLOC_PORTABLE: c_uint = 0x01;
/// Flag: maps the allocation into the CUDA address space (zero-copy).
pub const CUDA_HOST_ALLOC_MAPPED: c_uint = 0x02;
/// Flag: allocation is write-combined — faster host→device DMA, slow host reads.
pub const CUDA_HOST_ALLOC_WRITE_COMBINED: c_uint = 0x04;

extern "C" {
    /// Query free and total device memory for the current device.
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;

    /// Allocate `size` bytes of page-locked (pinned) host memory suitable for
    /// asynchronous `cudaMemcpyAsync` DMA transfers.
    pub fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t;

    /// Free host memory previously allocated with [`cudaHostAlloc`].
    pub fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t;

    /// Page-lock an existing host allocation in place (alternative to
    /// `cudaHostAlloc`; used when pinning an already-mmapped region).
    pub fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> cudaError_t;

    /// Un-pin a region previously registered with [`cudaHostRegister`].
    pub fn cudaHostUnregister(ptr: *mut c_void) -> cudaError_t;

    /// Allocate `size` bytes of device (VRAM) memory — a streaming-zone buffer.
    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> cudaError_t;

    /// Free device memory from [`cudaMalloc`].
    pub fn cudaFree(ptr: *mut c_void) -> cudaError_t;

    /// Create an asynchronous stream (one for compute, one for memory copies).
    pub fn cudaStreamCreate(stream: *mut cudaStream_t) -> cudaError_t;

    /// Destroy a stream.
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;

    /// Block the host until all work on `stream` completes.
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;

    /// Asynchronous memcpy queued on `stream`. With page-locked host memory this
    /// is a true DMA transfer overlapping the compute stream (`specs.md` §3.2).
    pub fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: cudaStream_t,
    ) -> cudaError_t;

    /// Create an event used to mark transfer-complete points for pointer swaps.
    pub fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;

    /// Destroy an event.
    pub fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;

    /// Record `event` on `stream`.
    pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;

    /// Make `stream` wait on `event` before proceeding (cross-stream ordering).
    pub fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: c_uint)
        -> cudaError_t;

    /// Return the string name of an error code (for diagnostics).
    pub fn cudaGetErrorString(error: cudaError_t) -> *const std::os::raw::c_char;
}
