//! AMD ROCm/HIP Runtime backend (compiled under the `rocm` feature).
//!
//! HIP mirrors the CUDA Runtime API almost 1:1, so this backend is a direct
//! analogue of [`super::ffi_cuda`] against `libamdhip64`. The same
//! vendor-neutral surface in [`super`] dispatches here when `dlm` is built for
//! AMD hardware. `hipify` maps the CUDA symbol names to these `hip*` names.

#![allow(non_camel_case_types)]
#![allow(dead_code)] // some symbols are bound ahead of the Phase 2 GPU exec path

use super::DeviceMemory;
use crate::error::{DlmError, Result};
use std::os::raw::{c_int, c_uint, c_void};
use std::ptr::NonNull;

pub type hipError_t = c_int;
pub const HIP_SUCCESS: hipError_t = 0;

/// Opaque `hipStream_t` handle.
pub type hipStream_t = *mut c_void;
/// Backend-neutral stream handle (the vendor-agnostic `device` module names it
/// `StreamRaw` regardless of backend).
pub type StreamRaw = hipStream_t;
/// Opaque `hipEvent_t` handle.
pub type hipEvent_t = *mut c_void;

pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;

pub const HIP_HOST_MALLOC_DEFAULT: c_uint = 0x00;
pub const HIP_HOST_MALLOC_PORTABLE: c_uint = 0x01;
pub const HIP_HOST_MALLOC_MAPPED: c_uint = 0x02;
pub const HIP_HOST_MALLOC_WRITE_COMBINED: c_uint = 0x04;

extern "C" {
    fn hipSetDevice(device: c_int) -> hipError_t;
    fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> hipError_t;
    fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> hipError_t;
    fn hipHostFree(ptr: *mut c_void) -> hipError_t;
    fn hipHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> hipError_t;
    fn hipHostUnregister(ptr: *mut c_void) -> hipError_t;
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
    fn hipFree(ptr: *mut c_void) -> hipError_t;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> hipError_t;
    fn hipDeviceSynchronize() -> hipError_t;
    fn hipStreamCreateWithFlags(stream: *mut hipStream_t, flags: c_uint) -> hipError_t;
    fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
    fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
    fn hipEventCreate(event: *mut hipEvent_t) -> hipError_t;
    fn hipEventDestroy(event: hipEvent_t) -> hipError_t;
    fn hipEventRecord(event: hipEvent_t, stream: hipStream_t) -> hipError_t;
    fn hipStreamWaitEvent(stream: hipStream_t, event: hipEvent_t, flags: c_uint) -> hipError_t;
}

/// Make GPU `id` the current device for subsequent allocations and launches.
pub(super) fn set_device(id: u32) -> Result<()> {
    // SAFETY: no pointers; just selects the active device index.
    let code = unsafe { hipSetDevice(id as c_int) };
    if code != HIP_SUCCESS {
        return Err(DlmError::Gpu {
            api: "hipSetDevice",
            code,
        });
    }
    Ok(())
}

/// Query free/total device memory.
pub(super) fn mem_get_info() -> Result<DeviceMemory> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    // SAFETY: both out-pointers reference live stack storage for the call.
    let code = unsafe { hipMemGetInfo(&mut free, &mut total) };
    if code != HIP_SUCCESS {
        return Err(DlmError::Gpu {
            api: "hipMemGetInfo",
            code,
        });
    }
    Ok(DeviceMemory {
        free: free as u64,
        total: total as u64,
    })
}

/// Allocate zero-initialized page-locked host memory.
pub(super) fn host_alloc(bytes: usize) -> Result<NonNull<u8>> {
    let mut raw: *mut c_void = std::ptr::null_mut();
    // SAFETY: valid out-pointer; `bytes` is a positive page multiple.
    let code = unsafe { hipHostMalloc(&mut raw, bytes, HIP_HOST_MALLOC_PORTABLE) };
    if code != HIP_SUCCESS {
        return Err(DlmError::Gpu {
            api: "hipHostMalloc",
            code,
        });
    }
    let ptr = NonNull::new(raw as *mut u8).ok_or(DlmError::HostAlloc {
        bytes,
        align: crate::memory::page::page_size(),
    })?;
    // hipHostMalloc does not zero; make the whole allocation defined.
    // SAFETY: `ptr` owns `bytes` writable bytes.
    unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, bytes) };
    Ok(ptr)
}

/// Free page-locked host memory.
pub(super) fn host_free(ptr: NonNull<u8>) {
    // SAFETY: `ptr` came from `hipHostMalloc` and is freed exactly once.
    unsafe {
        let _ = hipHostFree(ptr.as_ptr() as *mut c_void);
    }
}

// ── Device memory + streams (the surface the vendor-neutral `device` module
// dispatches to; mirrors `super::ffi_cuda` 1:1). ────────────────────────────

/// Allocate `bytes` of device (VRAM) memory.
pub(super) fn device_malloc(bytes: usize) -> Result<*mut c_void> {
    let mut raw: *mut c_void = std::ptr::null_mut();
    // SAFETY: valid out-pointer.
    let code = unsafe { hipMalloc(&mut raw, bytes.max(1)) };
    if code != HIP_SUCCESS {
        return Err(DlmError::Gpu { api: "hipMalloc", code });
    }
    Ok(raw)
}

/// Free device memory from [`device_malloc`].
pub(super) fn device_free(ptr: *mut c_void) {
    // SAFETY: `ptr` came from `hipMalloc` and is freed exactly once.
    unsafe {
        let _ = hipFree(ptr);
    }
}

/// Synchronous host→device copy.
pub(super) fn copy_h2d(dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
    // SAFETY: caller guarantees `bytes` valid on both sides.
    let code = unsafe { hipMemcpy(dst, src, bytes, HIP_MEMCPY_HOST_TO_DEVICE) };
    if code != HIP_SUCCESS {
        return Err(DlmError::Gpu { api: "hipMemcpy(H2D)", code });
    }
    Ok(())
}

/// Synchronous device→host copy.
pub(super) fn copy_d2h(dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
    // SAFETY: caller guarantees `bytes` valid on both sides.
    let code = unsafe { hipMemcpy(dst, src, bytes, HIP_MEMCPY_DEVICE_TO_HOST) };
    if code != HIP_SUCCESS {
        return Err(DlmError::Gpu { api: "hipMemcpy(D2H)", code });
    }
    Ok(())
}

/// Block until all device work completes.
pub(super) fn synchronize() -> Result<()> {
    // SAFETY: no arguments.
    let code = unsafe { hipDeviceSynchronize() };
    if code != HIP_SUCCESS {
        return Err(DlmError::Gpu { api: "hipDeviceSynchronize", code });
    }
    Ok(())
}

/// `hipStreamNonBlocking`: the stream doesn't implicitly sync with the null
/// stream, so its work overlaps null-stream kernels.
const HIP_STREAM_NON_BLOCKING: c_uint = 0x01;

/// Create a non-blocking stream (overlaps the default-stream compute).
pub(super) fn stream_create_nonblocking() -> Result<hipStream_t> {
    let mut s: hipStream_t = std::ptr::null_mut();
    // SAFETY: `s` is a valid out-pointer.
    let code = unsafe { hipStreamCreateWithFlags(&mut s, HIP_STREAM_NON_BLOCKING) };
    if code != HIP_SUCCESS {
        return Err(DlmError::Gpu { api: "hipStreamCreateWithFlags", code });
    }
    Ok(s)
}

/// Destroy a stream created by [`stream_create_nonblocking`].
pub(super) fn stream_destroy(stream: hipStream_t) {
    // SAFETY: `stream` was returned by a create call and is not used again.
    unsafe {
        hipStreamDestroy(stream);
    }
}

/// Block until `stream`'s queued work completes.
pub(super) fn stream_synchronize(stream: hipStream_t) -> Result<()> {
    // SAFETY: `stream` is a valid stream handle.
    let code = unsafe { hipStreamSynchronize(stream) };
    if code != HIP_SUCCESS {
        return Err(DlmError::Gpu { api: "hipStreamSynchronize", code });
    }
    Ok(())
}

/// Enqueue an async host→device copy on `stream` (`src` must be page-locked).
pub(super) fn copy_h2d_async(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    stream: hipStream_t,
) -> Result<()> {
    // SAFETY: caller guarantees `bytes` valid on both sides; `src` is pinned.
    let code = unsafe { hipMemcpyAsync(dst, src, bytes, HIP_MEMCPY_HOST_TO_DEVICE, stream) };
    if code != HIP_SUCCESS {
        return Err(DlmError::Gpu { api: "hipMemcpyAsync(H2D)", code });
    }
    Ok(())
}
