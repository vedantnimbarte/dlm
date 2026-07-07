//! NVIDIA CUDA Runtime backend (compiled under the `cuda` feature).
//!
//! Raw `extern "C"` bindings to `libcudart` plus the safe wrappers the
//! vendor-neutral [`super`] surface dispatches to. The raw symbols stay private
//! to this module.

#![allow(non_camel_case_types)]
#![allow(dead_code)] // some symbols are bound ahead of the Phase 2 GPU exec path

use super::DeviceMemory;
use crate::error::{DlmError, Result};
use std::os::raw::{c_int, c_uint, c_void};
use std::ptr::NonNull;

pub type cudaError_t = c_int;
pub const CUDA_SUCCESS: cudaError_t = 0;

/// Opaque `cudaStream_t` handle.
pub type cudaStream_t = *mut c_void;
/// Opaque `cudaEvent_t` handle.
pub type cudaEvent_t = *mut c_void;

pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

pub const CUDA_HOST_ALLOC_DEFAULT: c_uint = 0x00;
pub const CUDA_HOST_ALLOC_PORTABLE: c_uint = 0x01;
pub const CUDA_HOST_ALLOC_MAPPED: c_uint = 0x02;
pub const CUDA_HOST_ALLOC_WRITE_COMBINED: c_uint = 0x04;

extern "C" {
    fn cudaSetDevice(device: c_int) -> cudaError_t;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;
    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t;
    fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t;
    fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> cudaError_t;
    fn cudaHostUnregister(ptr: *mut c_void) -> cudaError_t;
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> cudaError_t;
    fn cudaFree(ptr: *mut c_void) -> cudaError_t;
    fn cudaStreamCreate(stream: *mut cudaStream_t) -> cudaError_t;
    fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: cudaStream_t,
    ) -> cudaError_t;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> cudaError_t;
    fn cudaDeviceSynchronize() -> cudaError_t;
    fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
    fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
    fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
    fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: c_uint) -> cudaError_t;
}

/// Make GPU `id` the current device for subsequent allocations and launches.
pub(super) fn set_device(id: u32) -> Result<()> {
    // SAFETY: no pointers; just selects the active device index.
    let code = unsafe { cudaSetDevice(id as c_int) };
    if code != CUDA_SUCCESS {
        return Err(DlmError::Gpu {
            api: "cudaSetDevice",
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
    let code = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    if code != CUDA_SUCCESS {
        return Err(DlmError::Gpu {
            api: "cudaMemGetInfo",
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
    let code = unsafe { cudaHostAlloc(&mut raw, bytes, CUDA_HOST_ALLOC_PORTABLE) };
    if code != CUDA_SUCCESS {
        return Err(DlmError::Gpu {
            api: "cudaHostAlloc",
            code,
        });
    }
    let ptr = NonNull::new(raw as *mut u8).ok_or(DlmError::HostAlloc {
        bytes,
        align: crate::memory::page::page_size(),
    })?;
    // cudaHostAlloc does not zero; make the whole allocation defined.
    // SAFETY: `ptr` owns `bytes` writable bytes.
    unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, bytes) };
    Ok(ptr)
}

/// Free page-locked host memory.
pub(super) fn host_free(ptr: NonNull<u8>) {
    // SAFETY: `ptr` came from `cudaHostAlloc` and is freed exactly once.
    unsafe {
        let _ = cudaFreeHost(ptr.as_ptr() as *mut c_void);
    }
}

/// Allocate `bytes` of device (VRAM) memory.
pub(super) fn device_malloc(bytes: usize) -> Result<*mut c_void> {
    let mut raw: *mut c_void = std::ptr::null_mut();
    // SAFETY: valid out-pointer.
    let code = unsafe { cudaMalloc(&mut raw, bytes.max(1)) };
    if code != CUDA_SUCCESS {
        return Err(DlmError::Gpu {
            api: "cudaMalloc",
            code,
        });
    }
    Ok(raw)
}

/// Free device memory from [`device_malloc`].
pub(super) fn device_free(ptr: *mut c_void) {
    // SAFETY: `ptr` came from `cudaMalloc` and is freed exactly once.
    unsafe {
        let _ = cudaFree(ptr);
    }
}

/// Synchronous host→device copy.
pub(super) fn copy_h2d(dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
    // SAFETY: caller guarantees `bytes` valid on both sides.
    let code = unsafe { cudaMemcpy(dst, src, bytes, CUDA_MEMCPY_HOST_TO_DEVICE) };
    if code != CUDA_SUCCESS {
        return Err(DlmError::Gpu {
            api: "cudaMemcpy(H2D)",
            code,
        });
    }
    Ok(())
}

/// Synchronous device→host copy.
pub(super) fn copy_d2h(dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
    // SAFETY: caller guarantees `bytes` valid on both sides.
    let code = unsafe { cudaMemcpy(dst, src, bytes, CUDA_MEMCPY_DEVICE_TO_HOST) };
    if code != CUDA_SUCCESS {
        return Err(DlmError::Gpu {
            api: "cudaMemcpy(D2H)",
            code,
        });
    }
    Ok(())
}

/// Block until all device work completes.
pub(super) fn synchronize() -> Result<()> {
    // SAFETY: no arguments.
    let code = unsafe { cudaDeviceSynchronize() };
    if code != CUDA_SUCCESS {
        return Err(DlmError::Gpu {
            api: "cudaDeviceSynchronize",
            code,
        });
    }
    Ok(())
}
