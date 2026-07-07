//! Vendor-neutral GPU backend surface.
//!
//! `dlm`'s streaming engine talks to the GPU through this thin, safe layer
//! rather than any one vendor's runtime. The backend is selected at compile time
//! by feature flag:
//!
//! * `cuda`  → NVIDIA CUDA Runtime (`cudart`).
//! * `rocm`  → AMD ROCm/HIP Runtime (`amdhip64`).
//! * neither → host fallback; GPU calls return [`DlmError::GpuUnavailable`] and
//!   pinned buffers degrade to page-aligned host allocations.
//!
//! CUDA and HIP expose near-identical runtime APIs (`hipMemGetInfo` mirrors
//! `cudaMemGetInfo`, and so on), so both backends implement the same private
//! surface — `mem_get_info`, `alloc_pinned_host`, `free_pinned_host` — and this
//! module dispatches to whichever is compiled in. Everything above this layer
//! (storage, profiler, pipeline schedule) is vendor-agnostic.

use crate::error::Result;
#[cfg(any(feature = "cuda", feature = "rocm"))]
use std::ptr::NonNull;

#[cfg(feature = "cuda")]
mod ffi_cuda;
#[cfg(feature = "rocm")]
mod ffi_hip;
#[cfg(feature = "cuda")]
pub mod device;

/// Which GPU runtime this binary was compiled against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    /// No GPU backend compiled in — host fallback only.
    None,
}

impl GpuVendor {
    /// Short human label, e.g. for the startup banner.
    pub const fn label(self) -> &'static str {
        match self {
            GpuVendor::Nvidia => "nvidia (cuda)",
            GpuVendor::Amd => "amd (rocm)",
            GpuVendor::None => "none (host fallback)",
        }
    }
}

/// The active GPU vendor for this build. CUDA takes precedence if both features
/// are somehow enabled (they are mutually exclusive in practice).
pub const fn active_vendor() -> GpuVendor {
    if cfg!(feature = "cuda") {
        GpuVendor::Nvidia
    } else if cfg!(feature = "rocm") {
        GpuVendor::Amd
    } else {
        GpuVendor::None
    }
}

/// Whether a real GPU backend (NVIDIA or AMD) is compiled in.
pub const fn is_available() -> bool {
    cfg!(any(feature = "cuda", feature = "rocm"))
}

/// Free/total device memory in bytes, as reported by the runtime.
///
/// This is the source of `M_free` in the profiler's layer-budget formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemory {
    pub free: u64,
    pub total: u64,
}

/// Make GPU `id` the current device, so subsequent allocations and kernel
/// launches target it (`cudaSetDevice`/`hipSetDevice`). Used by multi-GPU
/// pipeline parallelism to place each layer on the device that owns it. On a
/// host build (no GPU backend) this is a no-op, so the pipeline still runs (on
/// the CPU kernel) and stays testable off-GPU.
pub fn set_device(id: u32) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        ffi_cuda::set_device(id)
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    {
        ffi_hip::set_device(id)
    }
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        let _ = id;
        Ok(())
    }
}

/// Query the current device's free and total VRAM.
pub fn mem_get_info() -> Result<DeviceMemory> {
    #[cfg(feature = "cuda")]
    {
        ffi_cuda::mem_get_info()
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    {
        ffi_hip::mem_get_info()
    }
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        Err(crate::error::DlmError::GpuUnavailable("mem_get_info"))
    }
}

/// Allocate `bytes` of zero-initialized page-locked (pinned) host memory for
/// asynchronous DMA. Callers pass a page-rounded size. Only present in GPU
/// builds — the host fallback allocates page-aligned memory directly.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn alloc_pinned_host(bytes: usize) -> Result<NonNull<u8>> {
    #[cfg(feature = "cuda")]
    {
        ffi_cuda::host_alloc(bytes)
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    {
        ffi_hip::host_alloc(bytes)
    }
}

/// Free page-locked host memory from [`alloc_pinned_host`].
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn free_pinned_host(ptr: NonNull<u8>) {
    #[cfg(feature = "cuda")]
    {
        ffi_cuda::host_free(ptr);
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    {
        ffi_hip::host_free(ptr);
    }
}
