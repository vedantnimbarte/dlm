//! Owned device (VRAM) memory buffers.
//!
//! A thin RAII wrapper over device malloc/free with synchronous host↔device
//! copies, used by the GPU compute kernel to stage weights and activations in
//! VRAM. Backend-neutral: it dispatches to the CUDA or HIP runtime depending on
//! the enabled feature, so the whole GPU stack above it is vendor-agnostic.

#[cfg(feature = "cuda")]
use super::ffi_cuda as backend;
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
use super::ffi_hip as backend;
use crate::error::Result;
use std::os::raw::c_void;

/// A block of device memory holding `f32` elements.
#[derive(Debug)]
pub struct DeviceBuffer {
    ptr: *mut c_void,
    len: usize,
}

impl DeviceBuffer {
    /// Allocate space for `len` `f32`s (uninitialized).
    pub fn new(len: usize) -> Result<Self> {
        let ptr = backend::device_malloc(len * std::mem::size_of::<f32>())?;
        Ok(Self { ptr, len })
    }

    /// Allocate and upload `data`.
    pub fn from_slice(data: &[f32]) -> Result<Self> {
        let buf = Self::new(data.len().max(1))?;
        if !data.is_empty() {
            buf.upload(data)?;
        }
        Ok(buf)
    }

    /// Allocate and upload raw bytes **verbatim** — weights in their native
    /// checkpoint dtype (bf16/f16 bit patterns), which the kernel converts to f32
    /// in-register. Halves VRAM and PCIe traffic versus upsizing to f32 first.
    ///
    /// `len` is the logical *element* count (not the byte count), so `len()` keeps
    /// meaning the same thing it does for an f32 buffer.
    pub fn from_bytes(bytes: &[u8], len: usize) -> Result<Self> {
        let buf = Self::new_bytes(bytes.len(), len)?;
        if !bytes.is_empty() {
            backend::copy_h2d(buf.ptr, bytes.as_ptr() as *const c_void, bytes.len())?;
        }
        Ok(buf)
    }

    /// Allocate `bytes` of device memory holding `len` logical elements
    /// (uninitialized). For weights whose byte length is not `len * 4`.
    pub fn new_bytes(bytes: usize, len: usize) -> Result<Self> {
        let ptr = backend::device_malloc(bytes.max(1))?;
        Ok(Self { ptr, len })
    }

    /// Element count.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copy `data` from host into this buffer.
    pub fn upload(&self, data: &[f32]) -> Result<()> {
        backend::copy_h2d(
            self.ptr,
            data.as_ptr() as *const c_void,
            std::mem::size_of_val(data),
        )
    }

    /// Copy this buffer's contents into `out` on the host.
    pub fn download(&self, out: &mut [f32]) -> Result<()> {
        backend::copy_d2h(
            out.as_mut_ptr() as *mut c_void,
            self.ptr,
            std::mem::size_of_val(out),
        )
    }

    /// Raw device pointer (read-only view).
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr as *const f32
    }

    /// Raw device pointer (mutable view).
    pub fn as_mut_ptr(&self) -> *mut f32 {
        self.ptr as *mut f32
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        backend::device_free(self.ptr);
    }
}

// SAFETY: a DeviceBuffer uniquely owns its device allocation; the pointer is
// only handed to CUDA APIs. It is safe to move across threads.
unsafe impl Send for DeviceBuffer {}

// SAFETY: the buffer is a device pointer + length; `&self` methods either read
// (`as_ptr`, `download`) or issue synchronous CUDA copies, and the CUDA runtime
// is itself thread-safe (one process-shared primary context). The streaming GPU
// kernel shares resident *weight* buffers read-only across the compute thread
// and the prefetch worker; no two threads mutate the same buffer.
unsafe impl Sync for DeviceBuffer {}

/// A non-blocking CUDA stream: its work overlaps kernels on the default (null)
/// stream. Used as a dedicated *copy* stream so weight uploads run concurrently
/// with compute.
pub struct Stream {
    raw: backend::StreamRaw,
}

impl Stream {
    /// Create a non-blocking stream.
    pub fn new_nonblocking() -> Result<Self> {
        Ok(Self { raw: backend::stream_create_nonblocking()? })
    }

    /// Block the calling thread until this stream's queued work completes. The
    /// GPU still runs this stream concurrently with the default stream; only the
    /// caller waits.
    pub fn synchronize(&self) -> Result<()> {
        backend::stream_synchronize(self.raw)
    }

    fn raw(&self) -> backend::StreamRaw {
        self.raw
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        backend::stream_destroy(self.raw);
    }
}

// SAFETY: a CUDA stream handle is an opaque pointer into the thread-safe CUDA
// runtime; enqueuing/synchronizing from any thread is supported.
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl DeviceBuffer {
    /// Enqueue an async host→device copy on `stream`. `src` must point at
    /// page-locked host memory of `self.len` `f32`s for the copy to run truly
    /// asynchronously; the caller must not overwrite `src` until `stream` is
    /// synchronized.
    pub fn upload_async(&self, src: *const f32, stream: &Stream) -> Result<()> {
        backend::copy_h2d_async(
            self.ptr,
            src as *const c_void,
            self.len * std::mem::size_of::<f32>(),
            stream.raw(),
        )
    }

    /// Enqueue an async host→device copy of exactly `bytes` from page-locked
    /// `src`. Used for weights in their native dtype, whose byte length is not
    /// `len * 4`.
    pub fn upload_async_bytes(
        &self,
        src: *const c_void,
        bytes: usize,
        stream: &Stream,
    ) -> Result<()> {
        backend::copy_h2d_async(self.ptr, src, bytes, stream.raw())
    }
}

/// Block until the default (null) stream — the one kernels launch on —
/// completes. Unlike [`synchronize`], does *not* wait on other streams, so an
/// in-flight copy on a non-blocking [`Stream`] keeps running.
pub fn synchronize_default() -> Result<()> {
    backend::stream_synchronize(std::ptr::null_mut())
}

/// Block until all queued device work finishes.
pub fn synchronize() -> Result<()> {
    backend::synchronize()
}
