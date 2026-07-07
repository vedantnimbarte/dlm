//! Page-locked (pinned) host memory buffers.
//!
//! Per `specs.md` §3.2(2), the streaming pipeline stages weights through
//! **page-locked** host buffers so the PCIe controller can DMA them into VRAM
//! with `cudaMemcpyAsync` while the compute stream runs — pageable memory would
//! force a synchronous staging copy and stall the pipeline.
//!
//! Two backends, one API — selected by GPU feature (see [`crate::gpu`]):
//! * `--features cuda`/`rocm` → the vendor runtime's page-locking allocator
//!   (`cudaHostAlloc` / `hipHostMalloc`) returns true page-locked memory.
//! * default build → a page-*aligned* [`std::alloc`] buffer. It is not kernel-
//!   pinned, but it is laid out identically (page-aligned base, page-multiple
//!   length) so it satisfies the same pointer/alignment contract and can be
//!   promoted in place later via `cudaHostRegister`/`hipHostRegister`. This keeps
//!   every buffer allocation-compatible with async DMA streams from Phase 1 on.

use crate::error::{DlmError, Result};
use crate::memory::page::round_up_to_page;
use std::alloc::{dealloc, Layout};
#[cfg(not(any(feature = "cuda", feature = "rocm")))]
use std::alloc::alloc_zeroed;
use std::ptr::NonNull;

/// Which allocator produced a buffer's backing store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinKind {
    /// Real page-locked host memory from the GPU runtime (CUDA or ROCm).
    DevicePinned,
    /// Page-aligned `std::alloc` memory (off-GPU fallback / pre-pinning stage).
    PageAligned,
}

/// A page-locked, page-aligned host buffer used as a DMA staging area.
///
/// The buffer owns its allocation and frees it on drop through the matching
/// deallocator. It is `Send`/`Sync` (plain bytes, like `Vec<u8>`), so it can be
/// handed to a `tokio` copy task driving Stream B.
pub struct PinnedBuffer {
    ptr: NonNull<u8>,
    /// Bytes requested by the caller (the logical length).
    len: usize,
    /// Actual allocation size (rounded up to a page multiple).
    capacity: usize,
    kind: PinKind,
}

// SAFETY: `PinnedBuffer` uniquely owns a heap/pinned allocation of plain bytes.
// There is no interior aliasing; access is governed by Rust's borrow rules the
// same way `Vec<u8>` is, so it is safe to move and share across threads.
unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}

impl PinnedBuffer {
    /// Allocate a zero-initialized pinned buffer of at least `len` bytes.
    ///
    /// The allocation is rounded up to a whole number of pages. Under a GPU
    /// feature (`cuda`/`rocm`) this yields genuinely page-locked memory;
    /// otherwise a page-aligned host allocation with the same layout guarantees.
    pub fn with_len(len: usize) -> Result<Self> {
        if len == 0 {
            return Err(DlmError::HostAlloc {
                bytes: 0,
                align: crate::memory::page::page_size(),
            });
        }
        let capacity = round_up_to_page(len);
        Self::alloc_backend(len, capacity)
    }

    /// Allocate a pinned buffer sized to hold `tensor_bytes` and copy them in.
    /// This is the exact staging step before a `cudaMemcpyAsync` into a
    /// streaming-zone VRAM buffer.
    pub fn from_bytes(tensor_bytes: &[u8]) -> Result<Self> {
        let mut buf = Self::with_len(tensor_bytes.len())?;
        buf.as_mut_slice()[..tensor_bytes.len()].copy_from_slice(tensor_bytes);
        Ok(buf)
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    fn alloc_backend(len: usize, capacity: usize) -> Result<Self> {
        // Delegate to the active GPU runtime's page-locking allocator, which
        // returns zero-initialized pinned memory.
        let ptr = crate::gpu::alloc_pinned_host(capacity)?;
        Ok(Self {
            ptr,
            len,
            capacity,
            kind: PinKind::DevicePinned,
        })
    }

    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    fn alloc_backend(len: usize, capacity: usize) -> Result<Self> {
        let align = crate::memory::page::page_size();
        let layout = Layout::from_size_align(capacity, align).map_err(|_| DlmError::HostAlloc {
            bytes: capacity,
            align,
        })?;
        // SAFETY: `capacity > 0` (rounded from a non-zero `len`), and the layout
        // is valid. `alloc_zeroed` returns null on failure, handled below.
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).ok_or(DlmError::HostAlloc {
            bytes: capacity,
            align,
        })?;
        Ok(Self {
            ptr,
            len,
            capacity,
            kind: PinKind::PageAligned,
        })
    }

    /// Logical length in bytes (what the caller requested).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty (always false — zero-length is rejected).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Full page-aligned allocation size in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Which backend allocated this buffer.
    pub fn kind(&self) -> PinKind {
        self.kind
    }

    /// True if the memory is genuinely page-locked (DMA-ready without staging).
    pub fn is_pinned(&self) -> bool {
        self.kind == PinKind::DevicePinned
    }

    /// Raw host pointer — the `void*` handed to `cudaMemcpyAsync` as the source.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Mutable raw host pointer — the destination for a device→host copy.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Immutable view of the logical `len` bytes.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` owns `capacity >= len` initialized bytes for `self`'s life.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Mutable view of the logical `len` bytes.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: exclusive borrow guarantees no aliasing; bytes are initialized.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        match self.kind {
            PinKind::DevicePinned => {
                #[cfg(any(feature = "cuda", feature = "rocm"))]
                crate::gpu::free_pinned_host(self.ptr);
            }
            PinKind::PageAligned => {
                let align = crate::memory::page::page_size();
                if let Ok(layout) = Layout::from_size_align(self.capacity, align) {
                    // SAFETY: same layout used at allocation; freed exactly once.
                    unsafe { dealloc(self.ptr.as_ptr(), layout) };
                }
            }
        }
    }
}

impl std::fmt::Debug for PinnedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedBuffer")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .field("kind", &self.kind)
            .field("ptr", &self.ptr.as_ptr())
            .finish()
    }
}
