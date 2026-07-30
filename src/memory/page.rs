//! System page-size discovery.
//!
//! Page-locked host buffers must be aligned to (and sized in multiples of) the
//! OS page so that the PCIe DMA controller can pin whole pages. We query the
//! real value per-platform rather than assuming 4 KiB, since huge-page and
//! non-x86 systems differ.

use std::sync::OnceLock;

static PAGE_SIZE: OnceLock<usize> = OnceLock::new();

/// The system memory page size in bytes (cached after first query).
pub fn page_size() -> usize {
    *PAGE_SIZE.get_or_init(query_page_size)
}

/// Round `n` up to the next multiple of the system page size.
pub fn round_up_to_page(n: usize) -> usize {
    let p = page_size();
    // `n + p - 1` cannot overflow for any realistic allocation request; saturating
    // guards the theoretical edge near usize::MAX.
    n.saturating_add(p - 1) / p * p
}

#[cfg(windows)]
fn query_page_size() -> usize {
    #[repr(C)]
    struct SystemInfo {
        w_processor_architecture: u16,
        w_reserved: u16,
        dw_page_size: u32,
        lp_minimum_application_address: *mut core::ffi::c_void,
        lp_maximum_application_address: *mut core::ffi::c_void,
        dw_active_processor_mask: usize,
        dw_number_of_processors: u32,
        dw_processor_type: u32,
        dw_allocation_granularity: u32,
        w_processor_level: u16,
        w_processor_revision: u16,
    }

    extern "system" {
        fn GetSystemInfo(info: *mut SystemInfo);
    }

    // SAFETY: GetSystemInfo fully initializes the struct it is handed.
    let mut info: SystemInfo = unsafe { core::mem::zeroed() };
    unsafe { GetSystemInfo(&mut info) };
    let size = info.dw_page_size as usize;
    if size == 0 {
        4096
    } else {
        size
    }
}

#[cfg(unix)]
fn query_page_size() -> usize {
    // SAFETY: sysconf with a valid name is always safe; it returns -1 on error,
    // which we fall back from.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if size <= 0 {
        4096
    } else {
        size as usize
    }
}

#[cfg(not(any(windows, unix)))]
fn query_page_size() -> usize {
    4096
}

static TOTAL_RAM: OnceLock<Option<u64>> = OnceLock::new();

/// Total physical system RAM in bytes, or `None` if the platform cannot say.
///
/// Queried the same way as [`page_size`] — a direct platform call, no new
/// dependency — so budgets that would otherwise need a hard-coded ceiling can
/// scale to the machine they run on.
pub fn total_ram() -> Option<u64> {
    *TOTAL_RAM.get_or_init(query_total_ram)
}

#[cfg(windows)]
fn query_total_ram() -> Option<u64> {
    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }

    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    // SAFETY: `dw_length` must be set to the struct size before the call (the API
    // uses it to version the layout); the call then fills the rest.
    let mut status: MemoryStatusEx = unsafe { core::mem::zeroed() };
    status.dw_length = core::mem::size_of::<MemoryStatusEx>() as u32;
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok == 0 || status.ull_total_phys == 0 {
        None
    } else {
        Some(status.ull_total_phys)
    }
}

#[cfg(unix)]
fn query_total_ram() -> Option<u64> {
    // SAFETY: sysconf with a valid name is always safe; -1 signals "unknown",
    // which we report as `None` rather than guessing.
    let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if pages <= 0 || page <= 0 {
        return None;
    }
    (pages as u64).checked_mul(page as u64)
}

#[cfg(not(any(windows, unix)))]
fn query_total_ram() -> Option<u64> {
    None
}

#[cfg(test)]
mod ram_tests {
    /// Whatever the platform reports must be sane: either "unknown" or a figure
    /// big enough to be real RAM. A bogus small value would silently shrink every
    /// budget derived from it.
    #[test]
    fn total_ram_is_absent_or_plausible() {
        if let Some(bytes) = super::total_ram() {
            assert!(
                bytes >= 256 * 1024 * 1024,
                "implausible total RAM reported: {bytes} bytes"
            );
            // Cached: a second call must agree.
            assert_eq!(super::total_ram(), Some(bytes));
        }
    }
}
