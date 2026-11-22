//! Reading a checkpoint's bytes with positioned, unbuffered reads.
//!
//! A streamed model re-reads layers and experts from the file it is mapped
//! from. Copying them out of the map faults the pages in one at a time: on a
//! GTX 1650 box (Windows, NVMe) random 0.6-0.9 MB expert slices of a 22 GB
//! GGUF came out of the map at 461 MB/s, and at 1,344 MB/s with one unbuffered
//! read each. The unbuffered read also leaves the OS page cache alone, which
//! dlm's own RAM tier would only duplicate.
//!
//! Unbuffered I/O wants sector-aligned offsets, lengths and buffers, so a read
//! covers the aligned span around the wanted bytes and copies them out. Where
//! the OS or file system refuses unbuffered I/O, a plain positioned read is
//! used instead.

use std::fs::File;
use std::io;
use std::path::Path;

/// Sector alignment unbuffered reads keep to: 4 KiB covers both 512-byte and
/// 4K-native drives.
const ALIGN: usize = 4096;

/// A file opened for positioned reads, unbuffered when the platform allows.
pub(crate) struct DirectFile {
    /// Opened unbuffered (`FILE_FLAG_NO_BUFFERING` / `O_DIRECT`), if that
    /// worked.
    direct: Option<File>,
    /// A plain handle, for when the unbuffered one is missing or refuses.
    plain: File,
}

impl DirectFile {
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        Ok(Self {
            direct: open_unbuffered(path).ok(),
            plain: File::open(path)?,
        })
    }

    /// `len` bytes at `offset`.
    pub(crate) fn read(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        if let Some(f) = &self.direct {
            if let Ok(v) = read_aligned(f, offset, len) {
                return Ok(v);
            }
        }
        let mut v = vec![0u8; len];
        read_exact_at(&self.plain, &mut v, offset)?;
        Ok(v)
    }
}

/// `len` bytes at `offset` through an unbuffered handle: the aligned span
/// around them into an aligned buffer, then the wanted part copied out.
fn read_aligned(f: &File, offset: u64, len: usize) -> io::Result<Vec<u8>> {
    let start = offset & !(ALIGN as u64 - 1);
    let end = (offset + len as u64).next_multiple_of(ALIGN as u64);
    let span = (end - start) as usize;
    let mut buf = vec![0u8; span + ALIGN];
    let pad = buf.as_ptr().align_offset(ALIGN);
    let aligned = &mut buf[pad..pad + span];
    // A read that runs past the end of the file comes back short; only the
    // bytes asked for have to be there.
    let mut done = 0;
    let need = (offset - start) as usize + len;
    while done < need {
        let n = read_at(f, &mut aligned[done..], start + done as u64)?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        done += n;
    }
    let from = (offset - start) as usize;
    Ok(aligned[from..from + len].to_vec())
}

fn read_exact_at(f: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        let n = read_at(f, buf, offset)?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[n..];
        offset += n as u64;
    }
    Ok(())
}

#[cfg(windows)]
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(f, buf, offset)
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(f, buf, offset)
}

#[cfg(windows)]
fn open_unbuffered(path: &Path) -> io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_NO_BUFFERING)
        .open(path)
}

#[cfg(target_os = "linux")]
fn open_unbuffered(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn open_unbuffered(_: &Path) -> io::Result<File> {
    Err(io::ErrorKind::Unsupported.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every (offset, length) a direct read can take, unaligned ones and the
    /// file's tail included, gives exactly the file's bytes.
    #[test]
    fn reads_match_the_file_at_any_offset() {
        let dir = std::env::temp_dir().join(format!("dlm-direct-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.bin");
        let data: Vec<u8> = (0..3 * ALIGN as u32 + 123)
            .map(|i| (i * 7 + i / 251) as u8)
            .collect();
        std::fs::write(&path, &data).unwrap();
        let f = DirectFile::open(&path).unwrap();
        for &(off, len) in &[
            (0usize, 1usize),
            (1, ALIGN),
            (ALIGN - 1, 2),
            (ALIGN, ALIGN),
            (5000, 7000),
            (data.len() - 10, 10),
            (0, data.len()),
        ] {
            assert_eq!(
                f.read(off as u64, len).unwrap(),
                &data[off..off + len],
                "{off}+{len}"
            );
        }
        assert!(f.read(data.len() as u64 - 1, 2).is_err(), "past the end");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
