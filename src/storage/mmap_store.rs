//! Memory-mapped weight storage engine.
//!
//! Per `specs.md` §3.2(1), `dlm` maps model shards from the NVMe SSD directly
//! into the process address space, skipping the OS read-buffer copy. The kernel
//! then demand-pages 4 KiB regions straight from disk as the streaming pipeline
//! touches them. Tensor bytes are handed out as borrowed slices (`&[u8]`) that
//! point *into* the map — no intermediate heap allocation happens on the read
//! path. Those slices are what the pinned-memory staging buffers
//! (`cudaHostAlloc`) copy from before the async DMA into VRAM.

use crate::error::{DlmError, Result};
use crate::storage::safetensors::{SafetensorsHeader, TensorInfo};
use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::path::{Path, PathBuf};

/// A single memory-mapped safetensors shard plus its parsed header.
pub struct MmapShard {
    path: PathBuf,
    mmap: Mmap,
    header: SafetensorsHeader,
}

impl MmapShard {
    /// Map a `.safetensors` file and parse its header.
    ///
    /// The mapping is read-only and advisory; pages are faulted in lazily. On
    /// Unix we hint the kernel with `MADV_RANDOM` because the streaming
    /// scheduler jumps between layer blocks rather than reading front-to-back,
    /// so aggressive readahead would only evict pages we still need.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|source| DlmError::Io {
            path: path.clone(),
            source,
        })?;

        let file_len = file
            .metadata()
            .map_err(|source| DlmError::Io {
                path: path.clone(),
                source,
            })?
            .len() as usize;

        // SAFETY: the file stays open for the lifetime of the `Mmap`, and we
        // only ever hand out immutable slices, so the mapped bytes cannot be
        // aliased mutably from Rust. External truncation of the file is the one
        // hazard mmap always carries; the streaming layer treats shards as
        // immutable model artifacts for the process lifetime.
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|source| DlmError::Mmap {
                    path: path.clone(),
                    source,
                })?
        };

        #[cfg(unix)]
        {
            // Best-effort access-pattern hint; failure is non-fatal.
            let _ = mmap.advise(memmap2::Advice::Random);
        }

        let header = SafetensorsHeader::parse(&mmap, file_len)?;

        Ok(MmapShard {
            path,
            mmap,
            header,
        })
    }

    /// Path this shard was mapped from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Parsed header (tensor index + metadata).
    pub fn header(&self) -> &SafetensorsHeader {
        &self.header
    }

    /// Total mapped size in bytes.
    pub fn mapped_len(&self) -> usize {
        self.mmap.len()
    }

    /// Look up a tensor's metadata by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.header.tensors.get(name)
    }

    /// Zero-copy view of a tensor's raw bytes, borrowed straight from the map.
    ///
    /// The returned slice is valid as long as `self` lives. Copying it into a
    /// page-locked host buffer is what actually pulls the pages off disk.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensor_info(name)
            .ok_or_else(|| DlmError::UnknownTensor(name.to_string()))?;

        let start = self.header.data_offset + info.begin;
        let end = self.header.data_offset + info.end;

        // Defensive re-check; the header parser already validated ranges, but
        // this keeps the unsafe-free slice indexing panic-proof.
        self.mmap.get(start..end).ok_or_else(|| DlmError::TensorOutOfBounds {
            name: name.to_string(),
            start,
            end,
            len: self.mmap.len(),
        })
    }

    /// Iterate over every tensor in this shard.
    pub fn tensors(&self) -> impl Iterator<Item = &TensorInfo> {
        self.header.tensors.values()
    }
}

impl std::fmt::Debug for MmapShard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapShard")
            .field("path", &self.path)
            .field("mapped_len", &self.mmap.len())
            .field("tensors", &self.header.tensors.len())
            .finish()
    }
}

/// A collection of mmapped shards presenting one flat tensor namespace, as a
/// real sharded checkpoint (`model-00001-of-00003.safetensors`, ...) requires.
#[derive(Debug, Default)]
pub struct MmapStore {
    shards: Vec<MmapShard>,
}

impl MmapStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Open every `*.safetensors` file in a model directory and map them.
    /// Files are opened in sorted order for deterministic shard indices.
    pub fn open_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let read_dir = std::fs::read_dir(dir).map_err(|source| DlmError::Io {
            path: dir.to_path_buf(),
            source,
        })?;

        let mut shard_paths: Vec<PathBuf> = Vec::new();
        for entry in read_dir {
            let entry = entry.map_err(|source| DlmError::Io {
                path: dir.to_path_buf(),
                source,
            })?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                shard_paths.push(path);
            }
        }
        shard_paths.sort();

        if shard_paths.is_empty() {
            return Err(DlmError::InvalidConfig(format!(
                "no .safetensors shards found in {}",
                dir.display()
            )));
        }

        let mut store = MmapStore::new();
        for path in shard_paths {
            store.add_shard(MmapShard::open(path)?);
        }
        Ok(store)
    }

    /// Add an already-opened shard.
    pub fn add_shard(&mut self, shard: MmapShard) {
        self.shards.push(shard);
    }

    /// Number of mapped shards.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Total bytes mapped across all shards.
    pub fn total_mapped_bytes(&self) -> usize {
        self.shards.iter().map(MmapShard::mapped_len).sum()
    }

    /// Total number of tensors across all shards.
    pub fn num_tensors(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.header().tensors.len())
            .sum()
    }

    /// Resolve a tensor by name across all shards, returning the owning shard
    /// and its metadata. First match wins (checkpoint names are unique).
    pub fn locate(&self, name: &str) -> Option<(&MmapShard, &TensorInfo)> {
        self.shards
            .iter()
            .find_map(|shard| shard.tensor_info(name).map(|info| (shard, info)))
    }

    /// Zero-copy bytes for a tensor located in any shard.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let (shard, _) = self
            .locate(name)
            .ok_or_else(|| DlmError::UnknownTensor(name.to_string()))?;
        shard.tensor_bytes(name)
    }

    /// Iterate over every tensor across all shards.
    pub fn iter_tensors(&self) -> impl Iterator<Item = &TensorInfo> {
        self.shards.iter().flat_map(|s| s.tensors())
    }
}
