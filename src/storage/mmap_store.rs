//! Memory-mapped weight storage engine.
//!
//! Per(1), `dlm` maps model shards from the NVMe SSD directly
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

/// The tensor index of a multimodal checkpoint as its text model: the language
/// model's tensors under the names a text-only export uses, and the vision
/// tower dropped. A no-op for a text-only checkpoint.
///
/// Gemma 3 4B+ stores its language model as `language_model.model.*` (or, from
/// newer transformers, `model.language_model.*`) beside a SigLIP vision tower.
/// dlm serves text, so the tower is not merely unused: its encoder blocks are
/// named `...layers.N...`, which the catalog would count as transformer layers,
/// and its bytes would be planned into VRAM as pinned overhead.
fn text_model_view(
    tensors: std::collections::BTreeMap<String, TensorInfo>,
) -> std::collections::BTreeMap<String, TensorInfo> {
    const VISION: [&str; 4] = [
        "vision_tower.",
        "multi_modal_projector.",
        "model.vision_tower.",
        "model.multi_modal_projector.",
    ];
    const TEXT: [(&str, &str); 3] = [
        ("language_model.model.", "model."),
        ("model.language_model.", "model."),
        ("language_model.lm_head.", "lm_head."),
    ];
    tensors
        .into_iter()
        .filter(|(name, _)| !VISION.iter().any(|p| name.starts_with(p)))
        .map(|(name, mut info)| {
            let renamed = TEXT
                .iter()
                .find_map(|(from, to)| name.strip_prefix(from).map(|rest| format!("{to}{rest}")))
                .unwrap_or(name);
            info.name = renamed.clone();
            (renamed, info)
        })
        .collect()
}

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

        let mut header = SafetensorsHeader::parse(&mmap, file_len)?;
        header.tensors = text_model_view(std::mem::take(&mut header.tensors));

        Ok(MmapShard { path, mmap, header })
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
        self.mmap
            .get(start..end)
            .ok_or_else(|| DlmError::TensorOutOfBounds {
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
        self.shards.iter().map(|s| s.header().tensors.len()).sum()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Dtype;

    fn info(name: &str) -> (String, TensorInfo) {
        (
            name.to_string(),
            TensorInfo {
                name: name.to_string(),
                dtype: Dtype::F32,
                shape: vec![1],
                begin: 0,
                end: 4,
            },
        )
    }

    /// Both multimodal layouts come back as the text-only names, the vision
    /// tower is gone, and a text-only checkpoint is left exactly as it was.
    #[test]
    fn multimodal_checkpoint_is_viewed_as_its_text_model() {
        let names = [
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "language_model.lm_head.weight",
            "vision_tower.vision_model.encoder.layers.3.mlp.fc1.weight",
            "multi_modal_projector.mm_input_projection_weight",
            "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
        ];
        let view = text_model_view(names.iter().map(|n| info(n)).collect());
        let got: Vec<&str> = view.keys().map(String::as_str).collect();
        assert_eq!(
            got,
            [
                "lm_head.weight",
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.norm.weight"
            ]
        );
        assert!(
            view.iter().all(|(k, v)| *k == v.name),
            "TensorInfo.name follows the key"
        );

        let text_only: std::collections::BTreeMap<_, _> =
            ["model.layers.0.mlp.up_proj.weight", "lm_head.weight"]
                .iter()
                .map(|n| info(n))
                .collect();
        let before: Vec<String> = text_only.keys().cloned().collect();
        let after: Vec<String> = text_model_view(text_only).keys().cloned().collect();
        assert_eq!(before, after);
    }
}
