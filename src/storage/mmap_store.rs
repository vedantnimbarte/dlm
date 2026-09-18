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
use std::collections::BTreeMap;
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
    /// A GGUF shard's metadata (the model's config and tokenizer); `None` for a
    /// safetensors shard, which keeps those in files of their own.
    gguf: Option<BTreeMap<String, crate::storage::gguf::Value>>,
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

        // A GGUF file carries its config and tokenizer alongside the weights, so
        // its header yields metadata the safetensors path has no equivalent for.
        // Everything below this line treats the two identically: a tensor is a
        // name, a dtype and a byte range.
        let (mut header, gguf) = if crate::storage::gguf::is_gguf(&mmap) {
            let g = crate::storage::gguf::parse(&mmap, file_len)?;
            let metadata = g.metadata.clone();
            (g.into_safetensors_header(), Some(metadata))
        } else {
            (SafetensorsHeader::parse(&mmap, file_len)?, None)
        };
        header.tensors = text_model_view(std::mem::take(&mut header.tensors));

        Ok(MmapShard {
            path,
            mmap,
            header,
            gguf,
        })
    }

    /// The GGUF metadata this shard was opened with, or `None` for safetensors.
    pub fn gguf_metadata(&self) -> Option<&BTreeMap<String, crate::storage::gguf::Value>> {
        self.gguf.as_ref()
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

/// Mistral's whole-model file, next to the sharded copy of the same weights:
/// `consolidated.safetensors`, or `consolidated-00001-of-00002.safetensors` on
/// the repos that split it.
pub(crate) fn is_consolidated(file_name: &str) -> bool {
    file_name.starts_with("consolidated") && file_name.ends_with(".safetensors")
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

    /// Open a model: a directory of safetensors shards, or a single `.gguf` file.
    ///
    /// GGUF ships as one file, so `--model-path` naming it directly is how those
    /// models are used; a directory holding one is opened by naming the file.
    pub fn open_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.is_file() {
            let mut store = MmapStore::new();
            store.add_shard(MmapShard::open(path)?);
            return Ok(store);
        }
        Self::open_dir(path)
    }

    /// The GGUF metadata of the first shard that has any.
    pub fn gguf_metadata(&self) -> Option<&BTreeMap<String, crate::storage::gguf::Value>> {
        self.shards.iter().find_map(|s| s.gguf_metadata())
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
        let mut has_index = false;
        for entry in read_dir {
            let entry = entry.map_err(|source| DlmError::Io {
                path: dir.to_path_buf(),
                source,
            })?;
            let path = entry.path();
            has_index |= path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".safetensors.index.json"));
            if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                shard_paths.push(path);
            }
        }
        // Mistral repos ship the same weights twice: as `model-0000n-of-...`
        // shards with an index, and again as one `consolidated.safetensors` in
        // their own format. Mapping both doubles the address space and the page
        // cache for nothing, since the sharded copy answers every lookup.
        // Without an index, `consolidated` may be the only copy there is.
        if has_index {
            shard_paths.retain(|p| {
                !p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(is_consolidated)
            });
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

    /// A one-tensor safetensors file: `[header length][JSON header][data]`.
    fn write_shard(path: &Path, tensor: &str) {
        let header =
            format!(r#"{{"{tensor}":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#);
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        std::fs::write(path, bytes).unwrap();
    }

    /// Mistral repos hold the same weights twice, as sharded files with an index
    /// and again as `consolidated.safetensors` in a different tensor naming.
    /// Mapping both mapped every weight twice; the consolidated copy is skipped
    /// whenever the index is there to say the sharded one is complete.
    #[test]
    fn a_consolidated_copy_is_skipped_when_the_shards_have_an_index() {
        let dir = tempfile::tempdir().unwrap();
        write_shard(&dir.path().join("model-00001-of-00001.safetensors"), "kept");
        // Not a safetensors file at all: opening it is an error, so the test
        // fails loudly if the skip ever stops working.
        std::fs::write(dir.path().join("consolidated.safetensors"), b"junk").unwrap();

        std::fs::write(dir.path().join("model.safetensors.index.json"), b"{}").unwrap();
        let store = MmapStore::open_dir(dir.path()).expect("the consolidated copy is skipped");
        assert!(store.locate("kept").is_some());

        // Without an index, `consolidated` may be the only copy of the weights,
        // so it is still mapped -- here, failing on the junk contents.
        std::fs::remove_file(dir.path().join("model.safetensors.index.json")).unwrap();
        assert!(MmapStore::open_dir(dir.path()).is_err());
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
