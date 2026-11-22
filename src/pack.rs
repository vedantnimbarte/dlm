//! Layer packs: a safetensors checkpoint with its weights already quantized.
//!
//! `--quant int4`/`int8` quantizes a floating-point checkpoint at load. A
//! resident model pays that once; a streamed one pays it every time a layer
//! or expert misses the RAM cache, along with reading 16-bit weights it is
//! about to shrink. Quantizing ran at about 300 MB/s of bf16 on a 6-core
//! Ryzen, where an unbuffered read of the same bytes ran at 1.3 GB/s.
//!
//! `dlm pack` runs the loader once, layer by layer, and writes every weight
//! `load_native` quantized, exactly as it came out, into a new
//! `model.safetensors`. Everything else is copied as it is, along with the
//! config, tokenizer and template files. Loading the pack hands those bytes
//! back without converting them, so a packed model computes exactly what the
//! original does under the same `--quant`, from 3.5x (int4) fewer bytes.
//!
//! A packed weight is a `U8` tensor holding the [`Weights`] blob. The file's
//! `__metadata__` names the precision the pack was made for (`dlm.pack`), and
//! each packed tensor's form (`dlm.pack.<name>`: `int4|int8 <elements>
//! <group size>`). Tensors the loader reads some other way (norms, biases,
//! embeddings, rows sliced out of a fused projection) stay in their own dtype.

use crate::error::{DlmError, Result};
use crate::forward::Weights;
use crate::model::QuantScheme;
use crate::storage::{Dtype, MmapShard, MmapStore, TensorInfo};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;

/// `__metadata__` key naming the precision a pack was made for.
const PACK_KEY: &str = "dlm.pack";

thread_local! {
    /// While [`write_pack`] loads a layer: every weight `load_native`
    /// quantized, by tensor name.
    static RECORD: RefCell<Option<Vec<(String, Weights)>>> = const { RefCell::new(None) };
}

/// Called by `load_native` with each weight it quantized, kept while a pack
/// is being written.
pub(crate) fn record(name: &str, w: &Weights) {
    RECORD.with(|r| {
        if let Some(v) = r.borrow_mut().as_mut() {
            v.push((name.to_string(), w.clone()));
        }
    });
}

fn scheme_name(q: QuantScheme) -> Option<&'static str> {
    match q {
        QuantScheme::Int4 => Some("int4"),
        QuantScheme::Int8 => Some("int8"),
        _ => None,
    }
}

/// The precision `store` was packed for, if it is a pack.
pub fn pack_scheme(store: &MmapStore) -> Option<QuantScheme> {
    store
        .iter_shards()
        .find_map(|s| s.header().metadata.get(PACK_KEY))
        .and_then(|q| match q.as_str() {
            "int4" => Some(QuantScheme::Int4),
            "int8" => Some(QuantScheme::Int8),
            _ => None,
        })
}

/// Whether tensor `name` of `shard` is a packed weight.
pub(crate) fn is_packed(shard: &MmapShard, name: &str) -> bool {
    shard
        .header()
        .metadata
        .contains_key(&format!("{PACK_KEY}.{name}"))
}

/// Packed tensor `name`, as the [`Weights`] it was packed from; `None` when it
/// is not packed. Refused when the pack was made for another precision than
/// `quant`, or for another shape than `expected_len` elements.
pub(crate) fn read_packed(
    shard: &MmapShard,
    info: &TensorInfo,
    name: &str,
    expected_len: usize,
    quant: QuantScheme,
) -> Result<Option<Weights>> {
    let meta = &shard.header().metadata;
    let Some(spec) = meta.get(&format!("{PACK_KEY}.{name}")) else {
        return Ok(None);
    };
    let bad = || DlmError::InvalidConfig(format!("pack entry for {name:?} is malformed: {spec:?}"));
    let mut parts = spec.split(' ');
    let (kind, n, group) = (
        parts.next().ok_or_else(bad)?,
        parts
            .next()
            .and_then(|v| v.parse::<usize>().ok())
            .ok_or_else(bad)?,
        parts
            .next()
            .and_then(|v| v.parse::<usize>().ok())
            .ok_or_else(bad)?,
    );
    // What the loader asks for here, which is not always the pack's precision
    // (an int4 load keeps the end layers at int8), must be what was stored.
    if scheme_name(quant) != Some(kind) {
        let made_for = meta.get(PACK_KEY).map(String::as_str).unwrap_or("?");
        return Err(DlmError::InvalidConfig(format!(
            "this pack was made for --quant {made_for}, and {name:?} is stored as {kind}, not \
             the {quant:?} this load asks for: load it with --quant {made_for} (or no \
             --quant), or pack the original checkpoint again"
        )));
    }
    if n != expected_len {
        return Err(DlmError::InvalidConfig(format!(
            "packed tensor {name:?}: {n} elements, expected {expected_len}"
        )));
    }
    let blob = shard.read_tensor_range(name, 0..info.byte_len())?;
    let (layout, w) = match kind {
        "int4" => (
            crate::forward::QuantLayout::int4(n, group),
            Weights::Int4 {
                blob,
                group_size: group,
                num_elements: n,
            },
        ),
        "int8" => (
            crate::forward::QuantLayout::int8(n, group),
            Weights::Int8 {
                blob,
                group_size: group,
                num_elements: n,
            },
        ),
        _ => return Err(bad()),
    };
    if w.as_bytes().len() != layout.total_bytes {
        return Err(bad());
    }
    Ok(Some(w))
}

/// safetensors' name for `d`.
fn dtype_tag(d: Dtype) -> Result<&'static str> {
    Ok(match d {
        Dtype::Bool => "BOOL",
        Dtype::U8 => "U8",
        Dtype::I8 => "I8",
        Dtype::I16 => "I16",
        Dtype::U16 => "U16",
        Dtype::F16 => "F16",
        Dtype::BF16 => "BF16",
        Dtype::I32 => "I32",
        Dtype::U32 => "U32",
        Dtype::F32 => "F32",
        Dtype::I64 => "I64",
        Dtype::U64 => "U64",
        Dtype::F64 => "F64",
        other => {
            return Err(DlmError::UnsupportedQuant(format!(
                "a pack cannot hold a {other:?} tensor"
            )))
        }
    })
}

/// Write a pack of the safetensors checkpoint in `src` into `out` (a new or
/// empty directory), its weights at `quant` (int4 or int8). `progress` is
/// told each layer as it is written.
pub fn write_pack(
    src: &Path,
    out: &Path,
    quant: QuantScheme,
    mut progress: impl FnMut(u32, u32),
) -> Result<()> {
    let tag = scheme_name(quant).ok_or_else(|| {
        DlmError::InvalidConfig("a pack holds int4 or int8 weights: pass --quant int4|int8".into())
    })?;
    let io = |path: &Path| {
        let path = path.to_path_buf();
        move |source| DlmError::Io { path, source }
    };
    let (config, store) = crate::loader::open_model(src, quant)?;
    if store.gguf_metadata().is_some() {
        return Err(DlmError::InvalidConfig(
            "a GGUF file is already quantized, and dlm reads it as it is: packing it would \
             only copy it"
                .into(),
        ));
    }
    if pack_scheme(&store).is_some() {
        return Err(DlmError::InvalidConfig(format!(
            "{} is already a pack",
            src.display()
        )));
    }
    if config.packed_quant.is_some() {
        return Err(DlmError::InvalidConfig(
            "a GPTQ checkpoint is already 4-bit and loads without converting".into(),
        ));
    }
    std::fs::create_dir_all(out).map_err(io(out))?;
    if std::fs::read_dir(out).map_err(io(out))?.next().is_some() {
        return Err(DlmError::InvalidConfig(format!(
            "{} is not empty; a pack is written into a new directory",
            out.display()
        )));
    }

    // Tensor data goes to a scratch file first: the header, which comes first
    // in a safetensors file, lists every tensor's offsets.
    let part = out.join("model.safetensors.part");
    let mut data = std::io::BufWriter::new(std::fs::File::create(&part).map_err(io(&part))?);
    let mut offset = 0usize;
    let mut header = serde_json::Map::new();
    let mut meta = serde_json::Map::new();
    meta.insert(PACK_KEY.into(), tag.into());
    let mut put = |name: &str, dtype: &str, shape: Vec<usize>, bytes: &[u8]| -> Result<()> {
        data.write_all(bytes).map_err(io(&part))?;
        header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + bytes.len()],
            }),
        );
        offset += bytes.len();
        Ok(())
    };

    let cfg = crate::loader::pack_block_config(&config);
    let mut packed = BTreeSet::new();
    for layer in 0..config.num_layers {
        RECORD.with(|r| *r.borrow_mut() = Some(Vec::new()));
        let loaded = crate::loader::load_layer_tensors(
            &store,
            &cfg,
            layer,
            config.num_layers,
            quant,
            None,
            config.norm_add_one,
        );
        let recorded = RECORD.with(|r| r.borrow_mut().take()).unwrap_or_default();
        drop(loaded?);
        for (name, w) in recorded {
            if !packed.insert(name.clone()) {
                continue;
            }
            let (kind, group, n) = match &w {
                Weights::Int4 {
                    group_size,
                    num_elements,
                    ..
                } => ("int4", *group_size, *num_elements),
                Weights::Int8 {
                    group_size,
                    num_elements,
                    ..
                } => ("int8", *group_size, *num_elements),
                _ => continue,
            };
            let bytes = w.as_bytes();
            put(&name, "U8", vec![bytes.len()], bytes)?;
            meta.insert(
                format!("{PACK_KEY}.{name}"),
                format!("{kind} {n} {group}").into(),
            );
        }
        progress(layer + 1, config.num_layers);
    }
    // Everything the loader did not quantize, as it is.
    let mut rest: Vec<&TensorInfo> = store
        .iter_tensors()
        .filter(|t| !packed.contains(&t.name))
        .collect();
    rest.sort_by(|a, b| a.name.cmp(&b.name));
    for t in rest {
        put(
            &t.name,
            dtype_tag(t.dtype)?,
            t.shape.clone(),
            store.tensor_bytes(&t.name)?,
        )?;
    }
    data.flush().map_err(io(&part))?;
    drop(data);

    header.insert("__metadata__".into(), serde_json::Value::Object(meta));
    let mut json = serde_json::to_vec(&serde_json::Value::Object(header)).expect("serializable");
    // Data starts 8-byte aligned, as safetensors writers pad it.
    json.resize(json.len().next_multiple_of(8), b' ');
    let path = out.join("model.safetensors");
    let mut file = std::io::BufWriter::new(std::fs::File::create(&path).map_err(io(&path))?);
    file.write_all(&(json.len() as u64).to_le_bytes())
        .map_err(io(&path))?;
    file.write_all(&json).map_err(io(&path))?;
    std::io::copy(
        &mut std::fs::File::open(&part).map_err(io(&part))?,
        &mut file,
    )
    .map_err(io(&path))?;
    file.flush().map_err(io(&path))?;
    std::fs::remove_file(&part).map_err(io(&part))?;

    copy_side_files(src, out)
}

/// Copy everything in `src` but its weights (config, tokenizer, templates,
/// a sentence-transformers pooling config) into `out`.
fn copy_side_files(src: &Path, out: &Path) -> Result<()> {
    let io = |path: &Path, source| DlmError::Io {
        path: path.to_path_buf(),
        source,
    };
    for entry in std::fs::read_dir(src).map_err(|e| io(src, e))? {
        let entry = entry.map_err(|e| io(src, e))?;
        let from = entry.path();
        let name = entry.file_name();
        let lower = name.to_string_lossy().to_lowercase();
        let weights = [
            ".safetensors",
            ".safetensors.index.json",
            ".bin",
            ".gguf",
            ".pt",
            ".pth",
        ]
        .iter()
        .any(|ext| lower.ends_with(ext));
        let to = out.join(&name);
        if from.is_dir() {
            if !lower.starts_with('.') {
                std::fs::create_dir_all(&to).map_err(|e| io(&to, e))?;
                copy_side_files(&from, &to)?;
            }
        } else if !weights {
            std::fs::copy(&from, &to).map_err(|e| io(&from, e))?;
        }
    }
    Ok(())
}

/// The tensors of `store` that are packed, for tests.
pub fn packed_names(store: &MmapStore) -> BTreeMap<String, String> {
    store
        .iter_shards()
        .flat_map(|s| s.header().metadata.iter())
        .filter_map(|(k, v)| {
            k.strip_prefix(&format!("{PACK_KEY}."))
                .map(|n| (n.to_string(), v.clone()))
        })
        .collect()
}
