//! LoRA adapters (PEFT's format), merged into the weights as they load.
//!
//! An adapter is a low-rank update per projection: `W' = W + s * B @ A`, with
//! `A` `[r, in]`, `B` `[out, r]` and `s = alpha / r` (`alpha / sqrt(r)` under
//! rsLoRA). dlm adds it to each weight when the loader reads it, before
//! `--quant` quantizes it or the weight goes back to its checkpoint dtype, so
//! every path that loads layers (resident, streamed, split) serves the adapted
//! model at the base model's speed and size.
//!
//! What it merges is checked when the adapter is attached rather than as layers
//! load: each target must be one of a decoder layer's seven projections, stored
//! as a float `{name}.weight` of the adapter's shape. A projection dlm reads
//! some other way (a fused QKV, GPTQ's packed codes, a GGUF block) would take
//! no update and serve the base model without a word, so it is refused instead.
//! So are DoRA, adapter biases and fully trained modules, which are more than
//! `B @ A`.

use crate::error::{DlmError, Result};
use crate::storage::{bytes_to_f32, Dtype, MmapStore};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// A loaded adapter: its tensors, its scale, and which base weight each
/// `(A, B)` pair updates.
#[derive(Debug)]
pub struct Lora {
    store: MmapStore,
    scale: f32,
    /// Base weight name (`model.layers.0.self_attn.q_proj.weight`) → the
    /// adapter's `A` and `B` tensor names.
    targets: HashMap<String, (String, String)>,
}

#[derive(Deserialize)]
struct AdapterConfig {
    r: usize,
    lora_alpha: f32,
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    use_dora: bool,
    #[serde(default)]
    bias: Option<String>,
    #[serde(default)]
    modules_to_save: Option<serde_json::Value>,
}

/// The projections a target may be: a decoder layer's attention and MLP.
const PROJECTIONS: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

fn refuse(msg: String) -> DlmError {
    DlmError::InvalidConfig(format!("--lora: {msg}"))
}

impl Lora {
    /// Open the PEFT adapter in `dir` (`adapter_config.json` and
    /// `adapter_model.safetensors`) for the model in `base`, checking every
    /// target against it.
    pub fn open(dir: &Path, base: &MmapStore) -> Result<Self> {
        let config_path = dir.join("adapter_config.json");
        let text = std::fs::read(&config_path).map_err(|source| DlmError::Io {
            path: config_path.clone(),
            source,
        })?;
        let config: AdapterConfig =
            serde_json::from_slice(&text).map_err(|source| DlmError::Json {
                context: "adapter_config.json".into(),
                source,
            })?;
        if config.use_dora {
            return Err(refuse(
                "DoRA adapters are not supported, only plain LoRA".into(),
            ));
        }
        if config.bias.as_deref().is_some_and(|b| b != "none") {
            return Err(refuse(format!(
                "adapter biases ({:?}) are not supported",
                config.bias
            )));
        }
        if config
            .modules_to_save
            .as_ref()
            .is_some_and(|m| !m.is_null() && m.as_array().is_none_or(|a| !a.is_empty()))
        {
            return Err(refuse(
                "modules_to_save (whole trained modules) are not supported".into(),
            ));
        }
        if config.r == 0 {
            return Err(refuse("r is 0".into()));
        }
        if base.gguf_metadata().is_some() {
            return Err(refuse(
                "a GGUF model's weights are already quantized and cannot take the update; \
                 use the safetensors checkpoint"
                    .into(),
            ));
        }
        let scale = if config.use_rslora {
            config.lora_alpha / (config.r as f32).sqrt()
        } else {
            config.lora_alpha / config.r as f32
        };

        let store = MmapStore::open_path(dir)?;
        let mut targets = HashMap::new();
        for info in store.iter_tensors() {
            let Some(module) = info.name.strip_suffix(".lora_A.weight") else {
                continue;
            };
            let module = module.strip_prefix("base_model.model.").unwrap_or(module);
            let a_name = info.name.clone();
            let b_name = a_name.replace(".lora_A.weight", ".lora_B.weight");
            let weight = format!("{module}.weight");
            let known = module
                .strip_prefix("model.layers.")
                .and_then(|rest| rest.split_once('.'))
                .is_some_and(|(layer, proj)| {
                    layer.parse::<u32>().is_ok() && PROJECTIONS.contains(&proj)
                });
            if !known {
                return Err(refuse(format!(
                    "the adapter targets {module}, which is not one of a decoder layer's \
                     projections ({})",
                    PROJECTIONS.join(", ")
                )));
            }
            let (_, w) = base.locate(&weight).ok_or_else(|| {
                refuse(format!(
                    "the model has no float {weight} for the adapter to update (a fused, \
                     packed or GGUF projection cannot take it)"
                ))
            })?;
            if !matches!(w.dtype, Dtype::F32 | Dtype::BF16 | Dtype::F16) {
                return Err(refuse(format!(
                    "{weight} is {:?}, not a float weight",
                    w.dtype
                )));
            }
            let (_, b) = store
                .locate(&b_name)
                .ok_or_else(|| refuse(format!("{a_name} has no {b_name}")))?;
            let shapes_fit = matches!((&info.shape[..], &b.shape[..], &w.shape[..]),
                ([r, i], [o, r2], [o2, i2]) if r == r2 && *r == config.r && o == o2 && i == i2);
            if !shapes_fit {
                return Err(refuse(format!(
                    "{module}: A {:?} and B {:?} at r = {} do not update {weight} {:?}",
                    info.shape, b.shape, config.r, w.shape
                )));
            }
            targets.insert(weight, (a_name, b_name));
        }
        if targets.is_empty() {
            return Err(refuse(format!(
                "no lora_A/lora_B tensors in {}",
                dir.display()
            )));
        }
        Ok(Lora {
            store,
            scale,
            targets,
        })
    }

    /// How many weights the adapter updates.
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Whether it updates none (never true of an opened adapter).
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// `scale`: what `B @ A` is multiplied by.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// The update to base weight `name`, `[out, in]` row-major, if the adapter
    /// has one: `scale * B @ A`.
    pub fn delta(&self, name: &str) -> Result<Option<Vec<f32>>> {
        let Some((a_name, b_name)) = self.targets.get(name) else {
            return Ok(None);
        };
        let read = |n: &str| -> Result<(Vec<f32>, Vec<usize>)> {
            let (shard, info) = self.store.locate(n).expect("checked when opened");
            Ok((
                bytes_to_f32(shard.tensor_bytes(n)?, info.dtype)?,
                info.shape.clone(),
            ))
        };
        let (a, a_shape) = read(a_name)?;
        let (b, b_shape) = read(b_name)?;
        let (r, inn, out) = (a_shape[0], a_shape[1], b_shape[0]);
        let mut delta = vec![0.0f32; out * inn];
        let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
        let rows = out.div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            for (chunk, block) in delta.chunks_mut(rows * inn).enumerate() {
                let (a, b) = (&a, &b);
                scope.spawn(move || {
                    for (i, row) in block.chunks_mut(inn).enumerate() {
                        let o = chunk * rows + i;
                        for k in 0..r {
                            let bk = b[o * r + k] * self.scale;
                            for (d, &av) in row.iter_mut().zip(&a[k * inn..(k + 1) * inn]) {
                                *d += bk * av;
                            }
                        }
                    }
                });
            }
        });
        Ok(Some(delta))
    }
}

/// `x` rounded to the nearest bf16 (ties to even), as torch rounds.
pub(crate) fn f32_to_bf16(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7FC0;
    }
    let b = x.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-tensor-per-entry safetensors file at `path`.
    fn safetensors(path: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let begin = data.len();
            data.extend(values.iter().flat_map(|v| v.to_le_bytes()));
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [begin, data.len()]}),
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend(header);
        out.extend(data);
        std::fs::write(path, out).unwrap();
    }

    const Q: &str = "model.layers.0.self_attn.q_proj";

    /// A base model with one `[3, 2]` q_proj, and a rank-1 adapter for it.
    fn fixture(config: &str, target: &str) -> (tempfile::TempDir, MmapStore) {
        let tmp = tempfile::tempdir().unwrap();
        let (base, adapter) = (tmp.path().join("base"), tmp.path().join("adapter"));
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(&adapter).unwrap();
        safetensors(
            &base.join("model.safetensors"),
            &[(&format!("{Q}.weight"), vec![3, 2], vec![0.0; 6])],
        );
        safetensors(
            &adapter.join("adapter_model.safetensors"),
            &[
                (
                    &format!("base_model.model.{target}.lora_A.weight"),
                    vec![1, 2],
                    vec![1.0, 2.0],
                ),
                (
                    &format!("base_model.model.{target}.lora_B.weight"),
                    vec![3, 1],
                    vec![1.0, 0.5, -1.0],
                ),
            ],
        );
        std::fs::write(adapter.join("adapter_config.json"), config).unwrap();
        let store = MmapStore::open_path(&base).unwrap();
        (tmp, store)
    }

    #[test]
    fn the_update_is_scale_times_b_at_a() {
        let (tmp, base) = fixture(r#"{"r": 1, "lora_alpha": 4}"#, Q);
        let lora = Lora::open(&tmp.path().join("adapter"), &base).unwrap();
        assert_eq!((lora.len(), lora.scale()), (1, 4.0));
        let d = lora.delta(&format!("{Q}.weight")).unwrap().unwrap();
        // B @ A = [[1, 2], [0.5, 1], [-1, -2]], times 4.
        assert_eq!(d, vec![4.0, 8.0, 2.0, 4.0, -4.0, -8.0]);
        assert!(lora
            .delta("model.layers.0.mlp.up_proj.weight")
            .unwrap()
            .is_none());
        // rsLoRA scales by alpha / sqrt(r).
        let (tmp, base) = fixture(r#"{"r": 1, "lora_alpha": 4, "use_rslora": true}"#, Q);
        assert_eq!(
            Lora::open(&tmp.path().join("adapter"), &base)
                .unwrap()
                .scale(),
            4.0
        );
    }

    /// The loader reads the merged weight, whatever it does with it next.
    #[test]
    fn the_loader_reads_weights_with_the_update_in() {
        use crate::model::QuantScheme;
        let (tmp, base) = fixture(r#"{"r": 1, "lora_alpha": 4}"#, Q);
        let lora = Lora::open(&tmp.path().join("adapter"), &base).unwrap();
        let store = base.with_lora(lora);
        let name = format!("{Q}.weight");
        let want = [4.0, 8.0, 2.0, 4.0, -4.0, -8.0];
        let w = crate::loader::load_native(&store, &name, 6, QuantScheme::F32).unwrap();
        assert_eq!((0..6).map(|i| w.get(i)).collect::<Vec<_>>(), want);
        let w = crate::loader::load_native(&store, &name, 6, QuantScheme::Int8).unwrap();
        assert!((0..6).all(|i| (w.get(i) - want[i]).abs() < 0.05), "{w:?}");
    }

    #[test]
    fn what_would_not_merge_is_refused() {
        for (config, target, why) in [
            (r#"{"r": 1, "lora_alpha": 4, "use_dora": true}"#, Q, "DoRA"),
            (r#"{"r": 1, "lora_alpha": 4, "bias": "all"}"#, Q, "biases"),
            (
                r#"{"r": 1, "lora_alpha": 4, "modules_to_save": ["lm_head"]}"#,
                Q,
                "modules_to_save",
            ),
            (r#"{"r": 2, "lora_alpha": 4}"#, Q, "do not update"),
            (
                r#"{"r": 1, "lora_alpha": 4}"#,
                "model.layers.0.self_attn.qkv_proj",
                "not one of",
            ),
            (
                r#"{"r": 1, "lora_alpha": 4}"#,
                "model.layers.0.mlp.up_proj",
                "no float",
            ),
        ] {
            let (tmp, base) = fixture(config, target);
            let err = Lora::open(&tmp.path().join("adapter"), &base)
                .unwrap_err()
                .to_string();
            assert!(err.contains(why), "{why}: {err}");
        }
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        // Halfway between two bf16 values: ties go to the even one.
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8001)), 0x3F81);
    }
}
