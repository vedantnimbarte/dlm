//! An EAGLE-3 draft head on the GPU.
//!
//! EAGLE-3 (Li et al., 2025) drafts with one small decoder layer that reads the
//! target's own hidden states instead of re-deriving them from tokens. The target's
//! residual stream entering three of its layers (low, middle, high) is
//! concatenated and folded to the hidden width by `fc`; the layer's attention
//! then reads that feature beside the embedding of the next token, so position
//! `p` takes `(token p + 1, feature p)` and predicts token `p + 2`. Past the
//! first draft token there is no target feature yet, so the layer's own pre-norm
//! output stands in for it. The draft keeps its own KV cache, and a small
//! vocabulary (`draft_vocab_size`) whose ids map to the target's through `d2t`.
//!
//! The layout follows vLLM's `llama_eagle3.py` and SpecForge, which trained the
//! heads dlm reads (e.g. `AngelSlim/Qwen3-1.7B_eagle3`): `midlayer.*` holds the
//! layer, `fc`, `norm` and `lm_head` sit beside it, and `d2t[i]` is the offset
//! from draft id `i` to its target id.

use crate::error::{DlmError, Result};
use crate::forward::gpu::{upload_weight, GpuLmHead};
use crate::forward::Weights;
use crate::gpu::device::DeviceBuffer;
use crate::model::QuantScheme;
use crate::storage::MmapStore;
use std::ffi::c_void;
use std::path::Path;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn dlm_eagle3_step(
        hidden: i32,
        q_dim: i32,
        kv_dim: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        inter: i32,
        rms_eps: f32,
        w_dtype: *const i32,
        w_group: *const i32,
        fc: *const c_void,
        q_proj: *const c_void,
        k_proj: *const c_void,
        v_proj: *const c_void,
        o_proj: *const c_void,
        gate_proj: *const c_void,
        up_proj: *const c_void,
        down_proj: *const c_void,
        in_norm: *const f32,
        hidden_norm: *const f32,
        post_norm: *const f32,
        final_norm: *const f32,
        inv_freq: *const f32,
        embed_host: *const f32,
        feat_host: *const f32,
        h_dev: *mut f32,
        kv_keys: *mut f32,
        kv_values: *mut f32,
        kv_table: *const i32,
        position: i32,
        out_host: *mut f32,
    ) -> i32;
}

/// The shape of an EAGLE-3 head, from its `config.json`.
#[derive(Debug, Clone)]
pub struct Eagle3Config {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub draft_vocab: usize,
}

impl Eagle3Config {
    /// Read `config.json`. Only Llama-shaped EAGLE-3 heads are accepted
    /// (`LlamaForCausalLMEagle3`, one layer, no RoPE scaling).
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|source| DlmError::Json {
                context: "EAGLE-3 config.json".into(),
                source,
            })?;
        let is_eagle3 = v["architectures"].as_array().is_some_and(|a| {
            a.iter()
                .any(|x| x.as_str().is_some_and(|s| s.contains("Eagle3")))
        });
        if !is_eagle3 {
            return Err(DlmError::InvalidConfig(
                "not an EAGLE-3 head: config.json architectures names no *Eagle3 model".into(),
            ));
        }
        let int = |k: &str| {
            v[k].as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| DlmError::InvalidConfig(format!("EAGLE-3 config.json has no {k}")))
        };
        if v["num_hidden_layers"].as_u64().unwrap_or(1) != 1 {
            return Err(DlmError::InvalidConfig(
                "EAGLE-3 heads with more than one layer are not supported".into(),
            ));
        }
        if !v["rope_scaling"].is_null() {
            return Err(DlmError::InvalidConfig(
                "EAGLE-3 heads with RoPE scaling are not supported".into(),
            ));
        }
        let hidden = int("hidden_size")?;
        let heads = int("num_attention_heads")?;
        Ok(Self {
            hidden,
            heads,
            kv_heads: int("num_key_value_heads").unwrap_or(heads),
            head_dim: v["head_dim"]
                .as_u64()
                .map_or(hidden / heads, |x| x as usize),
            inter: int("intermediate_size")?,
            rms_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            rope_theta: v["rope_theta"].as_f64().unwrap_or(10_000.0) as f32,
            draft_vocab: v["draft_vocab_size"]
                .as_u64()
                .or(v["vocab_size"].as_u64())
                .ok_or_else(|| DlmError::InvalidConfig("EAGLE-3 config.json has no vocab".into()))?
                as usize,
        })
    }
}

/// Which target layers' inputs EAGLE-3 reads: low, middle and high, as vLLM and
/// SpecForge pick them when the head does not say.
pub fn aux_layers(target_layers: u32) -> [u32; 3] {
    [2, target_layers / 2, target_layers.saturating_sub(3)]
}

/// An EAGLE-3 head's weights, resident on the GPU and shared by every
/// sequence it drafts for; each sequence keeps an [`Eagle3Seq`].
pub struct Eagle3Head {
    cfg: Eagle3Config,
    /// fc, q, k, v, o, gate, up, down.
    w: Vec<DeviceBuffer>,
    w_dtype: [i32; 8],
    w_group: [i32; 8],
    in_norm: DeviceBuffer,
    hidden_norm: DeviceBuffer,
    post_norm: DeviceBuffer,
    final_norm: DeviceBuffer,
    inv_freq: DeviceBuffer,
    lm_head: GpuLmHead,
    /// Target id of each draft id.
    d2t: Vec<u32>,
}

/// One sequence's draft state: the head's KV cache, and the pre-norm output of
/// its last step (the next step's feature when the target has none for it).
pub struct Eagle3Seq {
    keys: DeviceBuffer,
    values: DeviceBuffer,
    /// `[0, 1, 2, ...]`: the block table of one-row blocks the attention
    /// kernels address KV rows through.
    table: DeviceBuffer,
    h: DeviceBuffer,
    max_positions: usize,
}

impl Eagle3Head {
    /// Load the head in `dir` (`config.json` + safetensors), its matrices in
    /// `quant` (`Fp16` keeps an f16 head as it is). `target_hidden` and
    /// `target_vocab` must match the target model.
    pub fn load(
        dir: &Path,
        quant: QuantScheme,
        target_hidden: usize,
        target_vocab: usize,
    ) -> Result<Self> {
        let config_path = dir.join("config.json");
        let bytes = std::fs::read(&config_path).map_err(|source| DlmError::Io {
            path: config_path,
            source,
        })?;
        let cfg = Eagle3Config::from_json(&bytes)?;
        if cfg.hidden != target_hidden {
            return Err(DlmError::InvalidConfig(format!(
                "EAGLE-3 head is {} wide, the target {}: it was trained for another model",
                cfg.hidden, target_hidden
            )));
        }
        let store = MmapStore::open_dir(dir)?;
        let (h, q, kv, inter) = (
            cfg.hidden,
            cfg.heads * cfg.head_dim,
            cfg.kv_heads * cfg.head_dim,
            cfg.inter,
        );
        let mat = |name: &str, rows: usize, cols: usize| -> Result<Weights> {
            crate::loader::load_native(&store, name, rows * cols, quant)
        };
        let mats = [
            mat("fc.weight", h, 3 * h)?,
            mat("midlayer.self_attn.q_proj.weight", q, 2 * h)?,
            mat("midlayer.self_attn.k_proj.weight", kv, 2 * h)?,
            mat("midlayer.self_attn.v_proj.weight", kv, 2 * h)?,
            mat("midlayer.self_attn.o_proj.weight", h, q)?,
            mat("midlayer.mlp.gate_proj.weight", inter, h)?,
            mat("midlayer.mlp.up_proj.weight", inter, h)?,
            mat("midlayer.mlp.down_proj.weight", h, inter)?,
        ];
        let mut w_dtype = [0i32; 8];
        let mut w_group = [0i32; 8];
        for (i, m) in mats.iter().enumerate() {
            w_dtype[i] = m.dtype_code();
            w_group[i] = m.group_size() as i32;
        }
        let w = mats.iter().map(upload_weight).collect::<Result<Vec<_>>>()?;
        let norm = |name: &str| -> Result<DeviceBuffer> {
            let v = crate::loader::load_floats(&store, name)?;
            if v.len() != h {
                return Err(DlmError::InvalidConfig(format!(
                    "{name}: {} values, expected {h}",
                    v.len()
                )));
            }
            DeviceBuffer::from_slice(&v)
        };
        let lm = mat("lm_head.weight", cfg.draft_vocab, h)?;
        let lm_head = GpuLmHead::new(&lm, cfg.draft_vocab, h, None)?;
        let d2t = read_d2t(&store, cfg.draft_vocab, target_vocab)?;
        let inv = crate::forward::cpu::rope_inv_freqs(cfg.head_dim, cfg.rope_theta, None);
        Ok(Self {
            in_norm: norm("midlayer.input_layernorm.weight")?,
            hidden_norm: norm("midlayer.hidden_norm.weight")?,
            post_norm: norm("midlayer.post_attention_layernorm.weight")?,
            final_norm: norm("norm.weight")?,
            inv_freq: DeviceBuffer::from_slice(&inv)?,
            lm_head,
            d2t,
            w,
            w_dtype,
            w_group,
            cfg,
        })
    }

    /// Fresh draft state for a sequence of up to `max_positions` tokens.
    pub fn new_seq(&self, max_positions: usize) -> Result<Eagle3Seq> {
        let kv = self.cfg.kv_heads * self.cfg.head_dim;
        let rows = max_positions.max(1);
        let table: Vec<u8> = (0..rows as i32).flat_map(i32::to_le_bytes).collect();
        Ok(Eagle3Seq {
            table: DeviceBuffer::from_bytes(&table, rows)?,
            keys: DeviceBuffer::new(max_positions.max(1) * kv)?,
            values: DeviceBuffer::new(max_positions.max(1) * kv)?,
            h: DeviceBuffer::new(self.cfg.hidden)?,
            max_positions,
        })
    }

    pub fn config(&self) -> &Eagle3Config {
        &self.cfg
    }

    /// Run position `position`: `embed` is the target's embedding of token
    /// `position + 1`, `feat` the target's three captured hidden states for
    /// `position` (`3 × hidden`), or `None` to read the previous step's output.
    /// Returns the final-normed hidden state for [`argmax`](Self::argmax) or
    /// [`logits`](Self::logits).
    pub fn step(
        &self,
        seq: &mut Eagle3Seq,
        embed: &[f32],
        feat: Option<&[f32]>,
        position: usize,
    ) -> Result<Vec<f32>> {
        let h = self.cfg.hidden;
        if embed.len() != h || feat.is_some_and(|f| f.len() != 3 * h) {
            return Err(DlmError::ShapeMismatch {
                expected: h,
                got: embed.len(),
            });
        }
        if position >= seq.max_positions {
            return Err(DlmError::InvalidConfig(format!(
                "EAGLE-3 draft KV holds {} positions; position {position} is past it",
                seq.max_positions
            )));
        }
        let mut out = vec![0.0f32; h];
        let c = &self.cfg;
        // SAFETY: every device pointer is a live buffer of the size the kernel
        // reads (checked at load and above); the host slices outlive the call,
        // which copies them synchronously.
        let code = unsafe {
            dlm_eagle3_step(
                h as i32,
                (c.heads * c.head_dim) as i32,
                (c.kv_heads * c.head_dim) as i32,
                c.heads as i32,
                c.kv_heads as i32,
                c.head_dim as i32,
                c.inter as i32,
                c.rms_eps,
                self.w_dtype.as_ptr(),
                self.w_group.as_ptr(),
                self.w[0].as_ptr() as *const c_void,
                self.w[1].as_ptr() as *const c_void,
                self.w[2].as_ptr() as *const c_void,
                self.w[3].as_ptr() as *const c_void,
                self.w[4].as_ptr() as *const c_void,
                self.w[5].as_ptr() as *const c_void,
                self.w[6].as_ptr() as *const c_void,
                self.w[7].as_ptr() as *const c_void,
                self.in_norm.as_ptr(),
                self.hidden_norm.as_ptr(),
                self.post_norm.as_ptr(),
                self.final_norm.as_ptr(),
                self.inv_freq.as_ptr(),
                embed.as_ptr(),
                feat.map_or(std::ptr::null(), |f| f.as_ptr()),
                seq.h.as_mut_ptr(),
                seq.keys.as_mut_ptr(),
                seq.values.as_mut_ptr(),
                seq.table.as_ptr() as *const i32,
                position as i32,
                out.as_mut_ptr(),
            )
        };
        if code != 0 {
            return Err(DlmError::Gpu {
                api: "dlm_eagle3_step",
                code,
            });
        }
        Ok(out)
    }

    /// The target id of the draft's most likely next token after `out`.
    pub fn argmax(&self, out: &[f32]) -> Result<u32> {
        let id = self.lm_head.argmax(out)?[0] as usize;
        Ok(self.d2t[id])
    }

    /// The draft's logits after `out`, as `(target id, logit)` per draft id.
    pub fn logits(&self, out: &[f32]) -> Result<Vec<(u32, f32)>> {
        Ok(self
            .lm_head
            .logits(out)?
            .into_iter()
            .enumerate()
            .map(|(i, l)| (self.d2t[i], l))
            .collect())
    }
}

/// `d2t` as target ids: the file stores, per draft id `i`, the offset
/// `target - i` (i64).
fn read_d2t(store: &MmapStore, draft_vocab: usize, target_vocab: usize) -> Result<Vec<u32>> {
    let (shard, info) = store
        .locate("d2t")
        .ok_or_else(|| DlmError::UnknownTensor("d2t".into()))?;
    let bytes = shard.tensor_bytes("d2t")?;
    if info.dtype != crate::storage::Dtype::I64 || bytes.len() != draft_vocab * 8 {
        return Err(DlmError::InvalidConfig(format!(
            "d2t: expected {draft_vocab} i64 offsets, got {:?} with {} bytes",
            info.dtype,
            bytes.len()
        )));
    }
    bytes
        .chunks_exact(8)
        .enumerate()
        .map(|(i, b)| {
            let off = i64::from_le_bytes(b.try_into().expect("8 bytes"));
            let t = i as i64 + off;
            if t < 0 || t as usize >= target_vocab {
                Err(DlmError::InvalidConfig(format!(
                    "d2t maps draft id {i} to {t}, outside the target's {target_vocab} tokens"
                )))
            } else {
                Ok(t as u32)
            }
        })
        .collect()
}
