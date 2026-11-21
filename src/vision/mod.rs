//! Image input for Gemma 3 (`Gemma3ForConditionalGeneration`, the 4B, 12B and
//! 27B checkpoints).
//!
//! A picture becomes `tokens_per_image` (256) vectors in the text model's
//! embedding space, which take the place of the image's soft tokens in the
//! prompt:
//!
//! 1. [`image`]: decode, resize to the tower's square input, map to `[-1, 1]`.
//! 2. The SigLIP tower: a 14x14 patch convolution (4096 patches at 896x896)
//!    plus learned position embeddings, then encoder layers of LayerNorm,
//!    bidirectional attention and a `gelu_pytorch_tanh` MLP, then a final
//!    LayerNorm.
//! 3. The projector: average-pool the patch grid down to 16x16, RMSNorm it
//!    with Gemma's `(1 + w)`, and multiply by `mm_input_projection_weight`.
//!
//! The tower is read from the model's own safetensors (the tensors
//! [`MmapStore::open_dir`] leaves out) or from a llama.cpp `mmproj` GGUF, and
//! its layers are read from the mapped file one at a time as they run, so it
//! holds no memory between images. On a `cuda-kernels` build each layer runs
//! on the GPU with buffers that are freed when the image is done.
//!
//! The text model then runs the image's tokens as one block that attends in
//! both directions ([`crate::generate::ImageEmbeds`]).

pub mod image;

#[cfg(any(feature = "cuda-kernels", feature = "rocm-kernels"))]
mod gpu;

use crate::error::{DlmError, Result};
use crate::forward::cpu::{layernorm, rmsnorm};
use crate::storage::MmapStore;
use crate::tokenizer::BpeTokenizer;
use std::path::{Path, PathBuf};

/// The vision tower's geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub eps: f32,
    /// Image tokens the projector pools the patch grid down to.
    pub tokens_per_image: usize,
    /// The text model's hidden size, the projection's output width.
    pub text_hidden: usize,
}

impl VisionConfig {
    fn patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    fn num_patches(&self) -> usize {
        self.patches_per_side() * self.patches_per_side()
    }

    fn validate(&self) -> Result<()> {
        let side = (self.tokens_per_image as f64).sqrt() as usize;
        let ok = self.patch_size > 0
            && self.image_size % self.patch_size == 0
            && self.heads > 0
            && self.hidden % self.heads == 0
            && side * side == self.tokens_per_image
            && side > 0
            && self.patches_per_side() % side == 0;
        if ok {
            Ok(())
        } else {
            Err(DlmError::InvalidConfig(format!(
                "vision tower geometry dlm cannot run: {self:?}"
            )))
        }
    }
}

/// Where the tower's tensors live, which decides their names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Naming {
    /// `vision_tower.vision_model.*` and `multi_modal_projector.*`.
    Hf,
    /// llama.cpp's `clip` names: `v.blk.N.*`, `mm.*`.
    Gguf,
}

/// One encoder layer's weights, as f32. Matrices are `[out, in]`.
pub(crate) struct VitLayer {
    pub ln1: (Vec<f32>, Vec<f32>),
    pub q: (Vec<f32>, Vec<f32>),
    pub k: (Vec<f32>, Vec<f32>),
    pub v: (Vec<f32>, Vec<f32>),
    pub o: (Vec<f32>, Vec<f32>),
    pub ln2: (Vec<f32>, Vec<f32>),
    pub fc1: (Vec<f32>, Vec<f32>),
    pub fc2: (Vec<f32>, Vec<f32>),
}

/// Gemma 3's SigLIP tower and multimodal projector, over a mapped checkpoint.
pub struct VisionTower {
    pub cfg: VisionConfig,
    store: MmapStore,
    naming: Naming,
    /// Where it was loaded from, for messages.
    pub source: PathBuf,
}

impl std::fmt::Debug for VisionTower {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionTower")
            .field("cfg", &self.cfg)
            .field("source", &self.source)
            .finish()
    }
}

fn invalid(msg: impl Into<String>) -> DlmError {
    DlmError::InvalidConfig(msg.into())
}

impl VisionTower {
    /// The tower for a model: the `mmproj` file when one is given, else the
    /// model's own safetensors when it is a multimodal checkpoint, else an
    /// `mmproj*.gguf` beside a GGUF model. `None` when there is none of those
    /// (a text-only model).
    pub fn find(model_path: &Path, mmproj: Option<&Path>) -> Result<Option<Self>> {
        if let Some(path) = mmproj {
            return Self::from_mmproj(path).map(Some);
        }
        if model_path.is_dir() {
            return Self::from_hf_dir(model_path);
        }
        let Some(dir) = model_path.parent() else {
            return Ok(None);
        };
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Ok(None);
        };
        let mut found: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("mmproj") && n.ends_with(".gguf"))
            })
            .collect();
        found.sort();
        match found.as_slice() {
            [] => Ok(None),
            [one] => Self::from_mmproj(one).map(Some),
            many => Err(invalid(format!(
                "several mmproj files beside {}: {}; pick one with --mmproj",
                model_path.display(),
                many.iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))),
        }
    }

    /// The tower in a multimodal safetensors checkpoint directory, or `None`
    /// when its `config.json` has no `vision_config`.
    pub fn from_hf_dir(dir: &Path) -> Result<Option<Self>> {
        let path = dir.join("config.json");
        let bytes = std::fs::read(&path).map_err(|source| DlmError::Io {
            path: path.clone(),
            source,
        })?;
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|source| DlmError::Json {
                context: path.display().to_string(),
                source,
            })?;
        let Some(v) = json.get("vision_config") else {
            return Ok(None);
        };
        let num = |o: &serde_json::Value, k: &str, d: u64| {
            o.get(k).and_then(serde_json::Value::as_u64).unwrap_or(d) as usize
        };
        // SiglipVisionConfig's defaults, for keys a config leaves out.
        let text_hidden = json
            .get("text_config")
            .and_then(|t| t.get("hidden_size"))
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| invalid("config.json: vision_config but no text_config.hidden_size"))?;
        let cfg = VisionConfig {
            image_size: num(v, "image_size", 224),
            patch_size: num(v, "patch_size", 16),
            hidden: num(v, "hidden_size", 768),
            intermediate: num(v, "intermediate_size", 3072),
            layers: num(v, "num_hidden_layers", 12),
            heads: num(v, "num_attention_heads", 12),
            eps: v
                .get("layer_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-6) as f32,
            tokens_per_image: num(&json, "mm_tokens_per_image", 256),
            text_hidden: text_hidden as usize,
        };
        cfg.validate()?;
        Ok(Some(Self {
            cfg,
            store: MmapStore::open_vision_dir(dir)?,
            naming: Naming::Hf,
            source: dir.to_path_buf(),
        }))
    }

    /// The tower in a llama.cpp `mmproj` GGUF (`general.architecture = clip`,
    /// `clip.projector_type = gemma3`).
    pub fn from_mmproj(path: &Path) -> Result<Self> {
        use crate::storage::gguf::Value;
        let store = MmapStore::open_path(path)?;
        let meta = store
            .gguf_metadata()
            .ok_or_else(|| invalid(format!("{}: not a GGUF file", path.display())))?;
        let projector = meta.get("clip.projector_type").and_then(Value::as_str);
        if projector != Some("gemma3") {
            return Err(invalid(format!(
                "{}: projector {projector:?}; dlm reads Gemma 3's (gemma3) mmproj files",
                path.display()
            )));
        }
        let int = |k: &str| {
            meta.get(&format!("clip.vision.{k}"))
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| invalid(format!("{}: no clip.vision.{k}", path.display())))
        };
        let cfg = VisionConfig {
            image_size: int("image_size")?,
            patch_size: int("patch_size")?,
            hidden: int("embedding_length")?,
            intermediate: int("feed_forward_length")?,
            layers: int("block_count")?,
            heads: int("attention.head_count")?,
            eps: meta
                .get("clip.vision.attention.layer_norm_epsilon")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
            // llama.cpp hard-codes Gemma 3's pooling to 16x16 tokens.
            tokens_per_image: 256,
            text_hidden: int("projection_dim")?,
        };
        cfg.validate()?;
        Ok(Self {
            cfg,
            store,
            naming: Naming::Gguf,
            source: path.to_path_buf(),
        })
    }

    fn name(&self, layer: Option<usize>, part: &str) -> String {
        let hf = |p: &str| match layer {
            Some(i) => format!("vision_tower.vision_model.encoder.layers.{i}.{p}"),
            None => p.to_string(),
        };
        match self.naming {
            Naming::Hf => match part {
                "patch.weight" => {
                    "vision_tower.vision_model.embeddings.patch_embedding.weight".into()
                }
                "patch.bias" => "vision_tower.vision_model.embeddings.patch_embedding.bias".into(),
                "pos" => "vision_tower.vision_model.embeddings.position_embedding.weight".into(),
                "post.weight" => "vision_tower.vision_model.post_layernorm.weight".into(),
                "post.bias" => "vision_tower.vision_model.post_layernorm.bias".into(),
                "soft_norm" => "multi_modal_projector.mm_soft_emb_norm.weight".into(),
                "proj" => "multi_modal_projector.mm_input_projection_weight".into(),
                p => hf(&p
                    .replace("ln1", "layer_norm1")
                    .replace("ln2", "layer_norm2")
                    .replace("attn_q", "self_attn.q_proj")
                    .replace("attn_k", "self_attn.k_proj")
                    .replace("attn_v", "self_attn.v_proj")
                    .replace("attn_out", "self_attn.out_proj")
                    .replace("fc1", "mlp.fc1")
                    .replace("fc2", "mlp.fc2")),
            },
            Naming::Gguf => match (part, layer) {
                ("patch.weight", _) => "v.patch_embd.weight".into(),
                ("patch.bias", _) => "v.patch_embd.bias".into(),
                ("pos", _) => "v.position_embd.weight".into(),
                ("post.weight", _) => "v.post_ln.weight".into(),
                ("post.bias", _) => "v.post_ln.bias".into(),
                ("soft_norm", _) => "mm.soft_emb_norm.weight".into(),
                ("proj", _) => "mm.input_projection.weight".into(),
                (p, Some(i)) => format!("v.blk.{i}.{p}"),
                (p, None) => p.to_string(),
            },
        }
    }

    /// A tensor as f32, checked to hold `len` values.
    fn floats(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        let (shard, info) = self
            .store
            .locate(name)
            .ok_or_else(|| DlmError::UnknownTensor(name.to_string()))?;
        let v = crate::storage::safetensors::bytes_to_f32(shard.tensor_bytes(name)?, info.dtype)?;
        if v.len() != len {
            return Err(invalid(format!(
                "{}: tensor {name:?} has {} values, expected {len}",
                self.source.display(),
                v.len()
            )));
        }
        Ok(v)
    }

    fn part(&self, layer: Option<usize>, part: &str, len: usize) -> Result<Vec<f32>> {
        self.floats(&self.name(layer, part), len)
    }

    /// The two MLP matrices of a GGUF layer, told apart by shape: llama.cpp's
    /// converter files SigLIP's `fc1` as `ffn_down` and `fc2` as `ffn_up`, and
    /// the shapes say which is which whatever a future converter calls them.
    fn gguf_mlp(&self, i: usize) -> Result<[String; 2]> {
        let (h, f) = (self.cfg.hidden, self.cfg.intermediate);
        let shape = |p: &str| {
            self.store
                .locate(&self.name(Some(i), &format!("{p}.weight")))
                .map(|(_, info)| info.shape.clone())
        };
        let (up, down) = (shape("ffn_up"), shape("ffn_down"));
        if down.as_deref() == Some(&[f, h][..]) && up.as_deref() == Some(&[h, f][..]) {
            Ok(["ffn_down".into(), "ffn_up".into()])
        } else if up.as_deref() == Some(&[f, h][..]) && down.as_deref() == Some(&[h, f][..]) {
            Ok(["ffn_up".into(), "ffn_down".into()])
        } else {
            Err(invalid(format!(
                "{}: layer {i}'s MLP is not [{f}, {h}] and [{h}, {f}]",
                self.source.display()
            )))
        }
    }

    pub(crate) fn layer(&self, i: usize) -> Result<VitLayer> {
        let (h, f) = (self.cfg.hidden, self.cfg.intermediate);
        let [fc1, fc2] = match self.naming {
            Naming::Hf => ["fc1".to_string(), "fc2".to_string()],
            Naming::Gguf => self.gguf_mlp(i)?,
        };
        let pair = |p: &str, rows: usize, cols: usize| -> Result<(Vec<f32>, Vec<f32>)> {
            let w = self.part(Some(i), &format!("{p}.weight"), rows * cols)?;
            let b = self.part(Some(i), &format!("{p}.bias"), rows)?;
            Ok((w, b))
        };
        Ok(VitLayer {
            ln1: pair("ln1", h, 1)?,
            q: pair("attn_q", h, h)?,
            k: pair("attn_k", h, h)?,
            v: pair("attn_v", h, h)?,
            o: pair("attn_out", h, h)?,
            ln2: pair("ln2", h, 1)?,
            fc1: pair(&fc1, f, h)?,
            fc2: pair(&fc2, h, f)?,
        })
    }

    /// Patch embeddings plus position embeddings: `[patches, hidden]`.
    fn embed_patches(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (p, side, size) = (c.patch_size, c.patches_per_side(), c.image_size);
        if pixels.len() != 3 * size * size {
            return Err(invalid(format!(
                "pixel_values has {} values, expected 3x{size}x{size}",
                pixels.len()
            )));
        }
        let k = 3 * p * p;
        // Each patch as the conv kernel reads it: channel, row, column.
        let mut patches = vec![0.0f32; c.num_patches() * k];
        for (n, patch) in patches.chunks_mut(k).enumerate() {
            let (py, px) = (n / side, n % side);
            for ch in 0..3 {
                for y in 0..p {
                    let src = ch * size * size + (py * p + y) * size + px * p;
                    patch[(ch * p + y) * p..(ch * p + y + 1) * p]
                        .copy_from_slice(&pixels[src..src + p]);
                }
            }
        }
        let w = self.part(None, "patch.weight", c.hidden * k)?;
        let b = self.part(None, "patch.bias", c.hidden)?;
        let mut x = linear(&patches, k, &w, c.hidden, Some(&b[..]));
        let pos = self.part(None, "pos", c.num_patches() * c.hidden)?;
        for (a, b) in x.iter_mut().zip(&pos) {
            *a += b;
        }
        Ok(x)
    }

    /// Post-LayerNorm, pooling and projection: `[tokens_per_image, text_hidden]`.
    fn project(&self, x: &[f32]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let h = c.hidden;
        let (pw, pb) = (
            self.part(None, "post.weight", h)?,
            self.part(None, "post.bias", h)?,
        );
        let normed: Vec<f32> = x
            .chunks(h)
            .flat_map(|row| layernorm(row, &pw, Some(&pb[..]), c.eps))
            .collect();
        // Average-pool the patch grid in k x k squares.
        let side = c.patches_per_side();
        let out_side = (c.tokens_per_image as f64).sqrt() as usize;
        let k = side / out_side;
        let mut pooled = vec![0.0f32; c.tokens_per_image * h];
        for (t, dst) in pooled.chunks_mut(h).enumerate() {
            let (oy, ox) = (t / out_side, t % out_side);
            for y in 0..k {
                for xx in 0..k {
                    let n = (oy * k + y) * side + ox * k + xx;
                    for (d, s) in dst.iter_mut().zip(&normed[n * h..(n + 1) * h]) {
                        *d += s;
                    }
                }
            }
            let inv = 1.0 / (k * k) as f32;
            dst.iter_mut().for_each(|v| *v *= inv);
        }
        // Gemma's RMSNorm scales by (1 + w); llama.cpp's converter stores the 1
        // already added.
        let mut soft = self.part(None, "soft_norm", h)?;
        if self.naming == Naming::Hf {
            soft.iter_mut().for_each(|v| *v += 1.0);
        }
        let normed: Vec<f32> = pooled
            .chunks(h)
            .flat_map(|row| rmsnorm(row, &soft, c.eps))
            .collect();
        // `mm_input_projection_weight` is [vision_hidden, text_hidden], applied
        // as `x @ W`: transposed here into the [out, in] the GEMM takes.
        let proj = self.part(None, "proj", h * c.text_hidden)?;
        let mut wt = vec![0.0f32; proj.len()];
        for i in 0..h {
            for j in 0..c.text_hidden {
                wt[j * h + i] = proj[i * c.text_hidden + j];
            }
        }
        Ok(linear(&normed, h, &wt, c.text_hidden, None))
    }

    /// Encode an image's `pixel_values` (see [`image::preprocess`]) into its
    /// `tokens_per_image` embeddings, `[tokens, text_hidden]`. `gpu` runs the
    /// encoder layers on the GPU (a `cuda-kernels` build), falling back to the
    /// CPU with a warning if it cannot.
    pub fn encode(&self, pixels: &[f32], gpu: bool) -> Result<Vec<f32>> {
        let mut x = self.embed_patches(pixels)?;
        #[cfg(any(feature = "cuda-kernels", feature = "rocm-kernels"))]
        let done = gpu
            && match gpu::encode_layers(self, &mut x) {
                Ok(()) => true,
                Err(e) => {
                    eprintln!("warning: the vision encoder runs on the CPU ({e})");
                    false
                }
            };
        #[cfg(not(any(feature = "cuda-kernels", feature = "rocm-kernels")))]
        let done = {
            let _ = gpu;
            false
        };
        if !done {
            for i in 0..self.cfg.layers {
                vit_layer_cpu(&self.cfg, &mut x, &self.layer(i)?);
            }
        }
        self.project(&x)
    }

    /// Decode, preprocess and encode an image file's bytes.
    pub fn encode_bytes(&self, bytes: &[u8], gpu: bool) -> Result<Vec<f32>> {
        let img = image::decode(bytes)?;
        self.encode(&image::preprocess(&img, self.cfg.image_size), gpu)
    }
}

/// `y = x W^T + b` for every row of `x` (`[rows, k]`), `W` being `[m, k]`.
/// Rows go to the CPU pool in blocks, each block reading `W` once.
pub(crate) fn linear(x: &[f32], k: usize, w: &[f32], m: usize, b: Option<&[f32]>) -> Vec<f32> {
    use crate::forward::simd::dot_f32;
    const BLOCK: usize = 32;
    let rows = x.len() / k;
    let mut y = vec![0.0f32; rows * m];
    let chunks: Vec<std::sync::Mutex<&mut [f32]>> =
        y.chunks_mut(BLOCK * m).map(std::sync::Mutex::new).collect();
    crate::forward::pool::parallel_for(chunks.len(), &|c| {
        let mut out = chunks[c].lock().unwrap_or_else(|e| e.into_inner());
        let r0 = c * BLOCK;
        let n = out.len() / m;
        for (o, wrow) in w.chunks(k).enumerate() {
            let bias = b.map_or(0.0, |b| b[o]);
            for r in 0..n {
                out[r * m + o] = dot_f32(wrow, &x[(r0 + r) * k..(r0 + r + 1) * k]) + bias;
            }
        }
    });
    y
}

/// One SigLIP encoder layer over every patch, on the CPU.
fn vit_layer_cpu(cfg: &VisionConfig, x: &mut [f32], w: &VitLayer) {
    let h = cfg.hidden;
    let n = x.len() / h;
    let ln = |x: &[f32], (lw, lb): &(Vec<f32>, Vec<f32>)| -> Vec<f32> {
        x.chunks(h)
            .flat_map(|row| layernorm(row, lw, Some(lb.as_slice()), cfg.eps))
            .collect()
    };
    let normed = ln(x, &w.ln1);
    let q = linear(&normed, h, &w.q.0, h, Some(&w.q.1[..]));
    let k = linear(&normed, h, &w.k.0, h, Some(&w.k.1[..]));
    let v = linear(&normed, h, &w.v.0, h, Some(&w.v.1[..]));
    let ctx = attention_cpu(&q, &k, &v, n, cfg.heads, h / cfg.heads);
    let attn = linear(&ctx, h, &w.o.0, h, Some(&w.o.1[..]));
    for (a, b) in x.iter_mut().zip(&attn) {
        *a += b;
    }
    let normed = ln(x, &w.ln2);
    let mut mid = linear(&normed, h, &w.fc1.0, cfg.intermediate, Some(&w.fc1.1[..]));
    mid.iter_mut()
        .for_each(|v| *v = crate::forward::cpu::gelu_tanh(*v));
    let out = linear(&mid, cfg.intermediate, &w.fc2.0, h, Some(&w.fc2.1[..]));
    for (a, b) in x.iter_mut().zip(&out) {
        *a += b;
    }
}

/// Unmasked multi-head attention over `n` tokens (`[n, heads * d]` each).
fn attention_cpu(q: &[f32], k: &[f32], v: &[f32], n: usize, heads: usize, d: usize) -> Vec<f32> {
    use crate::forward::simd::dot_f32;
    const BLOCK: usize = 16;
    let w = heads * d;
    let scale = 1.0 / (d as f32).sqrt();
    // Keys and values head-major, so each head's rows are contiguous.
    let by_head = |m: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; m.len()];
        for t in 0..n {
            for hh in 0..heads {
                out[(hh * n + t) * d..(hh * n + t + 1) * d]
                    .copy_from_slice(&m[t * w + hh * d..t * w + (hh + 1) * d]);
            }
        }
        out
    };
    let (kh, vh) = (by_head(k), by_head(v));
    let mut ctx = vec![0.0f32; n * w];
    let chunks: Vec<std::sync::Mutex<&mut [f32]>> = ctx
        .chunks_mut(BLOCK * w)
        .map(std::sync::Mutex::new)
        .collect();
    crate::forward::pool::parallel_for(chunks.len(), &|c| {
        let mut out = chunks[c].lock().unwrap_or_else(|e| e.into_inner());
        let mut scores = vec![0.0f32; n];
        for (r, row) in out.chunks_mut(w).enumerate() {
            let t = c * BLOCK + r;
            for hh in 0..heads {
                let qh = &q[t * w + hh * d..t * w + (hh + 1) * d];
                let keys = &kh[hh * n * d..(hh + 1) * n * d];
                let mut max = f32::NEG_INFINITY;
                for (s, key) in scores.iter_mut().zip(keys.chunks(d)) {
                    *s = dot_f32(qh, key) * scale;
                    max = max.max(*s);
                }
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let dst = &mut row[hh * d..(hh + 1) * d];
                for (s, val) in scores
                    .iter()
                    .zip(vh[hh * n * d..(hh + 1) * n * d].chunks(d))
                {
                    let p = s / sum;
                    for (o, x) in dst.iter_mut().zip(val) {
                        *o += p * x;
                    }
                }
            }
        }
    });
    ctx
}

/// The token ids Gemma 3 surrounds and fills an image with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageTokens {
    /// `<start_of_image>`: where the chat template puts an image.
    pub boi: u32,
    /// `<end_of_image>`.
    pub eoi: u32,
    /// The id written at each image position. Its embedding is never looked
    /// up -- the image's own embeddings replace it -- so it only has to be a
    /// valid id: `<image_soft_token>` when the vocabulary has it (safetensors),
    /// else the pad token (llama.cpp's vocabulary stops short of it).
    pub placeholder: u32,
    pub per_image: usize,
}

/// The chat-template marker for an image.
pub const IMAGE_MARKER: &str = "<start_of_image>";

impl ImageTokens {
    /// The ids `tokenizer` gives Gemma 3's image markers.
    pub fn new(tokenizer: &BpeTokenizer, vocab_size: usize, per_image: usize) -> Result<Self> {
        let id = |s: &str| tokenizer.id_of(s).filter(|&id| (id as usize) < vocab_size);
        let boi = id(IMAGE_MARKER)
            .ok_or_else(|| invalid("the tokenizer has no <start_of_image> token"))?;
        let eoi = id("<end_of_image>")
            .ok_or_else(|| invalid("the tokenizer has no <end_of_image> token"))?;
        Ok(Self {
            boi,
            eoi,
            placeholder: id("<image_soft_token>").unwrap_or(0),
            per_image,
        })
    }

    /// Tokenize a rendered prompt whose images are marked with
    /// [`IMAGE_MARKER`], laying each out as `Gemma3Processor` does:
    /// `"\n\n<start_of_image>"`, the image's tokens, `"<end_of_image>\n\n"`.
    /// Returns the ids and where each image's tokens start.
    pub fn encode(&self, tokenizer: &BpeTokenizer, prompt: &str) -> Result<(Vec<u32>, Vec<usize>)> {
        // The image tokens themselves are inserted after tokenizing: they are
        // special tokens, so the text around them tokenizes the same either way,
        // and llama.cpp's vocabulary has no `<image_soft_token>` to spell.
        let text = prompt.replace(IMAGE_MARKER, "\n\n<start_of_image><end_of_image>\n\n");
        let mut ids = Vec::new();
        let mut starts = Vec::new();
        for id in tokenizer.encode(&text)? {
            ids.push(id);
            if id == self.boi {
                starts.push(ids.len());
                ids.extend(std::iter::repeat_n(self.placeholder, self.per_image));
            }
        }
        Ok((ids, starts))
    }
}

/// An image-capable model's tower plus the ids its prompts use.
#[derive(Debug)]
pub struct Vision {
    pub tower: VisionTower,
    pub tokens: ImageTokens,
    /// Run the encoder layers on the GPU.
    pub gpu: bool,
}

impl Vision {
    /// Encode each image (file bytes) and place it in `prompt` (rendered with
    /// one [`IMAGE_MARKER`] per image): the prompt's ids and the images'
    /// embeddings, ready for [`crate::generate::Generator`].
    pub fn prompt(
        &self,
        tokenizer: &BpeTokenizer,
        prompt: &str,
        images: &[Vec<u8>],
    ) -> Result<(Vec<u32>, Vec<crate::generate::ImageEmbeds>)> {
        let (ids, starts) = self.tokens.encode(tokenizer, prompt)?;
        if starts.len() != images.len() {
            return Err(invalid(format!(
                "the prompt has {} image markers but {} images came with it",
                starts.len(),
                images.len()
            )));
        }
        let embeds = starts
            .into_iter()
            .zip(images)
            .map(|(start, bytes)| {
                Ok(crate::generate::ImageEmbeds {
                    start,
                    rows: self.tower.encode_bytes(bytes, self.gpu)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((ids, embeds))
    }
}
