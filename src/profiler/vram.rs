//! Dynamic VRAM profiling math engine (`specs.md` §3.1).
//!
//! Implements the layer-budget formula:
//!
//! ```text
//!                    ⌊ M_free − M_safety − M_kv_total ⌋
//! LayersToLoad  =    ────────────────────────────────
//!                            M_layer_weight
//! ```
//!
//! * `M_free`        — runtime free VRAM from `cudaMemGetInfo` (`crate::cuda`).
//! * `M_safety`      — fixed cushion for activation spikes (default 1.5 GiB).
//! * `M_kv_total`    — bytes the PagedAttention KV cache holds for the *whole*
//!   target context. In a layer-streaming engine the KV history for every layer
//!   stays resident in the Cache Zone even as weights are swapped, so this term
//!   sums across `num_layers` — see [`VramProfiler::kv_total_bytes`].
//! * `M_layer_weight`— average bytes of one streamed transformer block.
//!
//! All arithmetic is done in `u64`/`f64` with saturating subtraction so an
//! over-subscribed device yields a zero/one-layer budget instead of underflow.

use crate::error::Result;
use crate::gpu;
use crate::model::{ModelConfig, QuantScheme};
use crate::storage::LayerCatalog;

/// Default safety cushion: 1.5 GiB, per `specs.md` §3.1 (`M_safety`).
pub const DEFAULT_SAFETY_MARGIN_BYTES: u64 = 3 * 512 * 1024 * 1024; // 1.5 * 2^30

/// Bytes per KV element. KV cache is stored FP16 (`specs.md` §3.1: "2 bytes").
const KV_BYTES_PER_ELEMENT: u64 = 2;

/// Configurable inputs to the VRAM math.
#[derive(Debug, Clone)]
pub struct VramProfiler {
    /// Target context length `L_context` (tokens) the KV cache is sized for.
    pub target_context: u32,
    /// `M_safety` cushion in bytes.
    pub safety_margin_bytes: u64,
}

impl Default for VramProfiler {
    fn default() -> Self {
        Self {
            target_context: 8192,
            safety_margin_bytes: DEFAULT_SAFETY_MARGIN_BYTES,
        }
    }
}

/// A fully itemized profiling result. Every term is exposed so the decision is
/// auditable in logs rather than a bare layer count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramPlan {
    /// `M_free` observed at profile time.
    pub free_bytes: u64,
    /// `M_safety` applied.
    pub safety_bytes: u64,
    /// `M_kv_total` reserved for the KV cache across all layers.
    pub kv_total_bytes: u64,
    /// Bytes held permanently by the Pinned Zone (embedding, LM head, norms).
    /// Zero when profiling from a config estimate rather than a real catalog.
    pub pinned_bytes: u64,
    /// `M_layer_weight` — size of one streamed block.
    pub per_layer_weight_bytes: u64,
    /// `M_free − M_safety − M_kv_total − pinned`, saturated at 0.
    pub usable_bytes: u64,
    /// Final `LayersToLoad`, clamped to `[1, num_layers]`.
    pub layers_to_load: u32,
    /// The model's total layer count, for context.
    pub num_layers: u32,
}

impl VramPlan {
    /// Fraction of the model resident in VRAM at once (0.0–1.0).
    pub fn resident_fraction(&self) -> f64 {
        if self.num_layers == 0 {
            0.0
        } else {
            self.layers_to_load as f64 / self.num_layers as f64
        }
    }

    /// How many streaming passes one full forward pass requires.
    pub fn stream_passes(&self) -> u32 {
        if self.layers_to_load == 0 {
            0
        } else {
            self.num_layers.div_ceil(self.layers_to_load)
        }
    }
}

impl VramProfiler {
    /// Construct a profiler for a given context length using the default safety
    /// margin.
    pub fn new(target_context: u32) -> Self {
        Self {
            target_context,
            ..Self::default()
        }
    }

    /// Override the safety cushion (builder-style).
    pub fn with_safety_margin_bytes(mut self, bytes: u64) -> Self {
        self.safety_margin_bytes = bytes;
        self
    }

    /// KV-cache bytes for a **single** layer across the full target context:
    /// `2 (K,V) × N_kv_heads × D_head × 2 bytes × L_context`.
    pub fn kv_bytes_per_layer(&self, config: &ModelConfig) -> u64 {
        let per_token = 2 * config.num_kv_heads as u64
            * config.head_dim() as u64
            * KV_BYTES_PER_ELEMENT;
        per_token * self.target_context as u64
    }

    /// Total KV-cache footprint (`M_kv_total`) — per-layer KV summed over every
    /// layer, since all layers' histories stay resident while weights stream.
    pub fn kv_total_bytes(&self, config: &ModelConfig) -> u64 {
        self.kv_bytes_per_layer(config) * config.num_layers as u64
    }

    /// Average bytes of one streamed transformer block (`M_layer_weight`) — the
    /// unit that stays resident per layer in the streaming window.
    ///
    /// For an MoE model this is the layer **core** (attention + router + shared
    /// expert + norms), *not* the whole layer: the routed experts stream into a
    /// separate per-`(layer, expert)` cache budgeted apart, so planning the window
    /// as if each unit held all its experts would size it many times too small.
    pub fn per_layer_weight_bytes(&self, config: &ModelConfig) -> u64 {
        if config.is_moe() {
            return (config.resident_layer_params() as f64 * config.quant.bytes_per_param())
                .ceil() as u64;
        }
        let model_total_bytes =
            config.estimated_total_params() as f64 * config.quant.bytes_per_param();
        (model_total_bytes / config.num_layers as f64).ceil() as u64
    }

    /// Pure profiling step: compute the plan from an explicit `free_bytes`,
    /// using the parameter-count *estimate* for per-layer weight and no pinned
    /// overhead. Side-effect-free so it can be unit-tested without a GPU.
    pub fn plan_with_free(&self, config: &ModelConfig, free_bytes: u64) -> VramPlan {
        self.compute_plan(
            free_bytes,
            self.kv_total_bytes(config),
            self.per_layer_weight_bytes(config),
            0,
            config.num_layers,
        )
    }

    /// Profiling step using **measured** sizes from a [`LayerCatalog`]: the
    /// real largest-block weight for `M_layer_weight`, and the Pinned Zone's
    /// actual byte cost subtracted from free VRAM. Falls back to the config
    /// estimate for any figure the catalog can't supply (e.g. empty catalog).
    /// `native` is the checkpoint's own weight scheme. It is *not* always
    /// `config.quant`: with `--quant int4` against a bf16 file, the catalog
    /// measures the 16-bit bytes on disk while the layer that lands in VRAM is a
    /// quarter of that, having been quantized at load. Planning from the on-disk
    /// size would then leave 4x the VRAM unused.
    pub fn plan_from_catalog(
        &self,
        config: &ModelConfig,
        catalog: &LayerCatalog,
        native: QuantScheme,
        free_bytes: u64,
    ) -> VramPlan {
        let per_layer = if config.is_moe() {
            // The catalog measures a layer's *full* bytes (all experts inline), but
            // only the core stays resident on the MoE streaming path — experts
            // stream into their own budgeted cache. Use the core geometry estimate,
            // which the catalog can't separate out.
            self.per_layer_weight_bytes(config)
        } else {
            match catalog.max_layer_bytes() {
                0 => self.per_layer_weight_bytes(config),
                // Scale the measured on-disk bytes to the precision the engine will
                // actually hold the layer in.
                measured => {
                    let ratio = config.quant.bytes_per_param() / native.bytes_per_param();
                    (measured as f64 * ratio).ceil() as u64
                }
            }
        };
        let num_layers = if catalog.num_layers() == 0 {
            config.num_layers
        } else {
            catalog.num_layers()
        };
        self.compute_plan(
            free_bytes,
            self.kv_total_bytes(config),
            per_layer,
            catalog.pinned_bytes(),
            num_layers,
        )
    }

    /// Shared budget arithmetic behind both planning entry points.
    fn compute_plan(
        &self,
        free_bytes: u64,
        kv_total: u64,
        per_layer_weight: u64,
        pinned_bytes: u64,
        num_layers: u32,
    ) -> VramPlan {
        let per_layer_weight = per_layer_weight.max(1);
        let usable = free_bytes
            .saturating_sub(self.safety_margin_bytes)
            .saturating_sub(kv_total)
            .saturating_sub(pinned_bytes);

        let raw = usable / per_layer_weight;
        // Streaming requires at least one layer slot resident; cap at the model
        // size. Matches `specs.md` §3.1 `max(1, min(calc, num_layers))`.
        let layers_to_load = raw.clamp(1, num_layers as u64) as u32;

        VramPlan {
            free_bytes,
            safety_bytes: self.safety_margin_bytes,
            kv_total_bytes: kv_total,
            pinned_bytes,
            per_layer_weight_bytes: per_layer_weight,
            usable_bytes: usable,
            layers_to_load,
            num_layers,
        }
    }

    /// Profile against the live device, querying `M_free` via `cudaMemGetInfo`.
    /// Errors with [`crate::error::DlmError::GpuUnavailable`] on a non-GPU
    /// build — use [`plan_with_free`](Self::plan_with_free) for off-GPU planning.
    pub fn profile(&self, config: &ModelConfig) -> Result<VramPlan> {
        let dev = gpu::mem_get_info()?;
        Ok(self.plan_with_free(config, dev.free))
    }
}
