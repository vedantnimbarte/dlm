//! Tensor-name classification.
//!
//! Maps a checkpoint tensor name onto its role in the memory topography of
//! `specs.md` §2. The Pinned Zone (§2.1) holds the embedding, LM head, and
//! final norm permanently; the Streaming Zone (§2.2) cycles the per-layer
//! transformer blocks. To size either, we first have to know which tensor is
//! which — done here by structural name inspection, no model download required.

/// The role a tensor plays in the VRAM layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    /// Token embedding matrix — pinned in VRAM (§2.1).
    Embedding,
    /// Output projection / LM head — pinned (§2.1).
    LmHead,
    /// Final pre-head normalization — pinned (§2.1).
    FinalNorm,
    /// A weight belonging to transformer block `index` — streamed (§2.2).
    Layer(u32),
    /// Anything not recognized (rotary buffers, misc). Treated as pinned
    /// overhead so the budget never under-counts resident bytes.
    Other,
}

impl TensorRole {
    /// True if this tensor stays resident in VRAM for the whole run.
    pub fn is_pinned(self) -> bool {
        !matches!(self, TensorRole::Layer(_))
    }
}

/// Classify a tensor by its name.
///
/// Recognizes the common HuggingFace conventions used by Llama/Mistral/GPT-style
/// checkpoints: `model.layers.{i}.*`, `transformer.h.{i}.*`, `*.blocks.{i}.*`
/// for blocks, plus `embed_tokens`/`wte`, `lm_head`, and the top-level norm.
pub fn classify(name: &str) -> TensorRole {
    if let Some(index) = layer_index(name) {
        return TensorRole::Layer(index);
    }
    if name.contains("embed_tokens") || name.contains("wte") || name.contains("embeddings") {
        return TensorRole::Embedding;
    }
    if name.contains("lm_head") {
        return TensorRole::LmHead;
    }
    // Top-level norm (not inside a block, since blocks were caught above).
    if name.contains("ln_f") || name.ends_with("norm.weight") || name.ends_with("norm.bias") {
        return TensorRole::FinalNorm;
    }
    TensorRole::Other
}

/// Extract the block index from a name if it sits inside a transformer block,
/// i.e. a `layers`/`h`/`blocks` segment immediately followed by an integer.
fn layer_index(name: &str) -> Option<u32> {
    let mut segments = name.split('.').peekable();
    while let Some(seg) = segments.next() {
        if matches!(seg, "layers" | "h" | "blocks" | "block") {
            if let Some(next) = segments.peek() {
                if let Ok(idx) = next.parse::<u32>() {
                    return Some(idx);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_llama_style_names() {
        assert_eq!(classify("model.layers.0.self_attn.q_proj.weight"), TensorRole::Layer(0));
        assert_eq!(classify("model.layers.79.mlp.down_proj.weight"), TensorRole::Layer(79));
        assert_eq!(classify("model.embed_tokens.weight"), TensorRole::Embedding);
        assert_eq!(classify("lm_head.weight"), TensorRole::LmHead);
        assert_eq!(classify("model.norm.weight"), TensorRole::FinalNorm);
    }

    #[test]
    fn classifies_gpt_style_names() {
        assert_eq!(classify("transformer.h.11.attn.c_attn.weight"), TensorRole::Layer(11));
        assert_eq!(classify("transformer.wte.weight"), TensorRole::Embedding);
        assert_eq!(classify("transformer.ln_f.weight"), TensorRole::FinalNorm);
    }

    #[test]
    fn pinned_roles_are_pinned() {
        assert!(TensorRole::Embedding.is_pinned());
        assert!(TensorRole::LmHead.is_pinned());
        assert!(TensorRole::FinalNorm.is_pinned());
        assert!(TensorRole::Other.is_pinned());
        assert!(!TensorRole::Layer(3).is_pinned());
    }
}
