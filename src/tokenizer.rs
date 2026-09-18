//! Byte-level BPE tokenizer (GPT-2 / RoBERTa style).
//!
//! Turns text into the token ids the model consumes and back. It is *byte-level*
//! — every input byte is first mapped to a printable Unicode "byte char" via the
//! GPT-2 reversible mapping, so any UTF-8 text (or arbitrary bytes) tokenizes and
//! round-trips losslessly with no unknown token. BPE merges are then applied by
//! rank within each pre-tokenized chunk.
//!
//! Two ways to build one:
//! * [`BpeTokenizer::from_dir`] / [`from_files`](BpeTokenizer::from_files) — load
//!   a real vocabulary (`vocab.json` + `merges.txt`, the classic GPT-2 pair).
//! * [`BpeTokenizer::bytes_only`] — a trivial 256-token byte tokenizer (no
//!   merges), handy as a fallback and for testing the pipeline with no vocab.

use crate::error::{DlmError, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// A segment of input text: either a matched special token or ordinary text.
enum Seg {
    Special(u32),
    Text(String),
}

// ── HuggingFace `tokenizer.json` shape (only the BPE fields we use). ──────────

#[derive(Deserialize)]
struct HfTokenizer {
    #[serde(default)]
    added_tokens: Vec<HfAddedToken>,
    model: HfModel,
    /// Normalizer / pre-tokenizer graphs, kept raw. dlm reads them only to answer
    /// one question: does this tokenizer escape whitespace the SentencePiece way
    /// (space → ▁)? A `"type": "BPE"` model can be either byte-level (GPT-2,
    /// Qwen, Llama-3) or SentencePiece-style (Gemma, Mistral, Llama-2), and the
    /// model block alone cannot tell them apart — the answer lives out here.
    #[serde(default)]
    normalizer: serde_json::Value,
    #[serde(default)]
    pre_tokenizer: serde_json::Value,
}

#[derive(Deserialize)]
struct HfAddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(Deserialize)]
struct HfModel {
    /// "BPE" or "Unigram"; absent on older files (then inferred from shape).
    #[serde(default, rename = "type")]
    model_type: Option<String>,
    /// BPE: object `{piece: id}`. Unigram: array `[[piece, score], ...]`. Parsed
    /// per model type, so it is kept as a raw value until then.
    #[serde(default)]
    vocab: serde_json::Value,
    #[serde(default)]
    merges: Vec<HfMerge>,
    /// Unigram unknown-token id.
    #[serde(default)]
    unk_id: Option<u32>,
    /// Unigram: decompose unmatched chars into `<0xNN>` byte pieces.
    #[serde(default)]
    byte_fallback: bool,
}

/// Merges are `"a b"` in older files, `["a","b"]` in newer ones.
#[derive(Deserialize)]
#[serde(untagged)]
enum HfMerge {
    Str(String),
    Pair([String; 2]),
}

/// Build the GPT-2 reversible byte↔char mapping.
///
/// Printable byte ranges map to themselves; the rest map to code points starting
/// at 256, so all 256 bytes become distinct printable chars (space → 'Ġ').
fn byte_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u32> = Vec::new();
    bs.extend(b'!' as u32..=b'~' as u32);
    bs.extend(0xA1..=0xAC);
    bs.extend(0xAE..=0xFF);

    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut encoder = ['\0'; 256];
    let mut decoder = HashMap::new();
    for (&b, &c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(c).expect("valid code point");
        encoder[b as usize] = ch;
        decoder.insert(ch, b as u8);
    }
    (encoder, decoder)
}

/// A byte-level BPE tokenizer.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Token string → id.
    encoder: HashMap<String, u32>,
    /// Id → token string.
    decoder: HashMap<u32, String>,
    /// Merge rule `(a, b)` → rank (lower merges first).
    merges: HashMap<(String, String), u32>,
    /// Byte → printable char.
    byte_encoder: [char; 256],
    /// Printable char → byte.
    byte_decoder: HashMap<char, u8>,
    /// Special tokens (e.g. `<|eot_id|>`): literal string → id. These match as
    /// whole units before BPE and decode back to their literal text.
    special_encoder: HashMap<String, u32>,
    /// Id → special-token literal.
    special_decoder: HashMap<u32, String>,
    /// SentencePiece **Unigram** mode (Gemma, Llama-spm): when set, encode/decode
    /// use Viterbi segmentation over a scored vocabulary instead of BPE merges.
    /// `None` keeps the byte-level BPE path unchanged.
    unigram: Option<UnigramState>,
    /// SentencePiece-style **BPE** mode (Gemma, Mistral, Llama-2): merges run over
    /// literal characters with ▁ for spaces, not over GPT-2 byte chars. `None`
    /// keeps the byte-level BPE path unchanged.
    spm: Option<SpmBpe>,
    /// Beginning-of-sequence id to prepend when encoding, from the checkpoint's
    /// `add_bos_token`. Gemma and Llama/Mistral are trained with it always
    /// present and degenerate without it; Qwen sets `add_bos_token: false` and so
    /// leaves this `None`.
    bos_id: Option<u32>,
    /// How byte-level BPE splits text before merging (the tokenizer's regex).
    split: ByteLevelSplit,
}

/// Knobs for SentencePiece-style BPE, read from the tokenizer's normalizer and
/// pre-tokenizer.
#[derive(Debug, Clone)]
struct SpmBpe {
    /// Prepend one ▁ before encoding, and strip the matching leading space when
    /// decoding. Set by HF's `Metaspace{prepend_scheme}` (Mistral, Llama-2) and
    /// by a `Prepend` normalizer; Gemma has neither and so keeps this false.
    prepend: bool,
    /// Decompose an out-of-vocabulary character into `<0xNN>` byte pieces.
    byte_fallback: bool,
}

/// State for the SentencePiece Unigram model: per-piece log-prob scores plus the
/// knobs its Viterbi segmentation needs.
#[derive(Debug, Clone)]
struct UnigramState {
    /// Piece string → log-probability score (higher = more likely).
    scores: HashMap<String, f32>,
    /// Longest piece length in Unicode chars (bounds the Viterbi inner loop).
    max_piece_len: usize,
    /// Decompose an unmatched character into `<0xNN>` byte pieces (Gemma/Llama).
    byte_fallback: bool,
    /// Unknown-token id, used when a char matches no piece and byte-fallback is
    /// off or incomplete.
    unk_id: Option<u32>,
}

/// SentencePiece whitespace marker (▁, U+2581): spaces become this before Viterbi.
const SPM_SPACE: char = '\u{2581}';

impl BpeTokenizer {
    /// Build from an explicit vocabulary and an ordered merge list (rank = index).
    pub fn new(encoder: HashMap<String, u32>, merges_list: Vec<(String, String)>) -> Self {
        let decoder = encoder.iter().map(|(k, &v)| (v, k.clone())).collect();
        let merges = merges_list
            .into_iter()
            .enumerate()
            .map(|(rank, pair)| (pair, rank as u32))
            .collect();
        let (byte_encoder, byte_decoder) = byte_to_unicode();
        Self {
            encoder,
            decoder,
            merges,
            byte_encoder,
            byte_decoder,
            special_encoder: HashMap::new(),
            special_decoder: HashMap::new(),
            unigram: None,
            spm: None,
            bos_id: None,
            split: ByteLevelSplit::Gpt2,
        }
    }

    /// Register special tokens (literal string → id), matched as whole units
    /// before BPE and decoded back verbatim. Consumes and returns `self` for
    /// chaining.
    pub fn with_special(mut self, specials: impl IntoIterator<Item = (String, u32)>) -> Self {
        for (s, id) in specials {
            self.special_decoder.insert(id, s.clone());
            self.special_encoder.insert(s, id);
        }
        self
    }

    /// Switch this BPE tokenizer to SentencePiece-style merging (▁ for spaces,
    /// literal-character symbols) instead of GPT-2 byte-level merging.
    fn with_spm(mut self, spm: SpmBpe) -> Self {
        self.spm = Some(spm);
        self
    }

    /// Prepend `id` to every [`encode`](Self::encode), as the checkpoint's
    /// `add_bos_token` asks. Consumes and returns `self` for chaining.
    pub fn with_bos(mut self, id: Option<u32>) -> Self {
        self.bos_id = id;
        self
    }

    /// The beginning-of-sequence id this tokenizer prepends, if any.
    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    /// A trivial byte tokenizer: 256 tokens (one per byte), no merges. Every text
    /// round-trips; ids are just the raw bytes.
    pub fn bytes_only() -> Self {
        let (byte_encoder, byte_decoder) = byte_to_unicode();
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        for b in 0..256u32 {
            let s = byte_encoder[b as usize].to_string();
            encoder.insert(s.clone(), b);
            decoder.insert(b, s);
        }
        Self {
            encoder,
            decoder,
            merges: HashMap::new(),
            byte_encoder,
            byte_decoder,
            special_encoder: HashMap::new(),
            special_decoder: HashMap::new(),
            unigram: None,
            spm: None,
            bos_id: None,
            split: ByteLevelSplit::Gpt2,
        }
    }

    /// Build a SentencePiece **Unigram** tokenizer from a scored vocabulary
    /// (`pieces[i]` has id `i`), as shipped in a `tokenizer.json` Unigram model.
    /// `byte_fallback` decomposes unmatched characters into `<0xNN>` byte pieces
    /// (Gemma/Llama); `unk_id` is the last resort.
    pub fn from_unigram(
        pieces: Vec<(String, f32)>,
        byte_fallback: bool,
        unk_id: Option<u32>,
    ) -> Self {
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        let mut scores = HashMap::new();
        let mut max_piece_len = 1;
        for (id, (piece, score)) in pieces.into_iter().enumerate() {
            let id = id as u32;
            encoder.insert(piece.clone(), id);
            decoder.insert(id, piece.clone());
            max_piece_len = max_piece_len.max(piece.chars().count());
            scores.insert(piece, score);
        }
        let (byte_encoder, byte_decoder) = byte_to_unicode();
        Self {
            encoder,
            decoder,
            merges: HashMap::new(),
            byte_encoder,
            byte_decoder,
            special_encoder: HashMap::new(),
            special_decoder: HashMap::new(),
            unigram: Some(UnigramState {
                scores,
                max_piece_len,
                byte_fallback,
                unk_id,
            }),
            spm: None,
            bos_id: None,
            split: ByteLevelSplit::Gpt2,
        }
    }

    /// Load a `vocab.json` + `merges.txt` pair.
    pub fn from_files(vocab_path: &Path, merges_path: &Path) -> Result<Self> {
        let vocab_bytes = std::fs::read(vocab_path).map_err(|source| DlmError::Io {
            path: vocab_path.to_path_buf(),
            source,
        })?;
        let encoder: HashMap<String, u32> =
            serde_json::from_slice(&vocab_bytes).map_err(|source| DlmError::Json {
                context: "vocab.json".to_string(),
                source,
            })?;

        let merges_text = std::fs::read_to_string(merges_path).map_err(|source| DlmError::Io {
            path: merges_path.to_path_buf(),
            source,
        })?;
        let merges_list: Vec<(String, String)> = merges_text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .filter_map(|l| {
                let mut it = l.split_whitespace();
                Some((it.next()?.to_string(), it.next()?.to_string()))
            })
            .collect();

        Ok(Self::new(encoder, merges_list))
    }

    /// Load a HuggingFace `tokenizer.json` (the single-file "fast tokenizer"
    /// format modern models ship). Reads the BPE `model.vocab` + `model.merges`
    /// and registers `added_tokens` marked `special` (so chat-template control
    /// tokens like `<|eot_id|>` encode to their own id). Only BPE-model
    /// tokenizers are supported — SentencePiece/Unigram checkpoints are not.
    pub fn from_hf_json(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|source| DlmError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let hf: HfTokenizer = serde_json::from_slice(&bytes).map_err(|source| DlmError::Json {
            context: "tokenizer.json".to_string(),
            source,
        })?;
        let specials: Vec<(String, u32)> = hf
            .added_tokens
            .into_iter()
            .filter(|t| t.special)
            .map(|t| (t.content, t.id))
            .collect();

        // SentencePiece Unigram: `type: "Unigram"`, or a scored-array vocab.
        if hf.model.model_type.as_deref() == Some("Unigram") || hf.model.vocab.is_array() {
            let pieces: Vec<(String, f32)> =
                serde_json::from_value(hf.model.vocab).map_err(|source| DlmError::Json {
                    context: "tokenizer.json Unigram vocab".to_string(),
                    source,
                })?;
            return Ok(
                Self::from_unigram(pieces, hf.model.byte_fallback, hf.model.unk_id)
                    .with_special(specials),
            );
        }

        // BPE — byte-level (GPT-2/Qwen/Llama-3) or SentencePiece-style
        // (Gemma/Mistral/Llama-2), decided by the normalizer + pre-tokenizer.
        let vocab: HashMap<String, u32> =
            serde_json::from_value(hf.model.vocab).map_err(|source| DlmError::Json {
                context: "tokenizer.json BPE vocab".to_string(),
                source,
            })?;
        let merges_list = hf
            .model
            .merges
            .into_iter()
            .filter_map(|m| match m {
                HfMerge::Pair([a, b]) => Some((a, b)),
                HfMerge::Str(s) => {
                    let mut it = s.split_whitespace();
                    Some((it.next()?.to_string(), it.next()?.to_string()))
                }
            })
            .collect();
        let mut tok = Self::new(vocab, merges_list).with_special(specials);
        tok.split = detect_byte_level_split(&hf.pre_tokenizer);
        match detect_spm(&hf.normalizer, &hf.pre_tokenizer, hf.model.byte_fallback) {
            Some(spm) => Ok(tok.with_spm(spm)),
            None => Ok(tok),
        }
    }

    /// The tokenizer a GGUF file carries in its metadata.
    ///
    /// GGUF holds the whole tokenizer where a Hugging Face repo keeps
    /// `tokenizer.json`: the vocabulary as an ordered list (a token's index *is*
    /// its id), the BPE merges as "a b" lines, and a type per token that marks
    /// the control and user-defined ones. Reading it is what makes a downloaded
    /// `.gguf` self-contained — without it the file's ids mean nothing.
    pub fn from_gguf(
        metadata: &std::collections::BTreeMap<String, crate::storage::gguf::Value>,
    ) -> Result<Self> {
        use crate::storage::gguf::Value;
        let get = |key: &str| metadata.get(&format!("tokenizer.ggml.{key}"));
        let kind = get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| DlmError::InvalidConfig("gguf: no tokenizer.ggml.model".into()))?;
        let tokens = get("tokens")
            .and_then(Value::as_strings)
            .ok_or_else(|| DlmError::InvalidConfig("gguf: no tokenizer.ggml.tokens".into()))?;
        // Token types: 3 is a control token and 4 a user-defined one. Both must be
        // matched whole rather than merged into, which is what `with_special`
        // does; the rest are ordinary pieces.
        let types = get("token_type").and_then(Value::as_numbers);
        let is_special =
            |id: usize| types.is_some_and(|t| matches!(t.get(id).copied(), Some(3.0) | Some(4.0)));
        let specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(id, _)| is_special(*id))
            .map(|(id, t)| (t.clone(), id as u32))
            .collect();

        let mut tok = match kind {
            // Byte-level BPE (GPT-2, Llama 3, Qwen): the vocabulary is already in
            // the byte-level alphabet, and the merges are the same pairs a
            // `tokenizer.json` lists.
            "gpt2" => {
                let merges = get("merges").and_then(Value::as_strings).ok_or_else(|| {
                    DlmError::InvalidConfig(
                        "gguf: a gpt2-style tokenizer with no tokenizer.ggml.merges".into(),
                    )
                })?;
                let vocab: HashMap<String, u32> = tokens
                    .iter()
                    .enumerate()
                    .map(|(id, t)| (t.clone(), id as u32))
                    .collect();
                let merges_list: Vec<(String, String)> = merges
                    .iter()
                    .filter_map(|m| {
                        let mut it = m.split_whitespace();
                        Some((it.next()?.to_string(), it.next()?.to_string()))
                    })
                    .collect();
                let mut tok = Self::new(vocab, merges_list).with_special(specials);
                // Which regex the model splits with, by the name llama.cpp records
                // for it. An unknown one gets GPT-2's, the same fallback
                // `tokenizer.json` gets.
                tok.split = match get("pre").and_then(Value::as_str).unwrap_or("default") {
                    "qwen2" => ByteLevelSplit::Qwen2,
                    "llama-bpe" | "llama3" | "llama-v3" => ByteLevelSplit::Llama3,
                    _ => ByteLevelSplit::Gpt2,
                };
                tok
            }
            // SentencePiece (Llama 2, Mistral, Gemma): pieces scored against each
            // other, with ▁ for a space and byte tokens for anything unknown.
            "llama" => {
                let scores = get("scores").and_then(Value::as_numbers).ok_or_else(|| {
                    DlmError::InvalidConfig(
                        "gguf: a llama-style tokenizer with no tokenizer.ggml.scores".into(),
                    )
                })?;
                let byte_fallback = types.is_some_and(|t| t.contains(&6.0));
                let pieces: Vec<(String, f32)> = tokens
                    .iter()
                    .enumerate()
                    .map(|(id, t)| (t.clone(), scores.get(id).copied().unwrap_or(0.0) as f32))
                    .collect();
                let unk = get("unknown_token_id")
                    .and_then(Value::as_u64)
                    .map(|v| v as u32);
                Self::from_unigram(pieces, byte_fallback, unk).with_special(specials)
            }
            other => {
                return Err(DlmError::InvalidConfig(format!(
                    "gguf: tokenizer {other:?} is not one dlm implements. It reads the \
                     byte-level BPE (\"gpt2\") and SentencePiece (\"llama\") tokenizers."
                )))
            }
        };
        // A BOS token is prepended only when the file says to, which is how
        // llama.cpp decides: Gemma and Llama add one, Qwen does not.
        if get("add_bos_token").and_then(Value::as_u64).unwrap_or(0) != 0 {
            tok = tok.with_bos(
                get("bos_token_id")
                    .and_then(Value::as_u64)
                    .map(|v| v as u32),
            );
        }
        Ok(tok)
    }

    /// The tokenizer inside a `.gguf` file at `path`.
    pub fn from_gguf_path(path: &Path) -> Result<Self> {
        let shard = crate::storage::MmapShard::open(path)?;
        let metadata = shard.gguf_metadata().ok_or_else(|| {
            DlmError::InvalidConfig(format!("{} is not a GGUF file", path.display()))
        })?;
        Self::from_gguf(metadata)
    }

    /// Load a tokenizer from a model directory: prefer HF `tokenizer.json`, else
    /// fall back to the classic `vocab.json` + `merges.txt` pair.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        // `--tokenizer` naming a .gguf file: it holds one, and a user pointing at
        // the model they are running should get it.
        if dir.extension().and_then(|e| e.to_str()) == Some("gguf") {
            return Self::from_gguf_path(dir);
        }
        let hf = dir.join("tokenizer.json");
        let tok = if hf.exists() {
            Self::from_hf_json(&hf)?
        } else {
            Self::from_files(&dir.join("vocab.json"), &dir.join("merges.txt"))?
        };
        let bos = read_bos_config(&dir.join("tokenizer_config.json"), &tok);
        Ok(tok.with_bos(bos))
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// The id of a literal vocabulary piece, if the checkpoint has one.
    ///
    /// Lets a test state its expectation as the *pieces* a tokenizer should
    /// produce (`▁capital`) and resolve them against the checkpoint's own
    /// vocabulary, instead of hard-coding ids that came out of this encoder —
    /// which would only prove the encoder agrees with itself.
    pub fn id_of(&self, piece: &str) -> Option<u32> {
        self.encoder
            .get(piece)
            .or_else(|| self.special_encoder.get(piece))
            .copied()
    }

    /// Encode text into token ids. Registered special tokens are matched as whole
    /// units (longest-match) and emit their own id; the text between is BPE'd.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        // BOS goes on unless the caller already supplied it — a chat template that
        // spells `<bos>` literally must not end up with two.
        if let Some(bos) = self.bos_id {
            let already = self
                .special_decoder
                .get(&bos)
                .is_some_and(|lit| text.starts_with(lit.as_str()));
            if !already {
                ids.push(bos);
            }
        }
        for seg in self.split_special(text) {
            match seg {
                Seg::Special(id) => ids.push(id),
                Seg::Text(chunk_text) => match (&self.unigram, &self.spm) {
                    // SentencePiece Unigram: Viterbi over the whole text segment.
                    (Some(u), _) => self.encode_unigram(u, &chunk_text, &mut ids)?,
                    // SentencePiece-style BPE: merge over ▁-escaped characters.
                    (None, Some(spm)) => self.encode_spm_bpe(spm, &chunk_text, &mut ids)?,
                    // Byte-level BPE: pre-tokenize into chunks, merge each.
                    (None, None) => {
                        for chunk in pretokenize(&chunk_text, self.split) {
                            for symbol in self.bpe(chunk.as_bytes()) {
                                let id = self.encoder.get(&symbol).ok_or_else(|| {
                                    DlmError::Tokenizer(format!(
                                        "token {symbol:?} not in vocabulary"
                                    ))
                                })?;
                                ids.push(*id);
                            }
                        }
                    }
                },
            }
        }
        Ok(ids)
    }

    /// SentencePiece-style **BPE** encode: escape spaces to ▁ (optionally
    /// prefixing one), then merge over literal characters rather than GPT-2 byte
    /// chars, because the vocabulary is literal text — `▁capital`, not `Ġcapital`.
    ///
    /// A symbol left out of vocabulary after merging is decomposed into `<0xNN>`
    /// byte pieces when the model declares `byte_fallback`.
    fn encode_spm_bpe(&self, spm: &SpmBpe, text: &str, out: &mut Vec<u32>) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let mut norm = String::with_capacity(text.len() + 3);
        if spm.prepend {
            norm.push(SPM_SPACE);
        }
        for ch in text.chars() {
            norm.push(if ch == ' ' { SPM_SPACE } else { ch });
        }

        for chunk in pretokenize_spm(&norm) {
            let symbols: Vec<String> = chunk.chars().map(|c| c.to_string()).collect();
            for symbol in self.merge_symbols(symbols) {
                match self.encoder.get(&symbol) {
                    Some(&id) => out.push(id),
                    None if spm.byte_fallback => self.push_byte_pieces(&symbol, out)?,
                    None => {
                        return Err(DlmError::Tokenizer(format!(
                            "token {symbol:?} not in vocabulary"
                        )))
                    }
                }
            }
        }
        Ok(())
    }

    /// Append the `<0xNN>` byte pieces spelling `symbol` (SentencePiece byte
    /// fallback). Errors if the vocabulary is missing one of them.
    fn push_byte_pieces(&self, symbol: &str, out: &mut Vec<u32>) -> Result<()> {
        for b in symbol.as_bytes() {
            let piece = format!("<0x{b:02X}>");
            let id = self.encoder.get(&piece).ok_or_else(|| {
                DlmError::Tokenizer(format!(
                    "token {symbol:?} not in vocabulary and byte-fallback piece {piece:?} is missing"
                ))
            })?;
            out.push(*id);
        }
        Ok(())
    }

    /// SentencePiece Unigram encode: escape whitespace (space → ▁) with a leading
    /// dummy prefix, then Viterbi-segment the text to maximize the summed piece
    /// score. An unmatched character falls back to `<0xNN>` byte pieces
    /// (`byte_fallback`) or the unknown token. Appends ids to `out`.
    fn encode_unigram(&self, u: &UnigramState, text: &str, out: &mut Vec<u32>) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        // Normalize: spaces → ▁, and prepend one ▁ (add_dummy_prefix).
        let norm: Vec<char> = std::iter::once(SPM_SPACE)
            .chain(text.chars().map(|c| if c == ' ' { SPM_SPACE } else { c }))
            .collect();
        let n = norm.len();

        // Viterbi: best[i] = max summed score of a segmentation of norm[..i].
        // back[i] = (start, ids emitted for the piece ending at i).
        let neg = f32::NEG_INFINITY;
        let mut best = vec![neg; n + 1];
        best[0] = 0.0;
        let mut back: Vec<(usize, Vec<u32>)> = vec![(0, Vec::new()); n + 1];

        for i in 1..=n {
            // Multi-char vocab pieces ending at i.
            let lo = i.saturating_sub(u.max_piece_len);
            for j in lo..i {
                if best[j] == neg {
                    continue;
                }
                let piece: String = norm[j..i].iter().collect();
                if let (Some(&score), Some(&id)) = (u.scores.get(&piece), self.encoder.get(&piece))
                {
                    let cand = best[j] + score;
                    if cand > best[i] {
                        best[i] = cand;
                        back[i] = (j, vec![id]);
                    }
                }
            }
            // Single-character fallback (byte pieces or unk), so best[i] is always
            // reachable even for out-of-vocab characters. Heavily penalized so it
            // only wins when no real piece covers the character.
            if best[i - 1] != neg {
                let ch = norm[i - 1];
                if let Some(fallback) = self.unigram_char_fallback(u, ch) {
                    let cand = best[i - 1] - 10.0 + fallback.1; // penalty + byte scores
                    if cand > best[i] {
                        best[i] = cand;
                        back[i] = (i - 1, fallback.0);
                    }
                }
            }
        }

        if best[n] == neg {
            return Err(DlmError::Tokenizer(
                "unigram tokenizer could not segment input (no byte fallback or unk token)".into(),
            ));
        }
        // Reconstruct forward order.
        let mut pieces_rev: Vec<u32> = Vec::new();
        let mut i = n;
        while i > 0 {
            let (j, ref step_ids) = back[i];
            for &id in step_ids.iter().rev() {
                pieces_rev.push(id);
            }
            i = j;
        }
        pieces_rev.reverse();
        out.extend(pieces_rev);
        Ok(())
    }

    /// Ids covering a single unmatched character `ch`, plus their summed score:
    /// its `<0xNN>` byte pieces when `byte_fallback` and all are present, else the
    /// unk token. `None` if neither is available.
    fn unigram_char_fallback(&self, u: &UnigramState, ch: char) -> Option<(Vec<u32>, f32)> {
        if u.byte_fallback {
            let mut buf = [0u8; 4];
            let bytes = ch.encode_utf8(&mut buf).as_bytes();
            let mut ids = Vec::with_capacity(bytes.len());
            let mut score = 0.0;
            let mut ok = true;
            for &b in bytes {
                let piece = format!("<0x{b:02X}>");
                match (self.encoder.get(&piece), u.scores.get(&piece)) {
                    (Some(&id), Some(&s)) => {
                        ids.push(id);
                        score += s;
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                return Some((ids, score));
            }
        }
        u.unk_id.map(|id| (vec![id], 0.0))
    }

    /// Split `text` on registered special tokens (longest-match wins).
    fn split_special(&self, text: &str) -> Vec<Seg> {
        if self.special_encoder.is_empty() {
            return vec![Seg::Text(text.to_string())];
        }
        let mut out = Vec::new();
        let mut buf = String::new();
        let mut i = 0;
        while i < text.len() {
            let matched = if text.is_char_boundary(i) {
                self.special_encoder
                    .iter()
                    .filter(|(sp, _)| text[i..].starts_with(sp.as_str()))
                    .max_by_key(|(sp, _)| sp.len())
                    .map(|(sp, &id)| (sp.len(), id))
            } else {
                None
            };
            if let Some((len, id)) = matched {
                if !buf.is_empty() {
                    out.push(Seg::Text(std::mem::take(&mut buf)));
                }
                out.push(Seg::Special(id));
                i += len;
            } else {
                let ch = text[i..].chars().next().expect("valid char at boundary");
                buf.push(ch);
                i += ch.len_utf8();
            }
        }
        if !buf.is_empty() {
            out.push(Seg::Text(buf));
        }
        out
    }

    /// Decode token ids back into text (lossy on invalid UTF-8). Special-token
    /// ids render as their literal text; runs of byte tokens are byte-decoded.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        // Both SentencePiece modes decode the same way: pieces are literal text
        // with ▁ for spaces. Only Unigram always prepends at encode time, so only
        // it always strips the leading space back off.
        if self.unigram.is_some() {
            return self.decode_spm_pieces(ids, true);
        }
        if let Some(spm) = &self.spm {
            return self.decode_spm_pieces(ids, spm.prepend);
        }
        let mut result = String::new();
        let mut run = String::new();
        for &id in ids {
            if let Some(special) = self.special_decoder.get(&id) {
                self.flush_byte_run(&mut run, &mut result)?;
                result.push_str(special);
            } else {
                let tok = self
                    .decoder
                    .get(&id)
                    .ok_or_else(|| DlmError::Tokenizer(format!("unknown token id {id}")))?;
                run.push_str(tok);
            }
        }
        self.flush_byte_run(&mut run, &mut result)?;
        Ok(result)
    }

    /// SentencePiece decode (Unigram and SPM-style BPE alike): concatenate pieces
    /// (byte-fallback `<0xNN>` pieces reassemble into UTF-8), turn ▁ back into
    /// spaces, and — when the encoder prepended one — drop the leading space again.
    fn decode_spm_pieces(&self, ids: &[u32], strip_leading_space: bool) -> Result<String> {
        let mut pieces = String::new();
        let mut byte_run: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(sp) = self.special_decoder.get(&id) {
                flush_bytes(&mut byte_run, &mut pieces);
                pieces.push_str(sp);
                continue;
            }
            let piece = self
                .decoder
                .get(&id)
                .ok_or_else(|| DlmError::Tokenizer(format!("unknown token id {id}")))?;
            match parse_byte_piece(piece) {
                Some(b) => byte_run.push(b),
                None => {
                    flush_bytes(&mut byte_run, &mut pieces);
                    pieces.push_str(piece);
                }
            }
        }
        flush_bytes(&mut byte_run, &mut pieces);
        let text = pieces.replace(SPM_SPACE, " ");
        if strip_leading_space {
            return Ok(text.strip_prefix(' ').unwrap_or(&text).to_string());
        }
        Ok(text)
    }

    /// Byte-decode an accumulated run of byte-level tokens into `out`, clearing
    /// the run.
    fn flush_byte_run(&self, run: &mut String, out: &mut String) -> Result<()> {
        if run.is_empty() {
            return Ok(());
        }
        let mut bytes = Vec::with_capacity(run.len());
        for ch in run.chars() {
            let b = self
                .byte_decoder
                .get(&ch)
                .ok_or_else(|| DlmError::Tokenizer(format!("char {ch:?} is not a byte token")))?;
            bytes.push(*b);
        }
        out.push_str(&String::from_utf8_lossy(&bytes));
        run.clear();
        Ok(())
    }

    /// Apply BPE merges to one pre-tokenized chunk, returning its token strings.
    fn bpe(&self, chunk_bytes: &[u8]) -> Vec<String> {
        let symbols: Vec<String> = chunk_bytes
            .iter()
            .map(|&b| self.byte_encoder[b as usize].to_string())
            .collect();
        self.merge_symbols(symbols)
    }

    /// Apply BPE merges to an already-split symbol list, lowest rank first.
    ///
    /// Split out of [`bpe`] so the SentencePiece path can feed it literal
    /// characters while the byte-level path keeps feeding it byte chars — the
    /// merging itself is identical either way.
    fn merge_symbols(&self, mut symbols: Vec<String>) -> Vec<String> {
        while symbols.len() >= 2 {
            // Find the adjacent pair with the lowest merge rank.
            let mut best: Option<(usize, u32)> = None;
            for i in 0..symbols.len() - 1 {
                if let Some(&rank) = self
                    .merges
                    .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                {
                    if best.is_none_or(|(_, r)| rank < r) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((_, _)) = best else { break };
            let (a, b) = {
                let (i, _) = best.unwrap();
                (symbols[i].clone(), symbols[i + 1].clone())
            };

            // Merge every occurrence of that pair in one pass.
            let mut merged = Vec::with_capacity(symbols.len());
            let mut i = 0;
            while i < symbols.len() {
                if i + 1 < symbols.len() && symbols[i] == a && symbols[i + 1] == b {
                    merged.push(format!("{a}{b}"));
                    i += 2;
                } else {
                    merged.push(symbols[i].clone());
                    i += 1;
                }
            }
            symbols = merged;
        }
        symbols
    }
}

/// Flush accumulated byte-fallback bytes into `out` as UTF-8 (lossy), clearing
/// the run. Used by the SentencePiece Unigram decode path.
fn flush_bytes(byte_run: &mut Vec<u8>, out: &mut String) {
    if !byte_run.is_empty() {
        out.push_str(&String::from_utf8_lossy(byte_run));
        byte_run.clear();
    }
}

/// Parse a SentencePiece byte-fallback piece `<0xNN>` into its byte; `None` for
/// any ordinary piece.
fn parse_byte_piece(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() == 2 {
        u8::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

/// Resolve the BOS id to prepend, from `tokenizer_config.json`.
///
/// Returns `None` unless the checkpoint both sets `add_bos_token: true` and names
/// a `bos_token` that resolves to an id — so Qwen (`add_bos_token: false`) is
/// untouched while Gemma and Llama/Mistral get the token they were trained with.
/// A missing or malformed file is simply "no BOS", never an error: the tokenizer
/// itself already loaded, and refusing to run over an optional hint would be
/// worse than running without it.
fn read_bos_config(path: &Path, tok: &BpeTokenizer) -> Option<u32> {
    let bytes = std::fs::read(path).ok()?;
    let cfg: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    if cfg.get("add_bos_token")?.as_bool() != Some(true) {
        return None;
    }
    // `bos_token` is either a plain string or an AddedToken object.
    let bos = cfg.get("bos_token")?;
    let content = bos
        .as_str()
        .or_else(|| bos.get("content").and_then(|c| c.as_str()))?;
    tok.special_encoder
        .get(content)
        .or_else(|| tok.encoder.get(content))
        .copied()
}

/// Decide whether a `tokenizer.json` describes SentencePiece whitespace escaping,
/// by walking its normalizer and pre-tokenizer graphs.
///
/// The `model.type` is `"BPE"` for both conventions, so this is the only thing
/// that separates Gemma/Mistral/Llama-2 from Qwen/Llama-3. The two families
/// signal it in different places — Gemma with a `Replace(" " → ▁)` normalizer,
/// Mistral with a `Metaspace` pre-tokenizer and a *null* normalizer — so both
/// graphs have to be inspected, and neither alone is sufficient.
fn detect_spm(
    normalizer: &serde_json::Value,
    pre_tokenizer: &serde_json::Value,
    byte_fallback: bool,
) -> Option<SpmBpe> {
    let mut escapes = false;
    let mut prepend = false;
    walk_spm_nodes(normalizer, &mut escapes, &mut prepend);
    walk_spm_nodes(pre_tokenizer, &mut escapes, &mut prepend);
    escapes.then_some(SpmBpe {
        prepend,
        byte_fallback,
    })
}

/// Recurse through a normalizer/pre-tokenizer node (or a `Sequence` of them),
/// setting `escapes` when it rewrites spaces to ▁ and `prepend` when it also
/// prefixes one.
fn walk_spm_nodes(node: &serde_json::Value, escapes: &mut bool, prepend: &mut bool) {
    let Some(obj) = node.as_object() else { return };
    // A Sequence nests the real nodes under one of these keys.
    for key in ["normalizers", "pretokenizers"] {
        if let Some(list) = obj.get(key).and_then(|v| v.as_array()) {
            for child in list {
                walk_spm_nodes(child, escapes, prepend);
            }
        }
    }
    let spm_space = SPM_SPACE.to_string();
    match obj.get("type").and_then(|v| v.as_str()) {
        // Gemma: {"type":"Replace","pattern":{"String":" "},"content":"▁"}
        Some("Replace") => {
            let pattern = obj
                .get("pattern")
                .and_then(|p| p.get("String"))
                .and_then(|s| s.as_str());
            let content = obj.get("content").and_then(|c| c.as_str());
            if pattern == Some(" ") && content == Some(spm_space.as_str()) {
                *escapes = true;
            }
        }
        // Mistral / Llama-2: {"type":"Metaspace","replacement":"▁",
        //                     "prepend_scheme":"first"}. `replacement` defaults
        // to ▁, and the legacy spelling of the prefix flag is `add_prefix_space`.
        Some("Metaspace") => {
            let replacement = obj
                .get("replacement")
                .and_then(|r| r.as_str())
                .unwrap_or(spm_space.as_str());
            if replacement == spm_space {
                *escapes = true;
                let scheme = obj.get("prepend_scheme").and_then(|s| s.as_str());
                let legacy = obj.get("add_prefix_space").and_then(|b| b.as_bool());
                if matches!(scheme, Some("first") | Some("always")) || legacy == Some(true) {
                    *prepend = true;
                }
            }
        }
        // Llama-2: {"type":"Prepend","prepend":"▁"} ahead of the Replace.
        Some("Prepend")
            if obj.get("prepend").and_then(|p| p.as_str()) == Some(spm_space.as_str()) =>
        {
            *prepend = true;
        }
        _ => {}
    }
}

/// Split SentencePiece-normalized text so each chunk starts at a ▁ run.
///
/// Bounds the cost of merging (which is quadratic in chunk length) without
/// changing the result: the only vocabulary pieces that contain ▁ anywhere but
/// the front are *runs of ▁* (indentation), and a run is never split across
/// chunks here — so no reachable merge spans a chunk boundary.
fn pretokenize_spm(text: &str) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut prev_was_mark = false;
    for ch in text.chars() {
        let is_mark = ch == SPM_SPACE;
        // Break at the *start* of a ▁ run, so "a▁▁▁▁b" stays one chunk after "a".
        if is_mark && !prev_was_mark && !cur.is_empty() {
            chunks.push(std::mem::take(&mut cur));
        }
        cur.push(ch);
        prev_was_mark = is_mark;
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// Which regex a byte-level BPE tokenizer splits text with before merging.
///
/// Merges are learned only within the regex's pieces, so splitting any other
/// way yields tokens the model never saw in training. Splitting at spaces alone
/// turned a run of spaces (code indentation) into one token per space, where the
/// model's tokenizer makes one token of the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByteLevelSplit {
    /// GPT-2's built-in `ByteLevel` regex:
    /// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
    Gpt2,
    /// Qwen2, Qwen2.5 and Qwen3: [`QWEN2_PATTERN`].
    Qwen2,
    /// Llama 3: [`QWEN2_PATTERN`] with numbers in runs of up to three digits.
    Llama3,
}

const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const LLAMA3_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The split a `tokenizer.json` pre-tokenizer graph asks for: the Qwen2 or
/// Llama 3 pattern when a `Split` step carries it verbatim, else GPT-2's. Other
/// byte-level tokenizers (Falcon, DeepSeek) chain several splits of their own;
/// GPT-2's regex is the closest of the three to those.
fn detect_byte_level_split(pre: &serde_json::Value) -> ByteLevelSplit {
    fn patterns<'a>(v: &'a serde_json::Value, out: &mut Vec<&'a str>) {
        match v {
            serde_json::Value::Object(map) => {
                if let Some(p) = map
                    .get("pattern")
                    .and_then(|p| p.get("Regex"))
                    .and_then(|r| r.as_str())
                {
                    out.push(p);
                }
                for child in map.values() {
                    patterns(child, out);
                }
            }
            serde_json::Value::Array(items) => items.iter().for_each(|i| patterns(i, out)),
            _ => {}
        }
    }
    let mut found = Vec::new();
    patterns(pre, &mut found);
    if found.contains(&QWEN2_PATTERN) {
        ByteLevelSplit::Qwen2
    } else if found.contains(&LLAMA3_PATTERN) {
        ByteLevelSplit::Llama3
    } else {
        ByteLevelSplit::Gpt2
    }
}

/// Split `text` the way `split`'s regex does, leftmost alternative first, so
/// each piece is exactly what the model's tokenizer would merge within.
fn pretokenize(text: &str, split: ByteLevelSplit) -> Vec<&str> {
    use crate::unicode_class::{is_letter, is_number};
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let at = |i: usize| chars.get(i).map(|&(_, c)| c);
    let letter = |i: usize| at(i).is_some_and(is_letter);
    let number = |i: usize| at(i).is_some_and(is_number);
    let space = |i: usize| at(i).is_some_and(char::is_whitespace);
    let crlf = |i: usize| matches!(at(i), Some('\r' | '\n'));
    // Other: neither whitespace, letter nor number (`[^\s\p{L}\p{N}]`).
    let other = |i: usize| i < n && !space(i) && !letter(i) && !number(i);
    let run = |mut i: usize, f: &dyn Fn(usize) -> bool| {
        while i < n && f(i) {
            i += 1;
        }
        i
    };
    let contraction = |i: usize, fold: bool| -> Option<usize> {
        if at(i) != Some('\'') {
            return None;
        }
        let low = |j: usize| at(j).map(|c| if fold { c.to_ascii_lowercase() } else { c });
        match (low(i + 1), low(i + 2)) {
            (Some('s' | 't' | 'm' | 'd'), _) => Some(i + 2),
            (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => Some(i + 3),
            _ => None,
        }
    };
    // `\s+(?!\S)` then `\s+`: a whitespace run, less its last character when a
    // non-space follows (that character starts the next piece instead).
    let whitespace = |i: usize| -> usize {
        let end = run(i, &space);
        if end < n && end - i >= 2 {
            end - 1
        } else {
            end
        }
    };

    let mut pieces = Vec::new();
    let mut i = 0;
    while i < n {
        let end = match split {
            ByteLevelSplit::Gpt2 => contraction(i, false)
                .or_else(|| {
                    let from = if at(i) == Some(' ') && i + 1 < n {
                        i + 1
                    } else {
                        i
                    };
                    [&letter as &dyn Fn(usize) -> bool, &number, &other]
                        .into_iter()
                        .find(|f| f(from))
                        .map(|f| run(from, f))
                })
                .unwrap_or_else(|| whitespace(i)),
            ByteLevelSplit::Qwen2 | ByteLevelSplit::Llama3 => {
                if let Some(end) = contraction(i, true) {
                    end
                } else if letter(i) {
                    run(i, &letter)
                } else if !crlf(i) && !number(i) && letter(i + 1) {
                    // `[^\r\n\p{L}\p{N}]?\p{L}+`: one leading non-letter.
                    run(i + 1, &letter)
                } else if number(i) {
                    let max = if split == ByteLevelSplit::Llama3 {
                        3
                    } else {
                        1
                    };
                    run(i, &number).min(i + max)
                } else if other(i) || (at(i) == Some(' ') && other(i + 1)) {
                    // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
                    let from = if other(i) { i } else { i + 1 };
                    run(run(from, &other), &crlf)
                } else {
                    // `\s*[\r\n]+`: through the last line break of the run.
                    let end = run(i, &space);
                    match (i..end).rev().find(|&j| crlf(j)) {
                        Some(last) => last + 1,
                        None => whitespace(i),
                    }
                }
            }
        };
        // Every character matches some alternative; stepping one is only a guard.
        let end = end.max(i + 1);
        let start = chars[i].0;
        let stop = chars.get(end).map_or(text.len(), |&(b, _)| b);
        pieces.push(&text[start..stop]);
        i = end;
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each flavor splits exactly as Hugging Face `tokenizers` pre-tokenizes the
    /// same text (the expected pieces are its output, byte-level decoded).
    #[test]
    fn byte_level_split_matches_the_tokenizers_regexes() {
        type Case = (&'static str, &'static [&'static str]);
        let shared: [Case; 7] = [
            ("a   b\n\n\tc", &["a", "  ", " b", "\n\n", "\tc"]),
            (
                "def f(x):\n    return x**2  # square\n",
                &[
                    "def", " f", "(x", "):\n", "   ", " return", " x", "**", "2", " ", " #",
                    " square", "\n",
                ],
            ),
            (
                "HELLO'S   world'll  ",
                &["HELLO", "'S", "  ", " world", "'ll", "  "],
            ),
            (
                "line1\r\nline2 \r\n\n",
                &["line", "1", "\r\n", "line", "2", " \r\n\n"],
            ),
            ("   leading", &["  ", " leading"]),
            ("नमस्ते दुनिया", &["नमस", "्त", "े", " द", "ुन", "िय", "ा"]),
            ("'hello 'LL ...\n", &["'hello", " '", "LL", " ...\n"]),
        ];
        for (text, want) in shared {
            for split in [ByteLevelSplit::Qwen2, ByteLevelSplit::Llama3] {
                assert_eq!(pretokenize(text, split), want, "{split:?} {text:?}");
            }
        }
        // Digits: one at a time for Qwen2, runs of up to three for Llama 3.
        let digits = "I don't THINK it's 12345!";
        assert_eq!(
            pretokenize(digits, ByteLevelSplit::Qwen2),
            ["I", " don", "'t", " THINK", " it", "'s", " ", "1", "2", "3", "4", "5", "!"]
        );
        assert_eq!(
            pretokenize(digits, ByteLevelSplit::Llama3),
            ["I", " don", "'t", " THINK", " it", "'s", " ", "123", "45", "!"]
        );

        let gpt2: [Case; 6] = [
            ("a   b\n\n\tc", &["a", "  ", " b", "\n\n", "\t", "c"]),
            (
                "def f(x):\n    return x**2  # square\n",
                &[
                    "def", " f", "(", "x", "):", "\n   ", " return", " x", "**", "2", " ", " #",
                    " square", "\n",
                ],
            ),
            (
                digits,
                &["I", " don", "'t", " THINK", " it", "'s", " 12345", "!"],
            ),
            (
                "HELLO'S   world'll  ",
                &["HELLO", "'", "S", "  ", " world", "'ll", "  "],
            ),
            (
                "line1\r\nline2 \r\n\n",
                &["line", "1", "\r", "\n", "line", "2", " \r\n\n"],
            ),
            (
                "'hello 'LL ...\n",
                &["'", "hello", " '", "LL", " ...", "\n"],
            ),
        ];
        for (text, want) in gpt2 {
            assert_eq!(
                pretokenize(text, ByteLevelSplit::Gpt2),
                want,
                "GPT-2 {text:?}"
            );
        }
    }

    #[test]
    fn byte_map_is_a_bijection() {
        let (enc, dec) = byte_to_unicode();
        // All 256 bytes map to distinct chars that map back.
        let distinct: std::collections::HashSet<char> = enc.iter().copied().collect();
        assert_eq!(distinct.len(), 256);
        for b in 0..256usize {
            assert_eq!(dec[&enc[b]], b as u8);
        }
        // Space is remapped to 'Ġ' (U+0120).
        assert_eq!(enc[b' ' as usize], '\u{0120}');
    }

    #[test]
    fn bytes_only_round_trips_text() {
        let tok = BpeTokenizer::bytes_only();
        assert_eq!(tok.vocab_size(), 256);
        for text in ["Hello, world!", "héllo — Ünicode 🚀", "", "  spaced  "] {
            let ids = tok.encode(text).unwrap();
            assert_eq!(tok.decode(&ids).unwrap(), text);
        }
    }

    #[test]
    fn bytes_only_ids_are_raw_bytes() {
        let tok = BpeTokenizer::bytes_only();
        let ids = tok.encode("AB").unwrap();
        // 'A' = 0x41, 'B' = 0x42; byte-level ids equal the bytes.
        assert_eq!(ids, vec![0x41, 0x42]);
    }

    #[test]
    fn merges_combine_adjacent_symbols() {
        // Vocabulary: single byte-chars for a,b,c plus merged "ab", "abc".
        let (enc, _) = byte_to_unicode();
        let a = enc[b'a' as usize].to_string();
        let b = enc[b'b' as usize].to_string();
        let c = enc[b'c' as usize].to_string();
        let ab = format!("{a}{b}");
        let abc = format!("{ab}{c}");

        let mut vocab = HashMap::new();
        vocab.insert(a.clone(), 0);
        vocab.insert(b.clone(), 1);
        vocab.insert(c.clone(), 2);
        vocab.insert(ab.clone(), 3);
        vocab.insert(abc.clone(), 4);
        // Merge (a,b) first, then (ab,c).
        let merges = vec![(a.clone(), b.clone()), (ab.clone(), c.clone())];
        let tok = BpeTokenizer::new(vocab, merges);

        assert_eq!(tok.encode("ab").unwrap(), vec![3]); // "ab"
        assert_eq!(tok.encode("abc").unwrap(), vec![4]); // "abc"
        assert_eq!(tok.encode("aba").unwrap(), vec![3, 0]); // "ab" + "a"
        assert_eq!(tok.decode(&[4]).unwrap(), "abc");
    }

    #[test]
    fn loads_from_vocab_and_merges_files() {
        let tmp = tempfile::tempdir().unwrap();
        let (enc, _) = byte_to_unicode();
        let a = enc[b'a' as usize].to_string();
        let b = enc[b'b' as usize].to_string();
        let ab = format!("{a}{b}");

        let vocab = format!(r#"{{"{a}":0,"{b}":1,"{ab}":2}}"#);
        std::fs::write(tmp.path().join("vocab.json"), vocab).unwrap();
        std::fs::write(
            tmp.path().join("merges.txt"),
            format!("#version: 0.2\n{a} {b}\n"),
        )
        .unwrap();

        let tok = BpeTokenizer::from_dir(tmp.path()).unwrap();
        assert_eq!(tok.vocab_size(), 3);
        assert_eq!(tok.encode("ab").unwrap(), vec![2]);
        assert_eq!(tok.decode(&[2]).unwrap(), "ab");
    }

    #[test]
    fn encode_errors_on_missing_token() {
        // Vocabulary missing the byte-char for 'z'.
        let (enc, _) = byte_to_unicode();
        let mut vocab = HashMap::new();
        vocab.insert(enc[b'a' as usize].to_string(), 0);
        let tok = BpeTokenizer::new(vocab, vec![]);
        assert!(tok.encode("z").is_err());
    }

    #[test]
    fn special_tokens_encode_as_single_ids_and_round_trip() {
        let tok = BpeTokenizer::bytes_only().with_special([("<|eot|>".to_string(), 999u32)]);

        // A special token in the middle splits the surrounding text.
        let ids = tok.encode("hi<|eot|>x").unwrap();
        assert!(ids.contains(&999), "special id missing: {ids:?}");
        // The special token is exactly one id (not BPE'd into pieces).
        assert_eq!(ids.iter().filter(|&&i| i == 999).count(), 1);
        assert_eq!(tok.decode(&ids).unwrap(), "hi<|eot|>x");

        // Leading special token.
        let ids2 = tok.encode("<|eot|>done").unwrap();
        assert_eq!(ids2[0], 999);
        assert_eq!(tok.decode(&ids2).unwrap(), "<|eot|>done");
    }

    #[test]
    fn loads_hf_tokenizer_json_with_special_tokens() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("tokenizer.json");
        std::fs::write(
            &path,
            r#"{
                "added_tokens": [{"id": 5, "content": "<|end|>", "special": true}],
                "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}
            }"#,
        )
        .unwrap();

        let tok = BpeTokenizer::from_hf_json(&path).unwrap();
        // "ab" merges to id 2; the special token becomes id 5.
        assert_eq!(tok.encode("ab<|end|>").unwrap(), vec![2, 5]);
        assert_eq!(tok.decode(&[2, 5]).unwrap(), "ab<|end|>");
    }

    /// A `"type": "BPE"` model whose pre-tokenizer is `Metaspace` (Mistral,
    /// Llama-2) must merge over ▁-escaped characters, not GPT-2 byte chars.
    /// Encoding it byte-level produced `Ġ` symbols that these vocabularies do not
    /// contain, so the whole family failed to tokenize at all.
    #[test]
    fn metaspace_bpe_encodes_spm_pieces_not_byte_chars() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("tokenizer.json");
        std::fs::write(
            &path,
            r#"{
                "pre_tokenizer": {"type": "Metaspace", "replacement": "▁",
                                  "prepend_scheme": "first"},
                "model": {"type": "BPE", "byte_fallback": true,
                          "vocab": {"▁": 0, "h": 1, "i": 2, "▁hi": 3, "▁h": 4},
                          "merges": ["▁ h", "▁h i"]}
            }"#,
        )
        .unwrap();

        let tok = BpeTokenizer::from_hf_json(&path).unwrap();
        // "hi" gets the prepended ▁ and merges to the single piece "▁hi".
        assert_eq!(tok.encode("hi").unwrap(), vec![3]);
        // Decode inverts ▁ and drops the prefix the encoder added.
        assert_eq!(tok.decode(&[3]).unwrap(), "hi");
    }

    /// Gemma signals the same convention with a `Replace` *normalizer* and no
    /// Metaspace, and prepends nothing — so keying detection on the pre-tokenizer
    /// alone (or always prepending) would get Gemma wrong.
    #[test]
    fn replace_normalizer_is_spm_without_a_prepended_mark() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("tokenizer.json");
        std::fs::write(
            &path,
            r#"{
                "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
                "model": {"type": "BPE",
                          "vocab": {"a": 0, "▁b": 1, "▁": 2, "b": 3},
                          "merges": ["▁ b"]}
            }"#,
        )
        .unwrap();

        let tok = BpeTokenizer::from_hf_json(&path).unwrap();
        // No prepend: "a" keeps its bare form, and the space joins the next piece.
        assert_eq!(tok.encode("a b").unwrap(), vec![0, 1]);
        assert_eq!(tok.decode(&[0, 1]).unwrap(), "a b");
    }

    /// A byte-level tokenizer (Qwen, Llama-3) must be left alone: no normalizer
    /// and no Metaspace means spaces stay `Ġ`.
    #[test]
    fn byte_level_bpe_is_untouched_by_spm_detection() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("tokenizer.json");
        std::fs::write(
            &path,
            r#"{"model": {"type": "BPE", "vocab": {"a": 0, "Ġb": 1, "Ġ": 2, "b": 3},
                          "merges": ["Ġ b"]}}"#,
        )
        .unwrap();
        let tok = BpeTokenizer::from_hf_json(&path).unwrap();
        assert!(tok.spm.is_none(), "no SPM signal means byte-level");
        assert_eq!(tok.encode("a b").unwrap(), vec![0, 1]);
    }

    /// `add_bos_token` is honored from `tokenizer_config.json`, exactly once.
    /// Gemma is trained with `<bos>` always present and emits degenerate text
    /// without it; Qwen sets the flag false and must not get one.
    #[test]
    fn add_bos_token_prepends_once() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("tokenizer.json"),
            r#"{
                "added_tokens": [{"id": 9, "content": "<bos>", "special": true}],
                "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []}
            }"#,
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("tokenizer_config.json"),
            r#"{"add_bos_token": true, "bos_token": {"content": "<bos>"}}"#,
        )
        .unwrap();

        let tok = BpeTokenizer::from_dir(tmp.path()).unwrap();
        assert_eq!(tok.bos_id(), Some(9));
        assert_eq!(tok.encode("ab").unwrap(), vec![9, 0, 1]);
        // Text that already opens with the literal must not get a second one.
        assert_eq!(tok.encode("<bos>ab").unwrap(), vec![9, 0, 1]);

        // add_bos_token: false leaves encoding untouched.
        std::fs::write(
            tmp.path().join("tokenizer_config.json"),
            r#"{"add_bos_token": false, "bos_token": "<bos>"}"#,
        )
        .unwrap();
        let tok = BpeTokenizer::from_dir(tmp.path()).unwrap();
        assert_eq!(tok.bos_id(), None);
        assert_eq!(tok.encode("ab").unwrap(), vec![0, 1]);
    }

    /// SentencePiece Unigram: Viterbi picks the highest-scoring segmentation, and
    /// decode inverts the ▁-escaping + dummy prefix, so text round-trips.
    #[test]
    fn unigram_round_trips() {
        let pieces = vec![
            ("<unk>".to_string(), 0.0),
            ("\u{2581}hello".to_string(), -1.0),
            ("\u{2581}world".to_string(), -1.0),
            ("\u{2581}".to_string(), -5.0),
        ];
        let tok = BpeTokenizer::from_unigram(pieces, false, Some(0));
        let ids = tok.encode("hello world").unwrap();
        // Two whole-word pieces (▁hello, ▁world) beat any finer split.
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(tok.decode(&ids).unwrap(), "hello world");
    }

    /// A character with no vocab piece decomposes into `<0xNN>` byte pieces when
    /// byte-fallback is on, and those reassemble on decode.
    #[test]
    fn unigram_byte_fallback_round_trips() {
        let pieces = vec![
            ("<unk>".to_string(), 0.0),
            ("\u{2581}hi".to_string(), -1.0),
            ("<0x21>".to_string(), -5.0), // '!' = 0x21
        ];
        let tok = BpeTokenizer::from_unigram(pieces, true, Some(0));
        let ids = tok.encode("hi!").unwrap();
        assert_eq!(tok.decode(&ids).unwrap(), "hi!");
        // The '!' had no piece, so it came through the byte-fallback token.
        assert!(ids.contains(&2), "expected the <0x21> byte piece: {ids:?}");
    }

    /// A `tokenizer.json` declaring a Unigram model (array vocab) loads through the
    /// same entry point as BPE and tokenizes.
    #[test]
    fn loads_unigram_tokenizer_json() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("tokenizer.json");
        std::fs::write(
            &path,
            r#"{"model":{"type":"Unigram","unk_id":0,"byte_fallback":true,
                "vocab":[["<unk>",0.0],["▁hello",-1.0],["▁world",-1.0]]}}"#,
        )
        .unwrap();
        let tok = BpeTokenizer::from_hf_json(&path).unwrap();
        assert_eq!(
            tok.decode(&tok.encode("hello world").unwrap()).unwrap(),
            "hello world"
        );
    }
}
