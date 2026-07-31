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
        let tok = Self::new(vocab, merges_list).with_special(specials);
        match detect_spm(&hf.normalizer, &hf.pre_tokenizer, hf.model.byte_fallback) {
            Some(spm) => Ok(tok.with_spm(spm)),
            None => Ok(tok),
        }
    }

    /// Load a tokenizer from a model directory: prefer HF `tokenizer.json`, else
    /// fall back to the classic `vocab.json` + `merges.txt` pair.
    pub fn from_dir(dir: &Path) -> Result<Self> {
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
                        for chunk in pretokenize(&chunk_text) {
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

/// Split text so a leading space attaches to the following chunk (GPT-2 style:
/// " world" tokenizes as a "Ġworld" unit). Decoding is independent of this split.
fn pretokenize(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch == ' ' {
            if !cur.is_empty() {
                chunks.push(std::mem::take(&mut cur));
            }
            cur.push(ch);
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

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
