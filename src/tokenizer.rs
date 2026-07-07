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
    #[serde(default)]
    vocab: HashMap<String, u32>,
    #[serde(default)]
    merges: Vec<HfMerge>,
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
}

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
        let hf: HfTokenizer =
            serde_json::from_slice(&bytes).map_err(|source| DlmError::Json {
                context: "tokenizer.json".to_string(),
                source,
            })?;
        if !hf.model.vocab.is_empty() && hf.model.merges.is_empty() {
            // A vocab with no merges is almost certainly a Unigram model.
            return Err(DlmError::Tokenizer(
                "tokenizer.json has no BPE merges (Unigram/SentencePiece unsupported)".into(),
            ));
        }
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
        let specials: Vec<(String, u32)> = hf
            .added_tokens
            .into_iter()
            .filter(|t| t.special)
            .map(|t| (t.content, t.id))
            .collect();
        Ok(Self::new(hf.model.vocab, merges_list).with_special(specials))
    }

    /// Load a tokenizer from a model directory: prefer HF `tokenizer.json`, else
    /// fall back to the classic `vocab.json` + `merges.txt` pair.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let hf = dir.join("tokenizer.json");
        if hf.exists() {
            return Self::from_hf_json(&hf);
        }
        Self::from_files(&dir.join("vocab.json"), &dir.join("merges.txt"))
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// Encode text into token ids. Registered special tokens are matched as whole
    /// units (longest-match) and emit their own id; the text between is BPE'd.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for seg in self.split_special(text) {
            match seg {
                Seg::Special(id) => ids.push(id),
                Seg::Text(chunk_text) => {
                    for chunk in pretokenize(&chunk_text) {
                        for symbol in self.bpe(chunk.as_bytes()) {
                            let id = self.encoder.get(&symbol).ok_or_else(|| {
                                DlmError::Tokenizer(format!("token {symbol:?} not in vocabulary"))
                            })?;
                            ids.push(*id);
                        }
                    }
                }
            }
        }
        Ok(ids)
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
        let mut symbols: Vec<String> = chunk_bytes
            .iter()
            .map(|&b| self.byte_encoder[b as usize].to_string())
            .collect();

        while symbols.len() >= 2 {
            // Find the adjacent pair with the lowest merge rank.
            let mut best: Option<(usize, u32)> = None;
            for i in 0..symbols.len() - 1 {
                if let Some(&rank) = self.merges.get(&(symbols[i].clone(), symbols[i + 1].clone())) {
                    if best.map_or(true, |(_, r)| rank < r) {
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

    #[test]
    fn rejects_unigram_tokenizer_json() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("tokenizer.json");
        // Vocab present, no merges → Unigram/SentencePiece → unsupported.
        std::fs::write(&path, r#"{"model": {"vocab": {"a": 0}, "merges": []}}"#).unwrap();
        assert!(BpeTokenizer::from_hf_json(&path).is_err());
    }
}
