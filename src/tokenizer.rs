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

use crate::error::{FlipError, Result};
use std::collections::HashMap;
use std::path::Path;

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
        }
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
        }
    }

    /// Load a `vocab.json` + `merges.txt` pair.
    pub fn from_files(vocab_path: &Path, merges_path: &Path) -> Result<Self> {
        let vocab_bytes = std::fs::read(vocab_path).map_err(|source| FlipError::Io {
            path: vocab_path.to_path_buf(),
            source,
        })?;
        let encoder: HashMap<String, u32> =
            serde_json::from_slice(&vocab_bytes).map_err(|source| FlipError::Json {
                context: "vocab.json".to_string(),
                source,
            })?;

        let merges_text = std::fs::read_to_string(merges_path).map_err(|source| FlipError::Io {
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

    /// Load `vocab.json` + `merges.txt` from a model directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::from_files(&dir.join("vocab.json"), &dir.join("merges.txt"))
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// Encode text into token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for chunk in pretokenize(text) {
            for symbol in self.bpe(chunk.as_bytes()) {
                let id = self.encoder.get(&symbol).ok_or_else(|| {
                    FlipError::Tokenizer(format!("token {symbol:?} not in vocabulary"))
                })?;
                ids.push(*id);
            }
        }
        Ok(ids)
    }

    /// Decode token ids back into text (lossy on invalid UTF-8).
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut joined = String::new();
        for &id in ids {
            let tok = self
                .decoder
                .get(&id)
                .ok_or_else(|| FlipError::Tokenizer(format!("unknown token id {id}")))?;
            joined.push_str(tok);
        }
        // Map the byte-chars back to raw bytes, then interpret as UTF-8.
        let mut bytes = Vec::with_capacity(joined.len());
        for ch in joined.chars() {
            let b = self
                .byte_decoder
                .get(&ch)
                .ok_or_else(|| FlipError::Tokenizer(format!("char {ch:?} is not a byte token")))?;
            bytes.push(*b);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
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
}
