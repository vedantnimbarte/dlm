//! Hugging Face model hub client — `flip search` / `flip pull`.
//!
//! flip needs no HTTP/TLS crate of its own: it shells out to `curl`, which ships
//! built-in on Linux, macOS, and Windows 10 (1803+) / 11, and handles TLS,
//! redirects to the CDN, resume, and the progress bar. We only orchestrate:
//! query the public JSON API and download the handful of files the loader reads.
//!
//! Set `HF_ENDPOINT` to use a mirror (e.g. `https://hf-mirror.com`).

use crate::{FlipError, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Files the engine actually reads from a model directory. Anything else in the
/// repo (READMEs, `.gguf`, PyTorch `.bin`, images) is skipped.
const KEEP_EXACT: &[&str] = &[
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
];

fn base() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into())
}

/// Normalize any of `org/model`, a full `https://huggingface.co/org/model` URL,
/// or `.../tree/main` into the bare `org/model` repo id.
pub fn normalize_repo(input: &str) -> Result<String> {
    let repo = input
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("huggingface.co/")
        .split("/tree/")
        .next()
        .unwrap_or("")
        .trim_matches('/');
    if repo.split('/').filter(|s| !s.is_empty()).count() != 2 {
        return Err(FlipError::Hub(format!(
            "expected a repo id like `org/model`, got {input:?}"
        )));
    }
    Ok(repo.to_string())
}

#[derive(Debug, Deserialize)]
pub struct ModelHit {
    pub id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
    #[serde(default, rename = "pipeline_tag")]
    pub task: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    #[serde(default)]
    siblings: Vec<Sibling>,
}

#[derive(Debug, Deserialize)]
struct Sibling {
    rfilename: String,
}

/// Search the hub for models carrying safetensors weights, most-downloaded first.
/// An empty `query` lists the top models overall.
pub fn search(query: &str, limit: usize) -> Result<Vec<ModelHit>> {
    let url = format!(
        "{}/api/models?search={}&filter=safetensors&sort=downloads&direction=-1&limit={}",
        base(),
        urlencode(query),
        limit
    );
    let body = curl_json(&url, None)?;
    serde_json::from_slice(&body)
        .map_err(|e| FlipError::Hub(format!("could not parse search results: {e}")))
}

/// Download the files flip needs from `repo` into `dest` (default
/// `./models/<model>`). Returns the directory the model landed in.
pub fn pull(repo: &str, dest: Option<PathBuf>, token: Option<&str>) -> Result<PathBuf> {
    let repo = normalize_repo(repo)?;
    let info_url = format!("{}/api/models/{}", base(), repo);
    let body = curl_json(&info_url, token)?;
    let info: ModelInfo = serde_json::from_slice(&body).map_err(|e| {
        FlipError::Hub(format!("could not read model info for {repo}: {e} (private/gated? pass --token)"))
    })?;

    let wanted: Vec<&String> = info
        .siblings
        .iter()
        .map(|s| &s.rfilename)
        .filter(|f| is_wanted(f))
        .collect();

    if !wanted.iter().any(|f| f.ends_with(".safetensors")) {
        return Err(FlipError::Hub(format!(
            "{repo} has no .safetensors weights — flip cannot load GGUF/PyTorch-only repos"
        )));
    }

    let model_name = repo.rsplit('/').next().unwrap();
    let dir = dest.unwrap_or_else(|| Path::new("models").join(model_name));
    std::fs::create_dir_all(&dir)
        .map_err(|e| FlipError::Hub(format!("cannot create {}: {e}", dir.display())))?;

    println!("pulling {repo} → {} ({} files)", dir.display(), wanted.len());
    for file in &wanted {
        let url = format!("{}/{}/resolve/main/{}", base(), repo, file);
        let out = dir.join(file);
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        println!("  {file}");
        curl_download(&url, &out, token)?;
    }
    println!("done. run: flip serve --model-path {}", dir.display());
    Ok(dir)
}

fn is_wanted(f: &str) -> bool {
    f.ends_with(".safetensors")
        || f.ends_with(".safetensors.index.json")
        || KEEP_EXACT.contains(&f)
}

/// GET a URL and return the body, failing clearly if curl is missing or the
/// request errors.
fn curl_json(url: &str, token: Option<&str>) -> Result<Vec<u8>> {
    let mut cmd = Command::new("curl");
    cmd.args(["-sSfL", url]);
    if let Some(t) = token {
        cmd.arg("-H").arg(format!("Authorization: Bearer {t}"));
    }
    let out = cmd.output().map_err(curl_missing)?;
    if !out.status.success() {
        return Err(FlipError::Hub(format!(
            "request failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(out.stdout)
}

/// Download a URL to `out`, streaming curl's progress bar to the terminal.
fn curl_download(url: &str, out: &Path, token: Option<&str>) -> Result<()> {
    let mut cmd = Command::new("curl");
    cmd.args(["-fL", "--progress-bar", "-o"]);
    cmd.arg(out).arg(url);
    if let Some(t) = token {
        cmd.arg("-H").arg(format!("Authorization: Bearer {t}"));
    }
    let status = cmd.status().map_err(curl_missing)?;
    if !status.success() {
        return Err(FlipError::Hub(format!("download failed for {}", out.display())));
    }
    Ok(())
}

fn curl_missing(e: std::io::Error) -> FlipError {
    if e.kind() == std::io::ErrorKind::NotFound {
        FlipError::Hub(
            "`curl` not found. It ships with Windows 10 (1803+), macOS, and Linux — \
             install it, or download the model manually from the Hugging Face website."
                .into(),
        )
    } else {
        FlipError::Hub(format!("could not run curl: {e}"))
    }
}

/// Minimal percent-encoding for a query string (alphanumerics + `-_.~` pass
/// through, everything else becomes `%XX`). Enough for model-name searches.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_repo_forms() {
        for input in [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
            "huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main",
            "  Qwen/Qwen2.5-0.5B-Instruct/  ",
        ] {
            assert_eq!(normalize_repo(input).unwrap(), "Qwen/Qwen2.5-0.5B-Instruct");
        }
        assert!(normalize_repo("just-a-name").is_err());
        assert!(normalize_repo("a/b/c").is_err());
    }

    #[test]
    fn keeps_only_loadable_files() {
        assert!(is_wanted("model.safetensors"));
        assert!(is_wanted("model-00001-of-00002.safetensors"));
        assert!(is_wanted("model.safetensors.index.json"));
        assert!(is_wanted("config.json"));
        assert!(is_wanted("tokenizer.json"));
        assert!(!is_wanted("model.gguf"));
        assert!(!is_wanted("pytorch_model.bin"));
        assert!(!is_wanted("README.md"));
    }

    #[test]
    fn urlencodes_spaces() {
        assert_eq!(urlencode("llama 3.2"), "llama%203.2");
    }
}
