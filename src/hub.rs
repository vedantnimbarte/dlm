//! Hugging Face model hub client — `dlm search` / `dlm pull`.
//!
//! dlm needs no HTTP/TLS crate of its own: it shells out to `curl`, which ships
//! built-in on Linux, macOS, and Windows 10 (1803+) / 11, and handles TLS,
//! redirects to the CDN, resume, and the progress bar. We only orchestrate:
//! query the public JSON API and download the handful of files the loader reads.
//!
//! Set `HF_ENDPOINT` to use a mirror (e.g. `https://hf-mirror.com`).

use crate::{DlmError, Result};
use serde::Deserialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

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
    let segments: Vec<&str> = repo.split('/').filter(|s| !s.is_empty()).collect();
    if segments.len() != 2 {
        return Err(DlmError::Hub(format!(
            "expected a repo id like `org/model`, got {input:?}"
        )));
    }
    // Counting segments is not validating them. `a/..` has two, so it passed —
    // and the model name is then `..`, which makes the destination `models/..`,
    // i.e. the parent of the intended directory. `../x` likewise walked out of
    // the API URL. `is_safe_relative_path` has always guarded the *filenames* the
    // hub returns; this is the same rule for the *repo id the user types*, which
    // reaches the same `Path::join`.
    if !segments.iter().all(|s| is_safe_relative_path(s)) {
        return Err(DlmError::Hub(format!(
            "repo id {input:?} contains a path component that is not a plain name"
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
        .map_err(|e| DlmError::Hub(format!("could not parse search results: {e}")))
}

/// Download the files dlm needs from `repo` into `dest` (default
/// `./models/<model>`). Returns the directory the model landed in.
pub fn pull(repo: &str, dest: Option<PathBuf>, token: Option<&str>) -> Result<PathBuf> {
    let repo = normalize_repo(repo)?;
    let info_url = format!("{}/api/models/{}", base(), repo);
    let body = curl_json(&info_url, token)?;
    let info: ModelInfo = serde_json::from_slice(&body).map_err(|e| {
        DlmError::Hub(format!(
            "could not read model info for {repo}: {e} (private/gated? pass --token)"
        ))
    })?;

    let wanted: Vec<&String> = info
        .siblings
        .iter()
        .map(|s| &s.rfilename)
        .filter(|f| is_wanted(f))
        .collect();

    if !wanted.iter().any(|f| f.ends_with(".safetensors")) {
        return Err(DlmError::Hub(format!(
            "{repo} has no .safetensors weights — dlm cannot load GGUF/PyTorch-only repos"
        )));
    }

    let model_name = repo.rsplit('/').next().unwrap();
    let dir = dest.unwrap_or_else(|| Path::new("models").join(model_name));
    std::fs::create_dir_all(&dir)
        .map_err(|e| DlmError::Hub(format!("cannot create {}: {e}", dir.display())))?;

    println!(
        "pulling {repo} → {} ({} files)",
        dir.display(),
        wanted.len()
    );
    for file in &wanted {
        let url = format!("{}/{}/resolve/main/{}", base(), repo, file);
        let out = dir.join(file);
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        println!("  {file}");
        curl_download(&url, &out, token)?;
    }
    println!("done. run: dlm serve --model-path {}", dir.display());
    Ok(dir)
}

fn is_wanted(f: &str) -> bool {
    is_safe_relative_path(f)
        && (f.ends_with(".safetensors")
            || f.ends_with(".safetensors.index.json")
            || KEEP_EXACT.contains(&f))
}

/// Reject filenames from the hub API that would escape the destination directory.
///
/// `rfilename` is attacker-controlled data (anyone can publish a repo). Joining it
/// blindly is an arbitrary-file-write: `Path::join` with an absolute path *discards*
/// the base entirely, and `..` components walk out of it. A name only ever needs to
/// be a plain relative path, so require exactly that.
fn is_safe_relative_path(f: &str) -> bool {
    use std::path::{Component, Path};

    if f.is_empty() || f.contains('\\') || f.contains('\0') {
        return false;
    }
    // Rejects "/etc/x", "C:\x", "../x", "a/../../x" — and, on Windows, the
    // drive-relative and UNC forms `Path` also treats as non-normal.
    Path::new(f)
        .components()
        .all(|c| matches!(c, Component::Normal(_)))
}

/// Run `curl`, passing the auth header (if any) through a config file on **stdin**
/// rather than on the command line.
///
/// A token in argv is world-readable on Linux: `/proc/<pid>/cmdline` is not
/// privileged, so any local user can `ps` the token for as long as curl runs —
/// and for a model pull that is minutes to hours. curl documents this and offers
/// `--config <file>`; `-` reads the config from stdin, which never appears in the
/// process table. The token therefore exists only in this process's memory and on
/// the pipe.
///
/// `stdout` is returned; `stderr` is captured so the caller can report it.
fn curl_with_token(
    args: &[&str],
    token: Option<&str>,
    capture_stdout: bool,
) -> std::io::Result<std::process::Output> {
    let mut cmd = Command::new("curl");
    cmd.args(args);
    if token.is_some() {
        // Read the header line from stdin instead of argv.
        cmd.arg("--config").arg("-");
        cmd.stdin(Stdio::piped());
    } else {
        cmd.stdin(Stdio::null());
    }
    if !capture_stdout {
        // Progress bar / body go straight to the terminal or the -o file.
        cmd.stdout(Stdio::inherit());
        cmd.stderr(Stdio::inherit());
    }

    let mut child = cmd.spawn()?;
    if let Some(t) = token {
        // curl config syntax: one directive per line, value quoted.
        let line = format!("header = \"Authorization: Bearer {t}\"\n");
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(line.as_bytes())?;
            // Dropping closes the pipe, which curl needs to finish reading config.
        }
    }
    child.wait_with_output()
}

/// GET a URL and return the body, failing clearly if curl is missing or the
/// request errors.
fn curl_json(url: &str, token: Option<&str>) -> Result<Vec<u8>> {
    let out = curl_with_token(&["-sSfL", url], token, true).map_err(curl_missing)?;
    if !out.status.success() {
        return Err(DlmError::Hub(format!(
            "request failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(out.stdout)
}

/// The size the server reports for `url`, or `None` if it will not say.
///
/// Used to skip files already fully present and to catch a transfer that ended
/// early. A `HEAD` is cheap next to the multi-gigabyte bodies this module moves.
fn remote_size(url: &str, token: Option<&str>) -> Option<u64> {
    let out = curl_with_token(&["-sIfL", url], token, true).ok()?;
    if !out.status.success() {
        return None;
    }
    // Take the last Content-Length: redirects to the CDN emit one per hop.
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim()
                .eq_ignore_ascii_case("content-length")
                .then(|| v.trim().parse().ok())?
        })
        .next_back()
}

fn local_size(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Download a URL to `out`, streaming curl's progress bar to the terminal.
///
/// Resumable and retrying, because the bodies here are model shards: a single
/// dropped connection an hour into a 30 GB pull used to discard the whole file
/// and restart from zero. `--retry-all-errors` covers the connection resets and
/// truncated transfers that a plain `--retry` ignores, and `-C -` continues from
/// whatever already landed rather than starting over.
///
/// A file already present at its full size is left alone, so re-running `pull`
/// after a failure resumes the set instead of refetching it.
fn curl_download(url: &str, out: &Path, token: Option<&str>) -> Result<()> {
    let want = remote_size(url, token);
    if let Some(want) = want {
        if local_size(out) == want && want > 0 {
            println!("    already complete ({want} bytes)");
            return Ok(());
        }
    }

    let out_str = out.to_string_lossy();
    let status = curl_with_token(
        &[
            "-fL",
            "--progress-bar",
            "--retry",
            "10",
            "--retry-delay",
            "3",
            "--retry-all-errors",
            "--continue-at",
            "-",
            "-o",
            &out_str,
            url,
        ],
        token,
        false,
    )
    .map_err(curl_missing)?
    .status;
    if !status.success() {
        return Err(DlmError::Hub(format!(
            "download failed for {} — re-run `dlm pull` to resume from what was fetched",
            out.display()
        )));
    }

    // A transfer can end "successfully" and still be short (a proxy closing the
    // body early). Loading a truncated shard fails later with a confusing parse
    // error, so catch it here where the cause is obvious.
    if let Some(want) = want {
        let have = local_size(out);
        if have != want {
            return Err(DlmError::Hub(format!(
                "{} is {have} bytes but the server declared {want} — the download was truncated. \
                 Re-run `dlm pull` to resume it.",
                out.display()
            )));
        }
    }
    Ok(())
}

fn curl_missing(e: std::io::Error) -> DlmError {
    if e.kind() == std::io::ErrorKind::NotFound {
        DlmError::Hub(
            "`curl` not found. It ships with Windows 10 (1803+), macOS, and Linux — \
             install it, or download the model manually from the Hugging Face website."
                .into(),
        )
    } else {
        DlmError::Hub(format!("could not run curl: {e}"))
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
    fn rejects_repo_ids_that_escape_the_destination() {
        // Each of these has exactly two non-empty segments, so the old
        // segment-count check accepted every one. `a/..` is the sharp case: the
        // model name becomes `..` and the destination `models/..` -- the parent of
        // where the user asked for it.
        for evil in [
            "a/..",
            "../x",
            "../..",
            "a/../b",
            r"a\..",
            "org/model\0",
            "./x",
            "a/.",
        ] {
            assert!(
                normalize_repo(evil).is_err(),
                "should have rejected repo id {evil:?}"
            );
        }
        // Ordinary ids, including the URL forms, still normalize.
        assert_eq!(
            normalize_repo("Qwen/Qwen3-0.6B").unwrap(),
            "Qwen/Qwen3-0.6B"
        );
        assert_eq!(
            normalize_repo("meta-llama/Llama-3.2-1B-Instruct").unwrap(),
            "meta-llama/Llama-3.2-1B-Instruct"
        );
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
    }

    #[test]
    fn rejects_filenames_that_escape_the_destination() {
        // `rfilename` comes from the hub API — i.e. from whoever published the
        // repo. Each of these ends in an accepted suffix, so only the path check
        // stands between them and an arbitrary file write.
        for evil in [
            "../../../../etc/cron.d/x.safetensors",
            "/etc/cron.d/x.safetensors",
            "a/../../b.safetensors",
            r"..\..\x.safetensors",
            r"C:\Windows\Temp\x.safetensors",
            "",
        ] {
            assert!(!is_wanted(evil), "should have rejected {evil:?}");
        }
        // Plain relative paths, including subdirectories, stay allowed.
        assert!(is_wanted("model.safetensors"));
        assert!(is_safe_relative_path("subdir/model.safetensors"));
        assert!(!is_wanted("README.md"));
    }

    #[test]
    fn urlencodes_spaces() {
        assert_eq!(urlencode("llama 3.2"), "llama%203.2");
    }
}
