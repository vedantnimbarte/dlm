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
    // Newer exports keep the chat template here instead of in
    // tokenizer_config.json, and `--chat-template auto` reads it first.
    "chat_template.jinja",
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
    /// LFS metadata, present for the large files (the weights) and absent for
    /// small ones like `config.json`, which the hub stores inline.
    #[serde(default)]
    lfs: Option<Lfs>,
}

/// The LFS pointer the hub publishes for a large file. `oid` is the content's
/// SHA-256 in lowercase hex -- the digest we can check a download against.
#[derive(Debug, Deserialize)]
struct Lfs {
    #[serde(default)]
    oid: Option<String>,
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

/// Where a [`pull`] has got to.
///
/// Emitted per file rather than per byte: `curl_download` writes the file in
/// one call, so byte-level progress is not observable from here without taking
/// over the transfer. File granularity is still the difference between a UI
/// that shows "3 of 7 — model-00003-of-00007.safetensors" and one that shows a
/// frozen spinner for ten minutes.
#[derive(Debug, Clone)]
pub struct PullProgress<'a> {
    pub file: &'a str,
    /// 1-based index of the file being fetched.
    pub index: usize,
    pub total: usize,
    pub phase: PullPhase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullPhase {
    Downloading,
    /// Checking the downloaded bytes against the hub's SHA-256. On a
    /// multi-gigabyte shard this is slow enough to look like a hang if the UI
    /// does not say what it is doing.
    Verifying,
    Done,
}

/// Download the files dlm needs from `repo` into `dest` (default
/// `./models/<model>`). Returns the directory the model landed in.
///
/// Progress goes to stdout. Callers that need to render it themselves (a GUI)
/// should use [`pull_with_progress`].
pub fn pull(repo: &str, dest: Option<PathBuf>, token: Option<&str>) -> Result<PathBuf> {
    pull_with_progress(repo, dest, token, &mut |p: PullProgress| {
        if p.phase == PullPhase::Downloading {
            println!("  {}", p.file);
        }
    })
}

/// [`pull`], reporting progress through `on_progress`.
pub fn pull_with_progress(
    repo: &str,
    dest: Option<PathBuf>,
    token: Option<&str>,
    on_progress: &mut dyn FnMut(PullProgress),
) -> Result<PathBuf> {
    let repo = normalize_repo(repo)?;
    // `?blobs=true` is what makes the hub include each sibling's LFS block; the
    // bare endpoint returns filenames only, with no digest to verify against.
    let info_url = format!("{}/api/models/{}?blobs=true", base(), repo);
    let body = curl_json(&info_url, token)?;
    let info: ModelInfo = serde_json::from_slice(&body).map_err(|e| {
        DlmError::Hub(format!(
            "could not read model info for {repo}: {e} (private/gated? pass --token)"
        ))
    })?;

    // Mistral repos carry every weight twice: sharded with an index, and again
    // as `consolidated.safetensors`. dlm loads the sharded copy, so pulling the
    // other one doubles a multi-gigabyte download for a file nothing reads.
    let has_index = info
        .siblings
        .iter()
        .any(|s| s.rfilename.ends_with(".safetensors.index.json"));
    let wanted: Vec<(&String, Option<&str>)> = info
        .siblings
        .iter()
        .filter(|s| is_wanted(&s.rfilename))
        .filter(|s| !(has_index && crate::storage::mmap_store::is_consolidated(&s.rfilename)))
        .map(|s| (&s.rfilename, s.lfs.as_ref().and_then(|l| l.oid.as_deref())))
        .collect();

    if !wanted.iter().any(|(f, _)| f.ends_with(".safetensors")) {
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
    let total = wanted.len();
    for (index, (file, oid)) in wanted.iter().enumerate() {
        let url = format!("{}/{}/resolve/main/{}", base(), repo, file);
        let out = dir.join(file);
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let report = |phase| PullProgress {
            file,
            index: index + 1,
            total,
            phase,
        };
        on_progress(report(PullPhase::Downloading));
        curl_download(&url, &out, token)?;
        on_progress(report(PullPhase::Verifying));
        verify_sha256(&out, *oid)?;
        on_progress(report(PullPhase::Done));
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
    if capture_stdout {
        // `spawn` inherits both by default (unlike `output`), so capturing has to
        // be asked for, or `wait_with_output` returns empty buffers.
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
    } else {
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

/// Check a downloaded file against the digest the hub published for it.
///
/// A length match -- all `curl_download` could offer -- catches truncation and
/// nothing else. It does not catch a corrupted transfer that happens to be the
/// right size, a mirror serving different content, or a proxy rewriting the
/// body. The weights are the one input to a dlm run that nothing verified: the
/// binary is checksummed and minisigned by the installer and `Cargo.lock` pins
/// the build, while a multi-gigabyte shard arrived on trust.
///
/// Skipped, with a note, when the hub publishes no `oid` -- true for the small
/// non-LFS files. Silence there would be indistinguishable from a check that
/// passed.
fn verify_sha256(path: &Path, expected: Option<&str>) -> Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let got = crate::storage::sha256::file_hex(path)
        .map_err(|e| DlmError::Hub(format!("cannot read {} to verify: {e}", path.display())))?;
    if !got.eq_ignore_ascii_case(expected) {
        // Remove it: leaving a file that failed verification invites the next
        // `pull` to treat it as already complete and skip re-fetching it.
        let _ = std::fs::remove_file(path);
        return Err(DlmError::Hub(format!(
            "{} failed its checksum -- the hub published sha256 {expected}, the              download hashes to {got}. The file has been removed; re-run `dlm pull`.",
            path.display()
        )));
    }
    println!("    sha256 ok");
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

    /// `curl_json` must return the body, with and without a token. It once
    /// returned nothing — `spawn` inherits stdout, so the body went to the
    /// terminal — which broke every `dlm pull` at the first metadata request.
    /// A `file://` URL keeps this offline; skipped where curl is absent.
    #[test]
    fn curl_json_captures_the_body() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("info.json");
        std::fs::write(&path, br#"{"ok":true}"#).unwrap();
        let url = format!(
            "file:///{}",
            path.to_string_lossy()
                .replace('\\', "/")
                .trim_start_matches('/')
        );
        for token in [None, Some("secret")] {
            match curl_json(&url, token) {
                Ok(body) => assert_eq!(body, br#"{"ok":true}"#, "token {token:?}"),
                Err(DlmError::Hub(m)) if m.contains("curl") && m.contains("not found") => return,
                Err(e) => panic!("token {token:?}: {e}"),
            }
        }
    }

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
        assert!(is_wanted("chat_template.jinja"));
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

    /// `A10`: a download that hashes wrong is rejected AND removed, so the next
    /// `pull` re-fetches it rather than treating it as already complete.
    #[test]
    fn sha256_mismatch_is_rejected_and_the_file_removed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shard.safetensors");
        std::fs::write(&path, b"abc").unwrap();

        // Correct digest for "abc" passes and leaves the file alone.
        let ok = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert!(verify_sha256(&path, Some(ok)).is_ok());
        assert!(path.exists());
        // Case-insensitive: the hub publishes lowercase, but do not depend on it.
        assert!(verify_sha256(&path, Some(&ok.to_uppercase())).is_ok());

        // A wrong digest fails and takes the file with it.
        let err = verify_sha256(&path, Some(&"0".repeat(64))).unwrap_err();
        assert!(format!("{err}").contains("failed its checksum"), "{err}");
        assert!(
            !path.exists(),
            "a file that failed verification must not survive"
        );
    }

    /// No published `oid` (the small non-LFS files) is not an error.
    #[test]
    fn absent_digest_skips_verification() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, b"{}").unwrap();
        assert!(verify_sha256(&path, None).is_ok());
        assert!(path.exists());
    }

    #[test]
    fn urlencodes_spaces() {
        assert_eq!(urlencode("llama 3.2"), "llama%203.2");
    }
}
