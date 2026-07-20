#!/bin/sh
# dlm installer — download a prebuilt binary and drop it on your PATH.
#
#   curl -fsSL https://raw.githubusercontent.com/vedantnimbarte/dlm/main/install.sh | sh
#
# On Linux x86-64 with an NVIDIA GPU it installs the CUDA (GPU) build by default;
# everywhere else (and with DLM_CPU=1) it installs the portable CPU build.
#
# Env:
#   DLM_INSTALL_DIR   install location (default: $HOME/.local/bin)
#   DLM_CPU=1         force the portable CPU build even if a GPU is detected
set -eu

REPO="vedantnimbarte/dlm"
BIN="dlm"
# minisign public key for release authenticity. The sha256 check proves the
# download is intact; a signature proves it came from whoever holds this key.
# Empty until a signing key is published for the project (see release.yml) —
# while empty, signature verification is skipped and only the checksum applies.
# Override with DLM_MINISIGN_PUBKEY to pin your own key.
MINISIGN_PUBKEY="${DLM_MINISIGN_PUBKEY:-}"
INSTALL_DIR="${DLM_INSTALL_DIR:-$HOME/.local/bin}"

err() { printf 'error: %s\n' "$1" >&2; exit 1; }
info() { printf '%s\n' "$1"; }

os=$(uname -s)
arch=$(uname -m)

case "$os" in
  Linux) os_part="unknown-linux-gnu" ;;
  Darwin) os_part="apple-darwin" ;;
  *) err "unsupported OS '$os'. Prebuilt binaries cover Linux and macOS; on others build from source: cargo install --git https://github.com/$REPO" ;;
esac

case "$arch" in
  x86_64 | amd64) arch_part="x86_64" ;;
  aarch64 | arm64) arch_part="aarch64" ;;
  *) err "unsupported architecture '$arch'" ;;
esac

# No prebuilt Intel-Mac binary (GitHub retired the macOS Intel CI image). Apple
# Silicon and Linux are covered; Intel Macs build from source.
if [ "$os" = "Darwin" ] && [ "$arch_part" = "x86_64" ]; then
  err "Intel Macs have no prebuilt binary. Use an Apple Silicon Mac or Linux, or build from source: cargo install --git https://github.com/$REPO"
fi

target="${arch_part}-${os_part}"
cpu_asset="${BIN}-${target}.tar.gz"

# Build selection. The GPU (static-CUDA) build exists for Linux x86-64 only; its
# CUDA runtime is statically embedded, so it runs on any machine with just the
# NVIDIA driver (no toolkit needed). Pick it when an NVIDIA GPU is present
# (nvidia-smi answers), unless DLM_CPU=1 forces the portable CPU build.
#
# If nvidia-smi is absent we install the CPU build and say so; an AMD GPU lands
# here too — AMD GPU support is planned, so it runs on CPU for now.
asset="$cpu_asset"
kind="CPU"
gpu_capable=0
if [ "$os" = "Linux" ] && [ "$arch_part" = "x86_64" ]; then gpu_capable=1; fi

if [ "${DLM_CPU:-}" = "1" ]; then
  info "DLM_CPU=1 set — installing the CPU build."
elif [ "$gpu_capable" = "1" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  asset="${BIN}-${target}-cuda-static.tar.gz"
  kind="GPU (CUDA)"
  info "NVIDIA GPU detected — installing the GPU (CUDA) build."
else
  info "No NVIDIA GPU detected — installing the CPU build. (AMD GPU support is planned; it runs on CPU for now.)"
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# GET $1 into file $2. Fatal on any HTTP or network error.
download() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$1" -o "$2" || err "download failed: $1"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$2" "$1" || err "download failed: $1"
  else
    err "need curl or wget installed"
  fi
}

# Verify $tmp/$1 against its published .sha256. The archive is extracted and the
# binary executed moments later, so a corrupted or tampered asset must not get
# that far. Skipped only if the machine has no sha256 tool at all.
verify() {
  a="$1"
  # The checksum asset is named after the archive *stem*, with no archive
  # extension: `dlm-<target>.sha256`, not `dlm-<target>.tar.gz.sha256`.
  stem="${a%.tar.gz}"
  sum_url="https://github.com/${REPO}/releases/latest/download/${stem}.sha256"
  if command -v sha256sum >/dev/null 2>&1; then
    hash=$(sha256sum "$tmp/$a" | cut -d' ' -f1)
  elif command -v shasum >/dev/null 2>&1; then
    hash=$(shasum -a 256 "$tmp/$a" | cut -d' ' -f1)
  else
    # Fail closed: the README promises every download is checksum-verified
    # before it runs, so a machine with no hash tool must not silently install
    # an unverified binary. Escape hatch for the rare toolless box:
    # DLM_SKIP_CHECKSUM=1.
    if [ "${DLM_SKIP_CHECKSUM:-0}" = "1" ]; then
      info "warning: no sha256sum/shasum and DLM_SKIP_CHECKSUM=1 — installing WITHOUT verification"
      return 0
    fi
    err "no sha256sum/shasum found, so the download cannot be verified.
Install one (coreutils/perl) and re-run, or set DLM_SKIP_CHECKSUM=1 to install unverified."
  fi

  download "$sum_url" "$tmp/$stem.sha256"
  # The published file is "<hash>  <filename>" (or "<hash> *<filename>" when it
  # was produced in binary mode); the hash is the first whitespace-separated field.
  want=$(awk '{print $1; exit}' <"$tmp/$stem.sha256")
  [ -n "$want" ] || err "empty checksum file for $a"
  [ "$hash" = "$want" ] || err "checksum mismatch for $a
  expected: $want
  actual:   $hash
This means the download was corrupted or tampered with. Aborting."
}

# Verify $tmp/$1's minisign signature against MINISIGN_PUBKEY. The checksum
# already guarantees integrity; this adds authenticity (it came from the key
# holder). Opt-in and best-effort: skipped when no key is configured or minisign
# isn't installed, but a *present* signature that fails to verify aborts.
verify_signature() {
  a="$1"
  [ -n "$MINISIGN_PUBKEY" ] || return 0
  if ! command -v minisign >/dev/null 2>&1; then
    info "note: minisign not installed — skipping signature (authenticity) check; sha256 integrity still verified. Install minisign to enable it."
    return 0
  fi
  download "https://github.com/${REPO}/releases/latest/download/${a}.minisig" "$tmp/$a.minisig"
  printf '%s\n' "$MINISIGN_PUBKEY" >"$tmp/dlm.pub"
  minisign -Vm "$tmp/$a" -p "$tmp/dlm.pub" -x "$tmp/$a.minisig" >/dev/null 2>&1 \
    || err "signature verification failed for $a — it is not signed by the expected key. Aborting."
  info "release signature verified"
}

# Download + verify + extract the named asset into $tmp; sets $tmp/$BIN.
fetch() {
  a="$1"
  download "https://github.com/${REPO}/releases/latest/download/${a}" "$tmp/$a"
  verify "$a"
  verify_signature "$a"
  tar -xzf "$tmp/$a" -C "$tmp" || err "extract failed — is there a published release for ${a}?"
  [ -f "$tmp/$BIN" ] || err "binary '$BIN' not found in ${a}"
}

# If we picked the GPU build but it isn't published in this release (e.g. its CI
# build failed while the CPU build succeeded), fall back to CPU rather than
# aborting on a 404 — the user still gets a working install.
asset_exists() {
  url="https://github.com/${REPO}/releases/latest/download/$1"
  if command -v curl >/dev/null 2>&1; then curl -fsIL -o /dev/null "$url"
  elif command -v wget >/dev/null 2>&1; then wget -q --spider "$url"
  else return 0; fi
}
if [ "$asset" != "$cpu_asset" ] && ! asset_exists "$asset"; then
  info "GPU build not published in this release — falling back to the CPU build."
  asset="$cpu_asset"
  kind="CPU"
fi

info "Installing ${BIN} (${target}, ${kind} build)…"
fetch "$asset"

# The GPU build embeds the CUDA runtime but still needs a usable NVIDIA driver.
# If the driver is missing or too old the binary won't start — detect that and
# fall back to the CPU build so the install never leaves a binary that can't run.
if [ "$asset" != "$cpu_asset" ] && ! "$tmp/$BIN" --version >/dev/null 2>&1; then
  info "GPU build won't start here (NVIDIA driver missing or too old?) — falling back to the CPU build."
  info "Update your NVIDIA driver and re-run to get the GPU build."
  rm -f "$tmp/$BIN"
  fetch "$cpu_asset"
  kind="CPU"
fi

mkdir -p "$INSTALL_DIR"
if ! install -m 755 "$tmp/$BIN" "$INSTALL_DIR/$BIN" 2>/dev/null; then
  cp "$tmp/$BIN" "$INSTALL_DIR/$BIN" && chmod 755 "$INSTALL_DIR/$BIN" || err "could not write to $INSTALL_DIR"
fi

info "Installed the ${kind} build to $INSTALL_DIR/$BIN"

case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *)
    info ""
    info "  $INSTALL_DIR is not on your PATH. Add it:"
    info "    export PATH=\"$INSTALL_DIR:\$PATH\""
    info "  (append that to ~/.bashrc or ~/.zshrc to persist)"
    ;;
esac

info ""
if "$INSTALL_DIR/$BIN" --version >/dev/null 2>&1; then
  "$INSTALL_DIR/$BIN" --version
  info "Done. Try:  $BIN --help"
else
  info "Installed, but '$BIN --version' did not run — check the binary."
fi
