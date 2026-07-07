#!/bin/sh
# flip installer — download a prebuilt binary and drop it on your PATH.
#
#   curl -fsSL https://raw.githubusercontent.com/vedantnimbarte/Flip/main/install.sh | sh
#
# On Linux x86-64 with an NVIDIA GPU it installs the CUDA (GPU) build by default;
# everywhere else (and with FLIP_CPU=1) it installs the portable CPU build.
#
# Env:
#   FLIP_INSTALL_DIR   install location (default: $HOME/.local/bin)
#   FLIP_CPU=1         force the portable CPU build even if a GPU is detected
set -eu

REPO="vedantnimbarte/Flip"
BIN="flip"
INSTALL_DIR="${FLIP_INSTALL_DIR:-$HOME/.local/bin}"

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

target="${arch_part}-${os_part}"
cpu_asset="${BIN}-${target}.tar.gz"

# GPU build only exists for Linux x86-64. Pick it when an NVIDIA GPU is present
# (nvidia-smi answers) unless FLIP_CPU=1 forces the portable CPU build.
asset="$cpu_asset"
kind="CPU"
if [ "${FLIP_CPU:-}" != "1" ] && [ "$os" = "Linux" ] && [ "$arch_part" = "x86_64" ] \
   && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  asset="${BIN}-${target}-cuda.tar.gz"
  kind="GPU (CUDA)"
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Download + extract the named asset into $tmp; sets $tmp/$BIN. Fatal on failure.
fetch() {
  a="$1"
  u="https://github.com/${REPO}/releases/latest/download/${a}"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$u" -o "$tmp/$a" || err "download failed: $u"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$tmp/$a" "$u" || err "download failed: $u"
  else
    err "need curl or wget installed"
  fi
  tar -xzf "$tmp/$a" -C "$tmp" || err "extract failed — is there a published release for ${a}?"
  [ -f "$tmp/$BIN" ] || err "binary '$BIN' not found in ${a}"
}

info "Installing ${BIN} (${target}, ${kind} build)…"
fetch "$asset"

# The GPU build dynamically links the CUDA runtime; if it's absent the binary
# won't even start. Detect that here and fall back to the portable CPU build so
# the install never leaves a binary that can't run.
if [ "$asset" != "$cpu_asset" ] && ! "$tmp/$BIN" --version >/dev/null 2>&1; then
  info "GPU build won't start here (CUDA runtime missing?) — falling back to the CPU build."
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
