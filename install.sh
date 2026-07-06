#!/bin/sh
# flip installer — download a prebuilt binary and drop it on your PATH.
#
#   curl -fsSL https://raw.githubusercontent.com/vedantnimbarte/Flip/main/install.sh | sh
#
# Env:
#   FLIP_INSTALL_DIR   install location (default: $HOME/.local/bin)
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
asset="${BIN}-${target}.tar.gz"
url="https://github.com/${REPO}/releases/latest/download/${asset}"

info "Installing ${BIN} (${target})…"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$url" -o "$tmp/$asset" || err "download failed: $url"
elif command -v wget >/dev/null 2>&1; then
  wget -qO "$tmp/$asset" "$url" || err "download failed: $url"
else
  err "need curl or wget installed"
fi

tar -xzf "$tmp/$asset" -C "$tmp" || err "extract failed — is there a published release for ${target}?"
[ -f "$tmp/$BIN" ] || err "binary '$BIN' not found in the downloaded archive"

mkdir -p "$INSTALL_DIR"
if ! install -m 755 "$tmp/$BIN" "$INSTALL_DIR/$BIN" 2>/dev/null; then
  cp "$tmp/$BIN" "$INSTALL_DIR/$BIN" && chmod 755 "$INSTALL_DIR/$BIN" || err "could not write to $INSTALL_DIR"
fi

info "Installed to $INSTALL_DIR/$BIN"

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
