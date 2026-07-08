#!/bin/sh
# dlm uninstaller — remove the binary install.sh dropped on your PATH.
#
#   curl -fsSL https://raw.githubusercontent.com/vedantnimbarte/dlm/main/uninstall.sh | sh
#
# Env:
#   DLM_INSTALL_DIR   install location to clean (default: $HOME/.local/bin)
set -eu

BIN="dlm"
INSTALL_DIR="${DLM_INSTALL_DIR:-$HOME/.local/bin}"
target="$INSTALL_DIR/$BIN"

if [ -e "$target" ]; then
  rm -f "$target" || { printf 'error: could not remove %s\n' "$target" >&2; exit 1; }
  printf 'Removed %s\n' "$target"
else
  # Not where we install by default — maybe it's elsewhere on PATH.
  found=$(command -v "$BIN" 2>/dev/null || true)
  if [ -n "$found" ]; then
    printf 'No %s in %s, but found one at %s — remove it with:\n  rm %s\n' \
      "$BIN" "$INSTALL_DIR" "$found" "$found"
  else
    printf '%s is not installed (nothing at %s).\n' "$BIN" "$target"
  fi
fi
