#!/usr/bin/env bash
# Run the conformance program twice — once against the native numsol, once
# against the pure-Saule one — and diff.
#
# This is the oracle. The native package shadows `numsol` whenever it is
# installed (the resolver puts native packages ahead of every filesystem
# lookup), so the two cannot be loaded side by side; moving the library aside
# is what selects the reference. Identical output means the native kernels
# agree with the reference element for element.
#
#   ./examples/conformance.sh
#
# `SAULE_BIN` overrides the binary, `SAULE_HOME` the install directory.
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
saule="${SAULE_BIN:-$root/../saule/target/release/saule}"
packages="${SAULE_HOME:-$HOME/.saule}/native_packages"

case "$(uname -s)" in
  Darwin) ext=.dylib ;;
  MINGW* | MSYS* | CYGWIN*) ext=.dll ;;
  *) ext=.so ;;
esac
lib="libsaule_accel$ext"

if [ ! -x "$saule" ]; then
  echo "error: no saule binary at $saule — build it, or set SAULE_BIN" >&2
  exit 2
fi

built="$root/target/release/$lib"
if [ ! -f "$built" ]; then
  echo "error: $built is missing — run 'cargo build --release'" >&2
  exit 2
fi

out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
stash="$out/$lib"

# Remember whether the caller had it installed, so the original state is
# restored however this run ends.
had_installed=no
if [ -f "$packages/$lib" ]; then
  had_installed=yes
  mv "$packages/$lib" "$stash"
fi

# Reference: nothing installed, so `numsol` resolves to the Saule source.
(cd "$here/conformance" && "$saule" run .) > "$out/reference.txt" 2>&1
ref=$?

# Native: installed, shadowing it.
mkdir -p "$packages"
cp "$built" "$packages/$lib"
(cd "$here/conformance" && "$saule" run .) > "$out/native.txt" 2>&1
nat=$?

# Leave the install as the caller had it.
if [ "$had_installed" = no ]; then
  rm -f "$packages/$lib"
fi

if [ "$ref" -ne 0 ] || [ "$nat" -ne 0 ]; then
  echo "error: a run failed (reference=$ref native=$nat)" >&2
  echo "--- reference ---" >&2
  cat "$out/reference.txt" >&2
  echo "--- native ---" >&2
  cat "$out/native.txt" >&2
  exit 1
fi

if diff -u "$out/reference.txt" "$out/native.txt" > "$out/diff.txt"; then
  lines="$(wc -l < "$out/native.txt" | tr -d ' ')"
  echo "conformance: identical across $lines lines of output"
  exit 0
fi

echo "CONFORMANCE FAILED — the native kernels disagree with numsol:" >&2
cat "$out/diff.txt" >&2
exit 1
