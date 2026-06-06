#!/usr/bin/env bash

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

#
# Ensure the `pandoc` binary is available, installing it if needed.
#
# nbsphinx shells out to `pandoc` to convert notebook markdown cells during the
# docs build. On macOS it comes from Homebrew. On Linux the dnf/apt repos do
# not ship pandoc, so we download the upstream static binary into /usr/local.
# Override the version with the PANDOC_VERSION environment variable.
#
# Usage: ensure_pandoc.sh

set -euo pipefail

# shellcheck source=utils.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/utils.sh"

if command -v pandoc &>/dev/null; then
    echo "pandoc already installed: $(pandoc --version | head -n1)"
    exit 0
fi

echo "pandoc not found. Attempting installation..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    ensure_package pandoc
    exit
fi

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Error: unsupported OS ($OSTYPE). Install pandoc manually." >&2
    exit 1
fi

# Linux: dnf/apt repos do not provide pandoc — fetch the upstream binary.
pandoc_version="${PANDOC_VERSION:-3.9.0.2}"
case "$(uname -m)" in
x86_64) arch="amd64" ;;
aarch64 | arm64) arch="arm64" ;;
*)
    echo "Error: unsupported architecture $(uname -m) for pandoc." >&2
    exit 1
    ;;
esac

# SHA-256 digests keyed by "<version>-<arch>".
# Source: https://api.github.com/repos/jgm/pandoc/releases/tags/<VERSION>
# When bumping PANDOC_VERSION, add the new digests here.
declare -A PANDOC_SHA256=(
    ["3.9.0.2-amd64"]="a69abfababda8a56969a254b09f9553a7be89ddec00d4e0fe9fd585d71a67508"
    ["3.9.0.2-arm64"]="b6d21e8f9c3b15744f5a7ab40248019157ed7793875dbe0383d4c82ff572b528"
)
sha256_key="${pandoc_version}-${arch}"
expected_sha256="${PANDOC_SHA256[$sha256_key]:-}"
if [[ -z "$expected_sha256" ]]; then
    echo "Error: no pinned SHA-256 for pandoc ${pandoc_version} (${arch})." >&2
    echo "Update PANDOC_SHA256 in $(basename "$0") using the asset digests from:" >&2
    echo "  https://api.github.com/repos/jgm/pandoc/releases/tags/${pandoc_version}" >&2
    exit 1
fi

base_url="https://github.com/jgm/pandoc/releases/download/${pandoc_version}"
tarball="pandoc-${pandoc_version}-linux-${arch}.tar.gz"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "Downloading pandoc ${pandoc_version} (linux-${arch}) from GitHub..."
if ! curl -fsSL "${base_url}/${tarball}" -o "${tmpdir}/${tarball}"; then
    echo "Error: failed to download pandoc from ${base_url}/${tarball}." >&2
    exit 1
fi

echo "Verifying SHA-256 checksum..."
if ! echo "${expected_sha256}  ${tarball}" | (cd "$tmpdir" && sha256sum --check); then
    echo "Error: SHA-256 verification failed for ${tarball}." >&2
    exit 1
fi

echo "Extracting to /usr/local..."
if ! tar xz --strip-components=1 -C /usr/local -f "${tmpdir}/${tarball}"; then
    echo "Error: failed to extract ${tarball}." >&2
    exit 1
fi

if ! command -v pandoc &>/dev/null; then
    echo "Error: pandoc installation failed. Please install pandoc manually." >&2
    exit 1
fi
echo "pandoc installed successfully: $(pandoc --version | head -n1)"
