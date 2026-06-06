#!/usr/bin/env bash

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

#
# Ensure a Chrome binary matching mermaid-cli's bundled puppeteer is cached.
#
# mmdc (mermaid-cli) renders SVGs by driving headless Chrome through Puppeteer.
# Installing @mermaid-js/mermaid-cli alone does not guarantee a Chrome binary
# in Puppeteer's cache — the postinstall hook can be skipped or fail silently
# in some npm setups. This script triggers the install via mermaid-cli's own
# bundled puppeteer so the Chrome version always matches what mmdc expects.
#
# mmdc launches with `headless: 'shell'`, which selects the lightweight
# `chrome-headless-shell` browser binary rather than the full Chrome browser.
# We install that specific binary. Re-check on a mermaid-cli upgrade: a newer
# bundled puppeteer may switch browsers or expect a different version. Local
# devs can force a refresh with:
#   rm -rf "${PUPPETEER_CACHE_DIR:-$HOME/.cache/puppeteer}/chrome-headless-shell"
#
# Idempotent: short-circuits if a chrome-headless-shell binary is already
# present in the puppeteer cache.
#
# Usage: ensure_mmdc_chrome.sh

set -euo pipefail

CACHE_DIR="${PUPPETEER_CACHE_DIR:-$HOME/.cache/puppeteer}"
SHELL_DIR="$CACHE_DIR/chrome-headless-shell"

# Probe for the actual binary, not just a non-empty directory: an interrupted
# download (network failure, SIGINT, OOM) can leave a populated-but-broken
# directory that must be reinstalled rather than skipped.
if compgen -G "$SHELL_DIR/*/chrome-headless-shell*/chrome-headless-shell" >/dev/null; then
    exit 0
fi

NPM_ROOT="$(npm root -g)"
MMDC_PKG_DIR="$NPM_ROOT/@mermaid-js/mermaid-cli"
if [ ! -d "$MMDC_PKG_DIR" ]; then
    echo "ensure_mmdc_chrome.sh: $MMDC_PKG_DIR not found; install @mermaid-js/mermaid-cli first" >&2
    exit 1
fi

# `--no-install` keeps npx from fetching a different puppeteer — we want the
# puppeteer that mermaid-cli already depends on. The chrome-headless-shell binary
# itself downloads from Google's storage (not an npm registry); any incidental
# npm lookups use the registry configured in .npmrc / npm config.
echo "Installing chrome-headless-shell for mermaid-cli into ${CACHE_DIR}..."
(cd "$MMDC_PKG_DIR" && npx --no-install puppeteer browsers install chrome-headless-shell)
