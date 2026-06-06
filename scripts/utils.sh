#!/usr/bin/env bash

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

#
# Shared utility functions for CoreAI-Opt scripts

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Format: `2026-04-24 09:42:43.538635-0700 - LEVEL - msg`. Slice math drops the
# trailing 3 nanosecond digits from `%N` (9 digits) to produce 6-digit microseconds.
_log_ts() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S.%N%z')
    echo "${ts:0:26}${ts:29}"
}
log_info() { echo "$(_log_ts) - INFO - $*"; }
log_error() { echo "$(_log_ts) - ERROR - $*" >&2; }
log_success() { echo "$(_log_ts) - SUCCESS - $*"; }
log_warning() { echo "$(_log_ts) - WARNING - $*"; }
log_section() {
    echo ""
    echo "=== $* ==="
    echo ""
}

# ---------------------------------------------------------------------------
# Package management
# ---------------------------------------------------------------------------

# Ensure a package is installed, attempting installation if needed
#
# Usage: ensure_package <package_name> [binary_name]
#   package_name: Name of the package to install
#   binary_name: Name of the binary to check (defaults to package_name)
#
# Returns: 0 if package is available, 1 otherwise
ensure_package() {
    local package_name="$1"
    local binary_name="${2:-$package_name}"

    if command -v "$binary_name" &>/dev/null; then
        echo "$package_name already installed: $($binary_name --version 2>&1 | head -n1)"
        return 0
    fi

    echo "$package_name not found. Attempting installation..."
    install_package "$package_name" "$binary_name"

    if ! command -v "$binary_name" &>/dev/null; then
        echo "Error: $package_name installation failed. Please install $package_name manually."
        return 1
    fi

    echo "$package_name installed successfully: $($binary_name --version 2>&1 | head -n1)"
}

# Install package using appropriate package manager for the current OS
#
# Usage: install_package <package_name> [binary_name]
#   package_name: Name of the package to install
#   binary_name:  Name of the binary it provides (defaults to package_name)
#
# Returns: 0 on success, 1 on failure
install_package() {
    local package="$1"
    local binary="${2:-$package}"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [[ $EUID -eq 0 ]]; then
            # Homebrew refuses to run as root.
            # Use brew_install_from_root which runs brew as the 'local' user
            # and symlinks the binary into /usr/local/bin.
            brew_install_from_root "$package" "$binary"
        elif command -v brew &>/dev/null; then
            echo "Installing $package via Homebrew..."
            brew install "$package"
        else
            echo "Error: Homebrew not found. Install $package manually or install Homebrew first (https://brew.sh)."
            return 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v dnf &>/dev/null; then
            echo "Installing $package via dnf..."
            dnf install -y "$package"
        elif command -v apt-get &>/dev/null; then
            echo "Installing $package via apt-get..."
            apt-get update && apt-get install -y "$package"
        else
            echo "Error: No supported package manager found (dnf/apt-get). Install $package manually."
            return 1
        fi
    else
        echo "Error: Unsupported OS ($OSTYPE). Install $package manually."
        return 1
    fi
}

# Install a Homebrew package when running as root.
#
# Homebrew refuses to run as root. This function works around that by running
# `brew install` as the 'local' user via `sudo su`, then symlinking the binary
# into /usr/local/bin so root can use it. Retries on transient failures.
#
# Usage: brew_install_from_root <package_name> <executable_name>
#   package_name:     Homebrew formula (e.g., "python@3.10", "lychee")
#   executable_name:  Binary to check/symlink (e.g., "python3.10", "lychee")
#
# Returns: 0 on success, 1 on failure
brew_install_from_root() {
    local package_name="$1"
    local executable_name="$2"

    if [[ -z "$package_name" ]] || [[ -z "$executable_name" ]]; then
        log_error "brew_install_from_root: package_name and executable_name are required"
        return 1
    fi

    local max_retries=3
    local backoff_base=2

    # Detect brew binary: /opt/homebrew on Apple Silicon, /usr/local on Intel.
    local brew_bin
    if [[ -x /opt/homebrew/bin/brew ]]; then
        brew_bin=/opt/homebrew/bin/brew
    elif [[ -x /usr/local/bin/brew ]]; then
        brew_bin=/usr/local/bin/brew
    else
        log_error "Homebrew not found at /opt/homebrew/bin/brew or /usr/local/bin/brew"
        return 1
    fi
    local brew_prefix
    brew_prefix=$(dirname "$(dirname "$brew_bin")")

    if command -v "$executable_name" &>/dev/null; then
        local version_output
        version_output=$("$executable_name" --version 2>&1 || echo "unknown version")

        # For unversioned packages, any installed version is acceptable
        if [[ "$package_name" != *@* ]]; then
            log_info "${executable_name} is already installed: ${version_output}"
            return 0
        fi

        # For versioned packages (e.g., python@3.10), verify version matches
        local expected_version="${package_name##*@}"
        # Match version as a standalone token (bounded by space, dot, or end)
        # so "3.10" matches "Python 3.10.0" but not "Python 3.100".
        if [[ "$version_output" =~ (^|[[:space:]])${expected_version}([[:space:]]|\.|$) ]]; then
            log_info "${executable_name} is already installed with correct version: ${version_output}"
            return 0
        fi
        log_info "${executable_name} found but version mismatch. Expected ${expected_version}, found: ${version_output}"
    fi

    log_info "Installing ${package_name} via Homebrew..."

    local install_output
    for ((attempt = 1; attempt <= max_retries; attempt++)); do
        install_output=$(sudo su - local -c "${brew_bin} install -v \"${package_name}\"" 2>&1)
        local exit_code=$?

        if [[ $exit_code -ne 0 ]]; then
            echo "$install_output"
            if [[ $attempt -lt $max_retries ]]; then
                local backoff_seconds=$((backoff_base ** attempt))
                log_warning "Homebrew install of ${package_name} failed, retrying ($attempt/$max_retries) in ${backoff_seconds}s..."
                sleep "$backoff_seconds"
            fi
            continue
        fi

        log_info "Package installation details:"
        sudo su - local -c "${brew_bin} info \"${package_name}\"" 2>/dev/null | head -n1 || log_warning "Could not retrieve package info"

        # Create symlink for root user access
        local homebrew_path="${brew_prefix}/bin/${executable_name}"
        local symlink_path="/usr/local/bin/${executable_name}"

        if [[ ! -f "$homebrew_path" ]]; then
            log_error "Homebrew executable not found at ${homebrew_path}"
            return 1
        fi

        log_info "Creating symlink: ${symlink_path} -> ${homebrew_path}"
        ln -sf "$homebrew_path" "$symlink_path"

        if ! command -v "$symlink_path" &>/dev/null; then
            log_warning "Symlink created but ${executable_name} not found in PATH"
            log_warning "Current PATH: ${PATH}"
            log_warning "Symlink status: $(ls -la "$symlink_path" 2>&1)"
            return 1
        fi

        local final_version
        final_version=$("$symlink_path" --version 2>&1 || echo "unknown version")
        log_success "${package_name} installed successfully: ${final_version}"
        return 0
    done

    log_error "Failed to install ${package_name} after $max_retries attempts"
    return 1
}

# Ensure a binary provided by an npm package is installed globally.
#
# Unlike ensure_package (which uses the OS package manager directly), npm needs
# several prerequisites: the Node.js runtime, a configured registry, and
# authentication for private registries. This function tries to set each up
# automatically and only errors out for steps that genuinely require user input
# (e.g. `npm login`).
#
# Usage: ensure_npm_package <npm_package> [binary_name] [required_registry]
#   npm_package:       npm package to install (e.g. "@mermaid-js/mermaid-cli")
#   binary_name:       Binary the package provides (e.g. "mmdc"). Defaults to
#                      npm_package, since the package name often differs from
#                      the binary it ships.
#   required_registry: If set, install from this registry explicitly and verify
#                      auth (`npm whoami`) first. If omitted, npm uses its
#                      configured registry: internally the repo-root .npmrc
#                      selects the mirror; without one (e.g. OSS) the public default.
#
# Returns: 0 if binary is available, 1 otherwise
ensure_npm_package() {
    local npm_package="$1"
    local binary_name="${2:-$npm_package}"
    local required_registry="${3:-}"

    if command -v "$binary_name" &>/dev/null; then
        return 0
    fi

    echo "$binary_name not found — provided by npm package '$npm_package'."

    # `npm` ships bundled with the `node` package — installing node gives us both.
    # On RHEL/CentOS the package is named `nodejs`, not `node`.
    if ! command -v npm &>/dev/null; then
        local node_pkg="node"
        [[ "$OSTYPE" == "linux-gnu"* ]] && node_pkg="nodejs"
        ensure_package "$node_pkg" node || return 1
        # On Debian/Ubuntu, npm is often packaged separately from nodejs.
        if ! command -v npm &>/dev/null; then
            ensure_package npm || return 1
        fi
    fi

    # If a registry is explicitly required, install from it (verifying auth first).
    # Otherwise omit --registry so npm uses its configured registry (the repo-root
    # .npmrc internally; the public default otherwise) rather than forcing one here.
    local registry_args=()
    if [[ -n "$required_registry" ]]; then
        if ! npm whoami --registry "$required_registry" &>/dev/null; then
            echo "Error: npm is not authenticated for ${required_registry}. Run: npm login --registry ${required_registry}" >&2
            return 1
        fi
        registry_args=(--registry "$required_registry")
        echo "Installing $npm_package from ${required_registry}..."
    else
        echo "Installing $npm_package from the configured npm registry..."
    fi

    if ! npm install -g ${registry_args[@]+"${registry_args[@]}"} "$npm_package"; then
        echo "Error: npm install failed for $npm_package." >&2
        return 1
    fi

    if ! command -v "$binary_name" &>/dev/null; then
        echo "Error: $binary_name still not found after installation." >&2
        echo "Install manually: npm install -g $npm_package" >&2
        return 1
    fi

    echo "$binary_name installed: $($binary_name --version 2>&1 | head -n1)"
}
