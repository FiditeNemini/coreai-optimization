#!/usr/bin/env bash

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

set -euo pipefail

remove_if_exists() {
    local path="$1"
    if [ -e "$path" ]; then
        rm -rf "$path"
        echo "Removed $path"
    fi
    return 0
}

# Get current venv - check multiple sources
get_current_venv() {
    local venv_path=""

    # 1. Check VIRTUAL_ENV environment variable (most reliable)
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        venv_path="$VIRTUAL_ENV"
    # 2. Parse .envrc as fallback
    elif [ -f ".envrc" ]; then
        venv_path=$(grep -E "^source .*/bin/activate" .envrc | sed -E 's|^source (.+)/bin/activate|\1|' || echo "")
    # 3. Read .coreai-opt-venv marker file
    elif [ -f ".coreai-opt-venv" ]; then
        venv_path=$(sed -n 's/^DEFAULT_VENV *= *//p' .coreai-opt-venv)
    fi

    # Return only the basename if it's a local path (starts with . or is in current dir)
    if [ -n "$venv_path" ]; then
        # Normalize to absolute path if relative (portable approach)
        if [[ "$venv_path" != /* ]]; then
            if [ -d "$venv_path" ]; then
                venv_path="$(cd "$venv_path" && pwd)"
            else
                # Path doesn't exist - return empty to indicate invalid venv
                return 0
            fi
        fi

        # If it's a full path in current directory, extract basename
        local repo_path="$(pwd)"
        if [[ "$venv_path" == "$repo_path"/* ]]; then
            basename "$venv_path"
        else
            echo "$venv_path"
        fi
    fi
}

# Find all local venv directories (hidden directories with both bin/activate and pyvenv.cfg)
find_local_venvs() {
    find . -maxdepth 1 -type d -name ".*" -exec sh -c 'test -f "$0/bin/activate" && test -f "$0/pyvenv.cfg"' {} \; -print 2>/dev/null | sort || true
}

MODE="${1:-default}"
SWITCHED_TO_VENV=false
CURRENT_VENV=$(get_current_venv)

# Determine mode description for header
if [ "$MODE" = "all" ]; then
    MODE_DESC="ALL environments"
else
    if [ -n "$CURRENT_VENV" ]; then
        MODE_DESC="$CURRENT_VENV"
    else
        MODE_DESC="current environment"
    fi
fi

# Print header
echo "=========================================="
echo "Running deep clean ($MODE_DESC)"
echo "=========================================="
echo ""

echo "[1/4] Cleaning build artifacts..."
make clean
echo ""

if [ "$MODE" = "all" ]; then
    echo "[2/4] Removing virtual environments..."

    VENVS=$(find_local_venvs)
    if [ -n "$VENVS" ]; then
        echo "Found virtual environments:"
        echo "$VENVS"
        echo ""
        for venv in $VENVS; do
            remove_if_exists "$venv"
        done
        echo "✓ All virtual environments removed"
    else
        echo "No local virtual environments found"
    fi
    echo ""

    echo "[3/4] Removing configuration files..."
    remove_if_exists ".envrc"

else
    echo "[2/4] Detecting current virtual environment..."
    if [ -n "$CURRENT_VENV" ]; then
        echo "Detected: $CURRENT_VENV"
    else
        echo "No active virtual environment detected"
    fi
    echo ""

    echo "[3/4] Removing virtual environment..."
    # Check if it's a local venv (starts with . and is in current directory)
    if [ -n "$CURRENT_VENV" ] && [[ "$CURRENT_VENV" == .* ]] && [ -d "$CURRENT_VENV" ]; then
        remove_if_exists "$CURRENT_VENV"

        # Switch back to .venv if it exists and is different from current
        # Only update .envrc if it already exists (user has set up auto-venv)
        if [ "$CURRENT_VENV" != ".venv" ] && [ -d ".venv" ] && [ -f ".envrc" ]; then
            echo ""
            echo "Switching back to default .venv..."

            # Write new .envrc file
            cat >.envrc <<'EOF'
# direnv setup: auto-activate project .venv (created by `uv venv`) when entering the repo
# Steps:
#   1. Install direnv (macOS):  brew install direnv
#   2. Add this hook to your shell RC so direnv runs automatically in each shell:
#        eval "$(direnv hook zsh)"  # or bash
#   3. In your project root (contains .envrc and .venv):
#        direnv allow
#      (You only need to `allow` once per repo; rerun if .envrc changes)
# Behavior: activates .venv when entering the repo; unloads it when leaving.

source .venv/bin/activate
EOF

            # Check if write succeeded
            if [ $? -eq 0 ] && [ -f ".envrc" ]; then
                echo "✓ Updated .envrc to use .venv"
                SWITCHED_TO_VENV=true
            else
                echo "Error: Failed to write .envrc (check permissions)"
                exit 1
            fi

            # Re-allow direnv if it's available
            if command -v direnv &>/dev/null; then
                if direnv allow 2>/dev/null; then
                    echo "✓ direnv reloaded - environment will switch on next directory change"
                else
                    echo ""
                    echo "Note: Run 'direnv allow' to activate the new environment"
                fi
            fi
        elif [ "$CURRENT_VENV" != ".venv" ] && [ ! -d ".venv" ]; then
            # No .venv to fall back to, remove .envrc if it exists
            remove_if_exists ".envrc"
        fi
    else
        # Clean up default locations
        remove_if_exists ".venv"
        remove_if_exists ".envrc"
    fi
fi

echo ""
echo "[4/4] Removing configuration and lock files..."
remove_if_exists ".coreai-opt-venv"
remove_if_exists ".rio"
remove_if_exists "uv.lock"

echo ""
echo "=========================================="
echo "✅ Deep clean complete!"
echo "=========================================="

# Check if $VIRTUAL_ENV points to a non-existent directory
# Skip warning if direnv will handle the switch (.envrc exists and points to existing .venv)
if [ -n "${VIRTUAL_ENV:-}" ] && [ ! -d "$VIRTUAL_ENV" ]; then
    # Check if direnv will auto-switch to .venv
    if [ -f ".envrc" ] && [ -d ".venv" ] && grep -q "source \.venv/bin/activate" .envrc 2>/dev/null; then
        # direnv will automatically switch to .venv, no user action needed
        :
    else
        echo ""
        echo "⚠️  Warning: Your shell's \$VIRTUAL_ENV points to a removed directory"
        echo "   Run 'unset VIRTUAL_ENV' or start a new shell to clear the environment"
    fi
fi

# Only show "make env" message if we didn't switch back to .venv
if [ "$SWITCHED_TO_VENV" = false ]; then
    echo ""
    echo "To set up a new development environment, run:"
    echo "  make env"
fi

echo ""
