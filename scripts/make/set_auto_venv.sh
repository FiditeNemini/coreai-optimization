#!/usr/bin/env bash

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

set -euo pipefail

# Script to set up direnv for automatic virtual environment activation
# Arg 1 (optional): virtual environment path, e.g. ".venv" or "my_env" (defaults to .venv)
#   Usage: scripts/make/set_auto_venv.sh my_env
VENV="${1:-.venv}"
# Arg 2 (optional): shell RC file path, e.g. "~/.zshrc" or "~/.bashrc"
#   When provided, skips the interactive RC file prompt.
#   Usage: scripts/make/set_auto_venv.sh .venv ~/.zshrc
SHELL_RC_ARG="${2:-}"

# Check if the virtual environment exists
if [ ! -f "$VENV/bin/activate" ]; then
    echo "Error: Virtual environment '$VENV' does not exist"
    echo "Run 'make env VENV=$VENV' first to create it"
    exit 1
fi

echo "=========================================="
echo "Setting up automatic environment activation"
echo "Environment: $VENV"
echo "=========================================="
echo ""

# Check if running on macOS
OS_TYPE=$(uname -s)
if [ "$OS_TYPE" != "Darwin" ]; then
    echo "⚠️  Automatic setup is currently only supported on macOS."
    echo ""
    echo "For manual setup on $OS_TYPE, see [README](README.md)"
    echo ""
    exit 0
fi

DIRENV_SETUP_FAILED=0

# Create .envrc file with comments
echo "[1/4] Creating .envrc file..."
cat >.envrc <<EOF
# direnv setup: activates $VENV when entering the repo; unloads it when leaving.
# Automatically configured by scripts/make/set_auto_venv.sh
# To disable: remove the 'source $VENV/bin/activate' line and delete .envrc if no other settings exist.

source $VENV/bin/activate
EOF
echo "✓ Created .envrc for automatic virtual environment activation"

# Check if direnv is installed
echo ""
echo "[2/4] Checking direnv installation..."
if ! command -v direnv &>/dev/null; then
    echo "direnv is not installed. Would you like to install it for automatic environment activation?"
    read -p "Install direnv? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v brew &>/dev/null; then
            echo "Installing direnv via Homebrew..."
            if ! brew install direnv; then
                echo "Failed to install direnv"
                DIRENV_SETUP_FAILED=1
            else
                echo "✓ direnv installed successfully"
            fi
        else
            echo "Error: Homebrew is not installed. Please install Homebrew first: https://brew.sh"
            DIRENV_SETUP_FAILED=1
        fi
    else
        # User chose not to install direnv
        exit 0
    fi
else
    echo "✓ direnv is already installed"
fi

# Configure direnv hook if direnv is available
if command -v direnv &>/dev/null && [ $DIRENV_SETUP_FAILED -eq 0 ]; then
    # Detect default shell RC file based on user's shell
    USER_SHELL=$(basename "${SHELL:-/bin/zsh}")
    if [[ "$USER_SHELL" == "zsh" ]]; then
        DEFAULT_RC="$HOME/.zshrc"
        DEFAULT_HOOK_CMD='eval "$(direnv hook zsh)"'
    elif [[ "$USER_SHELL" == "bash" ]]; then
        DEFAULT_RC="$HOME/.bashrc"
        DEFAULT_HOOK_CMD='eval "$(direnv hook bash)"'
    else
        DEFAULT_RC="$HOME/.zshrc"
        DEFAULT_HOOK_CMD='eval "$(direnv hook zsh)"'
    fi

    echo ""
    echo "[3/4] Configuring direnv hook..."
    if [ -n "$SHELL_RC_ARG" ]; then
        RC_PATH="$SHELL_RC_ARG"
    else
        echo "direnv needs to be configured in your shell RC file."
        read -p "Enter custom RC file path (leave empty for $DEFAULT_RC): " RC_PATH
    fi

    # Use default if empty
    if [ -z "$RC_PATH" ]; then
        RC_PATH="$DEFAULT_RC"
        HOOK_CMD="$DEFAULT_HOOK_CMD"
    else
        # Determine hook command based on RC file name
        if [[ "$RC_PATH" == *"zshrc"* ]]; then
            HOOK_CMD='eval "$(direnv hook zsh)"'
        elif [[ "$RC_PATH" == *"bashrc"* ]] || [[ "$RC_PATH" == *"bash_profile"* ]]; then
            HOOK_CMD='eval "$(direnv hook bash)"'
        else
            HOOK_CMD='eval "$(direnv hook zsh)"'
        fi
    fi

    # Check if RC path exists
    if [ ! -f "$RC_PATH" ]; then
        echo "Error: RC file does not exist: $RC_PATH"
        DIRENV_SETUP_FAILED=1
    else
        # Check if direnv hook is already configured in the specified RC file
        if grep -q "direnv hook" "$RC_PATH"; then
            echo "✓ direnv hook already configured in $RC_PATH"
        else
            # Determine if we need to add a blank line before our direnv configuration
            # Three cases to handle:
            # 1. File ends with newline + non-empty last line → add blank line for separation
            # 2. File ends with newline + empty last line → no blank line needed (already separated)
            # 3. File doesn't end with newline or is empty → add blank line to ensure proper formatting
            if [ -s "$RC_PATH" ] && [ -z "$(tail -c 1 "$RC_PATH")" ]; then
                # File ends with newline, check if last line is empty
                if [ -n "$(tail -n 1 "$RC_PATH")" ]; then
                    # Case 1: Last line is not empty, add blank line
                    NEEDS_BLANK_LINE=true
                else
                    # Case 2: Last line is already empty, don't add another
                    NEEDS_BLANK_LINE=false
                fi
            else
                # Case 3: File doesn't end with newline or is empty, add blank line
                NEEDS_BLANK_LINE=true
            fi

            if ! {
                if [ "$NEEDS_BLANK_LINE" = true ]; then
                    echo "" >>"$RC_PATH"
                fi &&
                    echo "# direnv: automatic virtual environment activation when entering/exiting directories" >>"$RC_PATH" &&
                    echo "# Automatically configured by coreai_opt/scripts/make/set_auto_venv.sh" >>"$RC_PATH" &&
                    echo "$HOOK_CMD" >>"$RC_PATH" &&
                    echo "" >>"$RC_PATH"
            }; then
                echo "Failed to configure direnv hook"
                DIRENV_SETUP_FAILED=1
            else
                echo "✓ Added direnv hook to $RC_PATH"
            fi
        fi
    fi

    # Allow direnv for this repo
    if [ $DIRENV_SETUP_FAILED -eq 0 ]; then
        echo ""
        echo "[4/4] Allowing direnv for this repository..."
        if ! direnv allow; then
            echo "Failed to allow direnv for this repository"
            DIRENV_SETUP_FAILED=1
        else
            echo "✓ direnv has been allowed for this repository"
        fi
    fi
fi

# Clean up if setup failed
if [ $DIRENV_SETUP_FAILED -eq 1 ]; then
    echo ""
    echo "⚠️  Automatic direnv setup failed. Removing .envrc for retry on next run."
    echo "After fixing the issue, run 'make set-auto-venv' again to retry the setup."
    echo "Alternatively, see README.md for manual configuration instructions."
    rm -f .envrc
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Automatic environment activation setup complete!"
echo "=========================================="
echo ""
echo "direnv will now automatically activate $VENV when you enter this directory"
echo "and unload it when you leave."
echo ""
