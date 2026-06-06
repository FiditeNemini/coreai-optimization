#!/usr/bin/env python3

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Extract a specific version's notes from CHANGELOG.md.

Usage:
    python extract_changelog.py <version> [changelog-file]
    python extract_changelog.py 0.0.1
    python extract_changelog.py Unreleased CHANGELOG.md
"""

import re
import sys
from pathlib import Path


def extract_version(version: str, changelog_path: Path = Path("CHANGELOG.md")) -> str:
    """
    Extract release notes for a specific version from a Keep a Changelog formatted file.

    Args:
        version: The version to extract (e.g., "0.0.1" or "Unreleased")
        changelog_path: Path to the changelog file

    Returns:
        The release notes content for the specified version

    Raises:
        FileNotFoundError: If changelog file doesn't exist
        ValueError: If version is not found in changelog
    """
    if not changelog_path.exists():
        raise FileNotFoundError(f"Changelog not found: {changelog_path}")

    content = changelog_path.read_text()
    lines = content.splitlines()

    # Pattern to match version headers like "## [0.0.1] - 2026-01-27"
    version_pattern = re.compile(r"^## \[([^\]]+)\]")
    # Pattern to match link references like "[0.0.1]: https://..."
    link_pattern = re.compile(r"^\[.*\]:")

    in_section = False
    section_lines = []

    for line in lines:
        # Check if this is a version header
        match = version_pattern.match(line)
        if match:
            found_version = match.group(1)

            # If we're already in our target section, we've hit the next version
            if in_section:
                break

            # Check if this is our target version
            if found_version == version:
                in_section = True
                continue  # Skip the header line itself

        # Stop at link references section
        if link_pattern.match(line) and in_section:
            break

        # Collect lines while in the target section
        if in_section:
            section_lines.append(line)

    if not section_lines:
        raise ValueError(f"Version '{version}' not found in {changelog_path}")

    # Join lines and strip leading and trailing whitespace
    return "\n".join(section_lines).lstrip().rstrip()


def main():
    """Main entry point for CLI usage."""
    if len(sys.argv) < 2:
        print("Usage: extract_changelog.py <version> [changelog-file]", file=sys.stderr)
        print("Example: extract_changelog.py 0.0.1", file=sys.stderr)
        sys.exit(1)

    version = sys.argv[1]
    changelog_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("CHANGELOG.md")

    try:
        notes = extract_version(version, changelog_path)
        print(notes)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
