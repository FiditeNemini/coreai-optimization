#!/usr/bin/env python3

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Render LICENSE from a Jinja template, filling ``year`` with the current copyright span.

The start year is parsed from the first ``Copyright YYYY`` match in the existing
LICENSE; if missing, the current year is used. Output is ``YYYY`` when start equals
current, otherwise ``YYYY-YYYY``.
"""

import argparse
import datetime
import re
import sys
from pathlib import Path

from jinja2 import Template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--license-file", required=True, type=Path)
    args = parser.parse_args()

    try:
        existing = args.license_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = None

    current_year = datetime.datetime.now(datetime.UTC).year
    match = re.search(r"Copyright (\d{4})", existing) if existing else None
    start_year = int(match.group(1)) if match else current_year
    year_str = str(current_year) if start_year == current_year else f"{start_year}-{current_year}"

    template_text = args.template.read_text(encoding="utf-8")
    new_content = Template(template_text, keep_trailing_newline=True).render(year=year_str)
    if existing != new_content:
        args.license_file.write_text(new_content, encoding="utf-8", newline="\n")
        print(f"Updated {args.license_file} (year: {year_str})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
