#!/usr/bin/env python3

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Open a file or URL in the default web browser.

Supports two modes:

* Default: open the path as-is (file paths are converted to ``file://`` URLs).
* ``--serve``: spin up a local HTTP server on a free OS-assigned port that
  serves the file's parent directory, then open the browser at the file's
  URL on that server. Use this for static-site builds that depend on
  ``fetch()``, ``navigator.clipboard``, or other APIs that browsers
  restrict on ``file://`` origins (e.g. the docs Copy page button).
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

_BROWSER_OPEN_DELAY_SECONDS = 0.3


class _MarkdownAsTextHandler(http.server.SimpleHTTPRequestHandler):
    """Serve ``.md`` files as ``text/plain`` so browsers render them inline.

    Without this override, ``mimetypes`` returns either ``text/markdown`` or
    ``application/octet-stream`` for ``.md``, both of which trigger a download.
    """

    def guess_type(self, path: str) -> str:
        if path.endswith(".md"):
            return "text/plain; charset=utf-8"
        return super().guess_type(path)


def _open_in_browser(path: str) -> int:
    """Open ``path`` as a ``file://`` URL or pass through an existing URL."""
    if path.startswith(("http://", "https://", "file://")):
        url = path
    else:
        file_path = Path(path)
        if not file_path.exists():
            print(f"Error: File not found at {file_path}", file=sys.stderr)
            return 1
        url = file_path.resolve().as_uri()

    print(f"Opening in browser: {url}")
    if webbrowser.open(url, new=2):
        return 0

    print("Failed to open browser automatically.", file=sys.stderr)
    print(f"Please open manually: {url}", file=sys.stderr)
    return 1


def _serve_and_open(path: str) -> int:
    """Serve ``path``'s parent directory over HTTP, then open the browser there.

    Binds to port 0 so the OS picks a free ephemeral port — avoids collisions
    with whatever the user already has running on common ports. Runs in the
    foreground so Ctrl+C cleanly stops the server.
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return 1

    serve_dir = file_path.parent
    page = file_path.name

    handler = functools.partial(
        _MarkdownAsTextHandler,
        directory=str(serve_dir),
    )

    # ThreadingTCPServer so browser-issued parallel asset requests don't
    # serialize through a single connection — noticeable on docs pages with
    # many CSS/JS/SVG fetches. Bind to loopback rather than "" so the server
    # stays reachable only from this machine, matching the localhost URL below.
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler) as httpd:
        httpd.daemon_threads = True
        port = httpd.server_address[1]
        url = f"http://localhost:{port}/{page}"
        print(f"Serving {serve_dir} at http://localhost:{port}/ (Ctrl+C to stop)")
        print(f"Opening {url}")

        # Daemon so Ctrl+C exits cleanly without waiting for a pending
        # browser-open if the user kills the server within the delay window.
        timer = threading.Timer(
            _BROWSER_OPEN_DELAY_SECONDS,
            lambda: webbrowser.open(url, new=2),
        )
        timer.daemon = True
        timer.start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
    return 0


def main() -> int:
    """Parse arguments and dispatch to the open or serve handler."""
    parser = argparse.ArgumentParser(
        description="Open a file or URL in the default web browser.",
    )
    parser.add_argument("path", help="File path or URL to open.")
    parser.add_argument(
        "--serve",
        action="store_true",
        help=(
            "Serve the file's parent directory over HTTP on a free port "
            "instead of opening as file://. Needed for static sites that "
            "rely on fetch()/clipboard APIs."
        ),
    )
    args = parser.parse_args()

    if args.serve:
        return _serve_and_open(args.path)
    return _open_in_browser(args.path)


if __name__ == "__main__":
    sys.exit(main())
