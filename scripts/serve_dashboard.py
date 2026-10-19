"""Serve the dashboard locally, telling the browser to always revalidate.

``python3 -m http.server`` sends no caching headers, so browsers guess, and they
keep ``app.js`` and the data around. After an update the page then pairs the
new ``index.html`` with an old ``app.js`` and a new tab shows up empty. This
server sends ``Cache-Control: no-cache`` on every response, so the browser
checks for a newer file each time (an unchanged file still costs only a 304).

A browser that cached files from the plain server needs one hard reload
(Ctrl+Shift+R) before this takes effect.

Usage::

    uv run python scripts/serve_dashboard.py            # http://127.0.0.1:8765/
    uv run python scripts/serve_dashboard.py --port 9000
"""

from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"


class RevalidatingHandler(SimpleHTTPRequestHandler):
    """Static files, with ``Cache-Control: no-cache`` on every response."""

    def end_headers(self) -> None:
        """Add the revalidation header before the headers are closed."""
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()
    handler = functools.partial(RevalidatingHandler, directory=str(DASHBOARD))
    with ThreadingHTTPServer((args.bind, args.port), handler) as server:
        print(
            f"Serving {DASHBOARD.name}/ at http://{args.bind}:{args.port}/ (no-cache)"
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
