"""Render a single run as a self-contained HTML file.

Usage:
    python -m scripts.viewer.export <slug> <run_id> [-o trace.html]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .server import create_app


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.viewer.export")
    ap.add_argument("slug")
    ap.add_argument("run_id")
    ap.add_argument("-o", "--output", default=None,
                    help="Output HTML path (default <slug>_<run_id>.html).")
    args = ap.parse_args()

    out = Path(args.output) if args.output else Path(f"{args.slug}_{args.run_id}.html")
    app = create_app()
    client = app.test_client()
    resp = client.get(f"/export/{args.slug}/{args.run_id}")
    if resp.status_code != 200:
        print(f"export failed: HTTP {resp.status_code}", file=sys.stderr)
        sys.exit(1)
    out.write_bytes(resp.data)
    print(f"wrote {out} ({len(resp.data)} bytes)")


if __name__ == "__main__":
    main()
