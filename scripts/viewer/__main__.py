"""CLI entrypoint: `python -m scripts.viewer`."""

from __future__ import annotations

import argparse
import logging

from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.viewer",
        description="Local web viewer for Qgentic agent run transcripts.",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind host (default 127.0.0.1; never 0.0.0.0 by default).")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default 8765).")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = create_app()
    print(f"Qgentic viewer → http://{args.host}:{args.port}/  (task_root={app.config['TASK_ROOT']})")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=args.debug)


if __name__ == "__main__":
    main()
