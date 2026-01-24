#!/usr/bin/env python3
"""CLI entry point for the Popperian Discovery web viewer.

Usage:
    python scripts/run_webviewer.py --db results/discovery.db --port 5000

This starts a Flask development server for browsing discovery results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Run the Popperian Discovery web viewer"
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1, use 0.0.0.0 for external access)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode with auto-reload",
    )

    args = parser.parse_args()

    # Check database file exists
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}")
        sys.exit(1)

    # Import and initialize the app
    from src.web.app import init_app

    app = init_app(str(db_path))

    print(f"Starting web viewer for database: {db_path}")
    print(f"Server running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")

    # Run the Flask development server
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
