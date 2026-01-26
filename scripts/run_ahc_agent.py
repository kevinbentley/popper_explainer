#!/usr/bin/env python3
"""Entry point for the AHC-DS agent.

Usage:
    # Start new session
    python scripts/run_ahc_agent.py --db high_context.db

    # Resume existing session
    python scripts/run_ahc_agent.py --db high_context.db --resume SESSION_ID

    # With options
    python scripts/run_ahc_agent.py --db high_context.db --max-turns 1000 --model gemini-2.5-flash
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.ahc.agent import AgentLoop, AgentConfig
from src.ahc.llm import FunctionCallingClient, GeminiConfig


def setup_logging(verbose: bool = False) -> None:
    """Configure logging.

    Args:
        verbose: If True, set DEBUG level
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the AHC-DS discovery agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a new session
  python scripts/run_ahc_agent.py --db high_context.db

  # Resume an existing session
  python scripts/run_ahc_agent.py --db high_context.db --resume abc123

  # Run with limited turns (for testing)
  python scripts/run_ahc_agent.py --db high_context.db --max-turns 100
        """,
    )

    parser.add_argument(
        "--db",
        default="high_context.db",
        help="Path to SQLite database (default: high_context.db)",
    )

    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="Resume an existing session by ID",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=10000,
        help="Maximum number of turns (default: 10000)",
    )

    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="LLM model to use (default: gemini-2.5-flash)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--accuracy-target",
        type=float,
        default=1.0,
        help="Target prediction accuracy (default: 1.0 = 100%%)",
    )

    parser.add_argument(
        "--predictions-target",
        type=int,
        default=5000,
        help="Number of predictions for validation (default: 5000)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize but don't run (for testing)",
    )

    parser.add_argument(
        "--show-conversation",
        action="store_true",
        help="Print LLM interactions to console",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Create config
    config = AgentConfig(
        model=args.model,
        max_turns=args.max_turns,
        db_path=args.db,
        seed=args.seed,
        accuracy_target=args.accuracy_target,
        predictions_target=args.predictions_target,
    )

    # Create LLM client
    try:
        llm_config = GeminiConfig(model=args.model)
        llm_client = FunctionCallingClient(config=llm_config)
        logger.info(f"Using LLM model: {args.model}")
    except ValueError as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        logger.error("Make sure GEMINI_API_KEY or GOOGLE_API_KEY is set")
        return 1

    # Create agent
    agent = AgentLoop(
        config=config,
        llm_client=llm_client,
        show_conversation=args.show_conversation,
    )

    if args.dry_run:
        logger.info("Dry run mode - initialization complete")
        return 0

    try:
        if args.resume:
            logger.info(f"Resuming session: {args.resume}")
            session_id = agent.resume(args.resume)
        else:
            logger.info("Starting new AHC session")
            session_id = agent.run()

        logger.info(f"Session completed: {session_id}")
        return 0

    except KeyboardInterrupt:
        logger.info("Session interrupted by user")
        return 0

    except Exception as e:
        logger.exception(f"Session failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
