#!/usr/bin/env python3
"""Run the orchestrated discovery loop.

This script runs the full Popperian discovery loop using the orchestration engine:
1. DISCOVERY: Propose and test laws
2. THEOREM: Synthesize theorems from validated laws
3. EXPLANATION: Generate mechanistic explanations
4. PREDICTION: Verify predictions against held-out sets
5. FINALIZE: Generate final report

Supports:
- New runs: Start from scratch
- Resume: Continue from last checkpoint
- Status: Check run progress

Usage:
    # Start a new run
    python scripts/run_orchestrated.py --db popper.db

    # Resume an existing run
    python scripts/run_orchestrated.py --db popper.db --resume --run-id orch_abc123

    # Check status
    python scripts/run_orchestrated.py --db popper.db --status --run-id orch_abc123

    # Limit iterations (for testing)
    python scripts/run_orchestrated.py --db popper.db --max-iterations 10
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables (override=True ensures .env takes precedence over shell)
load_dotenv(override=True)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the orchestrated Popperian discovery loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--db",
        type=str,
        default="popper.db",
        help="Database path (default: popper.db)",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID (for resume/status)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show run status and exit",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum total iterations (overrides config)",
    )

    parser.add_argument(
        "--max-phase-iterations",
        type=int,
        default=None,
        help="Maximum iterations per phase",
    )

    parser.add_argument(
        "--discovery-threshold",
        type=float,
        default=85.0,
        help="Readiness threshold for DISCOVERY -> THEOREM (default: 85)",
    )

    parser.add_argument(
        "--laws-per-iteration",
        type=int,
        default=10,
        help="Laws to propose per iteration (default: 10)",
    )

    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel LLM workers for discovery (default: 1)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use mock LLM client (for testing)",
    )

    return parser


def show_status(repo, run_id: str) -> None:
    """Show status of a run.

    Args:
        repo: Database repository
        run_id: Run ID to check
    """
    from src.orchestration.phases import Phase

    run = repo.get_orchestration_run(run_id)
    if not run:
        print(f"Run {run_id} not found")
        return

    print(f"\n{'='*60}")
    print(f"Run: {run.run_id}")
    print(f"{'='*60}")
    print(f"Status: {run.status}")
    print(f"Current Phase: {run.current_phase}")
    print(f"Total Iterations: {run.total_iterations}")
    print(f"Started: {run.started_at}")
    if run.completed_at:
        print(f"Completed: {run.completed_at}")

    # Get phase iteration counts
    iterations = repo.list_iterations_for_run(run_id, limit=1000)
    phase_counts: dict[str, int] = {}
    for it in iterations:
        phase_counts[it.phase] = phase_counts.get(it.phase, 0) + 1

    print(f"\nPhase Iterations:")
    for phase in ["discovery", "theorem", "explanation", "prediction", "finalize"]:
        count = phase_counts.get(phase, 0)
        print(f"  {phase}: {count}")

    # Get latest readiness
    latest_snap = repo.get_latest_readiness_snapshot(run_id)
    if latest_snap:
        print(f"\nLatest Readiness ({latest_snap.phase}):")
        print(f"  Combined Score: {latest_snap.combined_score:.1f}")
        print(f"  s_pass: {latest_snap.s_pass:.2f}")
        print(f"  s_stability: {latest_snap.s_stability:.2f}")
        print(f"  s_novel_cex: {latest_snap.s_novel_cex:.2f}")
        print(f"  s_harness_health: {latest_snap.s_harness_health:.2f}")

    # Get phase transitions
    transitions = repo.list_phase_transitions(run_id)
    if transitions:
        print(f"\nPhase Transitions:")
        for t in transitions:
            print(f"  {t.from_phase} -> {t.to_phase} ({t.trigger}, score={t.readiness_score:.1f})")

    print()


def run_orchestration(args) -> int:
    """Run the orchestration engine.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    from src.db.repo import Repository
    from src.db.llm_logger import LLMLogger
    from src.discovery.novelty import NoveltyTracker
    from src.harness.config import HarnessConfig
    from src.harness.harness import Harness
    from src.orchestration.engine import OrchestratorEngine
    from src.orchestration.handlers.discovery_handler import (
        DiscoveryPhaseHandler,
        DiscoveryPhaseConfig,
    )
    from src.orchestration.phases import OrchestratorConfig, Phase
    from src.proposer.proposer import LawProposer, ProposerConfig

    # Connect to database
    repo = Repository(args.db)
    repo.connect()

    # Handle status command
    if args.status:
        if not args.run_id:
            # List recent runs
            runs = repo.list_orchestration_runs(limit=10)
            if not runs:
                print("No orchestration runs found")
                return 0

            print(f"\nRecent Runs:")
            print(f"{'Run ID':<20} {'Status':<12} {'Phase':<15} {'Iterations':<12} {'Started'}")
            print("-" * 80)
            for r in runs:
                print(f"{r.run_id:<20} {r.status:<12} {r.current_phase:<15} {r.total_iterations:<12} {r.started_at}")
            return 0
        else:
            show_status(repo, args.run_id)
            return 0

    # Build configuration
    config = OrchestratorConfig(
        discovery_to_theorem_threshold=args.discovery_threshold,
        laws_per_iteration=args.laws_per_iteration,
    )

    if args.max_iterations:
        config.max_total_iterations = args.max_iterations

    if args.max_phase_iterations:
        for phase in Phase:
            config.max_phase_iterations[phase] = args.max_phase_iterations

    # Create LLM loggers for capturing all LLM interactions
    model_name = "gemini-2.5-flash" if not args.mock_llm else "mock"
    proposer_logger = LLMLogger(repo, component="law_proposer", model_name=model_name)
    theorem_logger = LLMLogger(repo, component="theorem_generator", model_name=model_name)
    explanation_logger = LLMLogger(repo, component="explanation_generator", model_name=model_name)

    # Create components
    if args.mock_llm:
        from src.proposer.client import MockGeminiClient
        llm_client = MockGeminiClient()
    else:
        from src.proposer.client import GeminiClient, GeminiConfig
        llm_client = GeminiClient(GeminiConfig())

    proposer = LawProposer(
        client=llm_client,
        config=ProposerConfig(verbose=args.verbose),
        llm_logger=proposer_logger,
    )

    # Seed redundancy filter with existing laws from database
    existing_laws = repo.list_laws(limit=10000)
    seeded_count = 0
    for law_record in existing_laws:
        try:
            law_dict = json.loads(law_record.law_json)
            from src.claims.schema import CandidateLaw
            law = CandidateLaw(**law_dict)
            proposer.add_known_law(law)
            seeded_count += 1
        except Exception:
            pass  # Skip laws that can't be parsed
    if seeded_count > 0 and args.verbose:
        print(f"Seeded redundancy filter with {seeded_count} existing laws")

    harness = Harness(
        config=HarnessConfig(),
        repo=repo,
    )

    novelty_tracker = NoveltyTracker()

    # Create discovery handler
    discovery_config = DiscoveryPhaseConfig(num_workers=args.num_workers)
    discovery_handler = DiscoveryPhaseHandler(
        proposer=proposer,
        harness=harness,
        novelty_tracker=novelty_tracker,
        repo=repo,
        config=discovery_config,
    )

    # Create theorem handler
    from src.orchestration.handlers.theorem_handler import TheoremPhaseHandler
    from src.theorem.generator import TheoremGenerator, TheoremGeneratorConfig

    theorem_generator = TheoremGenerator(
        client=llm_client,
        config=TheoremGeneratorConfig(),
        llm_logger=theorem_logger,
    )
    theorem_handler = TheoremPhaseHandler(
        generator=theorem_generator,
        repo=repo,
    )

    # Create explanation handler
    from src.orchestration.handlers.explanation_handler import (
        ExplanationPhaseHandler,
        ExplanationPhaseConfig,
    )

    explanation_handler = ExplanationPhaseHandler(
        repo=repo,
        llm_client=llm_client,
        config=ExplanationPhaseConfig(min_theorems=3),
        llm_logger=explanation_logger,
    )

    # Create prediction handler
    from src.orchestration.handlers.prediction_handler import (
        PredictionPhaseHandler,
        PredictionPhaseConfig,
    )

    prediction_handler = PredictionPhaseHandler(
        repo=repo,
        config=PredictionPhaseConfig(
            held_out_count=100,
            adversarial_count=50,
        ),
    )

    # Create finalize handler
    from src.orchestration.handlers.finalize_handler import (
        FinalizePhaseHandler,
        FinalizePhaseConfig,
    )

    finalize_handler = FinalizePhaseHandler(
        repo=repo,
        config=FinalizePhaseConfig(),
    )

    # Create engine and register all handlers
    engine = OrchestratorEngine(repo=repo, config=config)
    engine.register_handler(discovery_handler)
    engine.register_handler(theorem_handler)
    engine.register_handler(explanation_handler)
    engine.register_handler(prediction_handler)
    engine.register_handler(finalize_handler)

    # Wire explanation predictor to prediction handler
    # This will be done dynamically when explanation phase completes
    def wire_predictor():
        """Wire the explanation predictor to prediction handler."""
        predictor_fn = explanation_handler.get_predictor()
        if predictor_fn:
            prediction_handler.set_predictor(predictor_fn)

    # Register callback on engine for phase transitions
    engine._wire_predictor = wire_predictor

    # Run
    print(f"\n{'='*60}")
    if args.resume and args.run_id:
        print(f"Resuming run: {args.run_id}")
    else:
        print("Starting new orchestration run")
    print(f"{'='*60}")
    print(f"Database: {args.db}")
    print(f"Max iterations: {config.max_total_iterations}")
    print(f"Discovery threshold: {config.discovery_to_theorem_threshold}")
    print(f"Laws per iteration: {config.laws_per_iteration}")
    if args.num_workers > 1:
        print(f"Parallel workers: {args.num_workers}")
    print()

    try:
        result = engine.run(
            run_id=args.run_id,
            resume=args.resume,
            max_iterations=args.max_iterations,
        )

        print(f"\n{'='*60}")
        print("Orchestration Complete")
        print(f"{'='*60}")
        print(f"Run ID: {result.run_id}")
        print(f"Status: {result.status}")
        print(f"Final Phase: {result.final_phase.value}")
        print(f"Total Iterations: {result.total_iterations}")

        if result.phase_iterations:
            print(f"\nPhase Iterations:")
            for phase, count in result.phase_iterations.items():
                phase_name = phase.value if hasattr(phase, 'value') else str(phase)
                print(f"  {phase_name}: {count}")

        if result.final_readiness:
            print(f"\nFinal Readiness:")
            print(f"  Combined Score: {result.final_readiness.combined_score * 100:.1f}")

        if result.error:
            print(f"\nError: {result.error}")
            return 1

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130

    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    finally:
        repo.close()


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    return run_orchestration(args)


if __name__ == "__main__":
    sys.exit(main())
