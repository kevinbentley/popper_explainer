#!/usr/bin/env python3
"""Run the law discovery loop.

This script orchestrates the full discovery cycle:
1. Propose laws using the LLM
2. Test laws with the harness
3. Record results
4. Repeat

Supports parallel workers (--workers N) where each worker independently
proposes and tests laws, synchronizing via the database.

Usage:
    python scripts/run_discovery.py [--iterations N] [--laws-per-iter K] [--workers W]
"""

import argparse
import json
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TranscriptLogger:
    """Logs all output to both console and transcript file. Thread-safe."""

    def __init__(self, transcript_path: Path):
        self.transcript_path = transcript_path
        self.file: TextIO | None = None
        self._lock = threading.Lock()

    def __enter__(self):
        self.file = open(self.transcript_path, "w")
        return self

    def __exit__(self, *args):
        if self.file:
            self.file.close()

    def log(self, message: str = "", worker_id: int | None = None):
        """Log message to both console and file. Thread-safe."""
        prefix = f"[W{worker_id}] " if worker_id is not None else ""
        full_message = f"{prefix}{message}"
        with self._lock:
            print(full_message)
            if self.file:
                self.file.write(full_message + "\n")
                self.file.flush()

    def log_json(self, label: str, data: dict):
        """Log JSON data to transcript only (not console). Thread-safe."""
        with self._lock:
            if self.file:
                self.file.write(f"\n=== {label} ===\n")
                self.file.write(json.dumps(data, indent=2))
                self.file.write("\n")
                self.file.flush()


@dataclass
class SharedDiscoveryState:
    """Thread-safe shared state for parallel discovery workers."""

    iterations_completed: int = 0
    max_iterations: int = 0
    total_proposed: int = 0
    total_provisional: int = 0
    total_promoted: int = 0
    total_falsified: int = 0
    total_unknown: int = 0
    total_rejected: int = 0
    total_redundant: int = 0
    unique_accepted_claims: set = field(default_factory=set)
    escalation_1_runs: int = 0
    escalation_2_runs: int = 0
    escalation_revoked: int = 0
    escalation_downgraded: int = 0
    # Escalation tracking
    last_escalation_1_iteration: int = 0
    last_escalation_2_iteration: int = 0
    _escalation_running: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def increment_iteration(self) -> int:
        """Atomically increment and return the new iteration count."""
        with self._lock:
            self.iterations_completed += 1
            return self.iterations_completed

    def can_continue(self) -> bool:
        """Check if more iterations should be run."""
        with self._lock:
            return self.iterations_completed < self.max_iterations

    def record_proposal_stats(
        self,
        proposed: int = 0,
        provisional: int = 0,
        falsified: int = 0,
        unknown: int = 0,
        rejected: int = 0,
        redundant: int = 0,
        claims: set | None = None,
    ):
        """Thread-safe update of proposal statistics."""
        with self._lock:
            self.total_proposed += proposed
            self.total_provisional += provisional
            self.total_falsified += falsified
            self.total_unknown += unknown
            self.total_rejected += rejected
            self.total_redundant += redundant
            if claims:
                self.unique_accepted_claims.update(claims)

    def record_escalation_stats(
        self,
        esc_1_runs: int = 0,
        esc_2_runs: int = 0,
        promoted: int = 0,
        revoked: int = 0,
        downgraded: int = 0,
    ):
        """Thread-safe update of escalation statistics."""
        with self._lock:
            self.escalation_1_runs += esc_1_runs
            self.escalation_2_runs += esc_2_runs
            self.total_promoted += promoted
            self.escalation_revoked += revoked
            self.escalation_downgraded += downgraded

    def get_stats(self) -> dict:
        """Get a snapshot of current stats."""
        with self._lock:
            return {
                "total_proposed": self.total_proposed,
                "total_provisional": self.total_provisional,
                "total_promoted": self.total_promoted,
                "total_falsified": self.total_falsified,
                "total_unknown": self.total_unknown,
                "total_rejected": self.total_rejected,
                "total_redundant": self.total_redundant,
                "unique_accepted_claims": list(self.unique_accepted_claims),
                "escalation_1_runs": self.escalation_1_runs,
                "escalation_2_runs": self.escalation_2_runs,
                "escalation_revoked": self.escalation_revoked,
                "escalation_downgraded": self.escalation_downgraded,
            }

    def try_start_escalation(self, iteration: int, config: "EscalationPolicyConfig") -> tuple[bool, bool, bool]:
        """Try to start escalation if not already running and due.

        Returns:
            Tuple of (acquired_lock, should_run_esc1, should_run_esc2)
        """
        with self._lock:
            if self._escalation_running:
                return False, False, False

            # Check if escalation is due
            iterations_since_esc1 = iteration - self.last_escalation_1_iteration
            iterations_since_esc2 = iteration - self.last_escalation_2_iteration

            run_esc_1 = (
                self.total_provisional >= config.min_accepted_laws
                and iterations_since_esc1 >= config.escalation_1_interval
            )
            run_esc_2 = (
                self.total_provisional >= config.min_accepted_laws
                and iterations_since_esc2 >= config.escalation_2_interval
            )

            if run_esc_1 or run_esc_2:
                self._escalation_running = True
                return True, run_esc_1, run_esc_2

            return False, False, False

    def finish_escalation(self, iteration: int, ran_esc_1: bool, ran_esc_2: bool):
        """Mark escalation as complete and update iteration tracking."""
        with self._lock:
            if ran_esc_1:
                self.last_escalation_1_iteration = iteration
            if ran_esc_2:
                self.last_escalation_2_iteration = iteration
            self._escalation_running = False


class UntestableLawsLog:
    """Logs untestable laws to a JSON Lines file for analysis."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.file: TextIO | None = None
        self.count = 0

    def __enter__(self):
        self.file = open(self.log_path, "a")  # Append mode
        return self

    def __exit__(self, *args):
        if self.file:
            self.file.close()

    def log(self, law: "CandidateLaw", verdict: "LawVerdict", error_msg: str | None = None):
        """Log an untestable law.

        Args:
            law: The candidate law that couldn't be tested
            verdict: The verdict with reason code
            error_msg: Optional error message
        """
        if self.file is None:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "law_id": law.law_id,
            "template": law.template.value,
            "claim": law.claim,
            "claim_ast": law.claim_ast,
            "observables": [{"name": o.name, "expr": o.expr} for o in law.observables],
            "preconditions": [{"lhs": p.lhs, "op": p.op.value, "rhs": p.rhs} for p in law.preconditions],
            "reason_code": verdict.reason_code.value if verdict.reason_code else "unknown",
            "notes": verdict.notes,
            "error": error_msg,
        }

        self.file.write(json.dumps(entry) + "\n")
        self.file.flush()
        self.count += 1

from src.claims.ast_schema import ast_to_string
from src.claims.schema import CandidateLaw
from src.db.discovery_persistence import DiscoveryPersistence
from src.harness.config import HarnessConfig
from src.harness.escalation import (
    EscalationLevel,
    EscalationPolicyConfig,
    EscalationState,
    FlipType,
    get_escalation_decisions,
    get_laws_for_escalation_1,
    get_promotion_status,
    run_escalation,
)
from src.harness.harness import Harness
from src.harness.verdict import LawVerdict
from src.proposer.client import GeminiClient, GeminiConfig
from src.proposer.memory import DiscoveryMemory
from src.proposer.prompt import UniverseContract
from src.proposer.proposer import LawProposer, ProposerConfig, ProposalRequest


def create_client() -> GeminiClient:
    """Create Gemini client with appropriate settings."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not set in environment")
        print("Please set it in .env file or environment")
        sys.exit(1)

    config = GeminiConfig(
        api_key=api_key,
        model="gemini-2.5-flash",  # Latest flash model with thinking
        temperature=0.7,
        max_output_tokens=65535,  # Maximum budget for JSON output
        top_p=0.95,
        top_k=40,
        thinking_budget=None,  # Let model decide (or set e.g. 8192)
        json_mode=True,  # Request JSON output
    )
    return GeminiClient(config)


def create_harness() -> Harness:
    """Create test harness with appropriate settings."""
    config = HarnessConfig(
        seed=42,
        max_cases=200,
        min_cases_used_for_pass=40,
        default_T=50,
        max_T=100,
        enable_adversarial_search=True,
        adversarial_budget=500,
        enable_counterexample_minimization=True,
    )
    return Harness(config)


def create_proposer(client: GeminiClient, verbose: bool = False) -> LawProposer:
    """Create law proposer."""
    contract = UniverseContract()
    config = ProposerConfig(
        max_token_budget=16000,  # Larger context for thinking
        include_counterexamples=True,
        strict_parsing=False,
        add_to_redundancy_filter=True,
        verbose=verbose,
    )
    return LawProposer(client=client, contract=contract, config=config)


def format_verdict(verdict: LawVerdict) -> str:
    """Format a verdict for display."""
    status_emoji = {"PASS": "✓", "FAIL": "✗", "UNKNOWN": "?"}
    emoji = status_emoji.get(verdict.status, "?")

    result = f"  {emoji} {verdict.status}"

    if verdict.status == "PASS":
        result += f" (cases={verdict.power_metrics.cases_used}, coverage={verdict.power_metrics.coverage_score:.2f})"
    elif verdict.status == "FAIL" and verdict.counterexample:
        cx = verdict.counterexample
        result += f" @ t={cx.t_fail}, state='{cx.initial_state}'"
    elif verdict.status == "UNKNOWN" and verdict.reason_code:
        result += f" ({verdict.reason_code.value})"

    return result


def discovery_worker(
    worker_id: int,
    db_path: Path,
    laws_per_iteration: int,
    shared_state: SharedDiscoveryState,
    logger: TranscriptLogger,
    verbose: bool = False,
    enable_escalation: bool = True,
    escalation_config: EscalationPolicyConfig | None = None,
) -> None:
    """Run discovery cycles in a worker thread.

    Each worker independently:
    1. Loads current state from database
    2. Proposes laws via LLM
    3. Tests laws with harness
    4. Saves results to database
    5. Repeats until max iterations reached

    Args:
        worker_id: Unique identifier for this worker
        db_path: Path to shared SQLite database
        laws_per_iteration: Laws to propose per cycle
        shared_state: Thread-safe shared state
        logger: Thread-safe logger
        verbose: Whether to log verbose output
    """
    # Each worker gets its own components
    client = create_client()
    harness = create_harness()
    proposer = create_proposer(client, verbose=verbose)

    # Set capabilities
    capabilities = {
        "primitive_observables": ["count(symbol)", "grid_length"],
        "transforms": ["mirror_swap", "shift_k", "swap_only", "mirror_only"],
        "generator_families": [
            "random_density_sweep",
            "constrained_pair_interactions",
            "edge_wrapping_cases",
            "symmetry_metamorphic_suite",
            "adversarial_mutation_search",
        ],
    }

    logger.log(f"Worker started", worker_id=worker_id)

    while shared_state.can_continue():
        iteration = shared_state.increment_iteration()
        if iteration > shared_state.max_iterations:
            break

        logger.log(f"Starting iteration {iteration}/{shared_state.max_iterations}", worker_id=worker_id)

        # Each cycle gets a fresh DB connection and memory
        try:
            with DiscoveryPersistence(db_path) as persistence:
                # Build fresh memory from current DB state
                memory = DiscoveryMemory()
                memory.set_capabilities(capabilities)

                # Load all current laws from DB
                try:
                    accepted_laws, falsified_laws = persistence.load_existing_laws()

                    for law in accepted_laws:
                        memory.record_evaluation(law, LawVerdict(
                            law_id=law.law_id,
                            status="PASS",
                        ))
                        proposer.add_known_law(law)

                    for law, cx in falsified_laws:
                        from src.harness.verdict import Counterexample
                        verdict = LawVerdict(
                            law_id=law.law_id,
                            status="FAIL",
                            counterexample=cx,
                        )
                        memory.record_evaluation(law, verdict)
                        proposer.add_known_law(law)

                    logger.log(f"Loaded {len(accepted_laws)} accepted, {len(falsified_laws)} falsified from DB", worker_id=worker_id)
                except Exception as e:
                    logger.log(f"Warning loading from DB: {e}", worker_id=worker_id)

                # Propose laws
                logger.log(f"Proposing {laws_per_iteration} laws...", worker_id=worker_id)
                try:
                    request = ProposalRequest(
                        count=laws_per_iteration,
                        temperature=0.7,
                    )
                    batch = proposer.propose(memory.get_snapshot(), request)
                except Exception as e:
                    logger.log(f"Proposal error: {e}", worker_id=worker_id)
                    continue

                logger.log(f"Proposed: {len(batch.laws)}, Rejected: {len(batch.rejections)}, Redundant: {len(batch.redundant)}", worker_id=worker_id)

                # Track stats for this cycle
                cycle_proposed = 0
                cycle_provisional = 0
                cycle_falsified = 0
                cycle_unknown = 0
                cycle_claims: set[str] = set()

                # Test each law
                for law in batch.laws:
                    try:
                        # Check if law already exists in DB (another worker may have tested it)
                        if persistence.law_exists(law):
                            logger.log(f"  {law.law_id}: Already in DB, skipping", worker_id=worker_id)
                            continue

                        verdict = harness.evaluate(law)
                    except Exception as e:
                        logger.log(f"  {law.law_id}: Error - {e}", worker_id=worker_id)
                        verdict = LawVerdict(
                            law_id=law.law_id,
                            status="UNKNOWN",
                            notes=[f"Evaluation error: {e}"],
                        )

                    # Log result
                    status_char = {"PASS": "✓", "FAIL": "✗", "UNKNOWN": "?"}.get(verdict.status, "?")
                    logger.log(f"  {law.law_id}: {status_char} {verdict.status}", worker_id=worker_id)

                    cycle_proposed += 1

                    if verdict.status == "PASS":
                        cycle_provisional += 1
                        claim_repr = ast_to_string(law.claim_ast) if law.claim_ast else law.claim
                        cycle_claims.add(claim_repr)
                        try:
                            persistence.save_accepted(law, verdict)
                        except Exception as e:
                            logger.log(f"    Save error: {e}", worker_id=worker_id)
                    elif verdict.status == "FAIL":
                        cycle_falsified += 1
                        try:
                            persistence.save_falsified(law, verdict)
                        except Exception as e:
                            logger.log(f"    Save error: {e}", worker_id=worker_id)
                    else:
                        cycle_unknown += 1

                # Update shared stats
                shared_state.record_proposal_stats(
                    proposed=cycle_proposed,
                    provisional=cycle_provisional,
                    falsified=cycle_falsified,
                    unknown=cycle_unknown,
                    rejected=len(batch.rejections),
                    redundant=len(batch.redundant),
                    claims=cycle_claims,
                )

                # Check if this worker should run periodic escalation
                if enable_escalation and escalation_config:
                    acquired, run_esc_1, run_esc_2 = shared_state.try_start_escalation(
                        iteration, escalation_config
                    )
                    if acquired:
                        try:
                            if run_esc_1:
                                logger.log(f"Running periodic escalation_1...", worker_id=worker_id)
                                esc_result = run_escalation(
                                    EscalationLevel.ESCALATION_1,
                                    persistence.repo,
                                )
                                if esc_result.laws_tested > 0:
                                    shared_state.record_escalation_stats(
                                        esc_1_runs=1,
                                        promoted=esc_result.stable_count,
                                        revoked=esc_result.revoked_count,
                                        downgraded=esc_result.downgraded_count,
                                    )
                                    logger.log(
                                        f"  Escalation_1: tested={esc_result.laws_tested}, "
                                        f"promoted={esc_result.stable_count}, revoked={esc_result.revoked_count}",
                                        worker_id=worker_id
                                    )

                            if run_esc_2:
                                logger.log(f"Running periodic escalation_2...", worker_id=worker_id)
                                esc_result = run_escalation(
                                    EscalationLevel.ESCALATION_2,
                                    persistence.repo,
                                )
                                if esc_result.laws_tested > 0:
                                    shared_state.record_escalation_stats(
                                        esc_2_runs=1,
                                        revoked=esc_result.revoked_count,
                                        downgraded=esc_result.downgraded_count,
                                    )
                                    logger.log(
                                        f"  Escalation_2: tested={esc_result.laws_tested}, "
                                        f"stable={esc_result.stable_count}, revoked={esc_result.revoked_count}",
                                        worker_id=worker_id
                                    )
                        except Exception as e:
                            logger.log(f"Escalation error: {e}", worker_id=worker_id)
                        finally:
                            shared_state.finish_escalation(iteration, run_esc_1, run_esc_2)

        except Exception as e:
            logger.log(f"Cycle error: {e}", worker_id=worker_id)
            import traceback
            traceback.print_exc()

    logger.log(f"Worker finished", worker_id=worker_id)


def run_discovery_parallel(
    iterations: int,
    laws_per_iteration: int,
    workers: int,
    output_dir: Path,
    db_path: Path,
    verbose: bool = False,
    enable_escalation: bool = True,
    escalation_config: EscalationPolicyConfig | None = None,
) -> dict:
    """Run parallel discovery with multiple workers.

    Args:
        iterations: Total number of discovery iterations across all workers
        laws_per_iteration: Laws to propose per iteration
        workers: Number of parallel workers
        output_dir: Directory to save results
        db_path: Path to SQLite database
        verbose: Whether to print verbose output
        enable_escalation: Whether to run escalation after all iterations
        escalation_config: Escalation policy configuration

    Returns:
        Summary statistics
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_path = output_dir / f"transcript_{timestamp}.txt"

    # Initialize escalation config
    if escalation_config is None:
        escalation_config = EscalationPolicyConfig()

    # Create shared state
    shared_state = SharedDiscoveryState(max_iterations=iterations)

    with TranscriptLogger(transcript_path) as logger:
        logger.log("=" * 60)
        logger.log("PARALLEL POPPERIAN LAW DISCOVERY")
        logger.log("=" * 60)
        logger.log(f"Timestamp: {timestamp}")
        logger.log(f"Total iterations: {iterations}")
        logger.log(f"Laws per iteration: {laws_per_iteration}")
        logger.log(f"Workers: {workers}")
        logger.log(f"Database: {db_path}")
        logger.log(f"Escalation: {'enabled' if enable_escalation else 'disabled'}")
        if enable_escalation:
            logger.log(f"  Trigger: {escalation_config.min_accepted_laws} laws")
            logger.log(f"  Escalation_1: every {escalation_config.escalation_1_interval} iterations")
            logger.log(f"  Escalation_2: every {escalation_config.escalation_2_interval} iterations")
        logger.log()

        # Ensure DB exists with schema
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with DiscoveryPersistence(db_path) as persistence:
            summary = persistence.get_summary()
            logger.log(f"Initial DB state: {summary}")
        logger.log()

        # Run workers in parallel
        logger.log(f"Starting {workers} parallel workers...")
        logger.log()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    discovery_worker,
                    worker_id=i,
                    db_path=db_path,
                    laws_per_iteration=laws_per_iteration,
                    shared_state=shared_state,
                    logger=logger,
                    verbose=verbose,
                    enable_escalation=enable_escalation,
                    escalation_config=escalation_config,
                )
                for i in range(workers)
            ]

            # Wait for all workers to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.log(f"Worker error: {e}")

        logger.log()
        logger.log("=" * 60)
        logger.log("ALL WORKERS COMPLETE")
        logger.log("=" * 60)

        # Run final escalation on any remaining provisional laws
        if enable_escalation:
            logger.log()
            logger.log("Running final escalation (catching any remaining provisional laws)...")
            with DiscoveryPersistence(db_path) as persistence:
                try:
                    # Run escalation_1 on any laws not yet escalated
                    esc_result = run_escalation(
                        EscalationLevel.ESCALATION_1,
                        persistence.repo,
                    )
                    if esc_result.laws_tested > 0:
                        shared_state.record_escalation_stats(
                            esc_1_runs=1,
                            promoted=esc_result.stable_count,
                            revoked=esc_result.revoked_count,
                            downgraded=esc_result.downgraded_count,
                        )
                        logger.log(f"Final escalation_1: tested={esc_result.laws_tested}, promoted={esc_result.stable_count}, revoked={esc_result.revoked_count}")
                    else:
                        logger.log("No provisional laws remaining for escalation")
                except Exception as e:
                    logger.log(f"Escalation error: {e}")

        # Final summary
        stats = shared_state.get_stats()
        logger.log()
        logger.log("=" * 60)
        logger.log("DISCOVERY COMPLETE")
        logger.log("=" * 60)
        logger.log(f"Total laws proposed: {stats['total_proposed']}")
        logger.log(f"  Provisional (baseline PASS): {stats['total_provisional']}")
        logger.log(f"  Promoted (escalation_1 PASS): {stats['total_promoted']}")
        logger.log(f"  Falsified (FAIL): {stats['total_falsified']}")
        logger.log(f"  Unknown: {stats['total_unknown']}")
        logger.log(f"Parse rejections: {stats['total_rejected']}")
        logger.log(f"Redundant filtered: {stats['total_redundant']}")
        logger.log(f"Unique claims: {len(stats['unique_accepted_claims'])}")

        # Get final DB summary
        with DiscoveryPersistence(db_path) as persistence:
            db_summary = persistence.get_summary()
            logger.log(f"\nDatabase summary ({db_path}):")
            logger.log(f"  Total PASS: {db_summary.get('PASS', 0)}")
            logger.log(f"  Total FAIL: {db_summary.get('FAIL', 0)}")
            logger.log(f"  Total UNKNOWN: {db_summary.get('UNKNOWN', 0)}")

        # Save results
        output_file = output_dir / f"discovery_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "iterations": iterations,
                "workers": workers,
                "laws_per_iteration": laws_per_iteration,
                "stats": stats,
            }, f, indent=2)

        logger.log(f"\nResults saved to: {output_file}")
        logger.log(f"Transcript saved to: {transcript_path}")

    return stats


def run_discovery(
    iterations: int = 5,
    laws_per_iteration: int = 5,
    output_dir: Path | None = None,
    verbose: bool = False,
    db_path: Path | None = None,
    enable_escalation: bool = True,
    escalation_config: EscalationPolicyConfig | None = None,
    workers: int = 1,
) -> dict:
    """Run the discovery loop.

    Args:
        iterations: Number of discovery iterations
        laws_per_iteration: Laws to propose per iteration
        output_dir: Directory to save results
        verbose: Whether to print full prompts and responses
        db_path: Path to SQLite database for persistence (default: results/discovery.db)
        enable_escalation: Whether to run power escalation during discovery
        escalation_config: Configuration for escalation policy (uses defaults if None)
        workers: Number of parallel workers (default: 1 for single-threaded)

    Returns:
        Summary statistics
    """
    # Set up output directory
    output_dir = Path(output_dir) if output_dir else Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default database path
    if db_path is None:
        db_path = output_dir / "discovery.db"

    # Use parallel mode if multiple workers requested
    if workers > 1:
        return run_discovery_parallel(
            iterations=iterations,
            laws_per_iteration=laws_per_iteration,
            workers=workers,
            output_dir=output_dir,
            db_path=db_path,
            verbose=verbose,
            enable_escalation=enable_escalation,
            escalation_config=escalation_config,
        )

    # Single-threaded mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_path = output_dir / f"transcript_{timestamp}.txt"

    untestable_path = output_dir / "untestable_laws.jsonl"

    with TranscriptLogger(transcript_path) as logger:
        with UntestableLawsLog(untestable_path) as untestable_log:
            with DiscoveryPersistence(db_path) as persistence:
                return _run_discovery_with_logging(
                    iterations, laws_per_iteration, output_dir, timestamp,
                    logger, verbose, untestable_log, persistence,
                    enable_escalation, escalation_config
                )


def _run_discovery_with_logging(
    iterations: int,
    laws_per_iteration: int,
    output_dir: Path,
    timestamp: str,
    logger: TranscriptLogger,
    verbose: bool = False,
    untestable_log: UntestableLawsLog | None = None,
    persistence: DiscoveryPersistence | None = None,
    enable_escalation: bool = True,
    escalation_config: EscalationPolicyConfig | None = None,
) -> dict:
    """Run discovery with transcript logging."""
    # Initialize escalation config early so we can log its values
    if escalation_config is None:
        escalation_config = EscalationPolicyConfig()

    logger.log("=" * 60)
    logger.log("POPPERIAN LAW DISCOVERY")
    logger.log("=" * 60)
    logger.log(f"Timestamp: {timestamp}")
    logger.log(f"Iterations: {iterations}")
    logger.log(f"Laws per iteration: {laws_per_iteration}")
    logger.log(f"Verbose: {verbose}")
    if persistence:
        logger.log(f"Database: {persistence.db_path}")
    logger.log(f"Escalation: {'enabled' if enable_escalation else 'disabled'}")
    if enable_escalation:
        logger.log(f"  Trigger: {escalation_config.min_accepted_laws} laws, {escalation_config.min_template_families} templates")
        logger.log(f"  Escalation_1: every {escalation_config.escalation_1_interval} iterations (last {escalation_config.escalation_1_window} laws)")
        logger.log(f"  Escalation_2: every {escalation_config.escalation_2_interval} iterations or novelty < {escalation_config.novelty_threshold}")
    logger.log()

    # Initialize components
    logger.log("Initializing...")
    client = create_client()
    harness = create_harness()
    proposer = create_proposer(client, verbose=verbose)
    memory = DiscoveryMemory()

    # Set capabilities in memory
    memory.set_capabilities({
        "primitive_observables": ["count(symbol)", "grid_length"],
        "transforms": ["mirror_swap", "shift_k", "swap_only", "mirror_only"],
        "generator_families": [
            "random_density_sweep",
            "constrained_pair_interactions",
            "edge_wrapping_cases",
            "symmetry_metamorphic_suite",
            "adversarial_mutation_search",
        ],
    })

    # Track results
    all_results = []
    stats = {
        "total_proposed": 0,
        "total_provisional": 0,  # Passed baseline (not yet escalated)
        "total_promoted": 0,  # Passed escalation_1 (fully accepted)
        "total_falsified": 0,
        "total_unknown": 0,
        "total_rejected": 0,
        "total_redundant": 0,
        "unique_accepted_claims": set(),
        "escalation_1_runs": 0,
        "escalation_2_runs": 0,
        "escalation_revoked": 0,
        "escalation_downgraded": 0,
    }

    # Initialize escalation state
    escalation_state = EscalationState()

    # Track accepted laws for escalation scope (initialized here so database load can populate it)
    accepted_laws_list: list[CandidateLaw] = []

    # Load existing laws from database
    if persistence:
        logger.log("Loading existing laws from database...")
        try:
            accepted_laws, falsified_laws = persistence.load_existing_laws()
            logger.log(f"  Loaded {len(accepted_laws)} accepted laws")
            logger.log(f"  Loaded {len(falsified_laws)} falsified laws")

            # Add to memory so proposer knows about them
            for law in accepted_laws:
                # Create a synthetic PASS verdict
                memory.record_evaluation(law, LawVerdict(
                    law_id=law.law_id,
                    status="PASS",
                ))
                # Add to proposer's redundancy filter
                proposer.add_known_law(law)
                # Track for escalation
                accepted_laws_list.append(law)

            for law, cx in falsified_laws:
                # Create a synthetic FAIL verdict
                from src.harness.verdict import Counterexample
                verdict = LawVerdict(
                    law_id=law.law_id,
                    status="FAIL",
                    counterexample=cx,
                )
                memory.record_evaluation(law, verdict)
                # Add to proposer's redundancy filter
                proposer.add_known_law(law)

            logger.log(f"  Memory initialized: {memory.stats}")
        except Exception as e:
            logger.log(f"  Warning: Could not load existing laws: {e}")
            import traceback
            logger.log_json("LOAD ERROR", {"error": str(e), "traceback": traceback.format_exc()})
        logger.log()

    logger.log("Starting discovery loop...")
    logger.log()

    for iteration in range(1, iterations + 1):
        logger.log(f"{'='*60}")
        logger.log(f"ITERATION {iteration}/{iterations}")
        logger.log(f"{'='*60}")

        # Propose laws
        logger.log(f"\n[1] Proposing {laws_per_iteration} laws...")
        try:
            request = ProposalRequest(
                count=laws_per_iteration,
                temperature=0.7,
            )
            batch = proposer.propose(memory.get_snapshot(), request)

            # Log full batch details to transcript
            logger.log_json(f"ITERATION {iteration} PROPOSAL BATCH", {
                "laws": [
                    {
                        "law_id": law.law_id,
                        "template": law.template.value,
                        "claim": law.claim,
                        "claim_ast": law.claim_ast,
                        "claim_ast_str": ast_to_string(law.claim_ast) if law.claim_ast else None,
                        "forbidden": law.forbidden,
                        "observables": [{"name": o.name, "expr": o.expr} for o in law.observables],
                        "preconditions": [{"lhs": p.lhs, "op": p.op.value, "rhs": p.rhs} for p in law.preconditions],
                    }
                    for law in batch.laws
                ],
                "rejections": batch.rejections,
                "warnings": batch.warnings,
            })

        except Exception as e:
            logger.log(f"ERROR proposing laws: {e}")
            import traceback
            logger.log_json("PROPOSAL ERROR", {"error": str(e), "traceback": traceback.format_exc()})
            continue

        # Verbose output: show full prompt and response
        if verbose:
            exchange = proposer.get_last_exchange()
            logger.log("\n" + "=" * 40)
            logger.log("SYSTEM INSTRUCTION:")
            logger.log("=" * 40)
            logger.log(exchange["system_instruction"])
            logger.log("\n" + "=" * 40)
            logger.log("PROMPT:")
            logger.log("=" * 40)
            logger.log(exchange["prompt"])
            logger.log("\n" + "=" * 40)
            logger.log("LLM RESPONSE:")
            logger.log("=" * 40)
            logger.log(exchange["response"])
            logger.log("=" * 40 + "\n")

        # Log token usage
        if client.last_usage:
            u = client.last_usage
            logger.log(f"    Tokens: prompt={u.prompt_tokens}, output={u.output_tokens}, thinking={u.thinking_tokens}, total={u.total_tokens}")

        logger.log(f"    Proposed: {len(batch.laws)}")
        logger.log(f"    Rejected (parse): {len(batch.rejections)}")
        logger.log(f"    Redundant: {len(batch.redundant)}")

        # Show redundancy details
        if batch.redundant:
            for law, match in batch.redundant:
                logger.log(f"      - {law.law_id} matched {match.matched_law_id} ({match.match_type}, sim={match.similarity:.2f})")

        stats["total_rejected"] += len(batch.rejections)
        stats["total_redundant"] += len(batch.redundant)

        if batch.warnings:
            for w in batch.warnings:
                logger.log(f"    WARNING: {w}")

        if not batch.laws:
            logger.log("    No valid laws to test, skipping...")
            continue

        # Test each law
        logger.log(f"\n[2] Testing {len(batch.laws)} laws...")
        iteration_results = []

        for i, (law, features) in enumerate(zip(batch.laws, batch.features)):
            logger.log(f"\n  [{i+1}/{len(batch.laws)}] {law.law_id}")
            logger.log(f"      Template: {law.template.value}")
            # Show AST-based claim if available
            if law.claim_ast:
                claim_str = ast_to_string(law.claim_ast)
            else:
                claim_str = law.claim
            logger.log(f"      Claim: {claim_str[:60]}{'...' if len(claim_str) > 60 else ''}")
            logger.log(f"      Score: {features.overall_score:.3f} (risk={features.risk:.2f}, novelty={features.novelty:.2f})")

            try:
                verdict = harness.evaluate(law)
            except Exception as e:
                logger.log(f"      ERROR: {e}")
                import traceback
                tb = traceback.format_exc()
                logger.log_json(f"EVALUATION ERROR for {law.law_id}", {
                    "error": str(e),
                    "traceback": tb,
                })
                verdict = LawVerdict(
                    law_id=law.law_id,
                    status="UNKNOWN",
                    notes=[f"Evaluation error: {e}"],
                )
                # Log the error as untestable
                if untestable_log:
                    untestable_log.log(law, verdict, error_msg=str(e))

            logger.log(format_verdict(verdict))

            # Log full verdict details to transcript
            logger.log_json(f"VERDICT for {law.law_id}", {
                "status": verdict.status,
                "reason_code": verdict.reason_code.value if verdict.reason_code else None,
                "counterexample": verdict.counterexample.initial_state if verdict.counterexample else None,
                "notes": verdict.notes,
                "power_metrics": verdict.power_metrics.to_dict() if verdict.power_metrics else None,
            })

            # Record result
            memory.record_evaluation(law, verdict)
            stats["total_proposed"] += 1

            if verdict.status == "PASS":
                stats["total_provisional"] += 1  # Baseline PASS = provisional
                # Track using AST string if available for better deduplication
                claim_repr = ast_to_string(law.claim_ast) if law.claim_ast else law.claim
                stats["unique_accepted_claims"].add(claim_repr)
                # Track for escalation
                accepted_laws_list.append(law)
                escalation_state.record_accepted(iteration, law.law_id)
                # Save to database
                if persistence:
                    try:
                        persistence.save_accepted(law, verdict)
                    except Exception as e:
                        logger.log(f"      Warning: Could not save to database: {e}")
            elif verdict.status == "FAIL":
                stats["total_falsified"] += 1
                # Save to database
                if persistence:
                    try:
                        persistence.save_falsified(law, verdict)
                    except Exception as e:
                        logger.log(f"      Warning: Could not save to database: {e}")
            else:
                stats["total_unknown"] += 1
                # Log untestable law for analysis
                if untestable_log:
                    untestable_log.log(law, verdict)

            iteration_results.append({
                "law_id": law.law_id,
                "template": law.template.value,
                "claim": law.claim,
                "claim_ast": law.claim_ast,
                "forbidden": law.forbidden,
                "observables": [{"name": o.name, "expr": o.expr} for o in law.observables],
                "status": verdict.status,
                "reason_code": verdict.reason_code.value if verdict.reason_code else None,
                "counterexample": verdict.counterexample.initial_state if verdict.counterexample else None,
            })

        all_results.append({
            "iteration": iteration,
            "laws_proposed": len(batch.laws),
            "results": iteration_results,
        })

        # Summary for this iteration
        logger.log(f"\n[3] Iteration {iteration} Summary:")
        logger.log(f"    Memory: {memory.stats}")

        # Check if escalation should run
        if enable_escalation and persistence:
            run_esc_1, run_esc_2 = get_escalation_decisions(
                iteration, escalation_state, escalation_config, accepted_laws_list
            )

            if run_esc_1 or run_esc_2:
                logger.log(f"\n[4] Escalation Check:")

            # Run escalation_1 (cheap, frequent, scoped to recent provisional laws)
            # Laws that pass are PROMOTED from provisional to accepted
            if run_esc_1:
                logger.log(f"    Running escalation_1 (promoting provisional laws)...")
                recent_law_ids = get_laws_for_escalation_1(
                    iteration, escalation_state, escalation_config, persistence.repo
                )
                if recent_law_ids:
                    try:
                        esc_result = run_escalation(
                            EscalationLevel.ESCALATION_1,
                            persistence.repo,
                            law_ids=recent_law_ids,
                        )
                        escalation_state.last_escalation_1_iteration = iteration
                        stats["escalation_1_runs"] += 1
                        stats["escalation_revoked"] += esc_result.revoked_count
                        stats["escalation_downgraded"] += esc_result.downgraded_count
                        stats["total_promoted"] += esc_result.stable_count  # Promoted to accepted

                        logger.log(f"      Tested: {esc_result.laws_tested}")
                        logger.log(f"      Promoted: {esc_result.stable_count}, Revoked: {esc_result.revoked_count}, Downgraded: {esc_result.downgraded_count}")

                        # Log promoted laws
                        for retest in esc_result.retests:
                            if retest.flip_type == FlipType.STABLE:
                                logger.log(f"      + PROMOTED: {retest.law_id}")

                        # Handle revoked laws
                        if esc_result.revoked_count > 0:
                            for retest in esc_result.retests:
                                if retest.flip_type == FlipType.REVOKED:
                                    logger.log(f"      ! REVOKED: {retest.law_id}")
                                    # Remove from accepted_laws_list
                                    accepted_laws_list = [l for l in accepted_laws_list if l.law_id != retest.law_id]
                    except Exception as e:
                        logger.log(f"      Escalation_1 error: {e}")
                else:
                    logger.log(f"      No provisional laws need escalation_1 testing")

            # Run escalation_2 (expensive, all laws)
            if run_esc_2:
                logger.log(f"    Running escalation_2 (all accepted laws)...")
                try:
                    esc_result = run_escalation(
                        EscalationLevel.ESCALATION_2,
                        persistence.repo,
                    )
                    escalation_state.last_escalation_2_iteration = iteration
                    stats["escalation_2_runs"] += 1
                    stats["escalation_revoked"] += esc_result.revoked_count
                    stats["escalation_downgraded"] += esc_result.downgraded_count

                    logger.log(f"      Tested: {esc_result.laws_tested}")
                    logger.log(f"      Stable: {esc_result.stable_count}, Revoked: {esc_result.revoked_count}, Downgraded: {esc_result.downgraded_count}")

                    # Handle revoked laws
                    if esc_result.revoked_count > 0:
                        for retest in esc_result.retests:
                            if retest.flip_type == FlipType.REVOKED:
                                logger.log(f"      ! REVOKED: {retest.law_id}")
                                # Remove from accepted_laws_list
                                accepted_laws_list = [l for l in accepted_laws_list if l.law_id != retest.law_id]
                except Exception as e:
                    logger.log(f"      Escalation_2 error: {e}")

    # Final summary
    logger.log()
    logger.log("=" * 60)
    logger.log("DISCOVERY COMPLETE")
    logger.log("=" * 60)
    logger.log(f"Total laws proposed: {stats['total_proposed']}")
    logger.log(f"  Provisional (baseline PASS): {stats['total_provisional']}")
    logger.log(f"  Promoted (escalation_1 PASS): {stats['total_promoted']}")
    logger.log(f"  Falsified (FAIL): {stats['total_falsified']}")
    logger.log(f"  Unknown: {stats['total_unknown']}")
    logger.log(f"Parse rejections: {stats['total_rejected']}")
    logger.log(f"Redundant filtered: {stats['total_redundant']}")
    logger.log(f"Unique claims (provisional+promoted): {len(stats['unique_accepted_claims'])}")

    # Escalation summary
    if enable_escalation and (stats["escalation_1_runs"] > 0 or stats["escalation_2_runs"] > 0):
        logger.log(f"\nEscalation Summary:")
        logger.log(f"  Escalation_1 runs: {stats['escalation_1_runs']}")
        logger.log(f"  Escalation_2 runs: {stats['escalation_2_runs']}")
        logger.log(f"  Laws promoted: {stats['total_promoted']}")
        logger.log(f"  Laws revoked: {stats['escalation_revoked']}")
        logger.log(f"  Laws downgraded: {stats['escalation_downgraded']}")

    if stats["unique_accepted_claims"]:
        logger.log("\nAccepted claims:")
        for claim in sorted(stats["unique_accepted_claims"]):
            logger.log(f"  ✓ {claim}")

    # Save results
    output_file = output_dir / f"discovery_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "iterations": iterations,
            "laws_per_iteration": laws_per_iteration,
            "stats": {
                **stats,
                "unique_accepted_claims": list(stats["unique_accepted_claims"]),
            },
            "results": all_results,
            "audit_log": proposer.get_audit_log(),
        }, f, indent=2)

    logger.log(f"\nResults saved to: {output_file}")
    logger.log(f"Transcript saved to: {output_dir / f'transcript_{timestamp}.txt'}")

    if untestable_log and untestable_log.count > 0:
        logger.log(f"Untestable laws logged to: {untestable_log.log_path} ({untestable_log.count} entries)")

    # Show database summary if using persistence
    if persistence:
        db_summary = persistence.get_summary()
        logger.log(f"\nDatabase summary ({persistence.db_path}):")
        logger.log(f"  Total PASS: {db_summary.get('PASS', 0)}")
        logger.log(f"  Total FAIL: {db_summary.get('FAIL', 0)}")
        logger.log(f"  Total UNKNOWN: {db_summary.get('UNKNOWN', 0)}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Run law discovery loop")
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=5,
        help="Number of discovery iterations (default: 5)",
    )
    parser.add_argument(
        "--laws-per-iter", "-l",
        type=int,
        default=5,
        help="Laws to propose per iteration (default: 5)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full prompts and LLM responses",
    )
    parser.add_argument(
        "--db", "-d",
        type=str,
        default=None,
        help="Path to SQLite database for persistence (default: results/discovery.db)",
    )
    parser.add_argument(
        "--no-escalation",
        action="store_true",
        help="Disable power escalation during discovery",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1). Each worker runs LLM queries independently.",
    )

    args = parser.parse_args()

    run_discovery(
        iterations=args.iterations,
        laws_per_iteration=args.laws_per_iter,
        output_dir=Path(args.output),
        verbose=args.verbose,
        db_path=Path(args.db) if args.db else None,
        enable_escalation=not args.no_escalation,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
