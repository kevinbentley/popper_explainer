#!/usr/bin/env python3
"""Run the law discovery loop.

This script orchestrates the full discovery cycle:
1. Propose laws using the LLM
2. Test laws with the harness
3. Record results
4. Repeat

Usage:
    python scripts/run_discovery.py [--iterations N] [--laws-per-iter K]
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import TextIO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TranscriptLogger:
    """Logs all output to both console and transcript file."""

    def __init__(self, transcript_path: Path):
        self.transcript_path = transcript_path
        self.file: TextIO | None = None

    def __enter__(self):
        self.file = open(self.transcript_path, "w")
        return self

    def __exit__(self, *args):
        if self.file:
            self.file.close()

    def log(self, message: str = ""):
        """Log message to both console and file."""
        print(message)
        if self.file:
            self.file.write(message + "\n")
            self.file.flush()

    def log_json(self, label: str, data: dict):
        """Log JSON data to transcript only (not console)."""
        if self.file:
            self.file.write(f"\n=== {label} ===\n")
            self.file.write(json.dumps(data, indent=2))
            self.file.write("\n")
            self.file.flush()

from src.claims.ast_schema import ast_to_string
from src.claims.schema import CandidateLaw
from src.harness.config import HarnessConfig
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
        max_output_tokens=8192,  # Allow longer responses
        top_p=0.95,
        top_k=40,
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


def create_proposer(client: GeminiClient) -> LawProposer:
    """Create law proposer."""
    contract = UniverseContract()
    config = ProposerConfig(
        max_token_budget=16000,  # Larger context for thinking
        include_counterexamples=True,
        strict_parsing=False,
        add_to_redundancy_filter=True,
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


def run_discovery(
    iterations: int = 5,
    laws_per_iteration: int = 5,
    output_dir: Path | None = None,
) -> dict:
    """Run the discovery loop.

    Args:
        iterations: Number of discovery iterations
        laws_per_iteration: Laws to propose per iteration
        output_dir: Directory to save results

    Returns:
        Summary statistics
    """
    # Set up output directory
    output_dir = Path(output_dir) if output_dir else Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_path = output_dir / f"transcript_{timestamp}.txt"

    with TranscriptLogger(transcript_path) as logger:
        return _run_discovery_with_logging(
            iterations, laws_per_iteration, output_dir, timestamp, logger
        )


def _run_discovery_with_logging(
    iterations: int,
    laws_per_iteration: int,
    output_dir: Path,
    timestamp: str,
    logger: TranscriptLogger,
) -> dict:
    """Run discovery with transcript logging."""
    logger.log("=" * 60)
    logger.log("POPPERIAN LAW DISCOVERY")
    logger.log("=" * 60)
    logger.log(f"Timestamp: {timestamp}")
    logger.log(f"Iterations: {iterations}")
    logger.log(f"Laws per iteration: {laws_per_iteration}")
    logger.log()

    # Initialize components
    logger.log("Initializing...")
    client = create_client()
    harness = create_harness()
    proposer = create_proposer(client)
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
        "total_accepted": 0,
        "total_falsified": 0,
        "total_unknown": 0,
        "total_rejected": 0,
        "total_redundant": 0,
        "unique_accepted_claims": set(),
    }

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

        logger.log(f"    Proposed: {len(batch.laws)}")
        logger.log(f"    Rejected (parse): {len(batch.rejections)}")
        logger.log(f"    Redundant: {len(batch.redundant)}")

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
                logger.log_json(f"EVALUATION ERROR for {law.law_id}", {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                verdict = LawVerdict(
                    law_id=law.law_id,
                    status="UNKNOWN",
                    notes=[f"Evaluation error: {e}"],
                )

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
                stats["total_accepted"] += 1
                # Track using AST string if available for better deduplication
                claim_repr = ast_to_string(law.claim_ast) if law.claim_ast else law.claim
                stats["unique_accepted_claims"].add(claim_repr)
            elif verdict.status == "FAIL":
                stats["total_falsified"] += 1
            else:
                stats["total_unknown"] += 1

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

    # Final summary
    logger.log()
    logger.log("=" * 60)
    logger.log("DISCOVERY COMPLETE")
    logger.log("=" * 60)
    logger.log(f"Total laws proposed: {stats['total_proposed']}")
    logger.log(f"  Accepted (PASS): {stats['total_accepted']}")
    logger.log(f"  Falsified (FAIL): {stats['total_falsified']}")
    logger.log(f"  Unknown: {stats['total_unknown']}")
    logger.log(f"Parse rejections: {stats['total_rejected']}")
    logger.log(f"Redundant filtered: {stats['total_redundant']}")
    logger.log(f"Unique accepted claims: {len(stats['unique_accepted_claims'])}")

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

    args = parser.parse_args()

    run_discovery(
        iterations=args.iterations,
        laws_per_iteration=args.laws_per_iter,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
