#!/usr/bin/env python3
"""CLI for running theorem generation.

Usage:
    python scripts/run_theorem_generation.py --db results/discovery.db
    python scripts/run_theorem_generation.py --db results/discovery.db --target-theorems 10
    python scripts/run_theorem_generation.py --db results/discovery.db --max-pass 50 --max-fail 50
    python scripts/run_theorem_generation.py --db results/discovery.db --mock  # Use mock client
    python scripts/run_theorem_generation.py --db results/discovery.db --distance-threshold 0.6 -v
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.db.models import (
    FailureClusterRecord,
    ObservableProposalRecord,
    TheoremRecord,
    TheoremRunRecord,
)
from src.db.repo import Repository
from src.theorem import (
    MockLLMClient,
    ObservableProposer,
    TheoremGenerator,
    TheoremGeneratorConfig,
    TwoPassClusterer,
    build_failure_signature,
    hash_signature,
)
from src.theorem.generator import create_gemini_client


def main():
    parser = argparse.ArgumentParser(
        description="Run theorem generation from discovery results"
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to the SQLite database",
    )
    parser.add_argument(
        "--max-pass",
        type=int,
        default=50,
        help="Maximum number of PASS laws to include (default: 50)",
    )
    parser.add_argument(
        "--max-fail",
        type=int,
        default=50,
        help="Maximum number of FAIL laws to include (default: 50)",
    )
    parser.add_argument(
        "--target-theorems",
        type=int,
        default=10,
        help="Target number of theorems to generate (default: 10)",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.6,
        help="Cosine distance threshold for TF-IDF clustering (default: 0.6)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM client (for testing)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="LLM model to use (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Check database exists
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    # Connect to database
    repo = Repository(db_path)
    repo.connect()

    try:
        run_theorem_generation(
            repo=repo,
            max_pass=args.max_pass,
            max_fail=args.max_fail,
            target_theorems=args.target_theorems,
            distance_threshold=args.distance_threshold,
            use_mock=args.mock,
            model=args.model,
            verbose=args.verbose,
        )
    finally:
        repo.close()


def run_theorem_generation(
    repo: Repository,
    max_pass: int,
    max_fail: int,
    target_theorems: int,
    distance_threshold: float,
    use_mock: bool,
    model: str,
    verbose: bool,
):
    """Run the full theorem generation pipeline."""
    print("=" * 60)
    print("Theorem Generation Pipeline (PHASE-C)")
    print("=" * 60)

    # Create run record
    run_id = f"thm_run_{uuid.uuid4().hex[:12]}"
    config = {
        "max_pass": max_pass,
        "max_fail": max_fail,
        "target_theorems": target_theorems,
        "distance_threshold": distance_threshold,
        "use_mock": use_mock,
        "model": model,
    }

    # Check available laws
    pass_count = len(repo.get_laws_with_status("PASS", limit=max_pass))
    fail_count = len(repo.get_laws_with_status("FAIL", limit=max_fail))

    print(f"\nInput Laws:")
    print(f"  PASS: {pass_count}")
    print(f"  FAIL: {fail_count}")
    print(f"\nClustering Settings:")
    print(f"  Distance threshold: {distance_threshold}")

    if pass_count == 0 and fail_count == 0:
        print("\nError: No laws found in database. Run discovery first.")
        return

    # Create run record
    run_record = TheoremRunRecord(
        run_id=run_id,
        status="running",
        config_json=json.dumps(config),
        pass_laws_count=pass_count,
        fail_laws_count=fail_count,
    )
    run_db_id = repo.insert_theorem_run(run_record)

    # Initialize generator
    config_obj = TheoremGeneratorConfig(
        max_pass_laws=max_pass,
        max_fail_laws=max_fail,
        target_theorem_count=target_theorems,
    )

    if use_mock:
        client = MockLLMClient()
        print("\nUsing mock LLM client")
    else:
        # Check for API key
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("\nError: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set.")
            print("Set the key in .env or use --mock for testing.")
            repo.update_theorem_run(run_id, status="aborted")
            return

        print(f"\nUsing Gemini model: {model}")
        client = create_gemini_client(api_key=api_key, model=model)

    generator = TheoremGenerator(client=client, config=config_obj)

    # Step 1: Build law snapshots
    print("\n[1/4] Building law snapshots...")
    start = time.time()
    law_snapshots = generator.build_law_snapshot(repo, max_pass, max_fail)
    print(f"      Built {len(law_snapshots)} snapshots in {time.time()-start:.2f}s")

    if verbose:
        for ls in law_snapshots[:5]:
            print(f"        - {ls.law_id} [{ls.status}]: {ls.claim[:50]}...")

    # Step 2: Generate theorems
    print("\n[2/4] Generating theorems...")
    start = time.time()
    batch, signatures = generator.generate_with_signatures(
        law_snapshots, target_theorems
    )
    print(f"      Generated {batch.accepted_count} theorems in {batch.runtime_ms}ms")
    print(f"      Rejected: {batch.rejected_count}")

    if batch.warnings:
        for warning in batch.warnings:
            print(f"      Warning: {warning}")

    # Store theorems
    for theorem in batch.theorems:
        sig_text, sig_hash = signatures.get(theorem.theorem_id, ("", ""))
        record = TheoremRecord(
            theorem_run_id=run_db_id,
            theorem_id=theorem.theorem_id,
            name=theorem.name,
            status=theorem.status.value,
            claim=theorem.claim,
            support_json=json.dumps([s.to_dict() for s in theorem.support]),
            failure_modes_json=json.dumps(theorem.failure_modes),
            missing_structure_json=json.dumps(theorem.missing_structure),
            failure_signature_text=sig_text,
            failure_signature_hash=sig_hash,
            bucket_tags_json=json.dumps(theorem.bucket_tags),
        )
        repo.insert_theorem(record)

    if verbose:
        for theorem in batch.theorems:
            print(f"\n        Theorem: {theorem.name}")
            print(f"        Status: {theorem.status.value}")
            print(f"        Claim: {theorem.claim[:80]}...")
            print(f"        Support: {len(theorem.support)} laws")

    # Step 3: Cluster failure signatures (PHASE-C: TF-IDF clustering)
    print("\n[3/4] Clustering failure signatures (TF-IDF)...")
    start = time.time()
    clusterer = TwoPassClusterer(distance_threshold=distance_threshold)
    clusters, cluster_stats = clusterer.cluster_with_stats(batch.theorems)
    print(f"      Found {len(clusters)} clusters in {time.time()-start:.2f}s")

    if verbose:
        print(f"      Bucket distribution: {cluster_stats['bucket_counts']}")
        print(f"      Action distribution: {cluster_stats['action_distribution']}")

    # Store clusters
    for cluster in clusters:
        record = FailureClusterRecord(
            theorem_run_id=run_db_id,
            cluster_id=cluster.cluster_id,
            bucket=cluster.bucket.value,
            bucket_tags_json=json.dumps(cluster.bucket_tags),
            semantic_cluster_idx=cluster.semantic_cluster_idx,
            theorem_ids_json=json.dumps(cluster.theorem_ids),
            cluster_size=len(cluster.theorem_ids),
            centroid_signature=cluster.centroid_signature,
            avg_similarity=cluster.avg_similarity,
            top_keywords_json=json.dumps(cluster.top_keywords),
            recommended_action=cluster.recommended_action,
            distance_threshold=cluster.distance_threshold,
        )
        repo.insert_failure_cluster(record)

    # Show top keywords per cluster in verbose mode
    if verbose and clusters:
        print("\n      Top cluster keywords:")
        for cluster in clusters[:5]:
            kw_str = ", ".join(kw for kw, _ in cluster.top_keywords[:4])
            print(f"        {cluster.bucket.value}_{cluster.semantic_cluster_idx}: {kw_str}")

    # Step 4: Propose observables
    print("\n[4/4] Proposing observables...")
    start = time.time()
    proposer = ObservableProposer()
    proposals = proposer.propose_from_all_clusters(clusters, dedupe=True)
    print(f"      Generated {len(proposals)} proposals in {time.time()-start:.2f}s")

    # Store proposals
    for proposal in proposals:
        record = ObservableProposalRecord(
            theorem_run_id=run_db_id,
            cluster_id=proposal.cluster_id,
            proposal_id=proposal.proposal_id,
            observable_name=proposal.observable_name,
            observable_expr=proposal.observable_expr,
            rationale=proposal.rationale,
            priority=proposal.priority,
            action_type=proposal.action_type,
        )
        repo.insert_observable_proposal(record)

    if verbose:
        high_priority = [p for p in proposals if p.priority == "high"]
        print(f"\n      High-priority proposals ({len(high_priority)}):")
        for p in high_priority[:5]:
            print(f"        - {p.observable_name}: {p.observable_expr[:50]}...")

    # Update run record
    repo.update_theorem_run(
        run_id=run_id,
        status="completed",
        theorems_generated=batch.accepted_count,
        clusters_found=len(clusters),
        observable_proposals=len(proposals),
        prompt_hash=batch.prompt_hash,
        completed_at=datetime.now().isoformat(),
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Run ID: {run_id}")
    print(f"  Theorems: {batch.accepted_count}")
    print(f"  Clusters: {len(clusters)}")

    # Action distribution
    action_dist = cluster_stats.get("action_distribution", {})
    if action_dist:
        print(f"\n  Clusters by action:")
        for action, count in sorted(action_dist.items()):
            print(f"    {action}: {count} clusters")

    # Proposal summary
    print(f"\n  Proposals: {len(proposals)}")
    print(f"    High priority: {len([p for p in proposals if p.priority == 'high'])}")
    print(f"    Medium priority: {len([p for p in proposals if p.priority == 'medium'])}")
    print(f"    Low priority: {len([p for p in proposals if p.priority == 'low'])}")

    # Action types in proposals
    action_types = {}
    for p in proposals:
        action_types[p.action_type] = action_types.get(p.action_type, 0) + 1
    if action_types:
        print(f"\n  Proposals by action type:")
        for action, count in sorted(action_types.items()):
            print(f"    {action}: {count}")


if __name__ == "__main__":
    main()
