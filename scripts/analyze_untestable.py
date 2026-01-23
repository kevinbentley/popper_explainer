#!/usr/bin/env python3
"""Analyze untestable laws log to identify capability gaps.

This script reads the untestable_laws.jsonl file and summarizes:
- Which reason codes are most common
- What observables/functions are being requested
- What templates are most problematic

Usage:
    python scripts/analyze_untestable.py [--file PATH]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze_untestable_laws(log_path: Path) -> dict:
    """Analyze untestable laws log.

    Args:
        log_path: Path to untestable_laws.jsonl

    Returns:
        Analysis summary
    """
    if not log_path.exists():
        print(f"No untestable laws log found at {log_path}")
        return {}

    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print("No untestable laws recorded.")
        return {}

    # Analyze by reason code
    reason_counts = Counter(e.get("reason_code", "unknown") for e in entries)

    # Analyze by template
    template_counts = Counter(e.get("template", "unknown") for e in entries)

    # Collect unique observables requested
    observables_requested = defaultdict(int)
    for e in entries:
        for obs in e.get("observables", []):
            key = f"{obs['name']}: {obs['expr']}"
            observables_requested[key] += 1

    # Collect error messages
    errors = []
    for e in entries:
        if e.get("error"):
            errors.append({
                "law_id": e.get("law_id"),
                "error": e.get("error"),
                "claim": e.get("claim"),
            })

    # Collect claims by reason code for detailed review
    claims_by_reason = defaultdict(list)
    for e in entries:
        reason = e.get("reason_code", "unknown")
        claims_by_reason[reason].append({
            "law_id": e.get("law_id"),
            "template": e.get("template"),
            "claim": e.get("claim"),
            "claim_ast": e.get("claim_ast"),
            "notes": e.get("notes"),
        })

    return {
        "total_untestable": len(entries),
        "by_reason_code": dict(reason_counts.most_common()),
        "by_template": dict(template_counts.most_common()),
        "observables_used": dict(sorted(observables_requested.items(), key=lambda x: -x[1])),
        "errors": errors[:20],  # Limit to first 20
        "claims_by_reason": {k: v[:10] for k, v in claims_by_reason.items()},  # First 10 per reason
    }


def print_analysis(analysis: dict):
    """Print analysis in a readable format."""
    if not analysis:
        return

    print("=" * 60)
    print("UNTESTABLE LAWS ANALYSIS")
    print("=" * 60)

    print(f"\nTotal untestable laws: {analysis['total_untestable']}")

    print("\n--- By Reason Code ---")
    for reason, count in analysis["by_reason_code"].items():
        print(f"  {reason}: {count}")

    print("\n--- By Template ---")
    for template, count in analysis["by_template"].items():
        print(f"  {template}: {count}")

    print("\n--- Observables Used ---")
    for obs, count in list(analysis["observables_used"].items())[:15]:
        print(f"  {obs} ({count}x)")

    if analysis["errors"]:
        print("\n--- Recent Errors ---")
        for err in analysis["errors"][:5]:
            print(f"  [{err['law_id']}] {err['error'][:80]}")

    print("\n--- Sample Claims by Reason ---")
    for reason, claims in analysis["claims_by_reason"].items():
        print(f"\n  {reason}:")
        for c in claims[:3]:
            claim_str = c.get("claim", "")[:60]
            print(f"    - [{c['template']}] {c['law_id']}: {claim_str}...")


def main():
    parser = argparse.ArgumentParser(description="Analyze untestable laws log")
    parser.add_argument(
        "--file", "-f",
        type=str,
        default="results/untestable_laws.jsonl",
        help="Path to untestable_laws.jsonl (default: results/untestable_laws.jsonl)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON instead of human-readable format",
    )

    args = parser.parse_args()

    analysis = analyze_untestable_laws(Path(args.file))

    if args.json:
        print(json.dumps(analysis, indent=2))
    else:
        print_analysis(analysis)


if __name__ == "__main__":
    main()
