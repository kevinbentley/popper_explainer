"""Fixture loading utilities for regression testing."""

import json
from pathlib import Path
from typing import Any

from src.claims.schema import (
    CandidateLaw,
    CapabilityRequirements,
    ComparisonOp,
    MonotoneDirection,
    Observable,
    Precondition,
    ProposedTest,
    Quantifiers,
    Template,
)
from src.harness.config import HarnessConfig


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def load_harness_config(path: Path) -> HarnessConfig:
    """Load harness config from JSON file."""
    data = load_json(path)
    return HarnessConfig(
        seed=data.get("seed", 42),
        max_runtime_ms_per_law=data.get("max_runtime_ms_per_law", 5000),
        max_cases=data.get("max_cases", 300),
        default_T=data.get("default_T", 50),
        max_T=data.get("max_T", 200),
        min_cases_used_for_pass=data.get("min_cases_used_for_pass", 50),
        enable_adversarial_search=data.get("enable_adversarial_search", True),
        adversarial_budget=data.get("adversarial_budget", 1500),
        enable_counterexample_minimization=data.get("enable_counterexample_minimization", True),
        minimization_budget=data.get("minimization_budget", 300),
        store_full_trajectories=data.get("store_full_trajectories", False),
        require_non_vacuous=data.get("require_non_vacuous", True),
    )


def parse_comparison_op(op_str: str) -> ComparisonOp:
    """Parse a comparison operator string."""
    mapping = {
        "==": ComparisonOp.EQ,
        "!=": ComparisonOp.NE,
        "<": ComparisonOp.LT,
        "<=": ComparisonOp.LE,
        ">": ComparisonOp.GT,
        ">=": ComparisonOp.GE,
    }
    return mapping.get(op_str, ComparisonOp.EQ)


def load_law(data: dict[str, Any]) -> CandidateLaw:
    """Load a single law from JSON data."""
    # Parse template
    template = Template(data["template"])

    # Parse quantifiers
    quant_data = data.get("quantifiers", {})
    quantifiers = Quantifiers(
        T=quant_data.get("T", 50),
        H=quant_data.get("H"),
    )

    # Parse preconditions
    preconditions = []
    for p in data.get("preconditions", []):
        preconditions.append(Precondition(
            lhs=p["lhs"],
            op=parse_comparison_op(p["op"]),
            rhs=p["rhs"],
        ))

    # Parse observables
    observables = []
    for o in data.get("observables", []):
        observables.append(Observable(
            name=o["name"],
            expr=o["expr"],
        ))

    # Parse proposed tests
    proposed_tests = []
    for t in data.get("proposed_tests", []):
        proposed_tests.append(ProposedTest(
            family=t["family"],
            params=t.get("params", {}),
        ))

    # Parse capability requirements
    cap_data = data.get("capability_requirements", {})
    capability_requirements = CapabilityRequirements(
        missing_observables=cap_data.get("missing_observables", []),
        missing_transforms=cap_data.get("missing_transforms", []),
        missing_generators=cap_data.get("missing_generators", []),
    )

    # Parse direction for monotone
    direction = None
    if "direction" in data:
        direction = MonotoneDirection(data["direction"])

    # Parse bound fields
    bound_value = data.get("bound_value")
    bound_op = None
    if "bound_op" in data:
        bound_op = parse_comparison_op(data["bound_op"])

    return CandidateLaw(
        schema_version=data.get("schema_version", "1.0.0"),
        law_id=data["law_id"],
        template=template,
        quantifiers=quantifiers,
        preconditions=preconditions,
        observables=observables,
        claim=data["claim"],
        forbidden=data["forbidden"],
        transform=data.get("transform"),
        direction=direction,
        bound_value=bound_value,
        bound_op=bound_op,
        proposed_tests=proposed_tests,
        capability_requirements=capability_requirements,
    )


def load_laws(path: Path) -> list[CandidateLaw]:
    """Load laws from a JSON file."""
    data = load_json(path)
    return [load_law(law_data) for law_data in data]


def load_expectations(path: Path) -> dict[str, dict[str, Any]]:
    """Load expectations from JSON file.

    Returns a dict mapping law_id to expected outcome.
    """
    data = load_json(path)
    return {
        exp["law_id"]: exp
        for exp in data.get("expected", [])
    }


class FixtureLoader:
    """Loader for test fixtures."""

    def __init__(self, fixtures_dir: Path | str):
        """Initialize fixture loader.

        Args:
            fixtures_dir: Path to fixtures directory
        """
        self.fixtures_dir = Path(fixtures_dir)

    def load_universe_contract(self) -> dict[str, Any]:
        """Load universe contract."""
        return load_json(self.fixtures_dir / "universe_contract.json")

    def load_harness_config(self) -> HarnessConfig:
        """Load harness configuration."""
        return load_harness_config(self.fixtures_dir / "harness_config.json")

    def load_true_laws(self) -> list[CandidateLaw]:
        """Load laws that should pass."""
        return load_laws(self.fixtures_dir / "laws_true.json")

    def load_false_laws(self) -> list[CandidateLaw]:
        """Load laws that should fail."""
        return load_laws(self.fixtures_dir / "laws_false.json")

    def load_unknown_laws(self) -> list[CandidateLaw]:
        """Load laws that should return unknown."""
        return load_laws(self.fixtures_dir / "laws_unknown.json")

    def load_all_laws(self) -> list[CandidateLaw]:
        """Load all fixture laws."""
        laws = []
        laws.extend(self.load_true_laws())
        laws.extend(self.load_false_laws())
        laws.extend(self.load_unknown_laws())
        return laws

    def load_expectations(self) -> dict[str, dict[str, Any]]:
        """Load expected outcomes."""
        return load_expectations(self.fixtures_dir / "expectations.json")
