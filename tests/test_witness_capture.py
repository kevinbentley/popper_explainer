"""Tests for PHASE-E witness capture functionality."""

import pytest

from src.harness.witness import (
    FormattedWitness,
    WitnessCapture,
    build_formatted_witness,
    compute_neighborhood_hash,
    extract_violation_info,
)
from src.claims.schema import CandidateLaw, Observable, Quantifiers, Template


def make_simple_law(law_id: str = "test_law", template: str = "invariant") -> CandidateLaw:
    """Create a simple test law."""
    return CandidateLaw(
        law_id=law_id,
        template=Template(template),
        quantifiers=Quantifiers(T=10),
        preconditions=[],
        observables=[
            Observable(name="left_count", expr="count('<')"),
            Observable(name="right_count", expr="count('>')"),
        ],
        claim="left_count(t) + right_count(t) == total_particles",
        forbidden="left_count(t) + right_count(t) != total_particles",
        proposed_tests=[],
    )


class TestFormattedWitness:
    def test_str_representation(self):
        """Test human-readable string format."""
        witness = FormattedWitness(
            law_id="test_law",
            t=5,
            lhs_expr="left_count",
            lhs_value=10,
            rhs_expr="right_count",
            rhs_value=5,
            violation_description="value changed",
            state_at_t=".><.X.",
            neighborhood_hash="abc123",
        )
        s = str(witness)
        assert "t=5" in s
        assert "left_count=10" in s
        assert "right_count=5" in s
        assert "value changed" in s

    def test_to_dict(self):
        """Test serialization."""
        witness = FormattedWitness(
            law_id="test_law",
            t=5,
            lhs_expr="lhs",
            lhs_value=10,
            rhs_expr="rhs",
            rhs_value=5,
            violation_description="violation",
            state_at_t=".><.",
            neighborhood_hash="abc123",
        )
        d = witness.to_dict()
        assert d["law_id"] == "test_law"
        assert d["t"] == 5
        assert d["lhs_value"] == 10

    def test_from_dict(self):
        """Test deserialization."""
        d = {
            "law_id": "test_law",
            "t": 10,
            "lhs_expr": "obs",
            "lhs_value": 42,
            "rhs_expr": "expected",
            "rhs_value": 50,
            "violation_description": "mismatch",
            "state_at_t": "..><..",
            "neighborhood_hash": "xyz789",
        }
        witness = FormattedWitness.from_dict(d)
        assert witness.t == 10
        assert witness.lhs_value == 42


class TestComputeNeighborhoodHash:
    def test_deterministic(self):
        """Same inputs should produce same hash."""
        state = "..><..X..<>"
        hash1 = compute_neighborhood_hash(state, position=5, radius=3)
        hash2 = compute_neighborhood_hash(state, position=5, radius=3)
        assert hash1 == hash2

    def test_different_positions_different_hash(self):
        """Different positions should produce different hashes."""
        state = "..><..X..<>"
        hash1 = compute_neighborhood_hash(state, position=2, radius=3)
        hash2 = compute_neighborhood_hash(state, position=6, radius=3)
        assert hash1 != hash2

    def test_global_hash_for_none_position(self):
        """None position should use entire state."""
        state = "..><..X..<>"
        hash1 = compute_neighborhood_hash(state, position=None)
        hash2 = compute_neighborhood_hash(state, position=None)
        assert hash1 == hash2

    def test_boundary_handling(self):
        """Should handle positions near boundaries."""
        state = ".><."
        # Position at start
        hash1 = compute_neighborhood_hash(state, position=0, radius=3)
        # Position at end
        hash2 = compute_neighborhood_hash(state, position=3, radius=3)
        # Both should work without error
        assert len(hash1) == 16
        assert len(hash2) == 16


class TestExtractViolationInfo:
    def test_invariant_template(self):
        """Test violation info extraction for invariant template."""
        law = make_simple_law(template="invariant")
        lhs, rhs, desc = extract_violation_info(law, None)
        assert desc == "value changed"

    def test_monotone_template(self):
        """Test violation info for monotone template."""
        law = make_simple_law(template="monotone")
        lhs, rhs, desc = extract_violation_info(law, None)
        assert "obs(t)" in lhs or "obs" in lhs.lower()
        assert "monotonicity" in desc.lower()

    def test_with_violation_details(self):
        """Violation details should override defaults."""
        law = make_simple_law()
        violation = {
            "details": {
                "lhs": 10,
                "rhs": 5,
                "reason": "custom violation reason",
            }
        }
        _, _, desc = extract_violation_info(law, violation)
        assert desc == "custom violation reason"


class TestBuildFormattedWitness:
    def test_basic_witness(self):
        """Test building a basic witness."""
        law = make_simple_law()
        trajectory = [".><.", "..X.", ".X..", "X..."]
        violation = {"t": 1, "details": {"actual": 5, "expected": 10}}

        witness = build_formatted_witness(
            law=law,
            trajectory=trajectory,
            t_fail=1,
            violation=violation,
        )

        assert witness.law_id == "test_law"
        assert witness.t == 1
        assert witness.state_at_t == "..X."
        assert witness.state_at_t1 == ".X.."
        assert len(witness.neighborhood_hash) == 16

    def test_witness_with_position(self):
        """Test witness with position for neighborhood hash."""
        law = make_simple_law()
        trajectory = [".><.", "..X.", ".X.."]

        witness = build_formatted_witness(
            law=law,
            trajectory=trajectory,
            t_fail=1,
            violation={"t": 1},
            position=2,
        )

        # Neighborhood hash should be computed from position
        assert len(witness.neighborhood_hash) == 16


class TestWitnessCapture:
    def test_add_witness(self):
        """Test adding a witness."""
        capture = WitnessCapture(max_witnesses_per_law=5)
        witness = FormattedWitness(
            law_id="law_001",
            t=1,
            lhs_expr="lhs",
            lhs_value=1,
            rhs_expr="rhs",
            rhs_value=2,
            violation_description="test",
            state_at_t="..><..",
            neighborhood_hash="hash001",
        )
        result = capture.add_witness(witness)
        assert result is True
        assert len(capture.get_witnesses("law_001")) == 1

    def test_duplicate_neighborhood_rejected(self):
        """Witnesses with same neighborhood hash should be rejected."""
        capture = WitnessCapture(max_witnesses_per_law=5)
        witness1 = FormattedWitness(
            law_id="law_001",
            t=1,
            lhs_expr="lhs",
            lhs_value=1,
            rhs_expr="rhs",
            rhs_value=2,
            violation_description="test",
            state_at_t="..><..",
            neighborhood_hash="same_hash",
        )
        witness2 = FormattedWitness(
            law_id="law_001",
            t=2,  # Different time
            lhs_expr="lhs",
            lhs_value=3,
            rhs_expr="rhs",
            rhs_value=4,
            violation_description="test",
            state_at_t=".><...",
            neighborhood_hash="same_hash",  # Same hash
        )

        capture.add_witness(witness1)
        result = capture.add_witness(witness2)

        assert result is False  # Rejected
        assert len(capture.get_witnesses("law_001")) == 1

    def test_capacity_limit(self):
        """Test max witnesses limit."""
        capture = WitnessCapture(max_witnesses_per_law=2)

        for i in range(5):
            witness = FormattedWitness(
                law_id="law_001",
                t=i,
                lhs_expr="lhs",
                lhs_value=i,
                rhs_expr="rhs",
                rhs_value=i,
                violation_description="test",
                state_at_t=f"state_{i}",
                neighborhood_hash=f"hash_{i}",
            )
            capture.add_witness(witness)

        # Should only have max_witnesses_per_law
        assert len(capture.get_witnesses("law_001")) == 2

    def test_get_primary_witness(self):
        """Test getting primary (first) witness."""
        capture = WitnessCapture()
        witness1 = FormattedWitness(
            law_id="law_001",
            t=1,
            lhs_expr="lhs",
            lhs_value=1,
            rhs_expr="rhs",
            rhs_value=1,
            violation_description="first",
            state_at_t="first",
            neighborhood_hash="hash1",
        )
        witness2 = FormattedWitness(
            law_id="law_001",
            t=2,
            lhs_expr="lhs",
            lhs_value=2,
            rhs_expr="rhs",
            rhs_value=2,
            violation_description="second",
            state_at_t="second",
            neighborhood_hash="hash2",
        )
        capture.add_witness(witness1)
        capture.add_witness(witness2)

        primary = capture.get_primary_witness("law_001")
        assert primary.t == 1
        assert primary.violation_description == "first"

    def test_diversity_count(self):
        """Test tracking unique neighborhood hashes."""
        capture = WitnessCapture()
        for i in range(5):
            witness = FormattedWitness(
                law_id="law_001",
                t=i,
                lhs_expr="lhs",
                lhs_value=i,
                rhs_expr="rhs",
                rhs_value=i,
                violation_description="test",
                state_at_t=f"state_{i}",
                neighborhood_hash=f"hash_{i}",
            )
            capture.add_witness(witness)

        assert capture.get_diversity_count("law_001") == 5

    def test_clear_specific_law(self):
        """Test clearing witnesses for a specific law."""
        capture = WitnessCapture()
        for law_id in ["law_001", "law_002"]:
            witness = FormattedWitness(
                law_id=law_id,
                t=1,
                lhs_expr="lhs",
                lhs_value=1,
                rhs_expr="rhs",
                rhs_value=1,
                violation_description="test",
                state_at_t="state",
                neighborhood_hash=f"hash_{law_id}",
            )
            capture.add_witness(witness)

        capture.clear("law_001")
        assert len(capture.get_witnesses("law_001")) == 0
        assert len(capture.get_witnesses("law_002")) == 1

    def test_clear_all(self):
        """Test clearing all witnesses."""
        capture = WitnessCapture()
        for law_id in ["law_001", "law_002"]:
            witness = FormattedWitness(
                law_id=law_id,
                t=1,
                lhs_expr="lhs",
                lhs_value=1,
                rhs_expr="rhs",
                rhs_value=1,
                violation_description="test",
                state_at_t="state",
                neighborhood_hash=f"hash_{law_id}",
            )
            capture.add_witness(witness)

        capture.clear()
        assert len(capture.get_witnesses("law_001")) == 0
        assert len(capture.get_witnesses("law_002")) == 0
