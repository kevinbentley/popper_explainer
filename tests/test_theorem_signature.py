"""Tests for theorem signature building."""

import pytest

from src.theorem.models import LawSupport, Theorem, TheoremStatus
from src.theorem.signature import (
    build_failure_signature,
    build_signature_with_law_context,
    compute_jaccard_similarity,
    extract_key_terms,
    hash_signature,
    normalize_text,
)


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("HELLO World") == "hello world"

    def test_strip_punctuation(self):
        assert normalize_text("Hello, World!") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_text("hello   world") == "hello world"
        assert normalize_text("  hello  world  ") == "hello world"

    def test_combined(self):
        assert normalize_text("  HELLO,  World!  ") == "hello world"

    def test_empty(self):
        assert normalize_text("") == ""

    def test_punctuation_only(self):
        assert normalize_text("...!!!") == ""


class TestBuildFailureSignature:
    def test_basic_signature(self):
        theorem = Theorem(
            theorem_id="thm_001",
            name="Test",
            status=TheoremStatus.ESTABLISHED,
            claim="Test claim",
            support=[],
            failure_modes=["Mode One", "Mode Two"],
            missing_structure=["Structure A"],
        )
        sig = build_failure_signature(theorem)

        assert "mode one" in sig
        assert "mode two" in sig
        assert "structure a" in sig

    def test_empty_modes(self):
        theorem = Theorem(
            theorem_id="thm_001",
            name="Test",
            status=TheoremStatus.ESTABLISHED,
            claim="Test",
            support=[],
            failure_modes=[],
            missing_structure=[],
        )
        sig = build_failure_signature(theorem)
        assert sig == ""

    def test_normalized_signature(self):
        theorem = Theorem(
            theorem_id="thm_001",
            name="Test",
            status=TheoremStatus.ESTABLISHED,
            claim="Test",
            support=[],
            failure_modes=["ADJACENT Cells!!"],
            missing_structure=["Local Pattern..."],
        )
        sig = build_failure_signature(theorem)

        assert "adjacent cells" in sig
        assert "local pattern" in sig
        # No punctuation
        assert "!" not in sig
        assert "." not in sig


class TestBuildSignatureWithLawContext:
    def test_includes_constraining_laws(self):
        theorem = Theorem(
            theorem_id="thm_001",
            name="Test",
            status=TheoremStatus.ESTABLISHED,
            claim="Test",
            support=[
                LawSupport("law_001", "confirms"),
                LawSupport("law_002", "constrains"),
            ],
            failure_modes=["Mode"],
            missing_structure=[],
        )
        sig = build_signature_with_law_context(theorem, include_fail_laws=True)

        assert "mode" in sig
        # PHASE-C: underscores are preserved for law IDs
        assert "law_002" in sig
        assert "law_001" not in sig  # confirms, not constrains

    def test_excludes_laws_when_disabled(self):
        theorem = Theorem(
            theorem_id="thm_001",
            name="Test",
            status=TheoremStatus.ESTABLISHED,
            claim="Test",
            support=[
                LawSupport("law_002", "constrains"),
            ],
            failure_modes=["Mode"],
            missing_structure=[],
        )
        sig = build_signature_with_law_context(theorem, include_fail_laws=False)

        assert "mode" in sig
        assert "law_002" not in sig


class TestHashSignature:
    def test_consistent_hash(self):
        sig = "test signature"
        hash1 = hash_signature(sig)
        hash2 = hash_signature(sig)
        assert hash1 == hash2

    def test_different_inputs_different_hashes(self):
        hash1 = hash_signature("signature one")
        hash2 = hash_signature("signature two")
        assert hash1 != hash2

    def test_hash_length(self):
        h = hash_signature("test")
        assert len(h) == 24


class TestExtractKeyTerms:
    def test_basic_extraction(self):
        terms = extract_key_terms("hello world test")
        assert terms == {"hello", "world", "test"}

    def test_min_length_filter(self):
        terms = extract_key_terms("a ab abc abcd", min_length=3)
        assert "a" not in terms
        assert "ab" not in terms
        assert "abc" in terms
        assert "abcd" in terms

    def test_empty_input(self):
        terms = extract_key_terms("")
        assert terms == set()

    def test_all_short_terms(self):
        terms = extract_key_terms("a b c", min_length=3)
        assert terms == set()


class TestComputeJaccardSimilarity:
    def test_identical_signatures(self):
        sig = "hello world test"
        sim = compute_jaccard_similarity(sig, sig)
        assert sim == 1.0

    def test_completely_different(self):
        sig1 = "hello world"
        sig2 = "foo bar baz"
        sim = compute_jaccard_similarity(sig1, sig2)
        assert sim == 0.0

    def test_partial_overlap(self):
        sig1 = "hello world test"
        sig2 = "hello world other"
        sim = compute_jaccard_similarity(sig1, sig2)
        # 2 common terms (hello, world), 4 total terms
        assert sim == 2 / 4

    def test_empty_signatures(self):
        assert compute_jaccard_similarity("", "") == 1.0
        assert compute_jaccard_similarity("hello", "") == 0.0
        assert compute_jaccard_similarity("", "hello") == 0.0

    def test_short_terms_filtered(self):
        # "a" and "b" are filtered out (< 3 chars)
        sig1 = "hello a"
        sig2 = "hello b"
        sim = compute_jaccard_similarity(sig1, sig2)
        # Only "hello" remains in both
        assert sim == 1.0
