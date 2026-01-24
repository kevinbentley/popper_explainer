"""Tests for theorem parser."""

import json

import pytest

from src.theorem.models import TheoremStatus
from src.theorem.parser import TheoremParser, TheoremParseResult


class TestTheoremParser:
    @pytest.fixture
    def parser(self):
        return TheoremParser()

    def test_parse_valid_json(self, parser):
        response = json.dumps([
            {
                "name": "Test Theorem",
                "status": "Established",
                "claim": "Test claim",
                "support": [
                    {"law_id": "law_001", "role": "confirms"},
                    {"law_id": "law_002", "role": "constrains"},
                ],
                "failure_modes": ["mode1"],
                "missing_structure": ["struct1"],
            }
        ])
        result = parser.parse(response)

        assert len(result.theorems) == 1
        assert result.theorems[0].name == "Test Theorem"
        assert result.theorems[0].status == TheoremStatus.ESTABLISHED
        assert len(result.theorems[0].support) == 2

    def test_parse_json_in_code_block(self, parser):
        response = """Here are the theorems:

```json
[
  {
    "name": "Block Theorem",
    "status": "Conditional",
    "claim": "Test claim",
    "support": [
      {"law_id": "law_001", "role": "confirms"},
      {"law_id": "law_002", "role": "confirms"}
    ],
    "failure_modes": [],
    "missing_structure": []
  }
]
```

That's all!"""
        result = parser.parse(response)

        assert len(result.theorems) == 1
        assert result.theorems[0].name == "Block Theorem"

    def test_parse_multiple_theorems(self, parser):
        response = json.dumps([
            {
                "name": "Theorem 1",
                "status": "Established",
                "claim": "Claim 1",
                "support": [
                    {"law_id": "law_001", "role": "confirms"},
                    {"law_id": "law_002", "role": "confirms"},
                ],
            },
            {
                "name": "Theorem 2",
                "status": "Conjectural",
                "claim": "Claim 2",
                "support": [
                    {"law_id": "law_003", "role": "confirms"},
                    {"law_id": "law_004", "role": "constrains"},
                ],
            },
        ])
        result = parser.parse(response)

        assert len(result.theorems) == 2
        assert result.theorems[0].status == TheoremStatus.ESTABLISHED
        assert result.theorems[1].status == TheoremStatus.CONJECTURAL

    def test_reject_invalid_status(self, parser):
        response = json.dumps([
            {
                "name": "Bad Status",
                "status": "InvalidStatus",
                "claim": "Test",
                "support": [
                    {"law_id": "law_001", "role": "confirms"},
                    {"law_id": "law_002", "role": "confirms"},
                ],
            }
        ])
        result = parser.parse(response)

        assert len(result.theorems) == 0
        assert len(result.rejections) == 1
        assert "invalid status" in result.rejections[0][1].lower()

    def test_reject_insufficient_support(self, parser):
        response = json.dumps([
            {
                "name": "Single Support",
                "status": "Established",
                "claim": "Test",
                "support": [
                    {"law_id": "law_001", "role": "confirms"},
                ],
            }
        ])
        result = parser.parse(response)

        assert len(result.theorems) == 0
        assert len(result.rejections) == 1
        assert "at least 2" in result.rejections[0][1].lower()

    def test_reject_missing_name(self, parser):
        response = json.dumps([
            {
                "status": "Established",
                "claim": "Test",
                "support": [
                    {"law_id": "law_001", "role": "confirms"},
                    {"law_id": "law_002", "role": "confirms"},
                ],
            }
        ])
        result = parser.parse(response)

        assert len(result.theorems) == 0
        assert len(result.rejections) == 1
        assert "name" in result.rejections[0][1].lower()

    def test_reject_missing_claim(self, parser):
        response = json.dumps([
            {
                "name": "No Claim",
                "status": "Established",
                "support": [
                    {"law_id": "law_001", "role": "confirms"},
                    {"law_id": "law_002", "role": "confirms"},
                ],
            }
        ])
        result = parser.parse(response)

        assert len(result.theorems) == 0
        assert len(result.rejections) == 1
        assert "claim" in result.rejections[0][1].lower()

    def test_invalid_json_warning(self, parser):
        response = "This is not valid JSON at all"
        result = parser.parse(response)

        assert len(result.theorems) == 0
        assert len(result.warnings) > 0

    def test_not_array_warning(self, parser):
        response = json.dumps({"name": "Single object"})
        result = parser.parse(response)

        assert len(result.theorems) == 0
        assert len(result.warnings) > 0
        assert "array" in result.warnings[0].lower()

    def test_partial_success(self, parser):
        response = json.dumps([
            {
                "name": "Good Theorem",
                "status": "Established",
                "claim": "Good claim",
                "support": [
                    {"law_id": "law_001", "role": "confirms"},
                    {"law_id": "law_002", "role": "confirms"},
                ],
            },
            {
                "name": "Bad Theorem",
                "status": "InvalidStatus",
                "claim": "Bad claim",
                "support": [],
            },
        ])
        result = parser.parse(response)

        assert len(result.theorems) == 1
        assert len(result.rejections) == 1

    def test_default_role_normalization(self, parser):
        response = json.dumps([
            {
                "name": "Test",
                "status": "Established",
                "claim": "Test",
                "support": [
                    {"law_id": "law_001", "role": "supports"},  # Invalid role
                    {"law_id": "law_002"},  # Missing role
                ],
            }
        ])
        result = parser.parse(response)

        assert len(result.theorems) == 1
        # Both should be normalized to "confirms"
        assert all(s.role == "confirms" for s in result.theorems[0].support)

    def test_generate_theorem_id(self, parser):
        response = json.dumps([
            {
                "name": "No ID",
                "status": "Established",
                "claim": "Test",
                "support": [
                    {"law_id": "law_001", "role": "confirms"},
                    {"law_id": "law_002", "role": "confirms"},
                ],
            }
        ])
        result = parser.parse(response)

        assert len(result.theorems) == 1
        assert result.theorems[0].theorem_id.startswith("thm_")

    def test_preserve_provided_theorem_id(self, parser):
        response = json.dumps([
            {
                "theorem_id": "custom_id_123",
                "name": "Custom ID",
                "status": "Established",
                "claim": "Test",
                "support": [
                    {"law_id": "law_001", "role": "confirms"},
                    {"law_id": "law_002", "role": "confirms"},
                ],
            }
        ])
        result = parser.parse(response)

        assert len(result.theorems) == 1
        assert result.theorems[0].theorem_id == "custom_id_123"
