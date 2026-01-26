"""Tests for Symbol Scrambler."""

import pytest

from src.proposer.scrambler import (
    SymbolScrambler,
    SymbolMapping,
    get_default_scrambler,
    to_abstract,
    to_physical,
)


class TestSymbolMapping:
    """Tests for SymbolMapping dataclass."""

    def test_valid_mapping(self):
        """Valid single-character mappings work."""
        mapping = SymbolMapping(".", "W", "Symbol W")
        assert mapping.physical == "."
        assert mapping.abstract == "W"
        assert mapping.description == "Symbol W"

    def test_invalid_physical_length(self):
        """Rejects multi-character physical symbols."""
        with pytest.raises(ValueError, match="single characters"):
            SymbolMapping("..", "_", "Test")

    def test_invalid_abstract_length(self):
        """Rejects multi-character abstract symbols."""
        with pytest.raises(ValueError, match="single characters"):
            SymbolMapping(".", "__", "Test")


class TestSymbolScrambler:
    """Tests for SymbolScrambler class."""

    @pytest.fixture
    def scrambler(self):
        return SymbolScrambler()

    def test_default_mappings(self, scrambler):
        """Default scrambler has correct mappings."""
        assert len(scrambler.mappings) == 4
        assert scrambler.abstract_symbols == ["W", "A", "B", "K"]
        assert scrambler.physical_symbols == [".", ">", "<", "X"]

    def test_to_abstract_state(self, scrambler):
        """Converts physical state to abstract."""
        physical = "..><.X.."
        abstract = scrambler.to_abstract(physical)
        assert abstract == "WWABWKWW"

    def test_to_physical_state(self, scrambler):
        """Converts abstract state to physical."""
        abstract = "WWABWKWW"
        physical = scrambler.to_physical(abstract)
        assert physical == "..><.X.."

    def test_roundtrip_state(self, scrambler):
        """Converting to abstract and back preserves state."""
        original = "...><><X..."
        roundtrip = scrambler.to_physical(scrambler.to_abstract(original))
        assert roundtrip == original

    def test_translate_observable_expr_to_physical(self, scrambler):
        """Translates observable expressions to physical symbols."""
        abstract_expr = "count('A') + count('B') + 2*count('K')"
        physical_expr = scrambler.translate_observable_expr(abstract_expr, to_physical=True)
        assert physical_expr == "count('>') + count('<') + 2*count('X')"

    def test_translate_observable_expr_to_abstract(self, scrambler):
        """Translates observable expressions to abstract symbols."""
        physical_expr = "count('>') + count('<') + 2*count('X')"
        abstract_expr = scrambler.translate_observable_expr(physical_expr, to_physical=False)
        assert abstract_expr == "count('A') + count('B') + 2*count('K')"

    def test_translate_pattern_to_physical(self, scrambler):
        """Translates neighbor patterns to physical."""
        abstract = "ABW"
        physical = scrambler.translate_pattern(abstract, to_physical=True)
        assert physical == "><."

    def test_translate_pattern_to_abstract(self, scrambler):
        """Translates neighbor patterns to abstract."""
        physical = "><."
        abstract = scrambler.translate_pattern(physical, to_physical=False)
        assert abstract == "ABW"

    def test_get_symbol_glossary(self, scrambler):
        """Glossary contains all symbols with descriptions."""
        glossary = scrambler.get_symbol_glossary()
        assert "W - Symbol W" in glossary
        assert "A - Symbol A" in glossary
        assert "B - Symbol B" in glossary
        assert "K - Symbol K" in glossary


class TestTranslateLawToPhysical:
    """Tests for law translation."""

    @pytest.fixture
    def scrambler(self):
        return SymbolScrambler()

    def test_translate_trigger_symbol(self, scrambler):
        """Translates trigger_symbol from abstract to physical."""
        law_data = {"trigger_symbol": "A"}
        result = scrambler.translate_law_to_physical(law_data)
        assert result["trigger_symbol"] == ">"

    def test_translate_result_symbol(self, scrambler):
        """Translates result_symbol from abstract to physical."""
        law_data = {"result_symbol": "B"}
        result = scrambler.translate_law_to_physical(law_data)
        assert result["result_symbol"] == "<"

    def test_translate_neighbor_pattern(self, scrambler):
        """Translates neighbor_pattern from abstract to physical."""
        law_data = {"neighbor_pattern": "ABW"}
        result = scrambler.translate_law_to_physical(law_data)
        assert result["neighbor_pattern"] == "><."

    def test_translate_observables(self, scrambler):
        """Translates observable expressions."""
        law_data = {
            "observables": [
                {"name": "R", "expr": "count('A')"},
                {"name": "L", "expr": "count('B')"},
            ]
        }
        result = scrambler.translate_law_to_physical(law_data)
        assert result["observables"][0]["expr"] == "count('>')"
        assert result["observables"][1]["expr"] == "count('<')"

    def test_translate_preconditions(self, scrambler):
        """Translates precondition lhs expressions."""
        law_data = {
            "preconditions": [
                {"lhs": "count('A')", "op": ">", "rhs": 0},
            ]
        }
        result = scrambler.translate_law_to_physical(law_data)
        assert result["preconditions"][0]["lhs"] == "count('>')"

    def test_preserves_other_fields(self, scrambler):
        """Translation preserves fields without symbols."""
        law_data = {
            "law_id": "test_law",
            "template": "invariant",
            "forbidden": "violation found",
        }
        result = scrambler.translate_law_to_physical(law_data)
        assert result["law_id"] == "test_law"
        assert result["template"] == "invariant"
        assert result["forbidden"] == "violation found"


class TestTranslateResultToAbstract:
    """Tests for result translation back to abstract."""

    @pytest.fixture
    def scrambler(self):
        return SymbolScrambler()

    def test_translate_counterexample_state(self, scrambler):
        """Translates counterexample states to abstract."""
        result_data = {
            "counterexample": {
                "initial_state": "..><..",
                "trajectory_excerpt": ["..><..", "..X...", ".>.<.."],
            }
        }
        result = scrambler.translate_result_to_abstract(result_data)
        assert result["counterexample"]["initial_state"] == "WWABWW"
        assert result["counterexample"]["trajectory_excerpt"][0] == "WWABWW"
        assert result["counterexample"]["trajectory_excerpt"][1] == "WWKWWW"
        assert result["counterexample"]["trajectory_excerpt"][2] == "WAWBWW"


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_default_scrambler(self):
        """get_default_scrambler returns a scrambler."""
        scrambler = get_default_scrambler()
        assert isinstance(scrambler, SymbolScrambler)

    def test_to_abstract(self):
        """Module-level to_abstract works."""
        assert to_abstract("..><..") == "WWABWW"

    def test_to_physical(self):
        """Module-level to_physical works."""
        assert to_physical("WWABWW") == "..><.."
