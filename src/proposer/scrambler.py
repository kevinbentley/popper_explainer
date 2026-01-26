"""Symbol Scrambler for integrity shielding.

This module implements a symbol abstraction layer to prevent the LLM from
inferring physical meaning from the actual universe symbols.

The Problem:
- Symbols like '>' and '<' visually suggest direction
- The LLM might use this to "guess" rather than discover through falsification
- This undermines the Popperian methodology

The Solution:
- Present the LLM with abstract symbols (W, A, B, K)
- Translate LLM proposals from abstract to physical before testing
- Translate results back to abstract for LLM feedback

Scramble Mapping (default):
    . → W
    > → A
    < → B
    X → K

No descriptive labels are attached to the abstract symbols. The LLM must
discover all properties through experimentation, not from symbol names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SymbolMapping:
    """Bidirectional mapping between physical and abstract symbols."""

    physical: str
    abstract: str
    description: str  # Neutral description for LLM

    def __post_init__(self):
        if len(self.physical) != 1 or len(self.abstract) != 1:
            raise ValueError("Symbols must be single characters")


@dataclass
class SymbolScrambler:
    """Scrambles symbols to prevent LLM from inferring physics.

    Usage:
        scrambler = SymbolScrambler()

        # In prompt generation:
        abstract_state = scrambler.to_abstract("..><..")

        # In response parsing (LLM uses abstract symbols):
        physical_state = scrambler.to_physical("WWABWW")

        # For law translation:
        physical_law = scrambler.translate_law_to_physical(abstract_law)
        abstract_result = scrambler.translate_result_to_abstract(physical_result)
    """

    mappings: list[SymbolMapping] = field(default_factory=lambda: [
        SymbolMapping(".", "W", "Symbol W"),
        SymbolMapping(">", "A", "Symbol A"),
        SymbolMapping("<", "B", "Symbol B"),
        SymbolMapping("X", "K", "Symbol K"),
    ])

    def __post_init__(self):
        # Build lookup tables
        self._to_abstract: dict[str, str] = {}
        self._to_physical: dict[str, str] = {}
        self._descriptions: dict[str, str] = {}

        for m in self.mappings:
            self._to_abstract[m.physical] = m.abstract
            self._to_physical[m.abstract] = m.physical
            self._descriptions[m.abstract] = m.description

    @property
    def abstract_symbols(self) -> list[str]:
        """Get list of abstract symbols."""
        return [m.abstract for m in self.mappings]

    @property
    def physical_symbols(self) -> list[str]:
        """Get list of physical symbols."""
        return [m.physical for m in self.mappings]

    def get_symbol_glossary(self) -> str:
        """Get a glossary of abstract symbols for the LLM prompt.

        Returns neutral descriptions that don't leak physics.
        """
        lines = ["Available symbols:"]
        for m in self.mappings:
            lines.append(f"  {m.abstract} - {m.description}")
        return "\n".join(lines)

    def to_abstract(self, text: str) -> str:
        """Convert physical symbols to abstract symbols.

        Args:
            text: String containing physical symbols

        Returns:
            String with physical symbols replaced by abstract symbols
        """
        result = text
        for phys, abst in self._to_abstract.items():
            result = result.replace(phys, abst)
        return result

    def to_physical(self, text: str) -> str:
        """Convert abstract symbols to physical symbols.

        Args:
            text: String containing abstract symbols

        Returns:
            String with abstract symbols replaced by physical symbols
        """
        result = text
        for abst, phys in self._to_physical.items():
            result = result.replace(abst, phys)
        return result

    def translate_state(self, state: str, to_abstract: bool = True) -> str:
        """Translate a state string between representations.

        Args:
            state: State string
            to_abstract: If True, physical→abstract; if False, abstract→physical

        Returns:
            Translated state string
        """
        if to_abstract:
            return self.to_abstract(state)
        else:
            return self.to_physical(state)

    def translate_observable_expr(self, expr: str, to_physical: bool = True) -> str:
        """Translate an observable expression between representations.

        Handles expressions like count('A') → count('>')

        Args:
            expr: Observable expression string
            to_physical: If True, abstract→physical; if False, physical→abstract

        Returns:
            Translated expression
        """
        if to_physical:
            # Replace abstract symbols in quoted contexts
            result = expr
            for abst, phys in self._to_physical.items():
                # Handle single-quoted symbols: 'A' → '>'
                result = result.replace(f"'{abst}'", f"'{phys}'")
                # Handle double-quoted symbols: "A" → ">"
                result = result.replace(f'"{abst}"', f'"{phys}"')
            return result
        else:
            result = expr
            for phys, abst in self._to_abstract.items():
                result = result.replace(f"'{phys}'", f"'{abst}'")
                result = result.replace(f'"{phys}"', f'"{abst}"')
            return result

    def translate_pattern(self, pattern: str, to_physical: bool = True) -> str:
        """Translate a neighbor pattern between representations.

        Args:
            pattern: 3-character neighborhood pattern
            to_physical: If True, abstract→physical; if False, physical→abstract

        Returns:
            Translated pattern
        """
        if to_physical:
            return self.to_physical(pattern)
        else:
            return self.to_abstract(pattern)

    def translate_law_to_physical(self, law_data: dict[str, Any]) -> dict[str, Any]:
        """Translate a law from abstract (LLM) format to physical format.

        This is called after parsing the LLM's response, before testing.

        Args:
            law_data: Law dictionary with abstract symbols

        Returns:
            Law dictionary with physical symbols
        """
        result = law_data.copy()

        # Translate trigger_symbol for local_transition
        if "trigger_symbol" in result:
            result["trigger_symbol"] = self.to_physical(result["trigger_symbol"])

        # Translate result_symbol for local_transition
        if "result_symbol" in result:
            result["result_symbol"] = self.to_physical(result["result_symbol"])

        # Translate neighbor_pattern
        if "neighbor_pattern" in result and result["neighbor_pattern"]:
            result["neighbor_pattern"] = self.translate_pattern(
                result["neighbor_pattern"], to_physical=True
            )

        # Translate observables
        if "observables" in result and result["observables"]:
            translated_obs = []
            for obs in result["observables"]:
                if obs is None:
                    continue
                new_obs = obs.copy()
                if "expr" in new_obs:
                    new_obs["expr"] = self.translate_observable_expr(
                        new_obs["expr"], to_physical=True
                    )
                translated_obs.append(new_obs)
            result["observables"] = translated_obs

        # Translate preconditions (lhs may contain observable expressions)
        if "preconditions" in result and result["preconditions"]:
            translated_prec = []
            for prec in result["preconditions"]:
                if prec is None:
                    continue
                new_prec = prec.copy()
                if "lhs" in new_prec and isinstance(new_prec["lhs"], str):
                    new_prec["lhs"] = self.translate_observable_expr(
                        new_prec["lhs"], to_physical=True
                    )
                translated_prec.append(new_prec)
            result["preconditions"] = translated_prec

        # Translate claim_ast (recursive)
        if "claim_ast" in result and result["claim_ast"]:
            result["claim_ast"] = self._translate_ast(
                result["claim_ast"], to_physical=True
            )

        return result

    def translate_result_to_abstract(
        self,
        result_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Translate test results from physical to abstract format.

        This is called before presenting results to the LLM.

        Args:
            result_data: Result dictionary with physical symbols

        Returns:
            Result dictionary with abstract symbols
        """
        result = result_data.copy()

        # Translate counterexample states
        if "counterexample" in result and result["counterexample"]:
            cx = result["counterexample"].copy()
            if "initial_state" in cx:
                cx["initial_state"] = self.to_abstract(cx["initial_state"])
            if "trajectory_excerpt" in cx and isinstance(cx["trajectory_excerpt"], list):
                cx["trajectory_excerpt"] = [
                    self.to_abstract(s) for s in cx["trajectory_excerpt"]
                ]
            result["counterexample"] = cx

        return result

    def _translate_ast(
        self,
        ast: dict[str, Any] | None,
        to_physical: bool = True,
    ) -> dict[str, Any] | None:
        """Recursively translate symbols in a claim AST.

        Args:
            ast: AST node dictionary
            to_physical: Direction of translation

        Returns:
            Translated AST
        """
        if ast is None:
            return None

        result = ast.copy()

        # Handle string values that might contain symbols
        if "obs" in result and isinstance(result["obs"], str):
            # Observable names might have symbols in them (rare but possible)
            pass  # Usually obs names are abstract, not translated

        # Recursively translate children
        for key in ["lhs", "rhs", "arg"]:
            if key in result and isinstance(result[key], dict):
                result[key] = self._translate_ast(result[key], to_physical)

        return result


# Module-level default scrambler
_default_scrambler = SymbolScrambler()


def get_default_scrambler() -> SymbolScrambler:
    """Get the default symbol scrambler."""
    return _default_scrambler


def to_abstract(text: str) -> str:
    """Convert physical symbols to abstract using default scrambler."""
    return _default_scrambler.to_abstract(text)


def to_physical(text: str) -> str:
    """Convert abstract symbols to physical using default scrambler."""
    return _default_scrambler.to_physical(text)
