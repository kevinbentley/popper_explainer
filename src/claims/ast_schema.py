"""Structured AST schema for law claims.

This module defines a JSON-serializable AST for claims that eliminates
parsing ambiguity. Time indexing is explicit, and only declared
observable names are valid references.

Example AST for "N(t) == N(0)":
{
    "op": "==",
    "lhs": {"obs": "N", "t": {"var": "t"}},
    "rhs": {"obs": "N", "t": {"const": 0}}
}

Example AST for "X_count(t) > 0 => X_count(t+1) == 0":
{
    "op": "=>",
    "lhs": {"op": ">", "lhs": {"obs": "X_count", "t": {"var": "t"}}, "rhs": {"const": 0}},
    "rhs": {"op": "==", "lhs": {"obs": "X_count", "t": {"t_plus_1": true}}, "rhs": {"const": 0}}
}
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Union


class ASTNodeType(Enum):
    """Types of AST nodes."""
    CONST = "const"           # Constant value: {"const": 5}
    VAR = "var"               # Variable reference: {"var": "t"}
    OBS = "obs"               # Observable at time: {"obs": "N", "t": ...}
    BINOP = "op"              # Binary operation: {"op": "+", "lhs": ..., "rhs": ...}
    T_PLUS_1 = "t_plus_1"     # Shorthand for t+1: {"t_plus_1": true}


# Valid binary operators
ARITHMETIC_OPS = {"+", "-", "*", "/"}
COMPARISON_OPS = {"==", "!=", "<", "<=", ">", ">="}
LOGICAL_OPS = {"=>", "and", "or", "not"}
ALL_OPS = ARITHMETIC_OPS | COMPARISON_OPS | LOGICAL_OPS


@dataclass
class ValidationError:
    """Error found during AST validation."""
    path: str  # JSON path to the error
    message: str


class ClaimAST:
    """Validator and interpreter for structured claim ASTs."""

    def __init__(self, declared_observables: set[str]):
        """Initialize with set of declared observable names.

        Args:
            declared_observables: Names of observables defined in the law
        """
        self.declared_observables = declared_observables
        self.errors: list[ValidationError] = []

    def validate(self, ast: dict[str, Any], path: str = "root") -> bool:
        """Validate an AST node.

        Args:
            ast: The AST node to validate
            path: Current path for error reporting

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(ast, dict):
            self.errors.append(ValidationError(path, f"Expected dict, got {type(ast).__name__}"))
            return False

        # Determine node type
        if "const" in ast:
            return self._validate_const(ast, path)
        elif "var" in ast:
            return self._validate_var(ast, path)
        elif "obs" in ast:
            return self._validate_obs(ast, path)
        elif "op" in ast:
            return self._validate_binop(ast, path)
        elif "t_plus_1" in ast:
            return self._validate_t_plus_1(ast, path)
        else:
            self.errors.append(ValidationError(path, f"Unknown node type. Keys: {list(ast.keys())}"))
            return False

    def _validate_const(self, ast: dict, path: str) -> bool:
        """Validate a constant node."""
        value = ast["const"]
        if not isinstance(value, (int, float)):
            self.errors.append(ValidationError(path, f"const must be a number, got {type(value).__name__}"))
            return False
        return True

    def _validate_var(self, ast: dict, path: str) -> bool:
        """Validate a variable node."""
        value = ast["var"]
        if value != "t":
            self.errors.append(ValidationError(path, f"Only 'var': 't' is allowed, got '{value}'"))
            return False
        return True

    def _validate_t_plus_1(self, ast: dict, path: str) -> bool:
        """Validate a t+1 shorthand node."""
        if ast.get("t_plus_1") is not True:
            self.errors.append(ValidationError(path, "t_plus_1 must be true"))
            return False
        return True

    def _validate_obs(self, ast: dict, path: str) -> bool:
        """Validate an observable reference node."""
        obs_name = ast.get("obs")
        if not isinstance(obs_name, str):
            self.errors.append(ValidationError(path, f"obs must be a string"))
            return False

        if obs_name not in self.declared_observables:
            self.errors.append(ValidationError(
                path,
                f"Unknown observable '{obs_name}'. Declared: {sorted(self.declared_observables)}"
            ))
            return False

        # Validate time index
        if "t" not in ast:
            self.errors.append(ValidationError(path, "obs node requires 't' field"))
            return False

        return self.validate(ast["t"], f"{path}.t")

    def _validate_binop(self, ast: dict, path: str) -> bool:
        """Validate a binary operation node."""
        op = ast.get("op")
        if op not in ALL_OPS:
            self.errors.append(ValidationError(path, f"Unknown operator '{op}'. Valid: {sorted(ALL_OPS)}"))
            return False

        # Check for required operands
        if op == "not":
            # Unary operator
            if "arg" not in ast:
                self.errors.append(ValidationError(path, "'not' requires 'arg' field"))
                return False
            return self.validate(ast["arg"], f"{path}.arg")
        else:
            # Binary operator
            if "lhs" not in ast:
                self.errors.append(ValidationError(path, f"'{op}' requires 'lhs' field"))
                return False
            if "rhs" not in ast:
                self.errors.append(ValidationError(path, f"'{op}' requires 'rhs' field"))
                return False

            valid_lhs = self.validate(ast["lhs"], f"{path}.lhs")
            valid_rhs = self.validate(ast["rhs"], f"{path}.rhs")
            return valid_lhs and valid_rhs


def validate_claim_ast(
    ast: dict[str, Any],
    declared_observables: set[str],
) -> tuple[bool, list[ValidationError]]:
    """Validate a claim AST.

    Args:
        ast: The AST to validate
        declared_observables: Set of declared observable names

    Returns:
        Tuple of (is_valid, list of errors)
    """
    validator = ClaimAST(declared_observables)
    is_valid = validator.validate(ast)
    return is_valid, validator.errors


def ast_to_string(ast: dict[str, Any]) -> str:
    """Convert an AST back to a human-readable string.

    Args:
        ast: The AST to convert

    Returns:
        Human-readable string representation
    """
    if "const" in ast:
        return str(ast["const"])
    elif "var" in ast:
        return ast["var"]
    elif "t_plus_1" in ast:
        return "t+1"
    elif "obs" in ast:
        t_str = ast_to_string(ast["t"])
        return f"{ast['obs']}({t_str})"
    elif "op" in ast:
        op = ast["op"]
        if op == "not":
            return f"not({ast_to_string(ast['arg'])})"
        elif op == "=>":
            return f"({ast_to_string(ast['lhs'])}) => ({ast_to_string(ast['rhs'])})"
        else:
            return f"({ast_to_string(ast['lhs'])} {op} {ast_to_string(ast['rhs'])})"
    return str(ast)
