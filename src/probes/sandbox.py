"""Secure execution environment for LLM-written Python probe functions.

A probe is a function `def probe(S) -> float | int` where S is a list of
single-character strings representing the universe state. Temporal probes
take two parameters: `def probe(S_current, S_next) -> float | int` to
measure transitions between timesteps. This module validates probe source
via AST inspection and executes it in a restricted namespace with timeout
enforcement.
"""

from __future__ import annotations

import ast
import signal
import threading
from typing import Union


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class ProbeError(Exception):
    """Base exception for all probe-related errors."""


class ProbeSyntaxError(ProbeError):
    """Probe source code has invalid Python syntax."""


class ProbeValidationError(ProbeError):
    """Probe source code uses disallowed constructs."""


class ProbeRuntimeError(ProbeError):
    """Probe raised an exception during execution."""


class ProbeTimeoutError(ProbeError):
    """Probe execution exceeded the allowed time budget."""


class ProbeReturnTypeError(ProbeError):
    """Probe returned a non-numeric value."""


# ---------------------------------------------------------------------------
# AST validation
# ---------------------------------------------------------------------------

_DISALLOWED_NAMES = frozenset({
    "import", "exec", "eval", "compile", "open",
    "__import__", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "hasattr",
    "breakpoint", "exit", "quit", "input", "print",
    "classmethod", "staticmethod", "property", "super",
    "type", "object", "memoryview", "bytearray", "bytes",
})

_DISALLOWED_ATTR_PREFIXES = ("__",)


class _ProbeASTValidator(ast.NodeVisitor):
    """Walk the AST and reject disallowed constructs."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    # --- Import statements ---------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        self.errors.append(
            f"Line {node.lineno}: 'import' statements are not allowed"
        )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.errors.append(
            f"Line {node.lineno}: 'from ... import' statements are not allowed"
        )
        self.generic_visit(node)

    # --- exec / eval / compile as Call nodes ----------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in _DISALLOWED_NAMES:
                self.errors.append(
                    f"Line {node.lineno}: '{node.func.id}()' is not allowed"
                )
        self.generic_visit(node)

    # --- Attribute access with __ prefix --------------------------------------
    def visit_Attribute(self, node: ast.Attribute) -> None:
        for prefix in _DISALLOWED_ATTR_PREFIXES:
            if node.attr.startswith(prefix):
                self.errors.append(
                    f"Line {node.lineno}: attribute access '{node.attr}' "
                    f"(starts with '{prefix}') is not allowed"
                )
        self.generic_visit(node)

    # --- Name references to disallowed builtins ------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _DISALLOWED_NAMES:
            # Only flag if used as a standalone name (not call - handled above)
            if not isinstance(getattr(node, '_parent', None), ast.Call):
                pass  # Call visitor handles calls; bare names are less risky
        if node.id.startswith("__") and node.id.endswith("__"):
            self.errors.append(
                f"Line {node.lineno}: dunder name '{node.id}' is not allowed"
            )
        self.generic_visit(node)

    # --- Global / Nonlocal ---------------------------------------------------
    def visit_Global(self, node: ast.Global) -> None:
        self.errors.append(
            f"Line {node.lineno}: 'global' statement is not allowed"
        )
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.errors.append(
            f"Line {node.lineno}: 'nonlocal' statement is not allowed"
        )
        self.generic_visit(node)


def validate_probe_source(source: str) -> tuple[bool, str | None]:
    """Validate probe source code via AST inspection.

    Returns:
        (True, None) if valid, or (False, error_message) if invalid.
    """
    # Parse
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"

    # Must define exactly one function named 'probe'
    func_defs = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "probe"
    ]
    if len(func_defs) == 0:
        return False, "Source must define a function named 'probe'"
    if len(func_defs) > 1:
        return False, "Source must define exactly one function named 'probe'"

    probe_func = func_defs[0]
    # Must accept 1 or 2 positional parameters
    args = probe_func.args
    positional = args.posonlyargs + args.args
    if len(positional) not in (1, 2):
        return False, (
            f"probe() must accept 1 or 2 parameters, "
            f"got {len(positional)}"
        )

    # AST security walk
    validator = _ProbeASTValidator()
    validator.visit(tree)
    if validator.errors:
        return False, "; ".join(validator.errors)

    return True, None


def detect_probe_arity(source: str) -> int:
    """Detect whether a probe takes 1 or 2 parameters.

    Args:
        source: Python source code defining `def probe(...)`.

    Returns:
        1 for single-state probes, 2 for temporal (transition) probes.

    Raises:
        ProbeSyntaxError: If source has syntax errors.
        ProbeValidationError: If no valid probe function is found.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ProbeSyntaxError(f"Syntax error: {exc}") from exc

    func_defs = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "probe"
    ]
    if not func_defs:
        raise ProbeValidationError("Source must define a function named 'probe'")

    probe_func = func_defs[0]
    positional = probe_func.args.posonlyargs + probe_func.args.args
    return len(positional)


# ---------------------------------------------------------------------------
# Restricted execution namespace
# ---------------------------------------------------------------------------

_ALLOWED_BUILTINS = {
    "range": range,
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "float": float,
    "int": int,
    "sum": sum,
    "True": True,
    "False": False,
    "None": None,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "reversed": reversed,
    "bool": bool,
    "str": str,
    "round": round,
    "any": any,
    "all": all,
    "map": map,
    "filter": filter,
    "isinstance": isinstance,
}


def _make_restricted_namespace() -> dict:
    """Create a namespace with only whitelisted builtins."""
    return {"__builtins__": _ALLOWED_BUILTINS}


# ---------------------------------------------------------------------------
# Execution with timeout
# ---------------------------------------------------------------------------

def execute_probe(
    source: str,
    state: list[str],
    timeout_ms: int = 100,
    next_state: list[str] | None = None,
) -> Union[float, int]:
    """Execute a probe function on the given state.

    Args:
        source: Python source code defining `def probe(S): ...` or
                `def probe(S_current, S_next): ...` for temporal probes.
        state: List of single-character strings (the universe state).
        timeout_ms: Maximum execution time in milliseconds.
        next_state: Optional next state for temporal (2-param) probes.

    Returns:
        Numeric result (float or int).

    Raises:
        ProbeSyntaxError: If source has syntax errors.
        ProbeValidationError: If source uses disallowed constructs.
        ProbeRuntimeError: If probe raises during execution, or if a
            2-param probe is called without next_state.
        ProbeTimeoutError: If execution exceeds timeout.
        ProbeReturnTypeError: If probe returns a non-numeric value.
    """
    # Validate first
    valid, error = validate_probe_source(source)
    if not valid:
        if "Syntax error" in (error or ""):
            raise ProbeSyntaxError(error)
        raise ProbeValidationError(error)

    arity = detect_probe_arity(source)
    if arity == 2 and next_state is None:
        raise ProbeRuntimeError(
            "Temporal probe requires next_state but none was provided"
        )

    # Compile
    try:
        code = compile(source, "<probe>", "exec")
    except SyntaxError as exc:
        raise ProbeSyntaxError(f"Compilation error: {exc}") from exc

    # Execute definition in restricted namespace
    namespace = _make_restricted_namespace()
    try:
        exec(code, namespace)  # noqa: S102 - intentional restricted exec
    except Exception as exc:
        raise ProbeRuntimeError(
            f"Error defining probe function: {exc}"
        ) from exc

    probe_fn = namespace.get("probe")
    if probe_fn is None or not callable(probe_fn):
        raise ProbeRuntimeError("probe() function not found after execution")

    # Execute probe with timeout
    result_box: list = []
    error_box: list = []

    def _run():
        try:
            if arity == 2:
                result_box.append(probe_fn(list(state), list(next_state)))
            else:
                result_box.append(probe_fn(list(state)))
        except Exception as exc:
            error_box.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_ms / 1000.0)

    if thread.is_alive():
        raise ProbeTimeoutError(
            f"Probe execution exceeded {timeout_ms}ms timeout"
        )

    if error_box:
        raise ProbeRuntimeError(
            f"Probe raised during execution: {error_box[0]}"
        ) from error_box[0]

    if not result_box:
        raise ProbeRuntimeError("Probe produced no result")

    result = result_box[0]

    # Type check
    if not isinstance(result, (int, float)):
        raise ProbeReturnTypeError(
            f"Probe must return int or float, got {type(result).__name__}: "
            f"{result!r}"
        )

    return result
