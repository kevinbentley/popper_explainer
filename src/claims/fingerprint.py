"""Semantic fingerprinting for law deduplication.

Creates canonical representations of laws to detect semantic duplicates
like "momentum_conserved" vs "net_momentum_conserved" or "A == B" vs "B == A".

Canonicalization rules:
1. Sort commutative operations (and, or, +, *, ==, !=)
2. Normalize comparisons (A < B becomes canonical B > A)
3. Normalize observable expressions to canonical form
4. Hash the result for efficient comparison
"""

import hashlib
import json
import re
from typing import Any

from src.claims.schema import CandidateLaw, Observable

# Commutative operators - operands can be sorted
COMMUTATIVE_OPS = {"and", "or", "+", "*", "==", "!="}

# Comparison operators and their flipped versions
COMPARISON_FLIPS = {
    "<": ">",
    "<=": ">=",
    ">": "<",
    ">=": "<=",
}


def canonicalize_ast(ast: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize an AST for comparison.

    Rules:
    - Sort operands of commutative operators
    - Normalize comparisons to canonical form (larger side on left)
    - Normalize algebraic equivalents (X*2 <= Y → X <= Y/2)
    - Recursively canonicalize sub-expressions

    Args:
        ast: The AST to canonicalize

    Returns:
        Canonicalized AST (new dict, doesn't modify input)
    """
    if not isinstance(ast, dict):
        return ast

    # Handle different node types
    if "const" in ast:
        return {"const": ast["const"]}

    if "var" in ast:
        return {"var": ast["var"]}

    if "t_plus_1" in ast:
        return {"t_plus_1": True}

    if "obs" in ast:
        return {
            "obs": ast["obs"],
            "t": canonicalize_ast(ast["t"]),
        }

    if "op" in ast:
        op = ast["op"]

        # Handle unary not
        if op == "not":
            return {"op": "not", "arg": canonicalize_ast(ast["arg"])}

        # Canonicalize operands first
        lhs = canonicalize_ast(ast["lhs"])
        rhs = canonicalize_ast(ast["rhs"])

        # Algebraic normalization for comparisons with multiplication/division
        # Transform: (A * k) op B  →  A op (B / k)  when k is a constant
        # Transform: A op (B / k)  →  (A * k) op B  (then pick canonical form)
        if op in ("<=", "<", ">=", ">"):
            lhs, rhs, op = _normalize_comparison_algebra(lhs, rhs, op)

        # For commutative operators, sort operands by their string representation
        if op in COMMUTATIVE_OPS:
            lhs_str = json.dumps(lhs, sort_keys=True)
            rhs_str = json.dumps(rhs, sort_keys=True)
            if lhs_str > rhs_str:
                lhs, rhs = rhs, lhs

        # For comparison operators, normalize to canonical form
        # Convention: put "larger" expression on left
        if op in COMPARISON_FLIPS:
            lhs_str = json.dumps(lhs, sort_keys=True)
            rhs_str = json.dumps(rhs, sort_keys=True)
            if lhs_str > rhs_str:
                # Flip the comparison
                lhs, rhs = rhs, lhs
                op = COMPARISON_FLIPS[op]

        return {"op": op, "lhs": lhs, "rhs": rhs}

    return ast


def _normalize_comparison_algebra(
    lhs: dict[str, Any],
    rhs: dict[str, Any],
    op: str
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Normalize algebraic expressions in comparisons.

    Transforms:
    - (A * k) <= B  →  A <= (B / k)
    - A <= (B / k)  →  A <= (B / k)  (canonical form: division on right)

    This allows detecting that (X * 2 <= Y) is equivalent to (X <= Y / 2).

    Args:
        lhs: Left operand
        rhs: Right operand
        op: Comparison operator

    Returns:
        Tuple of (normalized_lhs, normalized_rhs, op)
    """
    # Check if LHS is (A * k) where k is a constant
    if (isinstance(lhs, dict) and lhs.get("op") == "*"):
        lhs_left = lhs.get("lhs", {})
        lhs_right = lhs.get("rhs", {})

        # Find the constant (could be on either side due to commutativity)
        const_val = None
        other_operand = None

        if isinstance(lhs_right, dict) and "const" in lhs_right:
            const_val = lhs_right["const"]
            other_operand = lhs_left
        elif isinstance(lhs_left, dict) and "const" in lhs_left:
            const_val = lhs_left["const"]
            other_operand = lhs_right

        if const_val is not None and const_val != 0:
            # Transform: (A * k) op B  →  A op (B / k)
            new_lhs = other_operand
            new_rhs = {"op": "/", "lhs": rhs, "rhs": {"const": const_val}}
            return new_lhs, new_rhs, op

    # Check if RHS is (B / k) - this is already canonical form
    # No transformation needed

    return lhs, rhs, op


def canonicalize_observable_expr(expr: str) -> str:
    """Canonicalize an observable expression string.

    Normalizes:
    - Whitespace
    - Quoting style for symbols
    - Order of commutative operations

    Args:
        expr: Observable expression like "count('>')" or "count( \">\" )"

    Returns:
        Canonical form
    """
    # Normalize whitespace
    expr = re.sub(r'\s+', '', expr)

    # Normalize quotes (use single quotes)
    expr = expr.replace('"', "'")

    # Lowercase function names
    expr = re.sub(r'([a-zA-Z_]+)\(', lambda m: m.group(1).lower() + '(', expr)

    return expr


def extract_observable_semantics(obs: Observable) -> str:
    """Extract semantic meaning from an observable.

    Returns a canonical string representing what the observable measures.

    Args:
        obs: Observable definition

    Returns:
        Canonical semantic identifier
    """
    expr = canonicalize_observable_expr(obs.expr)

    # Normalize case variations
    expr_lower = expr.lower()

    # Map known equivalent expressions to canonical forms
    CANONICAL_FORMS = {
        # Particle counts (handle case variations)
        "count('>')": "count_right",
        "count('<')": "count_left",
        "count('x')": "count_collision",
        "count('.')": "count_empty",

        # Derived counts
        "count('>')+count('<')": "count_movers",
        "count('<')+count('>')": "count_movers",
        "count('>')+count('<')+count('x')": "count_active",
        "count('<')+count('>')+count('x')": "count_active",
        "count('x')+count('>')+count('<')": "count_active",

        # Momentum
        "count('>')-count('<')": "momentum",
        "-(count('<')-count('>'))": "momentum",

        # Total particles
        "count('>')+count('<')+2*count('x')": "total_particles",
        "count('<')+count('>')+2*count('x')": "total_particles",
        "2*count('x')+count('>')+count('<')": "total_particles",

        # Grid properties
        "grid_length": "grid_length",
        "len(state)": "grid_length",
    }

    # Try exact match first
    if expr in CANONICAL_FORMS:
        return CANONICAL_FORMS[expr]

    # Try lowercase match
    if expr_lower in CANONICAL_FORMS:
        return CANONICAL_FORMS[expr_lower]

    # Return the normalized expression as the semantic key
    return expr


def compute_semantic_fingerprint(law: CandidateLaw) -> str:
    """Compute a semantic fingerprint for a law.

    Laws with the same fingerprint are semantically equivalent.

    Args:
        law: The law to fingerprint

    Returns:
        Hex string fingerprint
    """
    parts = []

    # 1. Template type
    parts.append(f"template:{law.template.value}")

    # 2. Transform (for symmetry laws)
    if law.transform:
        parts.append(f"transform:{law.transform.lower()}")

    # 3. Observable semantics (sorted by semantic meaning, not name)
    # We only care about WHAT is measured, not what it's called
    obs_semantics = set()
    for obs in law.observables:
        semantic = extract_observable_semantics(obs)
        obs_semantics.add(semantic)
    parts.append(f"obs:[{','.join(sorted(obs_semantics))}]")

    # 4. Canonicalized claim AST
    if law.claim_ast:
        canonical_ast = canonicalize_ast(law.claim_ast)
        # Replace observable names with their semantic equivalents
        canonical_ast = _substitute_obs_semantics(canonical_ast, law.observables)
        ast_str = json.dumps(canonical_ast, sort_keys=True)
        parts.append(f"ast:{ast_str}")
    else:
        # Fall back to normalized claim string
        claim = law.claim.lower().replace(" ", "")
        parts.append(f"claim:{claim}")

    # 5. Preconditions (sorted, canonicalized)
    precond_strs = []
    for p in law.preconditions:
        lhs = p.lhs.lower().replace(" ", "")
        rhs = str(p.rhs)
        op = p.op.value

        # Normalize comparison direction
        if op in COMPARISON_FLIPS and lhs > rhs:
            lhs, rhs = rhs, lhs
            op = COMPARISON_FLIPS[op]

        precond_strs.append(f"{lhs}{op}{rhs}")
    precond_strs.sort()
    parts.append(f"pre:[{','.join(precond_strs)}]")

    # Combine and hash
    fingerprint_str = "|".join(parts)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:24]


def _substitute_obs_semantics(ast: dict[str, Any], observables: list[Observable]) -> dict[str, Any]:
    """Replace observable names with semantic equivalents in AST.

    Args:
        ast: The AST to modify
        observables: Observable definitions

    Returns:
        Modified AST with semantic names
    """
    if not isinstance(ast, dict):
        return ast

    # Build mapping from name to semantic
    name_to_semantic = {}
    for obs in observables:
        name_to_semantic[obs.name] = extract_observable_semantics(obs)

    return _substitute_obs_recursive(ast, name_to_semantic)


def _substitute_obs_recursive(ast: dict[str, Any], name_map: dict[str, str]) -> dict[str, Any]:
    """Recursively substitute observable names."""
    if not isinstance(ast, dict):
        return ast

    if "obs" in ast:
        obs_name = ast["obs"]
        semantic_name = name_map.get(obs_name, obs_name)
        return {
            "obs": semantic_name,
            "t": _substitute_obs_recursive(ast["t"], name_map),
        }

    if "op" in ast:
        op = ast["op"]
        if op == "not":
            return {"op": "not", "arg": _substitute_obs_recursive(ast["arg"], name_map)}
        return {
            "op": op,
            "lhs": _substitute_obs_recursive(ast["lhs"], name_map),
            "rhs": _substitute_obs_recursive(ast["rhs"], name_map),
        }

    if "const" in ast:
        return {"const": ast["const"]}
    if "var" in ast:
        return {"var": ast["var"]}
    if "t_plus_1" in ast:
        return {"t_plus_1": True}

    return ast


def laws_are_equivalent(law1: CandidateLaw, law2: CandidateLaw) -> bool:
    """Check if two laws are semantically equivalent.

    Args:
        law1: First law
        law2: Second law

    Returns:
        True if semantically equivalent
    """
    return compute_semantic_fingerprint(law1) == compute_semantic_fingerprint(law2)


def fingerprint_description(law: CandidateLaw) -> str:
    """Get a human-readable description of what the fingerprint captures.

    Useful for debugging why two laws match or don't match.

    Args:
        law: The law to describe

    Returns:
        Human-readable fingerprint components
    """
    parts = []

    parts.append(f"Template: {law.template.value}")

    if law.transform:
        parts.append(f"Transform: {law.transform}")

    for obs in law.observables:
        semantic = extract_observable_semantics(obs)
        parts.append(f"Observable {obs.name}: {semantic}")

    if law.claim_ast:
        canonical = canonicalize_ast(law.claim_ast)
        parts.append(f"Canonical AST: {json.dumps(canonical, sort_keys=True)}")

    return "\n".join(parts)
