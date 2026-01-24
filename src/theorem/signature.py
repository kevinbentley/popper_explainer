"""Failure signature building for theorem clustering."""

import hashlib
import re

from src.theorem.models import MissingStructureType, SupportRole, Theorem


def normalize_text(text: str) -> str:
    """Normalize text for signature comparison.

    - Lowercase
    - Strip punctuation EXCEPT underscores (for law IDs like law_001)
    - Collapse whitespace
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation but keep underscores for law IDs
    text = re.sub(r"[^a-z0-9_\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_failure_signature(theorem: Theorem) -> str:
    """Build a failure signature from a theorem.

    Concatenates failure_modes and missing_structure, then normalizes.
    """
    parts: list[str] = []

    # Add failure modes
    for fm in theorem.failure_modes:
        parts.append(normalize_text(fm))

    # Add missing structure
    for ms in theorem.missing_structure:
        parts.append(normalize_text(ms))

    # Join with space
    signature = " ".join(parts)

    return signature


def hash_signature(signature: str) -> str:
    """Compute SHA256 hash of a signature for indexing."""
    return hashlib.sha256(signature.encode()).hexdigest()[:24]


def build_signature_with_law_context(
    theorem: Theorem,
    include_fail_laws: bool = True,
) -> str:
    """Build an extended signature that includes context from supporting laws.

    Args:
        theorem: The theorem to build a signature for
        include_fail_laws: Whether to include FAIL law IDs in the signature

    Returns:
        Extended normalized signature
    """
    parts: list[str] = []

    # Add failure modes
    for fm in theorem.failure_modes:
        parts.append(normalize_text(fm))

    # Add missing structure
    for ms in theorem.missing_structure:
        parts.append(normalize_text(ms))

    # Optionally add fail law references (as tokens)
    if include_fail_laws:
        for sup in theorem.support:
            if sup.role == "constrains":
                # Include constraining law IDs as tokens
                parts.append(f"law_{normalize_text(sup.law_id)}")

    return " ".join(parts)


def extract_key_terms(signature: str, min_length: int = 3) -> set[str]:
    """Extract key terms from a signature for clustering.

    Args:
        signature: Normalized signature text
        min_length: Minimum term length to include

    Returns:
        Set of key terms
    """
    terms = signature.split()
    return {t for t in terms if len(t) >= min_length}


def compute_jaccard_similarity(sig1: str, sig2: str) -> float:
    """Compute Jaccard similarity between two signatures.

    Args:
        sig1: First signature
        sig2: Second signature

    Returns:
        Jaccard similarity coefficient (0.0 to 1.0)
    """
    terms1 = extract_key_terms(sig1)
    terms2 = extract_key_terms(sig2)

    if not terms1 and not terms2:
        return 1.0  # Both empty = identical
    if not terms1 or not terms2:
        return 0.0  # One empty = no overlap

    intersection = len(terms1 & terms2)
    union = len(terms1 | terms2)

    return intersection / union if union > 0 else 0.0


# =============================================================================
# PHASE-D: Role-Coded Signatures
# =============================================================================
# Role prefixes: C: (confirms), X: (constrains), R: (refutes_alternative)
# Structure prefixes: DEF:, LOC:, TMP:, MEC:


# Mapping from SupportRole to prefix
ROLE_PREFIX_MAP: dict[SupportRole, str] = {
    SupportRole.CONFIRMS: "C",
    SupportRole.CONSTRAINS: "X",
    SupportRole.REFUTES_ALTERNATIVE: "R",
}

# Mapping from MissingStructureType to prefix
STRUCTURE_PREFIX_MAP: dict[MissingStructureType, str] = {
    MissingStructureType.DEFINITION_MISSING: "DEF",
    MissingStructureType.LOCAL_STRUCTURE_MISSING: "LOC",
    MissingStructureType.TEMPORAL_STRUCTURE_MISSING: "TMP",
    MissingStructureType.MECHANISM_MISSING: "MEC",
}


def build_role_coded_signature(theorem: Theorem) -> str:
    """Build a role-coded signature for symbolic clustering.

    This signature encodes both support roles and missing structure types
    as prefixed tokens, enabling more precise clustering.

    Format:
        C:law_001 X:law_002 DEF:observable_name LOC:spatial_pattern

    Args:
        theorem: The theorem to build a signature for

    Returns:
        Role-coded signature string
    """
    tokens: list[str] = []

    # Add support with role prefixes
    for support in theorem.support:
        role = support.role if isinstance(support.role, SupportRole) else SupportRole(support.role)
        prefix = ROLE_PREFIX_MAP.get(role, "C")
        # Normalize law_id
        law_id = normalize_text(support.law_id)
        tokens.append(f"{prefix}:{law_id}")

    # Add typed missing structure with type prefixes
    for tms in theorem.typed_missing_structure:
        prefix = STRUCTURE_PREFIX_MAP.get(tms.type, "DEF")
        # Normalize target
        target = normalize_text(tms.target)
        # Extract key terms from target (first 2-3 significant words)
        key_words = [w for w in target.split() if len(w) >= 3][:3]
        if key_words:
            tokens.append(f"{prefix}:{'+'.join(key_words)}")

    return " ".join(tokens)


def build_role_coded_signature_with_context(
    theorem: Theorem,
    include_failure_modes: bool = True,
) -> str:
    """Build an extended role-coded signature including failure modes.

    Args:
        theorem: The theorem to build a signature for
        include_failure_modes: Whether to include failure mode keywords

    Returns:
        Extended role-coded signature string
    """
    base_sig = build_role_coded_signature(theorem)

    if not include_failure_modes:
        return base_sig

    # Add failure mode keywords with FM: prefix
    fm_tokens: list[str] = []
    for fm in theorem.failure_modes:
        normalized = normalize_text(fm)
        key_words = [w for w in normalized.split() if len(w) >= 3][:2]
        if key_words:
            fm_tokens.append(f"FM:{'+'.join(key_words)}")

    if fm_tokens:
        return f"{base_sig} {' '.join(fm_tokens)}"
    return base_sig


def extract_role_tokens(signature: str) -> dict[str, list[str]]:
    """Extract tokens by role from a role-coded signature.

    Args:
        signature: Role-coded signature string

    Returns:
        Dict mapping role prefixes to lists of tokens
    """
    result: dict[str, list[str]] = {
        "C": [],  # confirms
        "X": [],  # constrains
        "R": [],  # refutes
        "DEF": [],  # definition missing
        "LOC": [],  # local structure missing
        "TMP": [],  # temporal structure missing
        "MEC": [],  # mechanism missing
        "FM": [],  # failure mode
    }

    for token in signature.split():
        if ":" in token:
            prefix, value = token.split(":", 1)
            if prefix in result:
                result[prefix].append(value)

    return result


def compute_role_weighted_similarity(sig1: str, sig2: str) -> float:
    """Compute similarity between role-coded signatures with role weighting.

    Different roles have different importance for clustering:
    - Constraining laws (X) are more diagnostic
    - Missing structures are highly relevant

    Args:
        sig1: First role-coded signature
        sig2: Second role-coded signature

    Returns:
        Weighted similarity coefficient (0.0 to 1.0)
    """
    tokens1 = extract_role_tokens(sig1)
    tokens2 = extract_role_tokens(sig2)

    # Weights for different token types
    weights = {
        "C": 1.0,   # confirms - moderate
        "X": 2.0,   # constrains - high (more diagnostic)
        "R": 1.5,   # refutes - moderate-high
        "DEF": 2.0,  # definition gap - high
        "LOC": 1.5,  # local structure
        "TMP": 1.5,  # temporal structure
        "MEC": 1.5,  # mechanism
        "FM": 1.0,   # failure mode
    }

    total_weight = 0.0
    weighted_intersection = 0.0

    for role, weight in weights.items():
        set1 = set(tokens1.get(role, []))
        set2 = set(tokens2.get(role, []))

        if set1 or set2:
            union_size = len(set1 | set2)
            intersection_size = len(set1 & set2)
            if union_size > 0:
                role_similarity = intersection_size / union_size
                weighted_intersection += weight * role_similarity
                total_weight += weight

    if total_weight == 0:
        return 1.0 if not sig1 and not sig2 else 0.0

    return weighted_intersection / total_weight
