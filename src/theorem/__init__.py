"""Theorem generation module.

This module handles:
- Synthesizing theorems from PASS/FAIL laws
- Clustering failure signatures
- Proposing new observables to close explanatory gaps
"""

from src.theorem.clustering import (
    BUCKET_KEYWORDS,
    ClusterActionMapper,
    KeywordBucketAssigner,
    MultiLabelBucketAssigner,
    SemanticClusterer,
    TfidfClusterer,
    TwoPassClusterer,
    tag_buckets,
)
from src.theorem.generator import (
    LLMClient,
    MockLLMClient,
    THEOREM_SYSTEM_INSTRUCTION,
    TheoremGenerator,
    TheoremGeneratorConfig,
    create_gemini_client,
)
from src.theorem.models import (
    FailureBucket,
    FailureCluster,
    LawSnapshot,
    LawSupport,
    ObservableProposal,
    Theorem,
    TheoremBatch,
    TheoremStatus,
)
from src.theorem.observable_proposer import (
    BUCKET_OBSERVABLE_RULES,
    ObservableProposer,
    ObservableTemplate,
)
from src.theorem.parser import TheoremParser, TheoremParseResult
from src.theorem.prompt import (
    build_laws_section,
    build_prompt,
    compute_prompt_hash,
    format_law_snapshots_for_context,
)
from src.theorem.signature import (
    build_failure_signature,
    build_signature_with_law_context,
    compute_jaccard_similarity,
    extract_key_terms,
    hash_signature,
    normalize_text,
)

__all__ = [
    # Models
    "FailureBucket",
    "FailureCluster",
    "LawSnapshot",
    "LawSupport",
    "ObservableProposal",
    "Theorem",
    "TheoremBatch",
    "TheoremStatus",
    # Generator
    "LLMClient",
    "MockLLMClient",
    "THEOREM_SYSTEM_INSTRUCTION",
    "TheoremGenerator",
    "TheoremGeneratorConfig",
    "create_gemini_client",
    # Parser
    "TheoremParser",
    "TheoremParseResult",
    # Prompt
    "build_laws_section",
    "build_prompt",
    "compute_prompt_hash",
    "format_law_snapshots_for_context",
    # Signature
    "build_failure_signature",
    "build_signature_with_law_context",
    "compute_jaccard_similarity",
    "extract_key_terms",
    "hash_signature",
    "normalize_text",
    # Clustering
    "BUCKET_KEYWORDS",
    "ClusterActionMapper",
    "KeywordBucketAssigner",
    "MultiLabelBucketAssigner",
    "SemanticClusterer",
    "TfidfClusterer",
    "TwoPassClusterer",
    "tag_buckets",
    # Observable proposer
    "BUCKET_OBSERVABLE_RULES",
    "ObservableProposer",
    "ObservableTemplate",
]
