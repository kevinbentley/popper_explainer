"""Main theorem generator that orchestrates the generation pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from src.db.models import CounterexampleRecord, EvaluationRecord, LawRecord
from src.db.repo import Repository
from src.theorem.models import LawSnapshot, Theorem, TheoremBatch, TheoremGenerationArtifact
from src.theorem.parser import TheoremParser
from src.theorem.prompt import (
    PROMPT_TEMPLATE_VERSION,
    build_prompt,
    compute_prompt_hash,
    compute_snapshot_hash,
)
from src.theorem.signature import build_failure_signature, hash_signature

if TYPE_CHECKING:
    from src.db.llm_logger import LLMLogger


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    def generate(self, prompt: str) -> str:
        """Generate a response from a prompt."""
        ...


# System instruction for theorem generation
THEOREM_SYSTEM_INSTRUCTION = """You are a Popperian theory-construction agent. Your role is to synthesize empirically-tested laws into higher-level theorems that explain patterns in the data.

Key principles:
1. Theorems must combine multiple laws - never just restate a single law
2. Use both PASS laws (reliable patterns) and FAIL laws (boundaries/constraints)
3. Be explicit about failure modes and missing observables
4. Prefer local explanations over global-count-based ones when counts fail
5. Output valid JSON only - no markdown, no explanations outside JSON"""


@dataclass
class TheoremGeneratorConfig:
    """Configuration for theorem generation."""

    max_pass_laws: int = 50
    max_fail_laws: int = 50
    target_theorem_count: int = 10


class TheoremGenerator:
    """Generates theorems from law snapshots using an LLM."""

    def __init__(
        self,
        client: LLMClient,
        config: TheoremGeneratorConfig | None = None,
        system_instruction: str | None = None,
        llm_logger: LLMLogger | None = None,
    ):
        self.client = client
        self.config = config or TheoremGeneratorConfig()
        self.parser = TheoremParser()
        self.system_instruction = system_instruction or THEOREM_SYSTEM_INSTRUCTION
        self._llm_logger = llm_logger

    def build_law_snapshot(
        self,
        repo: Repository,
        max_pass: int | None = None,
        max_fail: int | None = None,
    ) -> list[LawSnapshot]:
        """Build law snapshots from the database.

        Args:
            repo: Repository instance
            max_pass: Maximum PASS laws to include
            max_fail: Maximum FAIL laws to include

        Returns:
            List of LawSnapshot objects
        """
        max_pass = max_pass or self.config.max_pass_laws
        max_fail = max_fail or self.config.max_fail_laws

        snapshots: list[LawSnapshot] = []

        # Get PASS laws with their evaluations
        pass_laws = repo.get_laws_with_status("PASS", limit=max_pass)
        for law_record, eval_record in pass_laws:
            snapshot = self._law_record_to_snapshot(
                law_record, eval_record, repo
            )
            snapshots.append(snapshot)

        # Get FAIL laws with their evaluations
        fail_laws = repo.get_laws_with_status("FAIL", limit=max_fail)
        for law_record, eval_record in fail_laws:
            snapshot = self._law_record_to_snapshot(
                law_record, eval_record, repo
            )
            snapshots.append(snapshot)

        return snapshots

    def _law_record_to_snapshot(
        self,
        law: LawRecord,
        evaluation: EvaluationRecord,
        repo: Repository,
    ) -> LawSnapshot:
        """Convert a law record and evaluation to a snapshot."""
        # Parse the law JSON
        try:
            law_data = json.loads(law.law_json)
        except json.JSONDecodeError:
            law_data = {}

        # Get counterexample if FAIL
        counterexample: dict[str, Any] | None = None
        if evaluation.status == "FAIL":
            cx_records = repo.get_counterexamples_for_law(law.law_id)
            if cx_records:
                cx = cx_records[0]  # Most recent
                counterexample = {
                    "initial_state": cx.initial_state,
                    "t_fail": cx.t_fail,
                    "t_max": cx.t_max,
                }

        # Parse power metrics
        power_metrics: dict[str, Any] | None = None
        if evaluation.power_metrics_json:
            try:
                power_metrics = json.loads(evaluation.power_metrics_json)
            except json.JSONDecodeError:
                pass

        # Extract claim text
        claim = law_data.get("claim_text", law_data.get("claim", str(law_data)))

        # Extract observables
        observables = law_data.get("observables", [])

        return LawSnapshot(
            law_id=law.law_id,
            template=law.template,
            claim=claim,
            status=evaluation.status,
            observables=observables,
            counterexample=counterexample,
            power_metrics=power_metrics,
        )

    def generate(
        self,
        law_snapshots: list[LawSnapshot],
        target_count: int | None = None,
    ) -> TheoremBatch:
        """Generate theorems from law snapshots.

        Args:
            law_snapshots: List of law snapshots to use as context
            target_count: Target number of theorems to generate

        Returns:
            TheoremBatch with generated theorems
        """
        target_count = target_count or self.config.target_theorem_count

        # Build prompt
        prompt = build_prompt(law_snapshots, target_count)
        prompt_hash = compute_prompt_hash(prompt)
        prompt_tokens = len(prompt) // 4  # Rough estimate

        # Call LLM - handle different client interfaces
        start_time = time.time()
        response = ""
        success = True
        error_message = None

        try:
            # Check if client supports system_instruction parameter (GeminiClient)
            if hasattr(self.client, 'generate') and 'system_instruction' in self.client.generate.__code__.co_varnames:
                response = self.client.generate(
                    prompt,
                    system_instruction=self.system_instruction,
                    temperature=0.7,
                )
            else:
                # Basic client - just pass prompt
                response = self.client.generate(prompt)
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            runtime_ms = int((time.time() - start_time) * 1000)

            # Log LLM call
            if self._llm_logger:
                self._llm_logger.log_call(
                    prompt=prompt,
                    response=response,
                    success=success,
                    system_instruction=self.system_instruction,
                    prompt_tokens=prompt_tokens,
                    duration_ms=runtime_ms,
                    error_message=error_message,
                )

        # Parse response
        parse_result = self.parser.parse(response)

        # Add failure signatures to theorems
        for theorem in parse_result.theorems:
            signature = build_failure_signature(theorem)
            # Store in theorem object (extend if needed)

        return TheoremBatch(
            theorems=parse_result.theorems,
            rejections=parse_result.rejections,
            prompt_hash=prompt_hash,
            runtime_ms=runtime_ms,
            warnings=parse_result.warnings,
        )

    def generate_with_signatures(
        self,
        law_snapshots: list[LawSnapshot],
        target_count: int | None = None,
    ) -> tuple[TheoremBatch, dict[str, tuple[str, str]]]:
        """Generate theorems and compute their failure signatures.

        Returns:
            Tuple of (TheoremBatch, dict mapping theorem_id to (signature, hash))
        """
        batch = self.generate(law_snapshots, target_count)

        signatures: dict[str, tuple[str, str]] = {}
        for theorem in batch.theorems:
            sig = build_failure_signature(theorem)
            sig_hash = hash_signature(sig)
            signatures[theorem.theorem_id] = (sig, sig_hash)

        return batch, signatures

    def set_llm_logger(self, logger: LLMLogger | None) -> None:
        """Set or update the LLM logger.

        Args:
            logger: LLM logger to use, or None to disable logging
        """
        self._llm_logger = logger

    def set_llm_logger_context(
        self,
        run_id: str | None = None,
        iteration_id: int | None = None,
        phase: str | None = None,
    ) -> None:
        """Update the LLM logger's context.

        Args:
            run_id: Orchestration run ID
            iteration_id: Current iteration index
            phase: Current phase name
        """
        if self._llm_logger:
            self._llm_logger.set_context(
                run_id=run_id,
                iteration_id=iteration_id,
                phase=phase,
            )

    def generate_with_artifact(
        self,
        law_snapshots: list[LawSnapshot],
        target_count: int | None = None,
        model_name: str = "unknown",
        model_params: dict[str, Any] | None = None,
    ) -> tuple[TheoremBatch, TheoremGenerationArtifact]:
        """Generate theorems with full artifact capture for reproducibility.

        This method captures all data needed to reproduce the generation:
        - Snapshot hash of input laws
        - Prompt template version
        - Model name and parameters
        - Raw LLM response
        - Parsed response

        Args:
            law_snapshots: List of law snapshots to use as context
            target_count: Target number of theorems to generate
            model_name: Name of the model being used
            model_params: Model parameters (temperature, max_tokens, etc.)

        Returns:
            Tuple of (TheoremBatch, TheoremGenerationArtifact)
        """
        target_count = target_count or self.config.target_theorem_count
        model_params = model_params or {}

        # Compute snapshot hash for reproducibility
        snapshot_hash = compute_snapshot_hash(law_snapshots)

        # Build prompt
        prompt = build_prompt(law_snapshots, target_count)
        prompt_hash = compute_prompt_hash(prompt)

        # Call LLM and capture raw response
        start_time = time.time()

        # Check if client supports system_instruction parameter (GeminiClient)
        if hasattr(self.client, 'generate') and 'system_instruction' in self.client.generate.__code__.co_varnames:
            raw_response = self.client.generate(
                prompt,
                system_instruction=self.system_instruction,
                temperature=model_params.get("temperature", 0.7),
            )
        else:
            # Basic client - just pass prompt
            raw_response = self.client.generate(prompt)

        runtime_ms = int((time.time() - start_time) * 1000)

        # Parse response
        parse_result = self.parser.parse(raw_response)

        # Extract parsed JSON for artifact
        parsed_response: list[dict[str, Any]] = []
        for theorem in parse_result.theorems:
            parsed_response.append(theorem.to_dict())

        # Compute artifact hash
        artifact_content = f"{snapshot_hash}:{raw_response}"
        artifact_hash = hashlib.sha256(artifact_content.encode()).hexdigest()[:32]

        # Create artifact
        artifact = TheoremGenerationArtifact(
            artifact_hash=artifact_hash,
            snapshot_hash=snapshot_hash,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            model_name=model_name,
            model_params=model_params,
            raw_response=raw_response,
            parsed_response=parsed_response,
            created_at=datetime.now(),
        )

        # Add failure signatures to theorems
        for theorem in parse_result.theorems:
            signature = build_failure_signature(theorem)
            # Store in theorem object (extend if needed)

        batch = TheoremBatch(
            theorems=parse_result.theorems,
            rejections=parse_result.rejections,
            prompt_hash=prompt_hash,
            runtime_ms=runtime_ms,
            warnings=parse_result.warnings,
        )

        return batch, artifact


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, response: str | None = None):
        self._response = response or self._default_response()

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self._response

    def _default_response(self) -> str:
        return json.dumps([
            {
                "name": "Conservation of Particle Count",
                "status": "Established",
                "claim": "The total number of particles (>, <, X) is conserved over time.",
                "support": [
                    {"law_id": "law_001", "role": "confirms"},
                    {"law_id": "law_002", "role": "confirms"},
                ],
                "failure_modes": [
                    "If particles can be created or destroyed at boundaries"
                ],
                "missing_structure": [],
            },
            {
                "name": "Local Collision Dynamics",
                "status": "Conditional",
                "claim": "Collisions only occur between adjacent opposite-moving particles.",
                "support": [
                    {"law_id": "law_003", "role": "confirms"},
                    {"law_id": "law_004", "role": "constrains"},
                ],
                "failure_modes": [
                    "May not hold for simultaneous multi-particle collisions",
                    "Boundary effects could alter collision rules",
                ],
                "missing_structure": [
                    "Local adjacency observable needed",
                    "Gap distribution between particles",
                ],
            },
        ])


def create_gemini_client(
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7,
):
    """Create a GeminiClient configured for theorem generation.

    Args:
        api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        model: Model to use
        temperature: Sampling temperature

    Returns:
        Configured GeminiClient
    """
    from src.proposer.client import GeminiClient, GeminiConfig

    config = GeminiConfig(
        api_key=api_key,
        model=model,
        temperature=temperature,
        json_mode=True,
        max_output_tokens=16384,
    )
    return GeminiClient(config)
