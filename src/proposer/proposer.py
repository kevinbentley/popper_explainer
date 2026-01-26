"""Main law proposer orchestrating LLM-driven law discovery."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.claims.schema import CandidateLaw
from src.proposer.client import GeminiClient, GeminiConfig, MockGeminiClient
from src.proposer.memory import DiscoveryMemory, DiscoveryMemorySnapshot
from src.proposer.parser import ParseResult, ResponseParser
from src.proposer.prompt import PromptBuilder, UniverseContract
from src.proposer.ranking import RankingFeatures, RankingModel
from src.proposer.redundancy import RedundancyDetector, RedundancyMatch

if TYPE_CHECKING:
    from src.db.llm_logger import LLMLogger


@dataclass
class ProposalRequest:
    """Request for law proposals.

    Attributes:
        count: Number of laws to request
        target_templates: Templates to focus on
        exclude_templates: Templates to avoid
        temperature: LLM temperature override
    """

    count: int = 5
    target_templates: list[str] | None = None
    exclude_templates: list[str] | None = None
    temperature: float | None = None


@dataclass
class ProposalBatch:
    """Result of a proposal iteration.

    Attributes:
        laws: Ranked candidate laws
        features: Ranking features for each law
        rejections: Laws rejected during parsing/validation
        redundant: Laws filtered as redundant
        prompt_hash: Hash of the prompt used
        prompt_tokens: Estimated prompt token count
        response_tokens: Estimated response token count
        runtime_ms: Time taken
        warnings: Non-fatal issues encountered
        research_log: LLM's research notes for continuity between iterations
    """

    laws: list[CandidateLaw] = field(default_factory=list)
    features: list[RankingFeatures] = field(default_factory=list)
    rejections: list[tuple[dict[str, Any], str]] = field(default_factory=list)
    redundant: list[tuple[CandidateLaw, RedundancyMatch]] = field(default_factory=list)
    prompt_hash: str = ""
    prompt_tokens: int = 0
    response_tokens: int = 0
    runtime_ms: int = 0
    warnings: list[str] = field(default_factory=list)
    research_log: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "laws_count": len(self.laws),
            "law_ids": [l.law_id for l in self.laws],
            "rejections_count": len(self.rejections),
            "redundant_count": len(self.redundant),
            "prompt_hash": self.prompt_hash,
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "runtime_ms": self.runtime_ms,
            "warnings": self.warnings,
            "has_research_log": self.research_log is not None,
        }


@dataclass
class ProposerConfig:
    """Configuration for law proposer.

    Attributes:
        max_token_budget: Maximum tokens for prompts
        include_counterexamples: Whether to include counterexample gallery
        strict_parsing: Whether to reject laws with any issues
        add_to_redundancy_filter: Whether to add proposed laws to filter
        verbose: Whether to store full prompts/responses for debugging
    """

    max_token_budget: int = 8000
    include_counterexamples: bool = True
    strict_parsing: bool = False
    add_to_redundancy_filter: bool = True
    verbose: bool = False


class LawProposer:
    """Main orchestrator for LLM-driven law proposal.

    Usage:
        proposer = LawProposer(client, contract)
        batch = proposer.propose(memory, request)
    """

    def __init__(
        self,
        client: GeminiClient | MockGeminiClient | None = None,
        contract: UniverseContract | None = None,
        config: ProposerConfig | None = None,
        llm_logger: LLMLogger | None = None,
    ):
        """Initialize law proposer.

        Args:
            client: LLM client (defaults to GeminiClient)
            contract: Universe contract (defaults to standard kinetic grid)
            config: Proposer configuration
            llm_logger: Optional LLM logger for capturing all LLM interactions
        """
        self.client = client or GeminiClient()
        self.contract = contract or UniverseContract()
        self.config = config or ProposerConfig()
        self._llm_logger = llm_logger

        self._prompt_builder = PromptBuilder(
            max_token_budget=self.config.max_token_budget,
            include_counterexamples=self.config.include_counterexamples,
        )
        self._parser = ResponseParser(strict=self.config.strict_parsing)
        self._redundancy = RedundancyDetector()
        self._ranking = RankingModel(redundancy_detector=self._redundancy)

        # Audit log
        self._iterations: list[dict[str, Any]] = []

        # Verbose logging storage
        self._last_system_instruction: str = ""
        self._last_prompt: str = ""
        self._last_response: str = ""

        # Research log for continuity across iterations
        self._research_log: str | None = None

    def propose(
        self,
        memory: DiscoveryMemorySnapshot | DiscoveryMemory,
        request: ProposalRequest | None = None,
    ) -> ProposalBatch:
        """Generate law proposals.

        Args:
            memory: Discovery memory (snapshot or full memory)
            request: Proposal request parameters

        Returns:
            ProposalBatch with ranked laws
        """
        start_time = time.time()
        request = request or ProposalRequest()

        # Get snapshot if full memory
        if isinstance(memory, DiscoveryMemory):
            snapshot = memory.get_snapshot()
        else:
            snapshot = memory

        # Inject previous research log into snapshot for continuity
        if self._research_log and not snapshot.previous_research_log:
            snapshot.previous_research_log = self._research_log

        # Build prompt
        prompt = self._prompt_builder.build(
            contract=self.contract,
            memory=snapshot,
            request_count=request.count,
            target_templates=request.target_templates,
            exclude_templates=request.exclude_templates,
        )

        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        prompt_tokens = self._prompt_builder.estimate_tokens(prompt)

        # Get LLM response
        system_instruction = self._prompt_builder.get_system_instruction()

        # Store for verbose logging
        self._last_system_instruction = system_instruction
        self._last_prompt = prompt
        self._last_response = ""

        llm_start = time.time()
        try:
            response = self.client.generate(
                prompt,
                system_instruction=system_instruction,
                temperature=request.temperature,
            )
            self._last_response = response
            llm_duration_ms = int((time.time() - llm_start) * 1000)
        except Exception as e:
            llm_duration_ms = int((time.time() - llm_start) * 1000)

            # Log failed LLM call
            if self._llm_logger:
                self._llm_logger.log_call(
                    prompt=prompt,
                    response="",
                    success=False,
                    system_instruction=system_instruction,
                    prompt_tokens=prompt_tokens,
                    duration_ms=llm_duration_ms,
                    error_message=str(e),
                )

            return ProposalBatch(
                prompt_hash=prompt_hash,
                prompt_tokens=prompt_tokens,
                runtime_ms=int((time.time() - start_time) * 1000),
                warnings=[f"LLM call failed: {e}"],
            )

        response_tokens = len(response) // 4  # Rough estimate

        # Parse response
        parse_result = self._parser.parse(response)

        # Extract and store research log for next iteration
        if parse_result.research_log:
            self._research_log = parse_result.research_log

        # Log successful LLM call (after parsing to include research_log)
        if self._llm_logger:
            self._llm_logger.log_call(
                prompt=prompt,
                response=response,
                success=True,
                system_instruction=system_instruction,
                research_log=parse_result.research_log,
                prompt_tokens=prompt_tokens,
                duration_ms=llm_duration_ms,
            )

        # Filter for redundancy
        non_redundant, redundant = self._redundancy.filter_batch(parse_result.laws)

        # Rank laws
        ranked = self._ranking.rank(non_redundant, snapshot)

        # Build result
        batch = ProposalBatch(
            laws=[law for law, _ in ranked],
            features=[features for _, features in ranked],
            rejections=parse_result.rejections,
            redundant=redundant,
            prompt_hash=prompt_hash,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            runtime_ms=int((time.time() - start_time) * 1000),
            warnings=parse_result.warnings,
            research_log=parse_result.research_log,
        )

        # Add to redundancy filter if configured
        if self.config.add_to_redundancy_filter:
            for law in batch.laws:
                self._redundancy.add_known_law(law)

        # Log iteration
        self._log_iteration(batch, request)

        return batch

    def add_known_law(self, law: CandidateLaw) -> None:
        """Add a law to the redundancy filter.

        Use this to seed the filter with existing accepted/falsified laws.

        Args:
            law: Law to add
        """
        self._redundancy.add_known_law(law)

    def add_known_laws(self, laws: list[CandidateLaw]) -> None:
        """Add multiple laws to the redundancy filter.

        Args:
            laws: Laws to add
        """
        for law in laws:
            self._redundancy.add_known_law(law)

    def _log_iteration(self, batch: ProposalBatch, request: ProposalRequest) -> None:
        """Log an iteration for audit."""
        self._iterations.append({
            "timestamp": time.time(),
            "request": {
                "count": request.count,
                "target_templates": request.target_templates,
                "exclude_templates": request.exclude_templates,
            },
            "result": batch.to_dict(),
        })

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Get the audit log of all iterations."""
        return self._iterations

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self._iterations = []

    @property
    def stats(self) -> dict[str, Any]:
        """Get proposer statistics."""
        total_laws = sum(len(it["result"]["law_ids"]) for it in self._iterations)
        total_rejections = sum(it["result"]["rejections_count"] for it in self._iterations)
        total_redundant = sum(it["result"]["redundant_count"] for it in self._iterations)

        return {
            "iterations": len(self._iterations),
            "total_laws_proposed": total_laws,
            "total_rejections": total_rejections,
            "total_redundant": total_redundant,
            "known_laws_in_filter": self._redundancy.known_count,
        }

    def get_last_exchange(self) -> dict[str, str]:
        """Get the last prompt/response exchange for debugging.

        Returns:
            Dict with system_instruction, prompt, and response
        """
        return {
            "system_instruction": self._last_system_instruction,
            "prompt": self._last_prompt,
            "response": self._last_response,
        }

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

    def get_research_log(self) -> str | None:
        """Get the current research log.

        Returns:
            The LLM's research notes from the most recent iteration
        """
        return self._research_log

    def set_research_log(self, log: str | None) -> None:
        """Set the research log for the next iteration.

        This can be used to seed the proposer with an initial research context
        or to restore state from persistence.

        Args:
            log: Research log content
        """
        self._research_log = log

    def clear_research_log(self) -> None:
        """Clear the research log, starting fresh."""
        self._research_log = None
