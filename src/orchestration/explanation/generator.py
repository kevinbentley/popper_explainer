"""Explanation generator using LLM synthesis.

Generates mechanistic explanations from validated theorems.
The generator analyzes theorem patterns and synthesizes
a coherent mechanistic model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.orchestration.explanation.models import (
    Criticism,
    Explanation,
    ExplanationBatch,
    ExplanationStatus,
    Mechanism,
    MechanismRule,
    MechanismType,
    OpenQuestion,
)
from src.proposer.scrambler import SymbolScrambler, get_default_scrambler

if TYPE_CHECKING:
    from src.db.llm_logger import LLMLogger
    from src.theorem.models import Theorem

logger = logging.getLogger(__name__)


@dataclass
class ExplanationGeneratorConfig:
    """Configuration for explanation generation.

    Attributes:
        max_explanations: Maximum explanations to generate per iteration
        min_theorem_support: Minimum theorems required to generate
        temperature: LLM temperature for generation
        verbose: Whether to log details
    """

    max_explanations: int = 3
    min_theorem_support: int = 2
    temperature: float = 0.7
    verbose: bool = False


# System instruction for explanation generation
EXPLANATION_SYSTEM_INSTRUCTION = """You are a Popperian scientist in the EXPLANATION PHASE - the final phase of discovery.

Your mission: Build a complete mechanistic model that can PREDICT the next state of the universe.

=== YOUR SCIENTIFIC IDENTITY ===

You have progressed through three phases of Popperian discovery:
1. LAW DISCOVERY: Tested empirical laws through falsification
2. THEOREM GENERATION: Synthesized laws into theoretical structure
3. EXPLANATION (you are here): Build predictive mechanistic models

Your explanation must be FALSIFIABLE. If your predictions fail, you may need to:
- Request better theorems (backtrack to theorem phase)
- Request new laws or observables (backtrack to discovery phase)

=== KEY PRINCIPLES ===

1. MECHANISMS, NOT JUST DESCRIPTIONS: Explain HOW and WHY, not just WHAT
2. PREDICTIVE POWER: Your model should predict the next state from the current state
3. FALSIFIABLE: State clearly what would prove your explanation wrong
4. COMPLETE: Account for all observed phenomena (state transitions, interactions, conservation)
5. BACKTRACK IF NEEDED: If theorems are insufficient, request what you need

Your research_log maintains continuity - record your evolving mechanistic understanding."""


# Prompt template for explanation generation
EXPLANATION_PROMPT_TEMPLATE = '''You are a Popperian scientist building a complete mechanistic explanation.

=== YOUR MISSION ===

Synthesize the validated theorems into a PREDICTIVE MECHANISM that can:
1. Explain HOW the universe evolves step by step
2. PREDICT what the next state will be given any current state
3. Be FALSIFIED if predictions fail

=== UNIVERSE DESCRIPTION ===

The universe is a 1D grid with periodic boundaries containing 4 abstract symbols: W, A, B, K

The symbol names carry no meaning. All properties must be derived from the theorems below.

=== VALIDATED THEOREMS ===

{theorems_text}

{previous_research_log_section}

=== YOUR TASK ===

Generate a mechanistic explanation that:
1. Explains HOW the universe evolves (movement rules, collision rules)
2. Can PREDICT the next state given any current state
3. Identifies failure modes - what would prove your mechanism wrong
4. Notes any open questions requiring more theorems or laws

=== RESPONSE FORMAT (JSON) ===

{{
    "research_log": "Your mechanistic notebook (max 300 words). Record: (1) Your current understanding of the mechanism, (2) What the theorems taught you, (3) What predictions your model makes, (4) What would falsify your explanation, (5) What additional theorems/laws you might need.",
    "hypothesis": "A natural language description of the complete mechanism",
    "mechanism": {{
        "description": "Overall mechanism description - how to predict the next state",
        "rules": [
            {{
                "rule_id": "rule_1",
                "rule_type": "movement|interaction|transformation|boundary|conservation",
                "condition": "When this rule applies",
                "effect": "What happens",
                "priority": 0
            }}
        ],
        "assumptions": ["List of assumptions"],
        "limitations": ["Known limitations"]
    }},
    "open_questions": [
        {{
            "question": "The question",
            "category": "definition|mechanism|boundary|prediction",
            "priority": "high|medium|low"
        }}
    ],
    "criticisms": [
        {{
            "criticism": "The criticism",
            "severity": "critical|major|minor",
            "source": "theorem|prediction|llm"
        }}
    ],
    "theorem_requests": ["Optional: specific theorems or laws needed from earlier phases"],
    "confidence": 0.8
}}

Generate the explanation:'''


class ExplanationGenerator:
    """Generates mechanistic explanations from theorems.

    Uses an LLM to synthesize theorems into a coherent
    mechanistic model that can make predictions.
    """

    def __init__(
        self,
        client: Any,  # LLM client
        config: ExplanationGeneratorConfig | None = None,
        llm_logger: LLMLogger | None = None,
        system_instruction: str | None = None,
        scrambler: SymbolScrambler | None = None,
    ):
        """Initialize the generator.

        Args:
            client: LLM client for generation
            config: Generator configuration
            llm_logger: Optional LLM logger for capturing all LLM interactions
            system_instruction: Optional custom system instruction
            scrambler: Symbol scrambler for translating physical symbols
        """
        self.client = client
        self.config = config or ExplanationGeneratorConfig()
        self._llm_logger = llm_logger
        self.system_instruction = system_instruction or EXPLANATION_SYSTEM_INSTRUCTION
        self._scrambler = scrambler or get_default_scrambler()

        # Research log for continuity across iterations
        self._research_log: str | None = None

    def generate(
        self,
        theorems: list[Theorem],
        previous_explanations: list[Explanation] | None = None,
        iteration_id: int | None = None,
    ) -> ExplanationBatch:
        """Generate explanations from validated theorems.

        Args:
            theorems: Validated theorems to synthesize
            previous_explanations: Previous explanations for context
            iteration_id: Current iteration index

        Returns:
            ExplanationBatch with generated explanations and research_log
        """
        start_time = time.time()
        warnings: list[str] = []

        # Check minimum theorem support
        if len(theorems) < self.config.min_theorem_support:
            warnings.append(
                f"Only {len(theorems)} theorems provided, "
                f"minimum is {self.config.min_theorem_support}"
            )

        # Build prompt with previous research log for continuity
        theorems_text = self._format_theorems(theorems)
        previous_research_log_section = self._build_previous_research_log_section()
        prompt = EXPLANATION_PROMPT_TEMPLATE.format(
            theorems_text=theorems_text,
            previous_research_log_section=previous_research_log_section,
        )
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        prompt_tokens = len(prompt) // 4  # Rough estimate

        # Generate via LLM
        response = ""
        llm_success = True
        llm_error = None
        research_log: str | None = None
        llm_start = time.time()

        try:
            # Check if client supports system_instruction parameter
            if hasattr(self.client, 'generate') and 'system_instruction' in self.client.generate.__code__.co_varnames:
                response = self.client.generate(
                    prompt,
                    system_instruction=self.system_instruction,
                    temperature=self.config.temperature,
                )
            else:
                response = self.client.generate(
                    prompt,
                    temperature=self.config.temperature,
                )
            explanations, research_log = self._parse_response(
                response, theorems, iteration_id
            )

            # Store research log for next iteration
            if research_log:
                self._research_log = research_log

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            warnings.append(f"LLM error: {str(e)}")
            llm_success = False
            llm_error = str(e)
            # Fall back to rule-based generation
            explanations = [
                self._generate_fallback_explanation(theorems, iteration_id)
            ]
        finally:
            llm_duration_ms = int((time.time() - llm_start) * 1000)

            # Log LLM call (after parsing to include research_log)
            if self._llm_logger:
                self._llm_logger.log_call(
                    prompt=prompt,
                    response=response,
                    success=llm_success,
                    system_instruction=self.system_instruction,
                    research_log=research_log,
                    prompt_tokens=prompt_tokens,
                    duration_ms=llm_duration_ms,
                    error_message=llm_error,
                )

        runtime_ms = int((time.time() - start_time) * 1000)

        return ExplanationBatch(
            explanations=explanations,
            prompt_hash=prompt_hash,
            runtime_ms=runtime_ms,
            warnings=warnings,
            research_log=research_log,
        )

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
            The LLM's mechanistic notes from the most recent iteration
        """
        return self._research_log

    def set_research_log(self, log: str | None) -> None:
        """Set the research log for the next iteration.

        This can be used to seed the generator with initial context
        or to restore state from persistence.

        Args:
            log: Research log content
        """
        self._research_log = log

    def clear_research_log(self) -> None:
        """Clear the research log, starting fresh."""
        self._research_log = None

    def _build_previous_research_log_section(self) -> str:
        """Build the previous research log section for continuity."""
        if not self._research_log:
            return ""

        return f"""
=== YOUR PREVIOUS RESEARCH LOG ===

Below are your mechanistic notes from the last iteration. Build on these insights
rather than starting fresh. Your understanding should deepen over time.

{self._research_log}

=== END OF PREVIOUS LOG ==="""

    def _format_theorems(self, theorems: list[Theorem]) -> str:
        """Format theorems for the prompt.

        Scrambles any physical symbols in theorem claims to abstract
        form, since existing theorems may have been generated before
        scrambling was applied to the theorem pipeline.
        """
        lines = []
        for i, theorem in enumerate(theorems, 1):
            # Scramble claim text: translate both quoted and bare symbols
            claim = self._scrambler.translate_observable_expr(
                theorem.claim, to_physical=False
            )
            # Also translate bare state symbols in prose text
            claim = self._scrambler.to_abstract(claim)

            name = self._scrambler.to_abstract(theorem.name)

            lines.append(f"{i}. [{theorem.status.value}] {name}")
            lines.append(f"   Claim: {claim}")
            if theorem.support:
                law_ids = [s.law_id for s in theorem.support]
                lines.append(f"   Supporting laws: {', '.join(law_ids)}")
            if theorem.missing_structure:
                lines.append(f"   Missing: {', '.join(theorem.missing_structure)}")
            lines.append("")
        return "\n".join(lines)

    def _parse_response(
        self,
        response: str,
        theorems: list[Theorem],
        iteration_id: int | None,
    ) -> tuple[list[Explanation], str | None]:
        """Parse LLM response into Explanation objects.

        Returns:
            Tuple of (explanations, research_log)
        """
        research_log: str | None = None

        try:
            # Extract JSON from response
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            # Extract research_log if present
            if isinstance(data, dict) and "research_log" in data:
                log_value = data["research_log"]
                if isinstance(log_value, str):
                    research_log = log_value

            # Build explanation
            explanation = self._build_explanation_from_data(
                data, theorems, iteration_id
            )
            return [explanation], research_log

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return [self._generate_fallback_explanation(theorems, iteration_id)], None

    def _extract_json(self, response: str) -> str:
        """Extract JSON from LLM response."""
        # Try to find JSON block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            return response[start:end].strip()
        elif "{" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            return response[start:end]
        return response

    def _build_explanation_from_data(
        self,
        data: dict[str, Any],
        theorems: list[Theorem],
        iteration_id: int | None,
    ) -> Explanation:
        """Build an Explanation from parsed data."""
        # Parse mechanism
        mech_data = data.get("mechanism", {})
        rules = [
            MechanismRule(
                rule_id=r.get("rule_id", f"rule_{i}"),
                rule_type=MechanismType(r.get("rule_type", "movement")),
                condition=r.get("condition", ""),
                effect=r.get("effect", ""),
                priority=r.get("priority", i),
            )
            for i, r in enumerate(mech_data.get("rules", []))
        ]

        mechanism = Mechanism(
            rules=rules,
            description=mech_data.get("description", ""),
            assumptions=mech_data.get("assumptions", []),
            limitations=mech_data.get("limitations", []),
        )

        # Parse open questions
        open_questions = [
            OpenQuestion(
                question=q.get("question", ""),
                category=q.get("category", "mechanism"),
                priority=q.get("priority", "medium"),
            )
            for q in data.get("open_questions", [])
        ]

        # Parse criticisms
        criticisms = [
            Criticism(
                criticism=c.get("criticism", ""),
                severity=c.get("severity", "minor"),
                source=c.get("source", "llm"),
            )
            for c in data.get("criticisms", [])
        ]

        # Generate ID
        explanation_id = f"exp_{hashlib.sha256(data.get('hypothesis', '').encode()).hexdigest()[:12]}"

        return Explanation(
            explanation_id=explanation_id,
            hypothesis_text=data.get("hypothesis", ""),
            mechanism=mechanism,
            supporting_theorems=[t.theorem_id for t in theorems],
            open_questions=open_questions,
            criticisms=criticisms,
            confidence=data.get("confidence", 0.5),
            status=ExplanationStatus.PROPOSED,
            iteration_id=iteration_id,
        )

    def _generate_fallback_explanation(
        self,
        theorems: list[Theorem],
        iteration_id: int | None,
    ) -> Explanation:
        """Generate a rule-based fallback explanation.

        This is used when LLM generation fails. It creates
        a basic mechanistic model using abstract symbols only.
        """
        rules = [
            MechanismRule(
                rule_id="movement_A",
                rule_type=MechanismType.MOVEMENT,
                condition="Cell contains A",
                effect="A moves one cell in its characteristic direction",
                priority=0,
            ),
            MechanismRule(
                rule_id="movement_B",
                rule_type=MechanismType.MOVEMENT,
                condition="Cell contains B",
                effect="B moves one cell in its characteristic direction",
                priority=0,
            ),
            MechanismRule(
                rule_id="interaction_formation",
                rule_type=MechanismType.INTERACTION,
                condition="A and B attempt to occupy the same cell",
                effect="K state forms at that cell",
                priority=1,
            ),
            MechanismRule(
                rule_id="interaction_resolution",
                rule_type=MechanismType.TRANSFORMATION,
                condition="Cell contains K",
                effect="A exits one direction, B exits the other, cell becomes W",
                priority=2,
            ),
            MechanismRule(
                rule_id="periodic_boundary",
                rule_type=MechanismType.BOUNDARY,
                condition="Particle at grid edge",
                effect="Wraps to opposite edge (periodic)",
                priority=0,
            ),
            MechanismRule(
                rule_id="particle_conservation",
                rule_type=MechanismType.CONSERVATION,
                condition="Always",
                effect="Total particle count is conserved",
                priority=0,
            ),
        ]

        mechanism = Mechanism(
            rules=rules,
            description=(
                "The universe evolves through simultaneous "
                "particle movement with K-state formation and resolution. "
                "Particles move at speed 1, K states form when particles "
                "meet, and resolve by separating in the next step."
            ),
            assumptions=[
                "All particles move at unit speed",
                "K states are always between exactly one A and one B",
                "Grid has periodic boundaries",
            ],
            limitations=[
                "Does not explain WHY particles move",
                "Does not explain particle creation/destruction",
            ],
        )

        return Explanation(
            explanation_id=f"exp_fallback_{iteration_id or 0}",
            hypothesis_text=(
                "The universe consists of particles moving in a 1D grid. "
                "A particles move in one direction, B particles move in the "
                "opposite direction, both at speed 1. When they meet, they "
                "form a K state which resolves in the next step with "
                "particles separating."
            ),
            mechanism=mechanism,
            supporting_theorems=[t.theorem_id for t in theorems],
            open_questions=[
                OpenQuestion(
                    question="What causes the initial particle distribution?",
                    category="mechanism",
                    priority="low",
                ),
            ],
            criticisms=[],
            confidence=0.8,  # High confidence for known rules
            status=ExplanationStatus.PROPOSED,
            iteration_id=iteration_id,
        )

    def refine_explanation(
        self,
        explanation: Explanation,
        prediction_results: dict[str, Any],
        theorems: list[Theorem],
    ) -> Explanation:
        """Refine an explanation based on prediction results.

        Args:
            explanation: Current explanation to refine
            prediction_results: Results from prediction verification
            theorems: Current validated theorems

        Returns:
            Refined explanation
        """
        # Update prediction accuracy
        accuracy = prediction_results.get("exact_match_rate", 0.0)
        explanation.prediction_accuracy = accuracy

        # Update status based on accuracy
        if accuracy >= 0.9:
            explanation.status = ExplanationStatus.VALIDATED
        elif accuracy < 0.5:
            explanation.status = ExplanationStatus.REFUTED
            explanation.criticisms.append(
                Criticism(
                    criticism=f"Prediction accuracy too low: {accuracy:.2%}",
                    severity="critical",
                    source="prediction",
                )
            )
        else:
            explanation.status = ExplanationStatus.TESTING

        # Update confidence based on accuracy
        explanation.confidence = min(explanation.confidence, accuracy)

        return explanation
