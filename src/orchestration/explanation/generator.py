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


# Prompt template for explanation generation
EXPLANATION_PROMPT_TEMPLATE = '''You are a physicist analyzing a 1D particle simulation.

Given the following validated theorems about the Kinetic Grid Universe, synthesize a mechanistic explanation.

## Universe Description
The Kinetic Grid Universe is a 1D grid with periodic boundaries containing:
- Empty cells (.)
- Right-moving particles (>)
- Left-moving particles (<)
- Collisions (X) - formed when > and < occupy the same cell

## Validated Theorems
{theorems_text}

## Task
Generate a mechanistic explanation that:
1. Explains HOW the universe evolves (movement rules, collision rules)
2. Identifies the fundamental mechanisms
3. Notes any open questions or limitations

## Response Format (JSON)
{{
    "hypothesis": "A natural language description of the mechanism",
    "mechanism": {{
        "description": "Overall mechanism description",
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
    ):
        """Initialize the generator.

        Args:
            client: LLM client for generation
            config: Generator configuration
            llm_logger: Optional LLM logger for capturing all LLM interactions
        """
        self.client = client
        self.config = config or ExplanationGeneratorConfig()
        self._llm_logger = llm_logger

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
            ExplanationBatch with generated explanations
        """
        start_time = time.time()
        warnings: list[str] = []

        # Check minimum theorem support
        if len(theorems) < self.config.min_theorem_support:
            warnings.append(
                f"Only {len(theorems)} theorems provided, "
                f"minimum is {self.config.min_theorem_support}"
            )

        # Build prompt
        theorems_text = self._format_theorems(theorems)
        prompt = EXPLANATION_PROMPT_TEMPLATE.format(theorems_text=theorems_text)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        prompt_tokens = len(prompt) // 4  # Rough estimate

        # Generate via LLM
        response = ""
        llm_success = True
        llm_error = None
        llm_start = time.time()

        try:
            response = self.client.generate(
                prompt,
                temperature=self.config.temperature,
            )
            explanations = self._parse_response(
                response, theorems, iteration_id
            )
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

            # Log LLM call
            if self._llm_logger:
                self._llm_logger.log_call(
                    prompt=prompt,
                    response=response,
                    success=llm_success,
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

    def _format_theorems(self, theorems: list[Theorem]) -> str:
        """Format theorems for the prompt."""
        lines = []
        for i, theorem in enumerate(theorems, 1):
            lines.append(f"{i}. [{theorem.status.value}] {theorem.name}")
            lines.append(f"   Claim: {theorem.claim}")
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
    ) -> list[Explanation]:
        """Parse LLM response into Explanation objects."""
        try:
            # Extract JSON from response
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            # Build explanation
            explanation = self._build_explanation_from_data(
                data, theorems, iteration_id
            )
            return [explanation]

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return [self._generate_fallback_explanation(theorems, iteration_id)]

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
        a basic mechanistic model based on known universe rules.
        """
        rules = [
            MechanismRule(
                rule_id="movement_right",
                rule_type=MechanismType.MOVEMENT,
                condition="Cell contains > (right-moving particle)",
                effect="Particle moves one cell to the right",
                priority=0,
            ),
            MechanismRule(
                rule_id="movement_left",
                rule_type=MechanismType.MOVEMENT,
                condition="Cell contains < (left-moving particle)",
                effect="Particle moves one cell to the left",
                priority=0,
            ),
            MechanismRule(
                rule_id="collision_formation",
                rule_type=MechanismType.INTERACTION,
                condition="> and < attempt to occupy the same cell",
                effect="Collision (X) forms at that cell",
                priority=1,
            ),
            MechanismRule(
                rule_id="collision_resolution",
                rule_type=MechanismType.TRANSFORMATION,
                condition="Cell contains X (collision)",
                effect="> exits right, < exits left, cell becomes empty",
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
                "The Kinetic Grid Universe evolves through simultaneous "
                "particle movement with collision formation and resolution. "
                "Particles move at speed 1, collisions form when particles "
                "meet, and resolve by separating in the next step."
            ),
            assumptions=[
                "All particles move at unit speed",
                "Collisions are always between exactly one > and one <",
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
                "Right-movers (>) move right, left-movers (<) move left, "
                "both at speed 1. When they meet, they form a collision (X) "
                "which resolves in the next step with particles separating."
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
