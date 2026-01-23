"""Main test harness for law evaluation.

The harness orchestrates:
- Case generation using multiple strategies
- Law evaluation against trajectories
- Adversarial search for enhanced falsification
- Counterexample minimization
- Verdict determination with power metrics
"""

import json
import time
from typing import Any

from src.claims.compiler import CompilationError
from src.claims.schema import CandidateLaw, Template
from src.db.models import CaseSetRecord, CounterexampleRecord, EvaluationRecord
from src.db.repo import Repository
from src.harness.adversarial import AdversarialSearcher
from src.harness.case import Case, CaseResult
from src.harness.config import HarnessConfig
from src.harness.evaluator import Evaluator, compute_density, has_collision, has_wrapping
from src.harness.generators import GeneratorRegistry
from src.harness.minimizer import Minimizer
from src.harness.power import PowerMetrics
from src.harness.vacuity import VacuityReport
from src.harness.verdict import Counterexample, LawVerdict, ReasonCode
from src.universe.simulator import version_hash as sim_version_hash
from src.universe.transforms import list_transforms


class Harness:
    """Main test harness for evaluating laws.

    Usage:
        harness = Harness(config)
        verdict = harness.evaluate(law)
    """

    def __init__(self, config: HarnessConfig | None = None, repo: Repository | None = None):
        """Initialize the harness.

        Args:
            config: Harness configuration
            repo: Optional database repository for persistence
        """
        self.config = config or HarnessConfig()
        self.repo = repo
        self._evaluator = Evaluator()
        self._minimizer = Minimizer(budget=self.config.minimization_budget)
        self._adversarial = AdversarialSearcher(
            budget=self.config.adversarial_budget,
            max_runtime_ms=self.config.max_runtime_ms_per_law // 2,
        )

    def evaluate(self, law: CandidateLaw) -> LawVerdict:
        """Evaluate a law and return a verdict.

        Args:
            law: The candidate law to evaluate

        Returns:
            LawVerdict with PASS, FAIL, or UNKNOWN status
        """
        start_time = time.time()

        # Check capabilities first
        capability_issue = self._check_capabilities(law)
        if capability_issue:
            return capability_issue

        # Try to compile the law
        try:
            self._evaluator.prepare(law)
        except CompilationError as e:
            return LawVerdict(
                law_id=law.law_id,
                status="UNKNOWN",
                reason_code=ReasonCode.AMBIGUOUS_CLAIM,
                notes=[f"Compilation failed: {e}"],
            )

        # Determine time horizon
        time_horizon = min(law.quantifiers.T, self.config.max_T)

        # Generate test cases
        cases = self._generate_cases(law)

        # Evaluate all cases
        results: list[CaseResult] = []
        counterexample: Counterexample | None = None
        power_metrics = PowerMetrics()

        for case in cases:
            if len(results) >= self.config.max_cases:
                break

            result = self._evaluator.evaluate_case(case, time_horizon)
            results.append(result)

            power_metrics.cases_attempted += 1
            if result.precondition_met:
                power_metrics.cases_used += 1

                # Track metrics
                if has_collision(result.trajectory):
                    power_metrics.cases_with_collisions += 1
                if has_wrapping(case.initial_state, result.trajectory):
                    power_metrics.cases_with_wrapping += 1

                density = compute_density(case.initial_state)
                density_bin = round(density, 1)
                if density_bin not in power_metrics.density_bins_hit:
                    power_metrics.density_bins_hit.append(density_bin)

            # Check for failure
            if not result.passed and result.precondition_met:
                # Found a counterexample
                counterexample = self._create_counterexample(result, time_horizon)
                break

        # Run adversarial search if enabled and no counterexample yet
        adversarial_result = None
        if (
            counterexample is None
            and self.config.enable_adversarial_search
            and power_metrics.cases_used >= self.config.min_cases_used_for_pass // 2
        ):
            adversarial_result = self._adversarial.search(
                law=law,
                evaluate_case=self._evaluator.evaluate_case,
                seed_results=results,
                time_horizon=time_horizon,
                seed=self.config.seed + 10000,
            )
            power_metrics.adversarial_cases_tried = adversarial_result.cases_tried

            if adversarial_result.found_counterexample:
                counterexample = adversarial_result.counterexample
                power_metrics.adversarial_found = True

        # Compute runtime
        runtime_ms = int((time.time() - start_time) * 1000)

        # Determine verdict
        verdict = self._determine_verdict(
            law, results, counterexample, power_metrics, runtime_ms
        )

        # Minimize counterexample if found
        if verdict.status == "FAIL" and counterexample and self.config.enable_counterexample_minimization:
            minimized = self._minimizer.minimize(law, counterexample, time_horizon)
            verdict.counterexample = minimized

        # Add adversarial search info to notes
        if adversarial_result and adversarial_result.cases_tried > 0:
            verdict.notes = verdict.notes or []
            if adversarial_result.found_counterexample:
                verdict.notes.append(
                    f"Counterexample found via adversarial search "
                    f"(tried {adversarial_result.cases_tried} mutations)"
                )
            else:
                verdict.notes.append(
                    f"Adversarial search: {adversarial_result.cases_tried} mutations tried, "
                    f"no counterexample found"
                )

        # Persist to database if repo available
        if self.repo:
            self._persist_evaluation(law, verdict, cases, results)

        return verdict

    def _check_capabilities(self, law: CandidateLaw) -> LawVerdict | None:
        """Check if we have capabilities to test this law.

        Returns a verdict if capabilities are missing, None otherwise.
        """
        # Check for missing observables (would need expression evaluation)
        if law.capability_requirements.missing_observables:
            return LawVerdict(
                law_id=law.law_id,
                status="UNKNOWN",
                reason_code=ReasonCode.MISSING_OBSERVABLE,
                notes=[f"Missing observables: {law.capability_requirements.missing_observables}"],
            )

        # Check for missing transforms
        if law.capability_requirements.missing_transforms:
            return LawVerdict(
                law_id=law.law_id,
                status="UNKNOWN",
                reason_code=ReasonCode.MISSING_TRANSFORM,
                notes=[f"Missing transforms: {law.capability_requirements.missing_transforms}"],
            )

        # For symmetry laws, check if transform is available
        if law.template == Template.SYMMETRY_COMMUTATION:
            if law.transform and law.transform not in list_transforms():
                return LawVerdict(
                    law_id=law.law_id,
                    status="UNKNOWN",
                    reason_code=ReasonCode.MISSING_TRANSFORM,
                    notes=[f"Transform '{law.transform}' not available"],
                )

        # Check for missing generators
        if law.capability_requirements.missing_generators:
            # Not fatal - we can use default generators
            pass

        return None

    def _generate_cases(self, law: CandidateLaw) -> list[Case]:
        """Generate test cases for evaluating a law."""
        cases: list[Case] = []
        seed = self.config.seed

        # CRITICAL: Always include pathological baseline cases first.
        # These catch false positives from generator coverage gaps (e.g., uniform grids).
        pathological_gen = GeneratorRegistry.create("pathological_cases")
        if pathological_gen:
            pathological_params = self._get_default_params("pathological_cases", law)
            # Always generate a baseline set of pathological cases
            baseline_count = min(20, self.config.max_cases // 5)
            pathological_cases = pathological_gen.generate(
                pathological_params, seed, baseline_count
            )
            cases.extend(pathological_cases)
            seed += len(pathological_cases)

        # Use proposed tests if available
        for proposed in law.proposed_tests:
            generator = GeneratorRegistry.create(proposed.family)
            if generator:
                new_cases = generator.generate(
                    proposed.params,
                    seed,
                    self.config.max_cases // max(len(law.proposed_tests), 1),
                )
                cases.extend(new_cases)
                seed += len(new_cases)

        # If no proposed tests or not enough cases, use default strategy
        if len(cases) < self.config.max_cases // 2:
            remaining = self.config.max_cases - len(cases)
            default_cases = self._generate_default_cases(law, seed, remaining)
            cases.extend(default_cases)

        return cases

    def _generate_default_cases(
        self, law: CandidateLaw, seed: int, count: int
    ) -> list[Case]:
        """Generate cases using default multi-stage strategy."""
        cases: list[Case] = []

        # Allocate cases based on weights
        weights = self.config.generator_weights
        total_weight = sum(weights.values())

        for family_name, weight in weights.items():
            generator = GeneratorRegistry.create(family_name)
            if generator is None:
                continue

            family_count = int(count * weight / total_weight)
            if family_count == 0:
                continue

            params = self._get_default_params(family_name, law)
            new_cases = generator.generate(params, seed, family_count)
            cases.extend(new_cases)
            seed += len(new_cases)

        return cases

    def _get_default_params(self, family_name: str, law: CandidateLaw) -> dict[str, Any]:
        """Get default parameters for a generator family."""
        if family_name == "random_density_sweep":
            return {
                "densities": [0.1, 0.2, 0.3, 0.5],
                "grid_lengths": [8, 16, 32],
                "include_collisions": True,
            }
        elif family_name == "constrained_pair_interactions":
            return {
                "patterns": ["approaching", "collision", "adjacent", "diverging"],
                "grid_lengths": [8, 16, 32],
                "include_collision_state": True,
            }
        elif family_name == "edge_wrapping_cases":
            return {
                "grid_lengths": [8, 16, 32],
                "include_collision_at_boundary": True,
            }
        elif family_name == "symmetry_metamorphic_suite":
            return {
                "grid_lengths": [8, 16, 32],
                "transform": law.transform or "mirror_swap",
                "bias_asymmetric": 0.7,
            }
        elif family_name == "pathological_cases":
            return {
                "min_length": 1,
                "max_length": 20,
                "include_empty": True,
            }
        return {}

    def _create_counterexample(
        self, result: CaseResult, time_horizon: int
    ) -> Counterexample:
        """Create a counterexample from a failed case result."""
        t_fail = result.violation.get("t", 0) if result.violation else 0

        # Extract trajectory excerpt around failure
        excerpt_start = max(0, t_fail - 2)
        excerpt_end = min(len(result.trajectory), t_fail + 3)
        trajectory_excerpt = result.trajectory[excerpt_start:excerpt_end]

        return Counterexample(
            initial_state=result.case.initial_state,
            config={
                "grid_length": result.case.config.grid_length,
                "boundary": result.case.config.boundary,
            },
            seed=result.case.seed,
            t_max=time_horizon,
            t_fail=t_fail,
            trajectory_excerpt=trajectory_excerpt,
            observables_at_fail=result.violation.get("details") if result.violation else None,
            witness=result.violation,
            minimized=False,
        )

    def _determine_verdict(
        self,
        law: CandidateLaw,
        results: list[CaseResult],
        counterexample: Counterexample | None,
        power_metrics: PowerMetrics,
        runtime_ms: int,
    ) -> LawVerdict:
        """Determine the final verdict from evaluation results."""
        # Get vacuity report
        vacuity = self._evaluator.get_vacuity_report(results)

        # Collect test families used
        tests_run = list(set(r.case.generator_family for r in results))

        # Compute coverage
        power_metrics.compute_coverage()

        # FAIL: counterexample found
        if counterexample is not None:
            return LawVerdict(
                law_id=law.law_id,
                status="FAIL",
                counterexample=counterexample,
                power_metrics=power_metrics,
                vacuity=vacuity,
                runtime_ms=runtime_ms,
                tests_run=tests_run,
            )

        # Check for insufficient testing
        if power_metrics.cases_used < self.config.min_cases_used_for_pass:
            return LawVerdict(
                law_id=law.law_id,
                status="UNKNOWN",
                reason_code=ReasonCode.UNMET_PRECONDITIONS,
                power_metrics=power_metrics,
                vacuity=vacuity,
                runtime_ms=runtime_ms,
                tests_run=tests_run,
                notes=[
                    f"Only {power_metrics.cases_used} cases satisfied preconditions "
                    f"(need {self.config.min_cases_used_for_pass})"
                ],
            )

        # Check for vacuous pass
        if self.config.require_non_vacuous and vacuity.is_vacuous:
            return LawVerdict(
                law_id=law.law_id,
                status="UNKNOWN",
                reason_code=ReasonCode.VACUOUS_PASS,
                power_metrics=power_metrics,
                vacuity=vacuity,
                runtime_ms=runtime_ms,
                tests_run=tests_run,
                notes=["Test was vacuous: antecedent never held"],
            )

        # Check for low power
        if power_metrics.coverage_score < 0.3:
            return LawVerdict(
                law_id=law.law_id,
                status="UNKNOWN",
                reason_code=ReasonCode.INCONCLUSIVE_LOW_POWER,
                power_metrics=power_metrics,
                vacuity=vacuity,
                runtime_ms=runtime_ms,
                tests_run=tests_run,
                notes=[f"Low coverage score: {power_metrics.coverage_score:.2f}"],
            )

        # PASS: no counterexample found with sufficient coverage
        return LawVerdict(
            law_id=law.law_id,
            status="PASS",
            power_metrics=power_metrics,
            vacuity=vacuity,
            runtime_ms=runtime_ms,
            tests_run=tests_run,
        )

    def _persist_evaluation(
        self,
        law: CandidateLaw,
        verdict: LawVerdict,
        cases: list[Case],
        results: list[CaseResult],
    ) -> None:
        """Persist evaluation results to database."""
        if self.repo is None:
            return

        # Store case set
        case_set_record = CaseSetRecord(
            generator_family="mixed",
            params_hash=self.config.content_hash(),
            seed=self.config.seed,
            cases_json=json.dumps([c.to_dict() for c in cases]),
            case_count=len(cases),
        )

        try:
            case_set_id = self.repo.insert_case_set(case_set_record)
        except Exception:
            # Case set might already exist
            existing = self.repo.get_case_set("mixed", self.config.content_hash(), self.config.seed)
            case_set_id = existing.id if existing else None

        # Store evaluation
        eval_record = EvaluationRecord(
            law_id=law.law_id,
            law_hash=law.content_hash(),
            status=verdict.status,
            reason_code=verdict.reason_code.value if verdict.reason_code else None,
            case_set_id=case_set_id,
            cases_attempted=verdict.power_metrics.cases_attempted,
            cases_used=verdict.power_metrics.cases_used,
            power_metrics_json=json.dumps(verdict.power_metrics.to_dict()),
            vacuity_json=json.dumps(verdict.vacuity.to_dict()),
            harness_config_hash=self.config.content_hash(),
            sim_hash=sim_version_hash(),
            runtime_ms=verdict.runtime_ms,
        )
        eval_id = self.repo.insert_evaluation(eval_record)

        # Store counterexample if present
        if verdict.counterexample:
            cx_record = CounterexampleRecord(
                evaluation_id=eval_id,
                law_id=law.law_id,
                initial_state=verdict.counterexample.initial_state,
                config_json=json.dumps(verdict.counterexample.config),
                seed=verdict.counterexample.seed,
                t_max=verdict.counterexample.t_max,
                t_fail=verdict.counterexample.t_fail,
                trajectory_excerpt_json=json.dumps(verdict.counterexample.trajectory_excerpt)
                if verdict.counterexample.trajectory_excerpt
                else None,
                observables_at_fail_json=json.dumps(verdict.counterexample.observables_at_fail)
                if verdict.counterexample.observables_at_fail
                else None,
                witness_json=json.dumps(verdict.counterexample.witness)
                if verdict.counterexample.witness
                else None,
                minimized=verdict.counterexample.minimized,
            )
            self.repo.insert_counterexample(cx_record)

        # Audit log
        self.repo.log_audit(
            operation="evaluate",
            entity_type="law",
            entity_id=law.law_id,
            details={
                "status": verdict.status,
                "cases_used": verdict.power_metrics.cases_used,
                "runtime_ms": verdict.runtime_ms,
            },
        )
