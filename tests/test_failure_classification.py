"""Tests for failure classification system."""

import pytest

from src.harness.failure_classification import (
    CounterexampleClassId,
    CounterexampleClassification,
    FailureClass,
    FailureClassificationResult,
    FailureClassifier,
    classify_failure,
    get_classifier,
)
from src.harness.power import PowerMetrics
from src.harness.verdict import (
    Counterexample,
    FailureType,
    LawVerdict,
    ReasonCode,
)


class TestCounterexampleClassification:
    """Tests for counterexample classification."""

    def test_converging_pair_detection(self):
        """Converging pair pattern should be detected."""
        classifier = FailureClassifier()
        ce = Counterexample(
            initial_state=">.<",
            config={"grid_length": 3, "boundary": "periodic"},
            seed=42,
            t_max=10,
            t_fail=1,
        )
        classification = classifier._classify_counterexample(ce)
        assert classification.class_id == CounterexampleClassId.CONVERGING_PAIR
        assert classification.confidence >= 0.9

    def test_x_self_collision_small_grid(self):
        """X in small grid should be detected as self-collision."""
        classifier = FailureClassifier()
        ce = Counterexample(
            initial_state="X",
            config={"grid_length": 1, "boundary": "periodic"},
            seed=42,
            t_max=5,
            t_fail=1,
        )
        classification = classifier._classify_counterexample(ce)
        assert classification.class_id == CounterexampleClassId.X_SELF_COLLISION

    def test_x_emission_early_violation(self):
        """X emission causing early violation should be detected."""
        classifier = FailureClassifier()
        ce = Counterexample(
            initial_state=".X....",
            config={"grid_length": 6, "boundary": "periodic"},
            seed=42,
            t_max=10,
            t_fail=1,
        )
        classification = classifier._classify_counterexample(ce)
        assert classification.class_id == CounterexampleClassId.X_EMISSION

    def test_high_density_detection(self):
        """High density states should be detected."""
        classifier = FailureClassifier()
        ce = Counterexample(
            initial_state="><><><",  # density = 1.0
            config={"grid_length": 6, "boundary": "periodic"},
            seed=42,
            t_max=10,
            t_fail=5,
        )
        classification = classifier._classify_counterexample(ce)
        assert classification.class_id == CounterexampleClassId.HIGH_DENSITY

    def test_sparse_state_detection(self):
        """Sparse states should be detected."""
        classifier = FailureClassifier()
        ce = Counterexample(
            initial_state=">...........",  # density ~0.08
            config={"grid_length": 12, "boundary": "periodic"},
            seed=42,
            t_max=10,
            t_fail=5,
        )
        classification = classifier._classify_counterexample(ce)
        assert classification.class_id == CounterexampleClassId.SPARSE_STATE

    def test_late_violation_detection(self):
        """Late violations should be detected."""
        classifier = FailureClassifier()
        ce = Counterexample(
            initial_state=">.....<",
            config={"grid_length": 7, "boundary": "periodic"},
            seed=42,
            t_max=50,
            t_fail=25,
        )
        classification = classifier._classify_counterexample(ce)
        assert classification.class_id == CounterexampleClassId.LATE_VIOLATION

    def test_immediate_violation_detection(self):
        """Immediate violations should be detected."""
        classifier = FailureClassifier()
        ce = Counterexample(
            initial_state=">.....",
            config={"grid_length": 6, "boundary": "periodic"},
            seed=42,
            t_max=10,
            t_fail=0,
        )
        classification = classifier._classify_counterexample(ce)
        assert classification.class_id == CounterexampleClassId.IMMEDIATE_VIOLATION

    def test_boundary_wrap_detection(self):
        """Particles at boundary should be detected."""
        classifier = FailureClassifier()
        ce = Counterexample(
            initial_state=">.....",  # particle at left edge
            config={"grid_length": 6, "boundary": "periodic"},
            seed=42,
            t_max=10,
            t_fail=3,
        )
        classification = classifier._classify_counterexample(ce)
        # Should be boundary_wrap or something else
        assert classification.class_id in (
            CounterexampleClassId.BOUNDARY_WRAP,
            CounterexampleClassId.IMMEDIATE_VIOLATION,
            CounterexampleClassId.SPARSE_STATE,
        )


class TestFailureClassifier:
    """Tests for the main failure classifier."""

    def test_type_d_harness_error_invalid_state(self):
        """Invalid initial state should be Type D."""
        classifier = FailureClassifier()
        verdict = LawVerdict(
            law_id="test_law",
            status="FAIL",
            failure_type=FailureType.INVALID_INITIAL_STATE,
        )
        result = classifier.classify(verdict)
        assert result.failure_class == FailureClass.TYPE_D_HARNESS_ERROR
        assert result.actionable is True

    def test_type_d_harness_error_evaluation_error(self):
        """Evaluation error should be Type D."""
        classifier = FailureClassifier()
        verdict = LawVerdict(
            law_id="test_law",
            status="FAIL",
            failure_type=FailureType.EVALUATION_ERROR,
        )
        result = classifier.classify(verdict)
        assert result.failure_class == FailureClass.TYPE_D_HARNESS_ERROR
        assert result.actionable is True

    def test_type_d_harness_error_timeout(self):
        """Timeout should be Type D."""
        classifier = FailureClassifier()
        verdict = LawVerdict(
            law_id="test_law",
            status="FAIL",
            failure_type=FailureType.TIMEOUT,
        )
        result = classifier.classify(verdict)
        assert result.failure_class == FailureClass.TYPE_D_HARNESS_ERROR

    def test_type_c_ambiguous_claim(self):
        """Ambiguous claim should be Type C."""
        classifier = FailureClassifier()
        verdict = LawVerdict(
            law_id="test_law",
            status="UNKNOWN",
            reason_code=ReasonCode.AMBIGUOUS_CLAIM,
        )
        result = classifier.classify(verdict)
        assert result.failure_class == FailureClass.TYPE_C_PROCESS_ISSUE
        assert result.actionable is True

    def test_type_c_missing_observable(self):
        """Missing observable should be Type C."""
        classifier = FailureClassifier()
        verdict = LawVerdict(
            law_id="test_law",
            status="UNKNOWN",
            reason_code=ReasonCode.MISSING_OBSERVABLE,
        )
        result = classifier.classify(verdict)
        assert result.failure_class == FailureClass.TYPE_C_PROCESS_ISSUE

    def test_type_c_missing_transform(self):
        """Missing transform should be Type C."""
        classifier = FailureClassifier()
        verdict = LawVerdict(
            law_id="test_law",
            status="UNKNOWN",
            reason_code=ReasonCode.MISSING_TRANSFORM,
        )
        result = classifier.classify(verdict)
        assert result.failure_class == FailureClass.TYPE_C_PROCESS_ISSUE

    def test_type_c_missing_generator(self):
        """Missing generator should be Type C."""
        classifier = FailureClassifier()
        verdict = LawVerdict(
            law_id="test_law",
            status="UNKNOWN",
            reason_code=ReasonCode.MISSING_GENERATOR,
        )
        result = classifier.classify(verdict)
        assert result.failure_class == FailureClass.TYPE_C_PROCESS_ISSUE

    def test_type_c_low_power_not_actionable(self):
        """Low power UNKNOWN should be Type C but not actionable."""
        classifier = FailureClassifier()
        verdict = LawVerdict(
            law_id="test_law",
            status="UNKNOWN",
            reason_code=ReasonCode.INCONCLUSIVE_LOW_POWER,
        )
        result = classifier.classify(verdict)
        assert result.failure_class == FailureClass.TYPE_C_PROCESS_ISSUE
        assert result.actionable is False

    def test_type_b_first_counterexample(self):
        """First counterexample of a class should be Type B (novel)."""
        classifier = FailureClassifier()
        verdict = LawVerdict(
            law_id="test_law",
            status="FAIL",
            failure_type=FailureType.LAW_COUNTEREXAMPLE,
            counterexample=Counterexample(
                initial_state=">.<",
                config={"grid_length": 3, "boundary": "periodic"},
                seed=42,
                t_max=10,
                t_fail=1,
            ),
        )
        result = classifier.classify(verdict)
        assert result.failure_class == FailureClass.TYPE_B_NOVEL_COUNTEREXAMPLE
        assert result.is_known_class is False
        assert result.counterexample_class is not None
        assert result.counterexample_class.class_id == CounterexampleClassId.CONVERGING_PAIR

    def test_type_a_known_counterexample(self):
        """Second counterexample of same class should be Type A (known)."""
        classifier = FailureClassifier()

        # First occurrence - should be Type B
        verdict1 = LawVerdict(
            law_id="test_law_1",
            status="FAIL",
            failure_type=FailureType.LAW_COUNTEREXAMPLE,
            counterexample=Counterexample(
                initial_state=">.<",
                config={"grid_length": 3, "boundary": "periodic"},
                seed=42,
                t_max=10,
                t_fail=1,
            ),
        )
        result1 = classifier.classify(verdict1)
        assert result1.failure_class == FailureClass.TYPE_B_NOVEL_COUNTEREXAMPLE

        # Second occurrence - should be Type A
        verdict2 = LawVerdict(
            law_id="test_law_2",
            status="FAIL",
            failure_type=FailureType.LAW_COUNTEREXAMPLE,
            counterexample=Counterexample(
                initial_state="..>.<..",
                config={"grid_length": 7, "boundary": "periodic"},
                seed=99,
                t_max=10,
                t_fail=1,
            ),
        )
        result2 = classifier.classify(verdict2)
        assert result2.failure_class == FailureClass.TYPE_A_KNOWN_COUNTEREXAMPLE
        assert result2.is_known_class is True

    def test_classifier_reset(self):
        """Classifier reset should clear known classes."""
        classifier = FailureClassifier()

        # Register a class
        verdict = LawVerdict(
            law_id="test_law",
            status="FAIL",
            failure_type=FailureType.LAW_COUNTEREXAMPLE,
            counterexample=Counterexample(
                initial_state=">.<",
                config={"grid_length": 3, "boundary": "periodic"},
                seed=42,
                t_max=10,
                t_fail=1,
            ),
        )
        classifier.classify(verdict)
        assert len(classifier.get_known_classes()) > 0

        classifier.reset()
        assert len(classifier.get_known_classes()) == 0

    def test_register_known_class(self):
        """Pre-registering a class should make it known."""
        classifier = FailureClassifier()
        classifier.register_known_class(CounterexampleClassId.CONVERGING_PAIR)

        verdict = LawVerdict(
            law_id="test_law",
            status="FAIL",
            failure_type=FailureType.LAW_COUNTEREXAMPLE,
            counterexample=Counterexample(
                initial_state=">.<",
                config={"grid_length": 3, "boundary": "periodic"},
                seed=42,
                t_max=10,
                t_fail=1,
            ),
        )
        result = classifier.classify(verdict)
        # Should be Type A because class was pre-registered
        assert result.failure_class == FailureClass.TYPE_A_KNOWN_COUNTEREXAMPLE


class TestFeatureExtraction:
    """Tests for feature extraction from counterexamples."""

    def test_basic_features(self):
        """Basic features should be extracted correctly."""
        classifier = FailureClassifier()
        ce = Counterexample(
            initial_state="><.X.",
            config={"grid_length": 5, "boundary": "periodic"},
            seed=42,
            t_max=10,
            t_fail=3,
        )
        features = classifier._extract_features(ce)

        assert features["grid_length"] == 5
        assert features["t_fail"] == 3
        assert features["t_max"] == 10
        assert features["count_right"] == 1
        assert features["count_left"] == 1
        assert features["count_x"] == 1
        assert features["count_empty"] == 2
        assert features["particle_count"] == 3
        assert features["density"] == 0.6
        assert features["has_x"] is True

    def test_pattern_detection_features(self):
        """Pattern detection features should work."""
        classifier = FailureClassifier()
        ce = Counterexample(
            initial_state=">.<",
            config={"grid_length": 3, "boundary": "periodic"},
            seed=42,
            t_max=10,
            t_fail=1,
        )
        features = classifier._extract_features(ce)

        assert features["has_converging_pair"] is True
        assert features["has_adjacent_opposite"] is False

    def test_adjacent_opposite_detection(self):
        """Adjacent opposite pattern should be detected."""
        classifier = FailureClassifier()
        ce = Counterexample(
            initial_state="><",
            config={"grid_length": 2, "boundary": "periodic"},
            seed=42,
            t_max=10,
            t_fail=1,
        )
        features = classifier._extract_features(ce)

        assert features["has_adjacent_opposite"] is True

    def test_trajectory_features(self):
        """Trajectory-based features should be extracted."""
        classifier = FailureClassifier()
        ce = Counterexample(
            initial_state=">.<",
            config={"grid_length": 3, "boundary": "periodic"},
            seed=42,
            t_max=10,
            t_fail=2,
            trajectory_excerpt=[">.<", ".X.", "><."],
        )
        features = classifier._extract_features(ce)

        assert features["trajectory_length"] == 3
        assert features["collision_count_in_excerpt"] == 1


class TestClassificationSerialization:
    """Tests for serialization of classification results."""

    def test_counterexample_classification_to_dict(self):
        """CounterexampleClassification should serialize correctly."""
        classification = CounterexampleClassification(
            class_id=CounterexampleClassId.CONVERGING_PAIR,
            confidence=0.95,
            features={"grid_length": 3, "density": 1.0},
            reasoning="Contains converging pair pattern",
        )
        d = classification.to_dict()

        assert d["class_id"] == "converging_pair"
        assert d["confidence"] == 0.95
        assert d["features"]["grid_length"] == 3
        assert d["reasoning"] == "Contains converging pair pattern"

    def test_counterexample_classification_from_dict(self):
        """CounterexampleClassification should deserialize correctly."""
        d = {
            "class_id": "converging_pair",
            "confidence": 0.95,
            "features": {"grid_length": 3},
            "reasoning": "Test",
        }
        classification = CounterexampleClassification.from_dict(d)

        assert classification.class_id == CounterexampleClassId.CONVERGING_PAIR
        assert classification.confidence == 0.95
        assert classification.features["grid_length"] == 3

    def test_failure_classification_result_to_dict(self):
        """FailureClassificationResult should serialize correctly."""
        result = FailureClassificationResult(
            failure_class=FailureClass.TYPE_B_NOVEL_COUNTEREXAMPLE,
            counterexample_class=CounterexampleClassification(
                class_id=CounterexampleClassId.CONVERGING_PAIR,
                confidence=0.95,
            ),
            is_known_class=False,
            reason="Novel counterexample",
            actionable=False,
        )
        d = result.to_dict()

        assert d["failure_class"] == "type_b_novel_counterexample"
        assert d["is_known_class"] is False
        assert d["counterexample_class"]["class_id"] == "converging_pair"

    def test_failure_classification_result_from_dict(self):
        """FailureClassificationResult should deserialize correctly."""
        d = {
            "failure_class": "type_a_known_counterexample",
            "counterexample_class": {
                "class_id": "x_emission",
                "confidence": 0.85,
                "features": {},
                "reasoning": "",
            },
            "is_known_class": True,
            "reason": "Known pattern",
            "actionable": False,
        }
        result = FailureClassificationResult.from_dict(d)

        assert result.failure_class == FailureClass.TYPE_A_KNOWN_COUNTEREXAMPLE
        assert result.is_known_class is True
        assert result.counterexample_class.class_id == CounterexampleClassId.X_EMISSION


class TestModuleLevelClassifier:
    """Tests for module-level classifier functions."""

    def test_classify_failure_function(self):
        """Module-level classify_failure should work."""
        verdict = LawVerdict(
            law_id="test_law",
            status="UNKNOWN",
            reason_code=ReasonCode.AMBIGUOUS_CLAIM,
        )
        result = classify_failure(verdict)
        assert result.failure_class == FailureClass.TYPE_C_PROCESS_ISSUE

    def test_get_classifier_function(self):
        """Module-level get_classifier should return classifier."""
        classifier = get_classifier()
        assert isinstance(classifier, FailureClassifier)
