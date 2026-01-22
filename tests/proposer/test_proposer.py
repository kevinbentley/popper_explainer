"""Tests for the law proposer subsystem."""

import json
import pytest

from src.claims.schema import (
    CandidateLaw,
    ComparisonOp,
    Observable,
    Precondition,
    Quantifiers,
    Template,
)
from src.proposer.client import GeminiConfig, MockGeminiClient
from src.proposer.memory import DiscoveryMemory, DiscoveryMemorySnapshot
from src.proposer.parser import ParseError, ParseResult, ResponseParser
from src.proposer.prompt import PromptBuilder, UniverseContract
from src.proposer.ranking import RankingFeatures, RankingModel, RankingWeights
from src.proposer.redundancy import RedundancyDetector, RedundancyMatch
from src.proposer.proposer import LawProposer, ProposalBatch, ProposalRequest, ProposerConfig


class TestUniverseContract:
    """Tests for UniverseContract."""

    def test_default_contract(self):
        """Default contract has expected values."""
        contract = UniverseContract()

        assert contract.universe_id == "kinetic_grid_v1"
        assert ">" in contract.symbols
        assert "<" in contract.symbols
        assert "mirror_swap" in contract.capabilities["transforms"]

    def test_to_dict(self):
        """Contract converts to dict."""
        contract = UniverseContract()
        d = contract.to_dict()

        assert d["universe_id"] == "kinetic_grid_v1"
        assert "symbols" in d
        assert "capabilities" in d

    def test_from_dict(self):
        """Contract creates from dict."""
        data = {
            "universe_id": "test",
            "symbols": ["a", "b"],
            "state_representation": "array",
            "capabilities": {"transforms": ["t1"]},
            "config_knobs": {"param": 10},
        }
        contract = UniverseContract.from_dict(data)

        assert contract.universe_id == "test"
        assert contract.symbols == ["a", "b"]


class TestPromptBuilder:
    """Tests for PromptBuilder."""

    def test_build_basic_prompt(self):
        """Builder creates a prompt."""
        builder = PromptBuilder()
        contract = UniverseContract()
        memory = DiscoveryMemorySnapshot()

        prompt = builder.build(contract, memory, request_count=3)

        assert "UNIVERSE CAPABILITIES" in prompt
        assert "YOUR TASK" in prompt
        assert "3" in prompt

    def test_includes_accepted_laws(self):
        """Prompt includes accepted laws."""
        builder = PromptBuilder()
        contract = UniverseContract()
        memory = DiscoveryMemorySnapshot(
            accepted_laws=[{
                "law_id": "test_law",
                "template": "invariant",
                "claim": "x(t) == x(0)",
                "observables": [],
                "preconditions": [],
            }]
        )

        prompt = builder.build(contract, memory)

        assert "ACCEPTED LAWS" in prompt
        assert "test_law" in prompt

    def test_includes_falsified_laws(self):
        """Prompt includes falsified laws."""
        builder = PromptBuilder()
        contract = UniverseContract()
        memory = DiscoveryMemorySnapshot(
            falsified_laws=[{
                "law_id": "false_law",
                "template": "invariant",
                "claim": "y(t) == y(0)",
                "counterexample": {"initial_state": ">..<", "t_fail": 2},
            }]
        )

        prompt = builder.build(contract, memory)

        assert "FALSIFIED LAWS" in prompt
        assert "false_law" in prompt
        assert ">..<" in prompt

    def test_target_templates(self):
        """Prompt includes target template guidance."""
        builder = PromptBuilder()
        contract = UniverseContract()
        memory = DiscoveryMemorySnapshot()

        prompt = builder.build(
            contract, memory,
            target_templates=["symmetry_commutation"],
        )

        assert "symmetry_commutation" in prompt

    def test_system_instruction(self):
        """Builder provides system instruction."""
        builder = PromptBuilder()
        instruction = builder.get_system_instruction()

        assert "falsifiable" in instruction.lower()
        assert "template" in instruction.lower()


class TestResponseParser:
    """Tests for ResponseParser."""

    def test_parse_valid_response(self):
        """Parser handles valid JSON response."""
        parser = ResponseParser()
        response = json.dumps([{
            "law_id": "test_law",
            "template": "invariant",
            "quantifiers": {"T": 50},
            "preconditions": [{"lhs": "grid_length", "op": ">=", "rhs": 4}],
            "observables": [{"name": "x", "expr": "count('>')"}],
            "claim": "x(t) == x(0)",
            "forbidden": "exists t where x(t) != x(0)",
            "proposed_tests": [{"family": "random_density_sweep"}],
        }])

        result = parser.parse(response)

        assert len(result.laws) == 1
        assert result.laws[0].law_id == "test_law"
        assert result.laws[0].template == Template.INVARIANT

    def test_parse_markdown_code_block(self):
        """Parser handles markdown code blocks."""
        parser = ResponseParser()
        response = """```json
        [{
            "law_id": "test",
            "template": "invariant",
            "claim": "x(t) == x(0)",
            "forbidden": "exists t where x(t) != x(0)"
        }]
        ```"""

        result = parser.parse(response)

        assert len(result.laws) == 1
        assert result.laws[0].law_id == "test"

    def test_parse_missing_template(self):
        """Parser rejects law without template."""
        parser = ResponseParser()
        response = json.dumps([{
            "law_id": "test",
            "claim": "x(t) == x(0)",
            "forbidden": "exists t",
        }])

        result = parser.parse(response)

        assert len(result.laws) == 0
        assert len(result.rejections) == 1

    def test_parse_invalid_template(self):
        """Parser rejects invalid template."""
        parser = ResponseParser()
        response = json.dumps([{
            "law_id": "test",
            "template": "not_a_template",
            "claim": "x(t) == x(0)",
            "forbidden": "exists t",
        }])

        result = parser.parse(response)

        assert len(result.laws) == 0
        assert len(result.rejections) == 1

    def test_parse_missing_forbidden(self):
        """Parser rejects law without forbidden."""
        parser = ResponseParser()
        response = json.dumps([{
            "law_id": "test",
            "template": "invariant",
            "claim": "x(t) == x(0)",
        }])

        result = parser.parse(response)

        assert len(result.laws) == 0
        assert len(result.rejections) == 1

    def test_parse_multiple_laws(self):
        """Parser handles multiple laws."""
        parser = ResponseParser()
        response = json.dumps([
            {
                "law_id": "law1",
                "template": "invariant",
                "claim": "x(t) == x(0)",
                "forbidden": "exists t where x(t) != x(0)",
            },
            {
                "law_id": "law2",
                "template": "monotone",
                "claim": "y(t+1) <= y(t)",
                "forbidden": "exists t where y(t+1) > y(t)",
            },
        ])

        result = parser.parse(response)

        assert len(result.laws) == 2
        assert result.laws[0].law_id == "law1"
        assert result.laws[1].law_id == "law2"

    def test_parse_generates_law_id(self):
        """Parser generates law_id if missing."""
        parser = ResponseParser()
        response = json.dumps([{
            "template": "invariant",
            "claim": "x(t) == x(0)",
            "forbidden": "exists t where x(t) != x(0)",
        }])

        result = parser.parse(response)

        assert len(result.laws) == 1
        assert result.laws[0].law_id == "proposed_0"


class TestRedundancyDetector:
    """Tests for RedundancyDetector."""

    def _make_law(self, law_id: str, template: Template, claim: str) -> CandidateLaw:
        """Helper to create a law."""
        return CandidateLaw(
            law_id=law_id,
            template=template,
            quantifiers=Quantifiers(T=50),
            preconditions=[],
            observables=[],
            claim=claim,
            forbidden=f"exists t where not ({claim})",
        )

    def test_exact_duplicate(self):
        """Detector finds exact duplicates."""
        detector = RedundancyDetector()
        law1 = self._make_law("law1", Template.INVARIANT, "x(t) == x(0)")
        law2 = self._make_law("law2", Template.INVARIANT, "x(t) == x(0)")

        detector.add_known_law(law1)
        match = detector.check(law2)

        assert match is not None
        assert match.match_type == "exact"
        assert match.similarity == 1.0

    def test_no_match(self):
        """Detector returns None for novel law."""
        detector = RedundancyDetector()
        law1 = self._make_law("law1", Template.INVARIANT, "x(t) == x(0)")
        law2 = self._make_law("law2", Template.MONOTONE, "y(t+1) <= y(t)")

        detector.add_known_law(law1)
        match = detector.check(law2)

        assert match is None

    def test_filter_batch(self):
        """Detector filters a batch."""
        detector = RedundancyDetector()
        law1 = self._make_law("law1", Template.INVARIANT, "x(t) == x(0)")
        detector.add_known_law(law1)

        batch = [
            self._make_law("law2", Template.INVARIANT, "x(t) == x(0)"),  # Duplicate
            self._make_law("law3", Template.MONOTONE, "y(t+1) <= y(t)"),  # Novel
        ]

        non_redundant, redundant = detector.filter_batch(batch)

        assert len(non_redundant) == 1
        assert len(redundant) == 1
        assert non_redundant[0].law_id == "law3"

    def test_within_batch_duplicate(self):
        """Detector finds duplicates within same batch."""
        detector = RedundancyDetector()

        batch = [
            self._make_law("law1", Template.INVARIANT, "x(t) == x(0)"),
            self._make_law("law2", Template.INVARIANT, "x(t) == x(0)"),  # Same content
        ]

        non_redundant, redundant = detector.filter_batch(batch)

        assert len(non_redundant) == 1
        assert len(redundant) == 1
        assert redundant[0][1].match_type == "batch_duplicate"


class TestRankingModel:
    """Tests for RankingModel."""

    def _make_law(
        self,
        law_id: str,
        template: Template,
        preconditions: list[Precondition] | None = None,
    ) -> CandidateLaw:
        """Helper to create a law."""
        return CandidateLaw(
            law_id=law_id,
            template=template,
            quantifiers=Quantifiers(T=50),
            preconditions=preconditions or [],
            observables=[],
            claim="test",
            forbidden="test",
        )

    def test_compute_features(self):
        """Model computes ranking features."""
        model = RankingModel()
        law = self._make_law("test", Template.INVARIANT)

        features = model.compute_features(law)

        assert 0 <= features.risk <= 1
        assert 0 <= features.novelty <= 1
        assert 0 <= features.testability <= 1

    def test_risk_higher_without_preconditions(self):
        """Laws without preconditions have higher risk."""
        model = RankingModel()

        law_no_preconds = self._make_law("law1", Template.INVARIANT, [])
        law_with_preconds = self._make_law("law2", Template.INVARIANT, [
            Precondition(lhs="x", op=ComparisonOp.GE, rhs=5),
            Precondition(lhs="y", op=ComparisonOp.LE, rhs=10),
            Precondition(lhs="z", op=ComparisonOp.EQ, rhs=1),
        ])

        risk1 = model._compute_risk(law_no_preconds)
        risk2 = model._compute_risk(law_with_preconds)

        assert risk1 > risk2

    def test_rank_orders_by_score(self):
        """Ranking orders laws by overall score."""
        model = RankingModel()

        laws = [
            self._make_law("low", Template.BOUND),  # Lower risk
            self._make_law("high", Template.INVARIANT),  # Higher risk
        ]

        ranked = model.rank(laws)

        # Higher overall score should come first
        assert len(ranked) == 2
        assert ranked[0][1].overall_score >= ranked[1][1].overall_score


class TestDiscoveryMemory:
    """Tests for DiscoveryMemory."""

    def test_record_accepted(self):
        """Memory records accepted laws."""
        from src.harness.verdict import LawVerdict
        from src.harness.power import PowerMetrics
        from src.harness.vacuity import VacuityReport

        memory = DiscoveryMemory()
        law = CandidateLaw(
            law_id="test",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            preconditions=[],
            observables=[],
            claim="x(t) == x(0)",
            forbidden="exists t where x(t) != x(0)",
        )
        verdict = LawVerdict(
            law_id="test",
            status="PASS",
            power_metrics=PowerMetrics(cases_used=100, coverage_score=0.8),
            vacuity=VacuityReport(),
        )

        memory.record_evaluation(law, verdict)

        assert memory.stats["accepted"] == 1

    def test_get_snapshot(self):
        """Memory provides snapshot."""
        memory = DiscoveryMemory()
        snapshot = memory.get_snapshot()

        assert isinstance(snapshot, DiscoveryMemorySnapshot)
        assert isinstance(snapshot.accepted_laws, list)


class TestMockGeminiClient:
    """Tests for MockGeminiClient."""

    def test_mock_generate(self):
        """Mock client returns responses."""
        responses = ['["response1"]', '["response2"]']
        client = MockGeminiClient(responses)

        r1 = client.generate("prompt1")
        r2 = client.generate("prompt2")

        assert r1 == '["response1"]'
        assert r2 == '["response2"]'
        assert len(client.calls) == 2

    def test_mock_generate_json(self):
        """Mock client parses JSON."""
        responses = ['[{"key": "value"}]']
        client = MockGeminiClient(responses)

        result = client.generate_json("prompt")

        assert result == [{"key": "value"}]


class TestLawProposer:
    """Tests for LawProposer."""

    def test_propose_basic(self):
        """Proposer generates proposals."""
        mock_response = json.dumps([{
            "law_id": "proposed_1",
            "template": "invariant",
            "claim": "x(t) == x(0)",
            "forbidden": "exists t where x(t) != x(0)",
            "proposed_tests": [{"family": "random_density_sweep"}],
        }])

        client = MockGeminiClient([mock_response])
        proposer = LawProposer(client=client)

        batch = proposer.propose(
            DiscoveryMemorySnapshot(),
            ProposalRequest(count=1),
        )

        assert len(batch.laws) == 1
        assert batch.laws[0].law_id == "proposed_1"
        assert batch.prompt_hash != ""

    def test_propose_filters_redundant(self):
        """Proposer filters redundant laws."""
        mock_response = json.dumps([
            {
                "law_id": "law1",
                "template": "invariant",
                "claim": "x(t) == x(0)",
                "forbidden": "exists t where x(t) != x(0)",
            },
            {
                "law_id": "law2",
                "template": "invariant",
                "claim": "x(t) == x(0)",  # Same as law1
                "forbidden": "exists t where x(t) != x(0)",
            },
        ])

        client = MockGeminiClient([mock_response])
        proposer = LawProposer(client=client)

        batch = proposer.propose(DiscoveryMemorySnapshot())

        assert len(batch.laws) == 1
        assert len(batch.redundant) == 1

    def test_propose_ranks_laws(self):
        """Proposer ranks laws."""
        mock_response = json.dumps([
            {
                "law_id": "law1",
                "template": "bound",  # Lower risk
                "claim": "x(t) <= 10",
                "forbidden": "exists t where x(t) > 10",
            },
            {
                "law_id": "law2",
                "template": "invariant",  # Higher risk
                "claim": "y(t) == y(0)",
                "forbidden": "exists t where y(t) != y(0)",
            },
        ])

        client = MockGeminiClient([mock_response])
        proposer = LawProposer(client=client)

        batch = proposer.propose(DiscoveryMemorySnapshot())

        assert len(batch.laws) == 2
        assert len(batch.features) == 2
        # Higher ranked law should have higher score
        assert batch.features[0].overall_score >= batch.features[1].overall_score

    def test_propose_logs_iterations(self):
        """Proposer logs iterations."""
        mock_response = json.dumps([{
            "law_id": "test",
            "template": "invariant",
            "claim": "x(t) == x(0)",
            "forbidden": "exists t",
        }])

        client = MockGeminiClient([mock_response])
        proposer = LawProposer(client=client)

        proposer.propose(DiscoveryMemorySnapshot())

        log = proposer.get_audit_log()
        assert len(log) == 1
        assert "timestamp" in log[0]
        assert "request" in log[0]
        assert "result" in log[0]

    def test_add_known_laws(self):
        """Proposer allows adding known laws."""
        proposer = LawProposer(client=MockGeminiClient())

        law = CandidateLaw(
            law_id="existing",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            preconditions=[],
            observables=[],
            claim="x(t) == x(0)",
            forbidden="exists t",
        )

        proposer.add_known_law(law)

        assert proposer.stats["known_laws_in_filter"] == 1
