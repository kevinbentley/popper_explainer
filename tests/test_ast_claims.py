"""Tests for structured claim AST schema and evaluator."""

import pytest

from src.claims.ast_schema import ClaimAST, ValidationError, ast_to_string, validate_claim_ast
from src.claims.ast_evaluator import ASTClaimEvaluator, ASTEvaluationError, create_ast_checker
from src.claims.schema import CandidateLaw, Observable, Quantifiers, Template


# === AST Schema Validation Tests ===

class TestASTValidation:
    """Tests for AST validation."""

    def test_valid_const(self):
        """Test constant node validation."""
        ast = {"const": 5}
        is_valid, errors = validate_claim_ast(ast, set())
        assert is_valid
        assert len(errors) == 0

    def test_invalid_const_type(self):
        """Test constant must be a number."""
        ast = {"const": "hello"}
        is_valid, errors = validate_claim_ast(ast, set())
        assert not is_valid
        assert any("number" in e.message for e in errors)

    def test_valid_var(self):
        """Test variable node validation."""
        ast = {"var": "t"}
        is_valid, errors = validate_claim_ast(ast, set())
        assert is_valid

    def test_invalid_var(self):
        """Test only 't' is allowed as variable."""
        ast = {"var": "x"}
        is_valid, errors = validate_claim_ast(ast, set())
        assert not is_valid
        assert any("'t'" in e.message for e in errors)

    def test_valid_t_plus_1(self):
        """Test t+1 shorthand validation."""
        ast = {"t_plus_1": True}
        is_valid, errors = validate_claim_ast(ast, set())
        assert is_valid

    def test_invalid_t_plus_1(self):
        """Test t_plus_1 must be true."""
        ast = {"t_plus_1": False}
        is_valid, errors = validate_claim_ast(ast, set())
        assert not is_valid

    def test_valid_obs(self):
        """Test observable reference validation."""
        ast = {"obs": "N", "t": {"var": "t"}}
        is_valid, errors = validate_claim_ast(ast, {"N"})
        assert is_valid

    def test_obs_unknown_observable(self):
        """Test unknown observable is rejected."""
        ast = {"obs": "X", "t": {"var": "t"}}
        is_valid, errors = validate_claim_ast(ast, {"N"})
        assert not is_valid
        assert any("Unknown observable" in e.message for e in errors)

    def test_obs_missing_t(self):
        """Test observable requires 't' field."""
        ast = {"obs": "N"}
        is_valid, errors = validate_claim_ast(ast, {"N"})
        assert not is_valid
        assert any("requires 't'" in e.message for e in errors)

    def test_valid_binop(self):
        """Test binary operation validation."""
        ast = {
            "op": "==",
            "lhs": {"obs": "N", "t": {"var": "t"}},
            "rhs": {"obs": "N", "t": {"const": 0}},
        }
        is_valid, errors = validate_claim_ast(ast, {"N"})
        assert is_valid

    def test_binop_missing_lhs(self):
        """Test binary op requires lhs."""
        ast = {"op": "==", "rhs": {"const": 0}}
        is_valid, errors = validate_claim_ast(ast, set())
        assert not is_valid
        assert any("'lhs'" in e.message for e in errors)

    def test_binop_missing_rhs(self):
        """Test binary op requires rhs."""
        ast = {"op": "==", "lhs": {"const": 0}}
        is_valid, errors = validate_claim_ast(ast, set())
        assert not is_valid
        assert any("'rhs'" in e.message for e in errors)

    def test_valid_unary_not(self):
        """Test unary not operation."""
        ast = {"op": "not", "arg": {"const": 0}}
        is_valid, errors = validate_claim_ast(ast, set())
        assert is_valid

    def test_unary_not_missing_arg(self):
        """Test not requires 'arg' field."""
        ast = {"op": "not"}
        is_valid, errors = validate_claim_ast(ast, set())
        assert not is_valid
        assert any("'arg'" in e.message for e in errors)

    def test_invalid_operator(self):
        """Test unknown operator is rejected."""
        ast = {"op": "xor", "lhs": {"const": 1}, "rhs": {"const": 0}}
        is_valid, errors = validate_claim_ast(ast, set())
        assert not is_valid
        assert any("Unknown operator" in e.message for e in errors)

    def test_nested_ast(self):
        """Test deeply nested AST validation."""
        ast = {
            "op": "=>",
            "lhs": {"op": ">", "lhs": {"obs": "X", "t": {"var": "t"}}, "rhs": {"const": 0}},
            "rhs": {"op": "==", "lhs": {"obs": "X", "t": {"t_plus_1": True}}, "rhs": {"const": 0}},
        }
        is_valid, errors = validate_claim_ast(ast, {"X"})
        assert is_valid


class TestASTToString:
    """Tests for AST to string conversion."""

    def test_const(self):
        assert ast_to_string({"const": 5}) == "5"

    def test_var(self):
        assert ast_to_string({"var": "t"}) == "t"

    def test_t_plus_1(self):
        assert ast_to_string({"t_plus_1": True}) == "t+1"

    def test_obs(self):
        ast = {"obs": "N", "t": {"var": "t"}}
        assert ast_to_string(ast) == "N(t)"

    def test_obs_at_zero(self):
        ast = {"obs": "N", "t": {"const": 0}}
        assert ast_to_string(ast) == "N(0)"

    def test_binop(self):
        ast = {
            "op": "==",
            "lhs": {"obs": "N", "t": {"var": "t"}},
            "rhs": {"obs": "N", "t": {"const": 0}},
        }
        assert ast_to_string(ast) == "(N(t) == N(0))"

    def test_implication(self):
        ast = {
            "op": "=>",
            "lhs": {"const": 1},
            "rhs": {"const": 0},
        }
        assert "=>" in ast_to_string(ast)

    def test_not(self):
        ast = {"op": "not", "arg": {"const": 1}}
        assert ast_to_string(ast) == "not(1)"


# === AST Evaluator Tests ===

class TestASTEvaluator:
    """Tests for AST claim evaluator."""

    def _make_law(
        self,
        observables: list[tuple[str, str]],
        claim_ast: dict,
        template: Template = Template.INVARIANT,
    ) -> CandidateLaw:
        """Helper to create a law with claim_ast."""
        return CandidateLaw(
            law_id="test_law",
            template=template,
            quantifiers=Quantifiers(T=50),
            observables=[Observable(name=n, expr=e) for n, e in observables],
            claim="(see claim_ast)",
            forbidden="test",
            claim_ast=claim_ast,
        )

    def test_evaluate_const(self):
        """Test constant evaluation."""
        law = self._make_law([], {"const": 5})
        evaluator = ASTClaimEvaluator(law)
        trajectory = [".>."]
        result = evaluator.evaluate_at_time({"const": 5}, trajectory, 0)
        assert result == 5

    def test_evaluate_var_t(self):
        """Test time variable evaluation."""
        law = self._make_law([], {"var": "t"})
        evaluator = ASTClaimEvaluator(law)
        trajectory = [".>."]
        assert evaluator.evaluate_at_time({"var": "t"}, trajectory, 3) == 3

    def test_evaluate_t_plus_1(self):
        """Test t+1 evaluation."""
        law = self._make_law([], {"t_plus_1": True})
        evaluator = ASTClaimEvaluator(law)
        trajectory = [".>."]
        assert evaluator.evaluate_at_time({"t_plus_1": True}, trajectory, 5) == 6

    def test_evaluate_obs(self):
        """Test observable evaluation."""
        law = self._make_law(
            [("N", "count('>') + count('<') + 2*count('X')")],
            {"obs": "N", "t": {"var": "t"}},
        )
        evaluator = ASTClaimEvaluator(law)
        trajectory = [".>.", "..>", ">..", ">.."]
        result = evaluator.evaluate_at_time({"obs": "N", "t": {"var": "t"}}, trajectory, 0)
        assert result == 1  # One right-moving particle

    def test_evaluate_obs_at_zero(self):
        """Test observable at t=0."""
        law = self._make_law(
            [("N", "count('>')")],
            {"obs": "N", "t": {"const": 0}},
        )
        evaluator = ASTClaimEvaluator(law)
        trajectory = [".>.", "..>", ">..", ">.."]
        # Always evaluates at t=0, regardless of current t
        result = evaluator.evaluate_at_time({"obs": "N", "t": {"const": 0}}, trajectory, 3)
        assert result == 1

    def test_evaluate_arithmetic(self):
        """Test arithmetic operations."""
        law = self._make_law([], {"const": 0})
        evaluator = ASTClaimEvaluator(law)
        trajectory = ["."]

        assert evaluator.evaluate_at_time({"op": "+", "lhs": {"const": 2}, "rhs": {"const": 3}}, trajectory, 0) == 5
        assert evaluator.evaluate_at_time({"op": "-", "lhs": {"const": 5}, "rhs": {"const": 3}}, trajectory, 0) == 2
        assert evaluator.evaluate_at_time({"op": "*", "lhs": {"const": 4}, "rhs": {"const": 3}}, trajectory, 0) == 12
        assert evaluator.evaluate_at_time({"op": "/", "lhs": {"const": 10}, "rhs": {"const": 2}}, trajectory, 0) == 5.0

    def test_evaluate_comparison(self):
        """Test comparison operations."""
        law = self._make_law([], {"const": 0})
        evaluator = ASTClaimEvaluator(law)
        trajectory = ["."]

        assert evaluator.evaluate_at_time({"op": "==", "lhs": {"const": 5}, "rhs": {"const": 5}}, trajectory, 0) is True
        assert evaluator.evaluate_at_time({"op": "!=", "lhs": {"const": 5}, "rhs": {"const": 3}}, trajectory, 0) is True
        assert evaluator.evaluate_at_time({"op": "<", "lhs": {"const": 3}, "rhs": {"const": 5}}, trajectory, 0) is True
        assert evaluator.evaluate_at_time({"op": "<=", "lhs": {"const": 5}, "rhs": {"const": 5}}, trajectory, 0) is True
        assert evaluator.evaluate_at_time({"op": ">", "lhs": {"const": 5}, "rhs": {"const": 3}}, trajectory, 0) is True
        assert evaluator.evaluate_at_time({"op": ">=", "lhs": {"const": 3}, "rhs": {"const": 3}}, trajectory, 0) is True

    def test_evaluate_logical(self):
        """Test logical operations."""
        law = self._make_law([], {"const": 0})
        evaluator = ASTClaimEvaluator(law)
        trajectory = ["."]

        # Implication: False => X is True
        assert evaluator.evaluate_at_time(
            {"op": "=>", "lhs": {"const": 0}, "rhs": {"const": 0}}, trajectory, 0
        ) is True

        # And
        assert evaluator.evaluate_at_time(
            {"op": "and", "lhs": {"const": 1}, "rhs": {"const": 1}}, trajectory, 0
        ) == 1

        # Or
        assert evaluator.evaluate_at_time(
            {"op": "or", "lhs": {"const": 0}, "rhs": {"const": 1}}, trajectory, 0
        ) == 1

        # Not
        assert evaluator.evaluate_at_time(
            {"op": "not", "arg": {"const": 0}}, trajectory, 0
        ) is True


class TestASTCheckInvariant:
    """Tests for invariant checking with AST."""

    def _make_invariant_law(self, obs_name: str, obs_expr: str) -> CandidateLaw:
        """Create an invariant law: obs(t) == obs(0)."""
        return CandidateLaw(
            law_id="test_invariant",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            observables=[Observable(name=obs_name, expr=obs_expr)],
            claim=f"{obs_name}(t) == {obs_name}(0)",
            forbidden=f"exists t where {obs_name}(t) != {obs_name}(0)",
            claim_ast={
                "op": "==",
                "lhs": {"obs": obs_name, "t": {"var": "t"}},
                "rhs": {"obs": obs_name, "t": {"const": 0}},
            },
        )

    def test_particle_count_conserved(self):
        """Test particle count conservation holds."""
        law = self._make_invariant_law("N", "count('>') + count('<') + 2*count('X')")
        evaluator = ASTClaimEvaluator(law)

        # Trajectory where particle count is conserved (1 right, 1 left -> collision -> separate)
        trajectory = [">.<.", ".X..", "<.>."]  # N = 2 throughout
        passed, t_fail, details, _ = evaluator.check_invariant(trajectory)
        assert passed
        assert t_fail is None

    def test_particle_count_violation(self):
        """Test particle count violation is detected."""
        law = self._make_invariant_law("N", "count('>')")
        evaluator = ASTClaimEvaluator(law)

        # Trajectory where right-movers change (collision absorbs them)
        trajectory = [".>.", "..>", ">.."]
        passed, t_fail, details, _ = evaluator.check_invariant(trajectory)
        # Actually this should pass since count('>') = 1 throughout
        assert passed

        # Now one that actually violates
        trajectory2 = [".>.", ".X.", ".<."]  # > -> X -> <
        passed2, t_fail2, details2, _ = evaluator.check_invariant(trajectory2)
        assert not passed2
        assert t_fail2 == 1  # Violation at t=1 where X appears

    def test_momentum_conserved(self):
        """Test momentum conservation."""
        law = self._make_invariant_law("P", "count('>') - count('<')")
        evaluator = ASTClaimEvaluator(law)

        trajectory = [">.<.", ".X..", "<.>."]  # P = 0 throughout
        passed, t_fail, details, _ = evaluator.check_invariant(trajectory)
        assert passed


class TestASTCheckImplication:
    """Tests for implication checking with AST."""

    def _make_implication_law(self) -> CandidateLaw:
        """Create implication law: X_count(t) > 0 => X_count(t+1) == 0."""
        return CandidateLaw(
            law_id="collision_resolves",
            template=Template.IMPLICATION_STEP,
            quantifiers=Quantifiers(T=50),
            observables=[Observable(name="X_count", expr="count('X')")],
            claim="X_count(t) > 0 => X_count(t+1) == 0",
            forbidden="exists t where X_count(t) > 0 and X_count(t+1) > 0",
            claim_ast={
                "op": "=>",
                "lhs": {"op": ">", "lhs": {"obs": "X_count", "t": {"var": "t"}}, "rhs": {"const": 0}},
                "rhs": {"op": "==", "lhs": {"obs": "X_count", "t": {"t_plus_1": True}}, "rhs": {"const": 0}},
            },
        )

    def test_implication_holds(self):
        """Test implication holds when consequent is true."""
        law = self._make_implication_law()
        evaluator = ASTClaimEvaluator(law)

        # Collision resolves in one step
        trajectory = [".>.<.", "..X..", ".<.>.", ".<..>"]
        passed, t_fail, details, _ = evaluator.check_implication_step(trajectory)
        assert passed

    def test_implication_vacuously_true(self):
        """Test implication holds when antecedent is never true."""
        law = self._make_implication_law()
        evaluator = ASTClaimEvaluator(law)

        # No collisions ever
        trajectory = [".>..", "..>.", "...>", ">..."]
        passed, t_fail, details, _ = evaluator.check_implication_step(trajectory)
        assert passed  # Vacuously true

    def test_implication_violation(self):
        """Test implication violation detection."""
        law = self._make_implication_law()
        evaluator = ASTClaimEvaluator(law)

        # Collision persists (hypothetically)
        trajectory = [".X..", ".X..", ".<>."]
        passed, t_fail, details, _ = evaluator.check_implication_step(trajectory)
        assert not passed
        assert t_fail == 0


class TestASTCheckMonotone:
    """Tests for monotone checking with AST."""

    def _make_monotone_law(self, direction: str = "<=") -> CandidateLaw:
        """Create monotone law: X_count(t+1) <= X_count(t)."""
        return CandidateLaw(
            law_id="x_decreases",
            template=Template.MONOTONE,
            quantifiers=Quantifiers(T=50),
            observables=[Observable(name="X_count", expr="count('X')")],
            claim=f"X_count(t+1) {direction} X_count(t)",
            forbidden=f"exists t where X_count(t+1) > X_count(t)" if direction == "<=" else "...",
            claim_ast={
                "op": direction,
                "lhs": {"obs": "X_count", "t": {"t_plus_1": True}},
                "rhs": {"obs": "X_count", "t": {"var": "t"}},
            },
        )

    def test_monotone_holds(self):
        """Test monotone decreasing holds."""
        law = self._make_monotone_law("<=")
        evaluator = ASTClaimEvaluator(law)

        # X count decreases or stays same
        trajectory = [".XX.", ".X..", "....", "...."]
        passed, t_fail, details, _ = evaluator.check_monotone(trajectory)
        assert passed

    def test_monotone_violation(self):
        """Test monotone violation detection."""
        law = self._make_monotone_law("<=")
        evaluator = ASTClaimEvaluator(law)

        # X count increases: t=0 has 0 X's, t=1 has 1 X
        # Comparing X(t+1) <= X(t) at t=0: X(1)=1 <= X(0)=0 is False
        trajectory = ["....", ".X..", ".XX."]
        passed, t_fail, details, _ = evaluator.check_monotone(trajectory)
        assert not passed
        assert t_fail == 0  # First violation at t=0 (X(1) > X(0))


class TestCreateASTChecker:
    """Tests for create_ast_checker factory."""

    def test_returns_none_without_claim_ast(self):
        """Test returns None when no claim_ast."""
        law = CandidateLaw(
            law_id="test",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            observables=[],
            claim="test",
            forbidden="test",
            claim_ast=None,
        )
        assert create_ast_checker(law) is None

    def test_returns_evaluator_with_claim_ast(self):
        """Test returns evaluator when claim_ast present."""
        law = CandidateLaw(
            law_id="test",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            observables=[Observable(name="N", expr="count('>')")],
            claim="N(t) == N(0)",
            forbidden="test",
            claim_ast={
                "op": "==",
                "lhs": {"obs": "N", "t": {"var": "t"}},
                "rhs": {"obs": "N", "t": {"const": 0}},
            },
        )
        checker = create_ast_checker(law)
        assert checker is not None
        assert isinstance(checker, ASTClaimEvaluator)

    def test_raises_on_invalid_ast(self):
        """Test raises error on invalid AST."""
        law = CandidateLaw(
            law_id="test",
            template=Template.INVARIANT,
            quantifiers=Quantifiers(T=50),
            observables=[],
            claim="test",
            forbidden="test",
            claim_ast={"obs": "unknown", "t": {"var": "t"}},  # Unknown observable
        )
        with pytest.raises(ASTEvaluationError):
            create_ast_checker(law)
