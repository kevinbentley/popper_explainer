"""Tests for expression parsing and evaluation (Phase 2A)."""

import pytest

from src.claims.expr_ast import BinOp, Count, GridLength, Literal, Operator, expr_to_string
from src.claims.expr_evaluator import ExpressionEvaluator, evaluate_expression
from src.claims.expr_parser import ExpressionParser, ParseError, parse_expression


class TestExpressionParser:
    """Tests for the expression parser."""

    def test_parse_literal(self):
        expr = parse_expression("42")
        assert expr == Literal(42)

    def test_parse_count_single_quote(self):
        expr = parse_expression("count('>')")
        assert expr == Count(">")

    def test_parse_count_double_quote(self):
        expr = parse_expression('count(">")')
        assert expr == Count(">")

    def test_parse_count_all_symbols(self):
        assert parse_expression("count('.')") == Count(".")
        assert parse_expression("count('>')") == Count(">")
        assert parse_expression("count('<')") == Count("<")
        assert parse_expression("count('X')") == Count("X")

    def test_parse_grid_length(self):
        expr = parse_expression("grid_length")
        assert expr == GridLength()

    def test_parse_addition(self):
        expr = parse_expression("count('>') + count('X')")
        assert isinstance(expr, BinOp)
        assert expr.op == Operator.ADD
        assert expr.left == Count(">")
        assert expr.right == Count("X")

    def test_parse_subtraction(self):
        expr = parse_expression("count('>') - count('<')")
        assert isinstance(expr, BinOp)
        assert expr.op == Operator.SUB

    def test_parse_multiplication(self):
        expr = parse_expression("2 * count('X')")
        assert isinstance(expr, BinOp)
        assert expr.op == Operator.MUL
        assert expr.left == Literal(2)
        assert expr.right == Count("X")

    def test_parse_complex_expression(self):
        # R_total = count('>') + count('X')
        expr = parse_expression("count('>') + count('X')")
        assert isinstance(expr, BinOp)

    def test_parse_particle_count(self):
        # particle_count = count('>') + count('<') + 2*count('X')
        expr = parse_expression("count('>') + count('<') + 2 * count('X')")
        assert isinstance(expr, BinOp)

    def test_parse_parentheses(self):
        expr = parse_expression("(count('>') + count('X'))")
        assert isinstance(expr, BinOp)
        assert expr.left == Count(">")
        assert expr.right == Count("X")

    def test_parse_nested_parentheses(self):
        expr = parse_expression("(count('>') + count('X')) - (count('<') + count('X'))")
        assert isinstance(expr, BinOp)
        assert expr.op == Operator.SUB

    def test_operator_precedence(self):
        # 2 * count('X') + count('>') should parse as (2 * count('X')) + count('>')
        expr = parse_expression("2 * count('X') + count('>')")
        assert isinstance(expr, BinOp)
        assert expr.op == Operator.ADD
        assert isinstance(expr.left, BinOp)
        assert expr.left.op == Operator.MUL

    def test_parse_whitespace_tolerance(self):
        expr1 = parse_expression("count('>')  +  count('X')")
        expr2 = parse_expression("count('>') + count('X')")
        assert expr1 == expr2

    def test_parse_invalid_symbol(self):
        with pytest.raises(ParseError):
            parse_expression("count('a')")

    def test_parse_empty_expression(self):
        with pytest.raises(ParseError):
            parse_expression("")

    def test_parse_unclosed_paren(self):
        with pytest.raises(ParseError):
            parse_expression("(count('>')")

    def test_parse_invalid_count_syntax(self):
        with pytest.raises(ParseError):
            parse_expression("count >")


class TestExpressionEvaluator:
    """Tests for expression evaluation."""

    def test_evaluate_literal(self):
        expr = Literal(42)
        assert evaluate_expression(expr, "...") == 42

    def test_evaluate_count(self):
        state = "..><.X.."
        assert evaluate_expression(Count("."), state) == 5
        assert evaluate_expression(Count(">"), state) == 1
        assert evaluate_expression(Count("<"), state) == 1
        assert evaluate_expression(Count("X"), state) == 1

    def test_evaluate_grid_length(self):
        assert evaluate_expression(GridLength(), "....") == 4
        assert evaluate_expression(GridLength(), "..><.X..") == 8

    def test_evaluate_addition(self):
        state = "..><.X.."
        # R_total = count('>') + count('X') = 1 + 1 = 2
        expr = BinOp(Count(">"), Operator.ADD, Count("X"))
        assert evaluate_expression(expr, state) == 2

    def test_evaluate_subtraction(self):
        state = ">>>.<"
        # momentum = count('>') - count('<') = 3 - 1 = 2
        expr = BinOp(Count(">"), Operator.SUB, Count("<"))
        assert evaluate_expression(expr, state) == 2

    def test_evaluate_multiplication(self):
        state = "..X.."
        # 2 * count('X') = 2 * 1 = 2
        expr = BinOp(Literal(2), Operator.MUL, Count("X"))
        assert evaluate_expression(expr, state) == 2

    def test_evaluate_particle_count(self):
        # particle_count = count('>') + count('<') + 2*count('X')
        state = "..><.X.."  # 1 >, 1 <, 1 X
        expr = parse_expression("count('>') + count('<') + 2 * count('X')")
        assert evaluate_expression(expr, state) == 4

    def test_evaluate_momentum(self):
        # momentum = count('>') - count('<')
        state = ">>><."  # 3 >, 1 <
        expr = parse_expression("count('>') - count('<')")
        assert evaluate_expression(expr, state) == 2


class TestExprToString:
    """Tests for converting AST back to string."""

    def test_literal_to_string(self):
        assert expr_to_string(Literal(42)) == "42"

    def test_count_to_string(self):
        assert expr_to_string(Count(">")) == "count('>')"

    def test_grid_length_to_string(self):
        assert expr_to_string(GridLength()) == "grid_length"

    def test_binop_to_string(self):
        expr = BinOp(Count(">"), Operator.ADD, Count("X"))
        assert expr_to_string(expr) == "count('>') + count('X')"


class TestIntegration:
    """Integration tests: parse then evaluate."""

    def test_r_total_conservation_observable(self):
        expr = parse_expression("count('>') + count('X')")
        state1 = "..>.."  # 1 >, 0 X
        state2 = "..X.."  # 0 >, 1 X
        assert evaluate_expression(expr, state1) == 1
        assert evaluate_expression(expr, state2) == 1

    def test_complex_expression(self):
        # (count('>') + count('X')) - (count('<') + count('X'))
        # This simplifies to count('>') - count('<') = momentum
        expr = parse_expression("(count('>') + count('X')) - (count('<') + count('X'))")
        state = ">>><X"  # 3 >, 1 <, 1 X
        # momentum = 3 - 1 = 2
        assert evaluate_expression(expr, state) == 2
