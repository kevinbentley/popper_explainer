"""Parser for observable expressions.

Parses strings like:
  count('>') + count('X')
  count('>') + count('<') + 2*count('X')
  grid_length
  (count('>') - count('<'))

Into an AST for evaluation.
"""

import re
from dataclasses import dataclass

from src.claims.expr_ast import BinOp, Count, Expr, GridLength, Literal, Operator
from src.universe.types import VALID_SYMBOLS


class ParseError(Exception):
    """Raised when expression parsing fails."""

    pass


@dataclass
class Token:
    """A lexical token."""

    type: str  # 'NUMBER', 'COUNT', 'GRID_LENGTH', 'SYMBOL', 'OP', 'LPAREN', 'RPAREN'
    value: str | int
    pos: int


class ExpressionParser:
    """Parser for observable expressions.

    Grammar (informal):
        expr     -> term (('+' | '-') term)*
        term     -> factor ('*' factor)*
        factor   -> NUMBER | COUNT | GRID_LENGTH | '(' expr ')'
        COUNT    -> 'count' '(' SYMBOL ')'
        SYMBOL   -> "'" CHAR "'" where CHAR in {'.', '>', '<', 'X'}
    """

    # Token patterns
    PATTERNS = [
        (r"\s+", None),  # Skip whitespace
        (r"\d+", "NUMBER"),
        (r"count", "COUNT"),
        (r"grid_length", "GRID_LENGTH"),
        (r"'([.><X])'", "SYMBOL"),  # Quoted symbol
        (r'"([.><X])"', "SYMBOL"),  # Double-quoted symbol also allowed
        (r"\+", "OP"),
        (r"-", "OP"),
        (r"\*", "OP"),
        (r"\(", "LPAREN"),
        (r"\)", "RPAREN"),
    ]

    def __init__(self):
        # Compile patterns
        self._patterns = [
            (re.compile(pattern), token_type) for pattern, token_type in self.PATTERNS
        ]

    def parse(self, text: str) -> Expr:
        """Parse an expression string into an AST.

        Args:
            text: The expression string

        Returns:
            The parsed expression AST

        Raises:
            ParseError: If parsing fails
        """
        tokens = self._tokenize(text)
        if not tokens:
            raise ParseError("Empty expression")

        expr, pos = self._parse_expr(tokens, 0)

        if pos < len(tokens):
            remaining = tokens[pos]
            raise ParseError(
                f"Unexpected token '{remaining.value}' at position {remaining.pos}"
            )

        return expr

    def _tokenize(self, text: str) -> list[Token]:
        """Convert text to tokens."""
        tokens = []
        pos = 0

        while pos < len(text):
            match = None
            for pattern, token_type in self._patterns:
                m = pattern.match(text, pos)
                if m:
                    match = m
                    if token_type is not None:
                        value: str | int
                        if token_type == "NUMBER":
                            value = int(m.group())
                        elif token_type == "SYMBOL":
                            # Extract the symbol from quotes
                            value = m.group(1)
                        else:
                            value = m.group()
                        tokens.append(Token(token_type, value, pos))
                    pos = m.end()
                    break

            if match is None:
                raise ParseError(f"Invalid character '{text[pos]}' at position {pos}")

        return tokens

    def _parse_expr(self, tokens: list[Token], pos: int) -> tuple[Expr, int]:
        """Parse: expr -> term (('+' | '-') term)*"""
        left, pos = self._parse_term(tokens, pos)

        while pos < len(tokens) and tokens[pos].type == "OP" and tokens[pos].value in ("+", "-"):
            op_token = tokens[pos]
            op = Operator.ADD if op_token.value == "+" else Operator.SUB
            pos += 1
            right, pos = self._parse_term(tokens, pos)
            left = BinOp(left, op, right)

        return left, pos

    def _parse_term(self, tokens: list[Token], pos: int) -> tuple[Expr, int]:
        """Parse: term -> factor ('*' factor)*"""
        left, pos = self._parse_factor(tokens, pos)

        while pos < len(tokens) and tokens[pos].type == "OP" and tokens[pos].value == "*":
            pos += 1
            right, pos = self._parse_factor(tokens, pos)
            left = BinOp(left, Operator.MUL, right)

        return left, pos

    def _parse_factor(self, tokens: list[Token], pos: int) -> tuple[Expr, int]:
        """Parse: factor -> NUMBER | COUNT | GRID_LENGTH | '(' expr ')'"""
        if pos >= len(tokens):
            raise ParseError("Unexpected end of expression")

        token = tokens[pos]

        if token.type == "NUMBER":
            return Literal(token.value), pos + 1

        elif token.type == "GRID_LENGTH":
            return GridLength(), pos + 1

        elif token.type == "COUNT":
            # Expect: count '(' SYMBOL ')'
            pos += 1
            if pos >= len(tokens) or tokens[pos].type != "LPAREN":
                raise ParseError(f"Expected '(' after 'count' at position {token.pos}")
            pos += 1
            if pos >= len(tokens) or tokens[pos].type != "SYMBOL":
                raise ParseError(f"Expected symbol in count() at position {token.pos}")
            symbol = tokens[pos].value
            if symbol not in VALID_SYMBOLS:
                raise ParseError(f"Invalid symbol '{symbol}' in count()")
            pos += 1
            if pos >= len(tokens) or tokens[pos].type != "RPAREN":
                raise ParseError(f"Expected ')' after symbol at position {token.pos}")
            return Count(symbol), pos + 1

        elif token.type == "LPAREN":
            pos += 1
            expr, pos = self._parse_expr(tokens, pos)
            if pos >= len(tokens) or tokens[pos].type != "RPAREN":
                raise ParseError(f"Expected ')' at position {token.pos}")
            return expr, pos + 1

        else:
            raise ParseError(f"Unexpected token '{token.value}' at position {token.pos}")


# Module-level parser instance for convenience
_parser = ExpressionParser()


def parse_expression(text: str) -> Expr:
    """Parse an expression string into an AST.

    Convenience function using module-level parser.
    """
    return _parser.parse(text)
