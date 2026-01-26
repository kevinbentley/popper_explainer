"""Parser for observable expressions.

Parses strings like:
  count('>') + count('X')
  count('>') + count('<') + 2*count('X')
  grid_length
  (count('>') - count('<'))
  leftmost('>') - leftmost('<')
  max_gap('.') * 2
  adjacent_pairs('>', '<')
  gap_pairs('>', '<', 1)
  spread('>')

Into an AST for evaluation.
"""

import re
from dataclasses import dataclass

from src.claims.expr_ast import (
    AdjacentPairs,
    BinOp,
    Count,
    CountAtParity,
    CountEven,
    CountOdd,
    CountPattern,
    Expr,
    GapPairs,
    GridLength,
    TransitionIndicator,
    Leftmost,
    Literal,
    MaxGap,
    Operator,
    Rightmost,
    Spread,
)
from src.universe.types import VALID_SYMBOLS


class ParseError(Exception):
    """Raised when expression parsing fails."""

    pass


@dataclass
class Token:
    """A lexical token."""

    type: str  # 'NUMBER', 'FUNC', 'GRID_LENGTH', 'SYMBOL', 'OP', 'LPAREN', 'RPAREN', 'COMMA'
    value: str | int
    pos: int


# Observable functions that take one symbol argument
SINGLE_SYMBOL_FUNCS = {"count", "leftmost", "rightmost", "max_gap", "spread", "count_even", "count_odd"}

# Observable functions that take two symbol arguments
TWO_SYMBOL_FUNCS = {"adjacent_pairs"}

# Observable functions that take two symbols and an integer
TWO_SYMBOL_INT_FUNCS = {"gap_pairs"}

# Observable functions that take a symbol and an integer (parity)
SYMBOL_INT_FUNCS = {"count_at_parity"}


class ExpressionParser:
    """Parser for observable expressions.

    Grammar (informal):
        expr     -> term (('+' | '-') term)*
        term     -> factor ('*' factor)*
        factor   -> NUMBER | FUNC | GRID_LENGTH | '(' expr ')'
        FUNC     -> FUNCNAME '(' SYMBOL (',' SYMBOL (',' NUMBER)?)? ')'
        FUNCNAME -> 'count' | 'leftmost' | 'rightmost' | 'max_gap' | 'spread'
                  | 'adjacent_pairs' | 'gap_pairs'
        SYMBOL   -> "'" CHAR "'" where CHAR in {'.', '>', '<', 'X'}
    """

    # Token patterns - order matters (longer matches first)
    PATTERNS = [
        (r"\s+", None),  # Skip whitespace
        (r"\d+", "NUMBER"),
        (r"adjacent_pairs", "FUNC"),
        (r"gap_pairs", "FUNC"),
        (r"count_at_parity", "FUNC"),  # Must be before count_even/count_odd/count
        (r"count_pattern", "FUNC"),  # Must be before count
        (r"count_even", "FUNC"),
        (r"count_odd", "FUNC"),
        (r"transition_indicator", "TRANSITION_INDICATOR"),  # Nullary, like grid_length
        (r"incoming_collisions", "TRANSITION_INDICATOR"),  # Backwards compatibility alias
        (r"grid_length", "GRID_LENGTH"),
        (r"leftmost", "FUNC"),
        (r"rightmost", "FUNC"),
        (r"max_gap", "FUNC"),
        (r"spread", "FUNC"),
        (r"count", "FUNC"),  # Must be after count_at_parity, count_even, count_odd, count_pattern
        # Accept both physical symbols (.><X) and abstract symbols (_ABK)
        (r"'([.><X_ABK]{3})'", "PATTERN"),  # 3-char neighborhood pattern (single quotes)
        (r'"([.><X_ABK]{3})"', "PATTERN"),  # 3-char neighborhood pattern (double quotes)
        (r"'([.><X_ABK])'", "SYMBOL"),  # Quoted symbol (1 char)
        (r'"([.><X_ABK])"', "SYMBOL"),  # Double-quoted symbol also allowed (1 char)
        (r"\+", "OP"),
        (r"-", "OP"),
        (r"\*", "OP"),
        (r"%", "OP"),
        (r",", "COMMA"),
        (r"\(", "LPAREN"),
        (r"\)", "RPAREN"),
    ]

    # Curly/smart quote mappings to straight quotes
    # Using explicit Unicode escapes to avoid encoding issues
    QUOTE_REPLACEMENTS = {
        "\u2018": "'",  # U+2018 LEFT SINGLE QUOTATION MARK '
        "\u2019": "'",  # U+2019 RIGHT SINGLE QUOTATION MARK '
        "\u201C": '"',  # U+201C LEFT DOUBLE QUOTATION MARK "
        "\u201D": '"',  # U+201D RIGHT DOUBLE QUOTATION MARK "
        "\u201A": "'",  # U+201A SINGLE LOW-9 QUOTATION MARK ‚
        "\u201E": '"',  # U+201E DOUBLE LOW-9 QUOTATION MARK „
        "\u2032": "'",  # U+2032 PRIME ′
        "\u2033": '"',  # U+2033 DOUBLE PRIME ″
    }

    def __init__(self):
        # Compile patterns
        self._patterns = [
            (re.compile(pattern), token_type) for pattern, token_type in self.PATTERNS
        ]

    def _sanitize_quotes(self, text: str) -> str:
        """Normalize curly/smart quotes to straight ASCII quotes.

        LLMs sometimes output fancy Unicode quotes which break parsing.
        """
        original = text
        for curly, straight in self.QUOTE_REPLACEMENTS.items():
            text = text.replace(curly, straight)

        # Also replace any remaining non-ASCII quote-like characters
        # that might slip through - comprehensive Unicode quote handling
        extra_replacements = {
            # Single quote replacements
            '\u00b4': "'",  # ACUTE ACCENT ´
            '\u0060': "'",  # GRAVE ACCENT `
            '\u02b9': "'",  # MODIFIER LETTER PRIME ʹ
            '\u02ba': '"',  # MODIFIER LETTER DOUBLE PRIME ʺ
            '\u02bc': "'",  # MODIFIER LETTER APOSTROPHE ʼ
            '\u02c8': "'",  # MODIFIER LETTER VERTICAL LINE ˈ
            '\u02ee': '"',  # MODIFIER LETTER DOUBLE APOSTROPHE ˮ
            '\u0301': "",   # COMBINING ACUTE ACCENT (strip)
            '\u2039': "'",  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK ‹
            '\u203a': "'",  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK ›
            '\u2035': "'",  # REVERSED PRIME ‵
            '\u275b': "'",  # HEAVY SINGLE TURNED COMMA QUOTATION MARK ORNAMENT ❛
            '\u275c': "'",  # HEAVY SINGLE COMMA QUOTATION MARK ORNAMENT ❜
            '\u275f': "'",  # HEAVY LOW SINGLE COMMA QUOTATION MARK ORNAMENT ❟
            '\uff07': "'",  # FULLWIDTH APOSTROPHE ＇
            # Double quote replacements
            '\u2036': '"',  # REVERSED DOUBLE PRIME ‶
            '\u275d': '"',  # HEAVY DOUBLE TURNED COMMA QUOTATION MARK ORNAMENT ❝
            '\u275e': '"',  # HEAVY DOUBLE COMMA QUOTATION MARK ORNAMENT ❞
            '\u2760': '"',  # HEAVY LOW DOUBLE COMMA QUOTATION MARK ORNAMENT ❠
            '\u301d': '"',  # REVERSED DOUBLE PRIME QUOTATION MARK 〝
            '\u301e': '"',  # DOUBLE PRIME QUOTATION MARK 〞
            '\u301f': '"',  # LOW DOUBLE PRIME QUOTATION MARK 〟
        }
        for curly, straight in extra_replacements.items():
            text = text.replace(curly, straight)

        return text

    def parse(self, text: str) -> Expr:
        """Parse an expression string into an AST.

        Args:
            text: The expression string

        Returns:
            The parsed expression AST

        Raises:
            ParseError: If parsing fails
        """
        # Sanitize curly quotes to straight quotes
        text = self._sanitize_quotes(text)
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
                        elif token_type == "PATTERN":
                            # Extract the 3-char pattern from quotes
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
        """Parse: term -> factor (('*' | '%') factor)*"""
        left, pos = self._parse_factor(tokens, pos)

        while pos < len(tokens) and tokens[pos].type == "OP" and tokens[pos].value in ("*", "%"):
            op_token = tokens[pos]
            op = Operator.MUL if op_token.value == "*" else Operator.MOD
            pos += 1
            right, pos = self._parse_factor(tokens, pos)
            left = BinOp(left, op, right)

        return left, pos

    def _parse_factor(self, tokens: list[Token], pos: int) -> tuple[Expr, int]:
        """Parse: factor -> NUMBER | FUNC | GRID_LENGTH | '(' expr ')'"""
        if pos >= len(tokens):
            raise ParseError("Unexpected end of expression")

        token = tokens[pos]

        if token.type == "NUMBER":
            return Literal(token.value), pos + 1

        elif token.type == "GRID_LENGTH":
            return GridLength(), pos + 1

        elif token.type == "TRANSITION_INDICATOR":
            return TransitionIndicator(), pos + 1

        elif token.type == "FUNC":
            func_name = token.value
            pos += 1

            # Expect: func_name '(' ...
            if pos >= len(tokens) or tokens[pos].type != "LPAREN":
                raise ParseError(f"Expected '(' after '{func_name}' at position {token.pos}")
            pos += 1

            # Handle count_pattern specially (takes 3-char PATTERN, not SYMBOL)
            if func_name == "count_pattern":
                if pos >= len(tokens) or tokens[pos].type != "PATTERN":
                    raise ParseError(
                        f"count_pattern() requires a 3-character pattern like '>.<' or '.X.'. "
                        f"Got token type: {tokens[pos].type if pos < len(tokens) else 'EOF'}"
                    )
                pattern = tokens[pos].value
                pos += 1
                if pos >= len(tokens) or tokens[pos].type != "RPAREN":
                    raise ParseError(f"Expected ')' after pattern in count_pattern()")
                return CountPattern(pattern), pos + 1

            # All other functions expect a SYMBOL first
            if pos >= len(tokens) or tokens[pos].type != "SYMBOL":
                raise ParseError(f"Expected symbol in {func_name}() at position {token.pos}")
            symbol1 = tokens[pos].value
            if symbol1 not in VALID_SYMBOLS:
                raise ParseError(f"Invalid symbol '{symbol1}' in {func_name}()")
            pos += 1

            # Handle count_at_parity specially (second arg is integer, not symbol)
            if func_name == "count_at_parity":
                if pos >= len(tokens) or tokens[pos].type != "COMMA":
                    raise ParseError(f"count_at_parity() requires two arguments: symbol and parity (0 or 1)")
                pos += 1
                if pos >= len(tokens) or tokens[pos].type != "NUMBER":
                    raise ParseError(f"count_at_parity() requires integer parity (0 or 1) as second argument")
                parity = tokens[pos].value
                if parity not in (0, 1):
                    raise ParseError(f"count_at_parity() parity must be 0 (even) or 1 (odd), got {parity}")
                pos += 1
                if pos >= len(tokens) or tokens[pos].type != "RPAREN":
                    raise ParseError(f"Expected ')' after arguments")
                return CountAtParity(symbol1, parity), pos + 1

            # Check for second argument (for adjacent_pairs, gap_pairs)
            symbol2 = None
            gap_value = None
            if pos < len(tokens) and tokens[pos].type == "COMMA":
                pos += 1
                if pos >= len(tokens) or tokens[pos].type != "SYMBOL":
                    raise ParseError(f"Expected second symbol after comma in {func_name}()")
                symbol2 = tokens[pos].value
                if symbol2 not in VALID_SYMBOLS:
                    raise ParseError(f"Invalid symbol '{symbol2}' in {func_name}()")
                pos += 1

                # Check for third argument (integer gap for gap_pairs)
                if pos < len(tokens) and tokens[pos].type == "COMMA":
                    pos += 1
                    if pos >= len(tokens) or tokens[pos].type != "NUMBER":
                        raise ParseError(f"Expected integer gap after second comma in {func_name}()")
                    gap_value = tokens[pos].value
                    pos += 1

            if pos >= len(tokens) or tokens[pos].type != "RPAREN":
                raise ParseError(f"Expected ')' after symbol(s) at position {token.pos}")
            pos += 1

            # Create appropriate AST node
            if func_name == "count":
                return Count(symbol1), pos
            elif func_name == "leftmost":
                return Leftmost(symbol1), pos
            elif func_name == "rightmost":
                return Rightmost(symbol1), pos
            elif func_name == "max_gap":
                return MaxGap(symbol1), pos
            elif func_name == "spread":
                return Spread(symbol1), pos
            elif func_name == "count_even":
                return CountEven(symbol1), pos
            elif func_name == "count_odd":
                return CountOdd(symbol1), pos
            elif func_name == "adjacent_pairs":
                if symbol2 is None:
                    raise ParseError(f"adjacent_pairs() requires two symbols")
                return AdjacentPairs(symbol1, symbol2), pos
            elif func_name == "gap_pairs":
                if symbol2 is None:
                    raise ParseError(f"gap_pairs() requires two symbols")
                if gap_value is None:
                    raise ParseError(f"gap_pairs() requires a gap integer as third argument")
                return GapPairs(symbol1, symbol2, gap_value), pos
            else:
                raise ParseError(f"Unknown function '{func_name}'")

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
