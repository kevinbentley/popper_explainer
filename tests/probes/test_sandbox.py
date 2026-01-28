"""Tests for the probe sandbox: validation, execution, security, timeout."""

import pytest

from src.probes.sandbox import (
    ProbeError,
    ProbeReturnTypeError,
    ProbeRuntimeError,
    ProbeSyntaxError,
    ProbeTimeoutError,
    ProbeValidationError,
    detect_probe_arity,
    execute_probe,
    validate_probe_source,
)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidateProbeSource:

    def test_valid_simple_count(self):
        source = "def probe(S):\n    return sum(1 for c in S if c == '>')"
        valid, error = validate_probe_source(source)
        assert valid is True
        assert error is None

    def test_valid_complex_probe(self):
        source = (
            "def probe(S):\n"
            "    pairs = 0\n"
            "    for i in range(len(S) - 1):\n"
            "        if S[i] == '>' and S[i+1] == '<':\n"
            "            pairs += 1\n"
            "    return pairs\n"
        )
        valid, error = validate_probe_source(source)
        assert valid is True

    def test_missing_probe_function(self):
        source = "def foo(S):\n    return 1"
        valid, error = validate_probe_source(source)
        assert valid is False
        assert "probe" in error

    def test_wrong_parameter_count_zero(self):
        source = "def probe():\n    return 1"
        valid, error = validate_probe_source(source)
        assert valid is False
        assert "1 or 2 parameters" in error

    def test_wrong_parameter_count_three(self):
        source = "def probe(S, T, U):\n    return 1"
        valid, error = validate_probe_source(source)
        assert valid is False
        assert "1 or 2 parameters" in error

    def test_two_param_probe_valid(self):
        source = "def probe(S_cur, S_nxt):\n    return sum(a != b for a, b in zip(S_cur, S_nxt))"
        valid, error = validate_probe_source(source)
        assert valid is True
        assert error is None

    def test_syntax_error(self):
        source = "def probe(S:\n    return 1"
        valid, error = validate_probe_source(source)
        assert valid is False
        assert "Syntax error" in error

    def test_import_rejected(self):
        source = "import os\ndef probe(S):\n    return 1"
        valid, error = validate_probe_source(source)
        assert valid is False
        assert "import" in error.lower()

    def test_from_import_rejected(self):
        source = "from os import path\ndef probe(S):\n    return 1"
        valid, error = validate_probe_source(source)
        assert valid is False
        assert "import" in error.lower()

    def test_exec_rejected(self):
        source = "def probe(S):\n    exec('x=1')\n    return 1"
        valid, error = validate_probe_source(source)
        assert valid is False
        assert "exec" in error

    def test_eval_rejected(self):
        source = "def probe(S):\n    return eval('1+1')"
        valid, error = validate_probe_source(source)
        assert valid is False
        assert "eval" in error

    def test_open_rejected(self):
        source = "def probe(S):\n    open('/etc/passwd')\n    return 1"
        valid, error = validate_probe_source(source)
        assert valid is False
        assert "open" in error

    def test_dunder_attribute_rejected(self):
        source = "def probe(S):\n    return S.__class__.__name__"
        valid, error = validate_probe_source(source)
        assert valid is False
        assert "__" in error

    def test_global_statement_rejected(self):
        source = "x = 1\ndef probe(S):\n    global x\n    return x"
        valid, error = validate_probe_source(source)
        assert valid is False
        assert "global" in error

    def test_compile_rejected(self):
        source = "def probe(S):\n    compile('x=1', '', 'exec')\n    return 1"
        valid, error = validate_probe_source(source)
        assert valid is False
        assert "compile" in error


# ---------------------------------------------------------------------------
# Execution tests
# ---------------------------------------------------------------------------

class TestExecuteProbe:

    def test_simple_count(self):
        source = "def probe(S):\n    return sum(1 for c in S if c == '>')"
        result = execute_probe(source, list("..><.."))
        assert result == 1

    def test_total_particles(self):
        source = "def probe(S):\n    return sum(1 for c in S if c != '.')"
        result = execute_probe(source, list("..><.X.."))
        assert result == 3

    def test_grid_length(self):
        source = "def probe(S):\n    return len(S)"
        result = execute_probe(source, list("..><.."))
        assert result == 6

    def test_even_index_count(self):
        source = (
            "def probe(S):\n"
            "    return sum(1 for i, c in enumerate(S) if c == '>' and i % 2 == 0)"
        )
        result = execute_probe(source, list(">.<.>."))
        # '>' at indices 0, 4 (both even) -> 2
        assert result == 2

    def test_returns_float(self):
        source = "def probe(S):\n    return len(S) / 2.0"
        result = execute_probe(source, list("..><.."))
        assert result == 3.0
        assert isinstance(result, float)

    def test_returns_int(self):
        source = "def probe(S):\n    return len(S)"
        result = execute_probe(source, list("..><.."))
        assert isinstance(result, int)

    def test_empty_state(self):
        source = "def probe(S):\n    return len(S)"
        result = execute_probe(source, [])
        assert result == 0

    def test_non_numeric_return_raises(self):
        source = "def probe(S):\n    return 'hello'"
        with pytest.raises(ProbeReturnTypeError):
            execute_probe(source, list(".."))

    def test_list_return_raises(self):
        source = "def probe(S):\n    return [1, 2, 3]"
        with pytest.raises(ProbeReturnTypeError):
            execute_probe(source, list(".."))

    def test_runtime_error_raises(self):
        source = "def probe(S):\n    return 1 / 0"
        with pytest.raises(ProbeRuntimeError):
            execute_probe(source, list(".."))

    def test_index_error_raises(self):
        source = "def probe(S):\n    return S[999]"
        with pytest.raises(ProbeRuntimeError):
            execute_probe(source, list(".."))

    def test_timeout_on_infinite_loop(self):
        source = "def probe(S):\n    while True:\n        pass\n    return 0"
        with pytest.raises(ProbeTimeoutError):
            execute_probe(source, list(".."), timeout_ms=50)

    def test_validation_failure_raises(self):
        source = "import os\ndef probe(S):\n    return 1"
        with pytest.raises(ProbeValidationError):
            execute_probe(source, list(".."))

    def test_syntax_error_raises(self):
        source = "def probe(S:\n    return 1"
        with pytest.raises(ProbeSyntaxError):
            execute_probe(source, list(".."))

    def test_builtins_restricted(self):
        """Verify that non-whitelisted builtins are not available."""
        source = "def probe(S):\n    print('hello')\n    return 1"
        # 'print' is in _DISALLOWED_NAMES, so validation fails
        with pytest.raises(ProbeValidationError):
            execute_probe(source, list(".."))

    def test_whitelisted_builtins_available(self):
        """Verify that whitelisted builtins work."""
        source = (
            "def probe(S):\n"
            "    return max(0, min(len(S), abs(-5)))"
        )
        result = execute_probe(source, list("..><.."))
        assert result == 5  # min(6, 5) = 5

    def test_sorted_available(self):
        source = "def probe(S):\n    return len(sorted(S))"
        result = execute_probe(source, list("..><.."))
        assert result == 6

    def test_enumerate_available(self):
        source = "def probe(S):\n    return sum(i for i, c in enumerate(S) if c == '>')"
        result = execute_probe(source, list("..>..."))
        assert result == 2


# ---------------------------------------------------------------------------
# Temporal (2-param) probe tests
# ---------------------------------------------------------------------------

class TestTemporalProbes:

    def test_execute_two_param_probe(self):
        source = "def probe(S_cur, S_nxt):\n    return sum(a != b for a, b in zip(S_cur, S_nxt))"
        result = execute_probe(
            source,
            list("..><.."),
            next_state=list("..X..."),
        )
        # Differences: position 2 (> vs X), position 3 (< vs .) = 2
        assert result == 2

    def test_two_param_without_next_state_raises(self):
        source = "def probe(S_cur, S_nxt):\n    return 0"
        with pytest.raises(ProbeRuntimeError, match="next_state"):
            execute_probe(source, list("..><.."))

    def test_detect_arity_one(self):
        source = "def probe(S):\n    return len(S)"
        assert detect_probe_arity(source) == 1

    def test_detect_arity_two(self):
        source = "def probe(S_cur, S_nxt):\n    return 0"
        assert detect_probe_arity(source) == 2

    def test_one_param_probe_ignores_next_state(self):
        """A 1-param probe should work even if next_state is passed (it's ignored)."""
        source = "def probe(S):\n    return len(S)"
        # next_state is provided but arity=1, so it should not be passed
        result = execute_probe(source, list("..><.."), next_state=list("......"))
        assert result == 6

    def test_temporal_probe_counts_changed_cells(self):
        """Temporal probe that counts cells that changed."""
        source = (
            "def probe(S_cur, S_nxt):\n"
            "    return sum(1 for a, b in zip(S_cur, S_nxt) if a != b)"
        )
        state1 = list("..><..")
        state2 = list("..><..")  # identical
        assert execute_probe(source, state1, next_state=state2) == 0

        state3 = list("...X..")  # 2 changes from state1
        assert execute_probe(source, state1, next_state=state3) == 2
