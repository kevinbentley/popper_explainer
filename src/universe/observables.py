"""Observable computations for the Kinetic Grid Universe.

Observables are functions that extract numeric values from states.
These are the building blocks for law claims.

HARNESS SEMANTICS - CRITICAL INVARIANTS:
========================================
1. incoming_collisions MUST match the simulator's collision condition:
   - Cell j has incoming collision iff:
     state[(j-1) % L] in {'>', 'X'} AND state[(j+1) % L] in {'<', 'X'}
   - This is because BOTH '>' AND 'X' emit right-movers (to their right)
   - And BOTH '<' AND 'X' emit left-movers (to their left)
   - The bridge law CollisionCells(t+1) == IncomingCollisions(t) MUST hold

2. Observable definitions must be consistent with simulator.py:
   - '>' moves right: dest = (i + 1) % L
   - '<' moves left: dest = (i - 1) % L
   - 'X' emits both: right to (i+1)%L, left to (i-1)%L
   - Collision forms when both a right-mover AND left-mover arrive at same cell
"""

from src.universe.types import State, Symbol, VALID_SYMBOLS
from src.universe.validation import validate_state


def count_symbol(state: State, symbol: str) -> int:
    """Count occurrences of a symbol in the state.

    This is the primary primitive observable.

    Args:
        state: State string
        symbol: Symbol to count (one of . > < X)

    Returns:
        Count of the symbol

    Raises:
        ValueError: If symbol is not valid
    """
    if symbol not in VALID_SYMBOLS:
        raise ValueError(f"Invalid symbol '{symbol}'. Valid symbols: {sorted(VALID_SYMBOLS)}")

    return state.count(symbol)


def grid_length(state: State) -> int:
    """Return the length of the grid.

    Args:
        state: State string

    Returns:
        Length of the state string
    """
    return len(state)


# Derived observable helpers (these can be composed in expression language)

def count_right(state: State) -> int:
    """Count right-moving particles (>)."""
    return count_symbol(state, Symbol.RIGHT.value)


def count_left(state: State) -> int:
    """Count left-moving particles (<)."""
    return count_symbol(state, Symbol.LEFT.value)


def count_collision(state: State) -> int:
    """Count collision cells (X)."""
    return count_symbol(state, Symbol.COLLISION.value)


def count_empty(state: State) -> int:
    """Count empty cells (.)."""
    return count_symbol(state, Symbol.EMPTY.value)


def right_component(state: State) -> int:
    """Count right-moving components: count('>') + count('X').

    This is a conserved quantity.
    """
    return count_right(state) + count_collision(state)


def left_component(state: State) -> int:
    """Count left-moving components: count('<') + count('X').

    This is a conserved quantity.
    """
    return count_left(state) + count_collision(state)


def particle_count(state: State) -> int:
    """Total particle count: count('>') + count('<') + 2*count('X').

    This is a conserved quantity. Each X contains 2 particles (one > and one <).
    Also known as: TotalParticles
    """
    return count_right(state) + count_left(state) + 2 * count_collision(state)


def free_movers(state: State) -> int:
    """Count free (non-colliding) movers: count('>') + count('<').

    This is NOT conserved - it decreases when particles collide (form X)
    and increases when collisions resolve.

    Also known as: FreeMovers
    """
    return count_right(state) + count_left(state)


def momentum(state: State) -> int:
    """Net momentum: right_component - left_component.

    Simplifies to count('>') - count('<').
    This is a conserved quantity.
    """
    return count_right(state) - count_left(state)


def occupied_cells(state: State) -> int:
    """Count occupied cells: count('>') + count('<') + count('X').

    This is NOT conserved (changes during collisions).
    """
    return count_right(state) + count_left(state) + count_collision(state)


# --- Position-based observables (enable spatial reasoning) ---

def leftmost(state: State, symbol: str) -> int:
    """Find the position of the leftmost occurrence of a symbol.

    Args:
        state: State string
        symbol: Symbol to find

    Returns:
        0-indexed position, or -1 if not found
    """
    if symbol not in VALID_SYMBOLS:
        raise ValueError(f"Invalid symbol '{symbol}'")
    pos = state.find(symbol)
    return pos


def rightmost(state: State, symbol: str) -> int:
    """Find the position of the rightmost occurrence of a symbol.

    Args:
        state: State string
        symbol: Symbol to find

    Returns:
        0-indexed position, or -1 if not found
    """
    if symbol not in VALID_SYMBOLS:
        raise ValueError(f"Invalid symbol '{symbol}'")
    pos = state.rfind(symbol)
    return pos


def max_gap(state: State, symbol: str = ".") -> int:
    """Find the length of the longest contiguous run of a symbol.

    Args:
        state: State string
        symbol: Symbol to measure gaps of (default: empty cells)

    Returns:
        Length of longest contiguous run, or 0 if none
    """
    if symbol not in VALID_SYMBOLS:
        raise ValueError(f"Invalid symbol '{symbol}'")

    max_run = 0
    current_run = 0

    for c in state:
        if c == symbol:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return max_run


def adjacent_pairs(state: State, sym1: str, sym2: str) -> int:
    """Count adjacent pairs of symbols (sym1 immediately followed by sym2).

    Args:
        state: State string
        sym1: First symbol
        sym2: Second symbol

    Returns:
        Count of adjacent pairs
    """
    if sym1 not in VALID_SYMBOLS or sym2 not in VALID_SYMBOLS:
        raise ValueError(f"Invalid symbols")

    count = 0
    for i in range(len(state) - 1):
        if state[i] == sym1 and state[i + 1] == sym2:
            count += 1
    return count


def gap_pairs(state: State, sym1: str, sym2: str, gap: int) -> int:
    """Count pairs of symbols with exactly `gap` cells between them.

    This is essential for detecting converging particles:
    - gap_pairs('>', '<', 1) counts '>.<' patterns (will collide in 1 step)
    - gap_pairs('>', '<', 0) is equivalent to adjacent_pairs('>', '<')

    NOTE: This does NOT account for wrap-around. For collision detection
    with periodic boundaries, use incoming_collisions() instead.

    Args:
        state: State string
        sym1: First symbol
        sym2: Second symbol
        gap: Number of cells between sym1 and sym2 (distance = gap + 1)

    Returns:
        Count of pairs at the specified distance

    Raises:
        ValueError: If symbols are invalid or gap is negative
    """
    if sym1 not in VALID_SYMBOLS or sym2 not in VALID_SYMBOLS:
        raise ValueError(f"Invalid symbols")
    if gap < 0:
        raise ValueError(f"Gap must be non-negative, got {gap}")

    distance = gap + 1  # gap=0 means adjacent (distance 1)
    count = 0
    for i in range(len(state) - distance):
        if state[i] == sym1 and state[i + distance] == sym2:
            count += 1
    return count


def incoming_collisions(state: State) -> int:
    """Count cells that will have a collision at the next timestep.

    This is THE canonical observable for collision prediction. A cell j
    has an incoming collision if both:
    - state[(j-1) % L] in {'>', 'X'} (right-mover heading into j)
    - state[(j+1) % L] in {'<', 'X'} (left-mover heading into j)

    Note: X contributes BOTH a right-mover (to its right) AND a left-mover
    (to its left) when it resolves. So X acts as a source for both directions.

    This properly handles wrap-around with periodic boundaries.

    Key laws using this observable:
    - Bridge: CollisionCells(t+1) == IncomingCollisions(t)
    - Conservation: (IncomingCollisions(t)==0 AND CollisionCells(t)==0)
                    => FreeMovers(t+1) == FreeMovers(t)
    - Exchange: FreeMovers(t+1) - FreeMovers(t) == 2*(CollisionCells(t) - IncomingCollisions(t))

    Args:
        state: State string

    Returns:
        Count of cells that will become X at t+1
    """
    L = len(state)
    if L == 0:
        return 0

    # Sources of right-movers: > and X (both send right-mover to their right)
    # Sources of left-movers: < and X (both send left-mover to their left)
    RIGHT_SOURCES = {'>', 'X'}
    LEFT_SOURCES = {'<', 'X'}

    count = 0
    for j in range(L):
        left_neighbor = (j - 1) % L
        right_neighbor = (j + 1) % L
        # Cell j has incoming collision if right-mover from left AND left-mover from right
        has_right_mover = state[left_neighbor] in RIGHT_SOURCES
        has_left_mover = state[right_neighbor] in LEFT_SOURCES
        if has_right_mover and has_left_mover:
            count += 1
    return count


def spread(state: State, symbol: str) -> int:
    """Measure the spread of a symbol: rightmost - leftmost position.

    Returns 0 if symbol appears 0 or 1 times.

    Args:
        state: State string
        symbol: Symbol to measure

    Returns:
        Spread (rightmost - leftmost), or 0
    """
    left = leftmost(state, symbol)
    if left == -1:
        return 0
    right = rightmost(state, symbol)
    return right - left


# Registry of primitive observables for capability checking
PRIMITIVE_OBSERVABLES: dict[str, str] = {
    "count": "count(symbol) - count occurrences of a symbol",
    "grid_length": "grid_length - length of the state",
    "leftmost": "leftmost(symbol) - position of first occurrence (-1 if none)",
    "rightmost": "rightmost(symbol) - position of last occurrence (-1 if none)",
    "max_gap": "max_gap(symbol) - longest contiguous run of symbol",
    "adjacent_pairs": "adjacent_pairs(sym1, sym2) - count of sym1 followed by sym2",
    "gap_pairs": "gap_pairs(sym1, sym2, gap) - count of sym1 followed by sym2 with gap cells between",
    "incoming_collisions": "incoming_collisions - count of cells that will have collision at t+1",
    "spread": "spread(symbol) - rightmost - leftmost position",
}


def evaluate_observable(name: str, state: State, **kwargs) -> int:
    """Evaluate a named observable on a state.

    This is a dispatch function for the expression evaluator.

    Args:
        name: Observable name
        state: State to evaluate on
        **kwargs: Additional arguments (e.g., symbol for count)

    Returns:
        Observable value

    Raises:
        ValueError: If observable is unknown
    """
    if name == "count":
        symbol = kwargs.get("symbol")
        if symbol is None:
            raise ValueError("count() requires a 'symbol' argument")
        return count_symbol(state, symbol)

    elif name == "grid_length":
        return grid_length(state)

    else:
        raise ValueError(f"Unknown observable: {name}")


# =============================================================================
# CANONICAL OBSERVABLES REGISTRY
# =============================================================================
# This registry defines the standard observable names and their semantics.
# Using these canonical names ensures consistency and helps the linter
# catch semantic errors.
#
# CONSERVATION KEY:
#   [C] = Conserved (invariant under evolution)
#   [~] = Not conserved (changes during collisions)
# =============================================================================

from dataclasses import dataclass
from enum import Enum


class ConservationStatus(str, Enum):
    """Whether a quantity is conserved under time evolution."""
    CONSERVED = "conserved"
    NOT_CONSERVED = "not_conserved"
    CONDITIONAL = "conditional"  # Conserved under certain conditions


@dataclass
class CanonicalObservable:
    """A canonical observable with documented semantics."""
    name: str
    expression: str
    description: str
    conservation: ConservationStatus
    quantity_type: str  # from QuantityType enum
    notes: str = ""


CANONICAL_OBSERVABLES: dict[str, CanonicalObservable] = {
    # === CONSERVED QUANTITIES ===
    "TotalParticles": CanonicalObservable(
        name="TotalParticles",
        expression="count('>') + count('<') + 2*count('X')",
        description="Total number of particles (X contains 2 particles)",
        conservation=ConservationStatus.CONSERVED,
        quantity_type="particle_count",
        notes="Each X is a collision of one > and one <, so counts as 2",
    ),
    "RightComponent": CanonicalObservable(
        name="RightComponent",
        expression="count('>') + count('X')",
        description="Right-moving component count",
        conservation=ConservationStatus.CONSERVED,
        quantity_type="component_count",
        notes="The number of right-moving particles, including those in collision",
    ),
    "LeftComponent": CanonicalObservable(
        name="LeftComponent",
        expression="count('<') + count('X')",
        description="Left-moving component count",
        conservation=ConservationStatus.CONSERVED,
        quantity_type="component_count",
        notes="The number of left-moving particles, including those in collision",
    ),
    "Momentum": CanonicalObservable(
        name="Momentum",
        expression="count('>') - count('<')",
        description="Net momentum (right - left)",
        conservation=ConservationStatus.CONSERVED,
        quantity_type="momentum_like",
        notes="Equivalent to RightComponent - LeftComponent",
    ),
    "GridLength": CanonicalObservable(
        name="GridLength",
        expression="grid_length",
        description="Length of the state string",
        conservation=ConservationStatus.CONSERVED,
        quantity_type="length",
    ),

    # === NOT CONSERVED QUANTITIES ===
    "FreeMovers": CanonicalObservable(
        name="FreeMovers",
        expression="count('>') + count('<')",
        description="Free (non-colliding) movers",
        conservation=ConservationStatus.NOT_CONSERVED,
        quantity_type="cell_count",
        notes="Decreases when particles collide, increases when collisions resolve",
    ),
    "OccupiedCells": CanonicalObservable(
        name="OccupiedCells",
        expression="count('>') + count('<') + count('X')",
        description="Number of non-empty cells",
        conservation=ConservationStatus.NOT_CONSERVED,
        quantity_type="cell_count",
        notes="Changes during collisions: 2 cells (><) become 1 cell (X)",
    ),
    "EmptyCells": CanonicalObservable(
        name="EmptyCells",
        expression="count('.')",
        description="Number of empty cells",
        conservation=ConservationStatus.NOT_CONSERVED,
        quantity_type="cell_count",
        notes="Changes during collisions",
    ),
    "CollisionCells": CanonicalObservable(
        name="CollisionCells",
        expression="count('X')",
        description="Number of cells with active collisions",
        conservation=ConservationStatus.NOT_CONSERVED,
        quantity_type="cell_count",
        notes="X cells resolve in the next time step",
    ),

    # === CONDITIONALLY CONSERVED ===
    "RightMovers": CanonicalObservable(
        name="RightMovers",
        expression="count('>')",
        description="Count of free right-moving particles",
        conservation=ConservationStatus.CONDITIONAL,
        quantity_type="component_count",
        notes="Conserved only if no collisions occur",
    ),
    "LeftMovers": CanonicalObservable(
        name="LeftMovers",
        expression="count('<')",
        description="Count of free left-moving particles",
        conservation=ConservationStatus.CONDITIONAL,
        quantity_type="component_count",
        notes="Conserved only if no collisions occur",
    ),

    # === COLLISION PREDICTION OBSERVABLES ===
    "IncomingCollisions": CanonicalObservable(
        name="IncomingCollisions",
        expression="incoming_collisions",
        description="Count of cells that will have collision at t+1",
        conservation=ConservationStatus.NOT_CONSERVED,
        quantity_type="collision_count",
        notes="THE canonical collision predictor. Cell j collides if "
              "state[(j-1)%L] in {>,X} AND state[(j+1)%L] in {<,X}. "
              "X contributes both a right-mover and left-mover when resolving.",
    ),
    "AdjacentRL": CanonicalObservable(
        name="AdjacentRL",
        expression="adjacent_pairs('>', '<')",
        description="Count of >< patterns",
        conservation=ConservationStatus.NOT_CONSERVED,
        quantity_type="pair_count",
        notes="Counts adjacent >< pairs. Note: >< does NOT cause collision "
              "(particles pass through). Use IncomingCollisions for collision prediction.",
    ),
}


# =============================================================================
# CANONICAL LAWS (derived from dynamics)
# =============================================================================
# These are the key laws that should be verified by the harness.
# They follow directly from the simulator's update rules.
#
# BRIDGE LAW (collision prediction):
#   CollisionCells(t+1) == IncomingCollisions(t)
#
# CONSERVATION (no activity):
#   (IncomingCollisions(t) == 0 AND CollisionCells(t) == 0)
#     => FreeMovers(t+1) == FreeMovers(t)
#   (IncomingCollisions(t) == 0 AND CollisionCells(t) == 0)
#     => RightMovers(t+1) == RightMovers(t)
#   (IncomingCollisions(t) == 0 AND CollisionCells(t) == 0)
#     => LeftMovers(t+1) == LeftMovers(t)
#
# EXCHANGE RATE LAW (always true):
#   FreeMovers(t+1) - FreeMovers(t) == 2*(CollisionCells(t) - IncomingCollisions(t))
#
# INVARIANTS (always true):
#   TotalParticles(t) == TotalParticles(0)           # conservation
#   RightComponent(t) == RightComponent(0)           # conservation
#   LeftComponent(t) == LeftComponent(0)             # conservation
#   Momentum(t) == Momentum(0)                       # conservation
#   TotalParticles(t) == FreeMovers(t) + 2*CollisionCells(t)  # identity
# =============================================================================


# Mapping from common "Movers" expressions to canonical names
# Use this to suggest fixes when linting
EXPRESSION_TO_CANONICAL: dict[str, str] = {
    "count('>') + count('<')": "FreeMovers",
    "count('>') + count('<') + count('X')": "OccupiedCells",
    "count('>') + count('<') + 2*count('X')": "TotalParticles",
    "count('>') + count('X')": "RightComponent",
    "count('<') + count('X')": "LeftComponent",
    "count('>') - count('<')": "Momentum",
    "count('.')": "EmptyCells",
    "count('X')": "CollisionCells",
    "count('>')": "RightMovers",
    "count('<')": "LeftMovers",
    "grid_length": "GridLength",
}


def get_canonical_name(expression: str) -> str | None:
    """Get the canonical name for an expression, if one exists.

    Args:
        expression: The expression string

    Returns:
        Canonical name or None if not a standard expression
    """
    # Normalize expression: remove all spaces
    normalized = expression.replace(" ", "")

    # Try direct lookup
    if normalized in EXPRESSION_TO_CANONICAL:
        return EXPRESSION_TO_CANONICAL[normalized]

    # Try with normalized keys (remove spaces from canonical expressions too)
    for key, name in EXPRESSION_TO_CANONICAL.items():
        if key.replace(" ", "") == normalized:
            return name

    return None


def suggest_canonical_observable(observable_name: str, expression: str) -> str | None:
    """Suggest a canonical observable name if the expression matches a known pattern.

    Args:
        observable_name: Current name given to the observable
        expression: The expression defining the observable

    Returns:
        Suggested canonical name, or None if no suggestion
    """
    canonical = get_canonical_name(expression)
    if canonical and canonical.lower() != observable_name.lower():
        return canonical
    return None


# =============================================================================
# PHASE-D: Observable Glossary Generation
# =============================================================================


def generate_glossary_entries() -> list[dict]:
    """Extract glossary entries from CANONICAL_OBSERVABLES.

    Returns:
        List of glossary entry dicts with keys:
        - name: Observable name
        - expression: Definition expression
        - description: Human-readable description
        - conservation: Conservation status string
        - notes: Additional notes
    """
    entries = []
    for name, obs in CANONICAL_OBSERVABLES.items():
        entries.append({
            "name": name,
            "expression": obs.expression,
            "description": obs.description,
            "conservation": obs.conservation.value,
            "notes": obs.notes,
        })
    return entries


def format_glossary_block() -> str:
    """Format glossary as markdown for prompt injection.

    Returns:
        Formatted markdown string with observable glossary
    """
    lines = [
        "### OBSERVABLE GLOSSARY",
        "",
        "The following observables are available. Use canonical names in theorems.",
        "",
        "| Name | Definition | Status |",
        "|------|------------|--------|",
    ]

    # Sort by conservation status (conserved first), then by name
    def sort_key(item: tuple) -> tuple:
        name, obs = item
        status_order = {
            ConservationStatus.CONSERVED: 0,
            ConservationStatus.CONDITIONAL: 1,
            ConservationStatus.NOT_CONSERVED: 2,
        }
        return (status_order.get(obs.conservation, 3), name)

    sorted_observables = sorted(CANONICAL_OBSERVABLES.items(), key=sort_key)

    for name, obs in sorted_observables:
        # Format conservation status as compact indicator
        status_map = {
            ConservationStatus.CONSERVED: "[C] Conserved",
            ConservationStatus.CONDITIONAL: "[?] Conditional",
            ConservationStatus.NOT_CONSERVED: "[~] Not conserved",
        }
        status_str = status_map.get(obs.conservation, "[?]")

        # Escape pipes in expression
        expr = obs.expression.replace("|", "\\|")

        lines.append(f"| {name} | `{expr}` | {status_str} |")

    lines.append("")
    return "\n".join(lines)
