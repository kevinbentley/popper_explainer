"""Main agent loop for AHC-DS."""

import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from src.ahc.agent.journal import JournalManager
from src.ahc.agent.termination import TerminationChecker, TerminationStatus
from src.ahc.agent.context import ContextBuilder, ContextConfig, estimate_tokens
from src.ahc.agent.compaction import CompactionManager, CompactionConfig
from src.ahc.db.models import (
    SessionRecord,
    SessionStatus,
    ConversationTurnRecord,
)
from src.ahc.db.repo import AHCRepository
from src.ahc.tools.registry import ToolRegistry, create_popperian_registry
from src.proposer.scrambler import SymbolScrambler, get_default_scrambler

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the AHC agent.

    Attributes:
        model: LLM model identifier
        max_turns: Maximum number of turns before stopping
        db_path: Path to the SQLite database
        seed: Random seed for reproducibility
        accuracy_target: Target prediction accuracy for termination
        predictions_target: Number of predictions required for termination
        rolling_window_size: Number of recent turns in context (Tier 3)
        compaction_token_threshold: Trigger compaction at this token count
        compaction_turn_threshold: Or after this many turns
        enable_compaction: Whether to enable automatic compaction
    """
    model: str = "gemini-2.5-flash"
    max_turns: int = 10000
    db_path: str = "high_context.db"
    seed: int = 42
    accuracy_target: float = 1.0
    predictions_target: int = 5000
    # Context management
    rolling_window_size: int = 15
    compaction_token_threshold: int = 50000
    compaction_turn_threshold: int = 50
    enable_compaction: bool = True

    def to_json(self) -> str:
        """Serialize config to JSON."""
        return json.dumps({
            "model": self.model,
            "max_turns": self.max_turns,
            "db_path": self.db_path,
            "seed": self.seed,
            "accuracy_target": self.accuracy_target,
            "predictions_target": self.predictions_target,
            "rolling_window_size": self.rolling_window_size,
            "compaction_token_threshold": self.compaction_token_threshold,
            "compaction_turn_threshold": self.compaction_turn_threshold,
            "enable_compaction": self.enable_compaction,
        })


class AgentLoop:
    """Main agent loop implementing the Think-Act-Learn cycle.

    The agent maintains a persistent conversation with the LLM,
    using tools to interact with the physics simulator and
    building understanding over time.
    """

    _NON_LOCAL_TEMPLATES = {
        "invariant", "monotone", "implication_step", "implication_state",
        "eventually", "symmetry_commutation", "bound",
    }

    def __init__(
        self,
        config: AgentConfig | None = None,
        llm_client: Any = None,
        show_conversation: bool = False,
    ):
        """Initialize the agent loop.

        Args:
            config: Agent configuration
            llm_client: LLM client for generation (if None, uses default)
            show_conversation: If True, print LLM interactions to console
        """
        self.config = config or AgentConfig()
        self._llm_client = llm_client
        self._show_conversation = show_conversation
        self._repo: AHCRepository | None = None
        self._session: SessionRecord | None = None
        self._tools: ToolRegistry | None = None
        self._journal: JournalManager | None = None
        self._termination: TerminationChecker | None = None
        self._context_builder: ContextBuilder | None = None
        self._compaction_manager: CompactionManager | None = None
        self._turn_number = 0
        self._conversation_history: list[dict[str, Any]] = []
        self._scrambler = get_default_scrambler()
        self._consecutive_no_tool_calls = 0
        self._consecutive_no_content = 0
        self._just_compacted = False
        self._recent_templates: list[str] = []
        self._last_non_local_turn: int = 0

    def run(self) -> str:
        """Start a new session and run until termination.

        Returns:
            Session ID of the completed session
        """
        # Initialize
        self._init_session()
        session_id = self._session.session_id

        logger.info(f"Starting new AHC session: {session_id}")

        try:
            # Run the main loop
            self._run_loop()
        except Exception as e:
            logger.exception(f"Session failed: {e}")
            self._session.status = SessionStatus.FAILED
            self._session.termination_reason = str(e)
            self._repo.update_session(self._session)
        finally:
            self._repo.close()

        return session_id

    def resume(self, session_id: str) -> str:
        """Resume an existing session.

        Args:
            session_id: Session ID to resume

        Returns:
            Session ID of the resumed session
        """
        # Connect to database
        self._repo = AHCRepository(self.config.db_path)
        self._repo.connect()

        # Load session
        self._session = self._repo.get_session(session_id)
        if self._session is None:
            raise ValueError(f"Session not found: {session_id}")

        if self._session.status == SessionStatus.COMPLETED:
            logger.warning(f"Session {session_id} is already completed")
            return session_id

        # Restore state
        self._session.status = SessionStatus.RUNNING
        self._repo.update_session(self._session)

        # Initialize components
        self._tools = create_popperian_registry(self._repo)
        self._journal = JournalManager(self._repo, self._session.id)
        self._termination = TerminationChecker(
            self._repo,
            self._session.id,
            self.config.accuracy_target,
            self.config.predictions_target,
        )

        # Initialize context management
        context_config = ContextConfig(
            rolling_window_size=self.config.rolling_window_size,
        )
        self._context_builder = ContextBuilder(
            self._repo,
            self._session.id,
            context_config,
        )

        if self.config.enable_compaction:
            compaction_config = CompactionConfig(
                token_threshold=self.config.compaction_token_threshold,
                turn_threshold=self.config.compaction_turn_threshold,
            )
            self._compaction_manager = CompactionManager(
                self._repo,
                self._llm_client,
                self._session.id,
                compaction_config,
            )

        # Restore conversation history
        self._restore_conversation()

        logger.info(f"Resuming AHC session: {session_id} from turn {self._turn_number}")

        try:
            self._run_loop()
        except Exception as e:
            logger.exception(f"Session failed: {e}")
            self._session.status = SessionStatus.FAILED
            self._session.termination_reason = str(e)
            self._repo.update_session(self._session)
        finally:
            self._repo.close()

        return session_id

    def _init_session(self) -> None:
        """Initialize a new session."""
        # Create database connection
        self._repo = AHCRepository(self.config.db_path)
        self._repo.connect()

        # Create session record
        session_id = str(uuid.uuid4())[:8]
        self._session = SessionRecord(
            session_id=session_id,
            status=SessionStatus.RUNNING,
            config_json=self.config.to_json(),
            model_id=self.config.model,
            seed=self.config.seed,
        )
        self._repo.insert_session(self._session)

        # Initialize components
        self._tools = create_popperian_registry(self._repo)
        self._journal = JournalManager(self._repo, self._session.id)
        self._termination = TerminationChecker(
            self._repo,
            self._session.id,
            self.config.accuracy_target,
            self.config.predictions_target,
        )

        # Initialize context management
        context_config = ContextConfig(
            rolling_window_size=self.config.rolling_window_size,
        )
        self._context_builder = ContextBuilder(
            self._repo,
            self._session.id,
            context_config,
        )

        if self.config.enable_compaction:
            compaction_config = CompactionConfig(
                token_threshold=self.config.compaction_token_threshold,
                turn_threshold=self.config.compaction_turn_threshold,
            )
            self._compaction_manager = CompactionManager(
                self._repo,
                self._llm_client,
                self._session.id,
                compaction_config,
            )

        # Set tool contexts
        self._set_tool_contexts()

        # Initialize conversation with system prompt
        self._init_conversation()

    def _set_tool_contexts(self) -> None:
        """Set session context on all tools."""
        for tool_name in self._tools.list_tools():
            tool = self._tools.get_tool(tool_name)
            if hasattr(tool, "set_context"):
                tool.set_context(self._session.id)

    def _update_tool_turns(self) -> None:
        """Update turn number on tools that track it."""
        for tool_name in self._tools.list_tools():
            tool = self._tools.get_tool(tool_name)
            if hasattr(tool, "set_turn"):
                tool.set_turn(self._turn_number)

    def _init_conversation(self) -> None:
        """Initialize the conversation with system prompt."""
        system_prompt = self._build_system_prompt()

        self._conversation_history = [
            {"role": "system", "content": system_prompt},
        ]

        # Set system prompt on context builder for Tier 1
        if self._context_builder:
            self._context_builder.set_system_prompt(system_prompt)

        # Print and save system message
        self._print_conversation("system", system_prompt)
        self._save_turn("system", system_prompt)

        # Add initial user prompt
        initial_prompt = self._build_initial_prompt()
        self._conversation_history.append({"role": "user", "content": initial_prompt})
        self._print_conversation("user", initial_prompt)
        self._save_turn("user", initial_prompt)

    def _restore_conversation(self) -> None:
        """Restore conversation history from database."""
        turns = self._repo.get_conversation_history(self._session.id)

        self._conversation_history = []
        for turn in turns:
            self._conversation_history.append({
                "role": turn.role,
                "content": turn.content,
            })

            # Set system prompt on context builder if this is the system turn
            if turn.role == "system" and self._context_builder:
                self._context_builder.set_system_prompt(turn.content)

            if turn.tool_calls_json:
                # Restore tool calls if present
                pass

        # Set turn number to continue from
        self._turn_number = self._repo.get_last_turn_number(self._session.id)

    def _run_loop(self) -> None:
        """Run the main think-act-learn loop."""
        while self._turn_number < self.config.max_turns:
            self._turn_number += 1
            self._journal.set_turn(self._turn_number)
            self._update_tool_turns()

            logger.debug(f"Turn {self._turn_number}")

            # Check termination
            status = self._termination.check()
            if status.terminated:
                self._complete_session(status)
                return

            # Get LLM response
            response = self._get_llm_response()
            if response is None:
                logger.error("Failed to get LLM response")
                continue

            # Process response (may include tool calls)
            self._process_response(response)

            # Update session stats
            self._update_session_stats()

        # Max turns reached
        logger.info(f"Max turns ({self.config.max_turns}) reached")
        self._session.status = SessionStatus.COMPLETED
        self._session.termination_reason = "Max turns reached"
        self._session.terminated_at = datetime.now()
        self._repo.update_session(self._session)

    def _get_llm_response(self) -> dict[str, Any] | None:
        """Get a response from the LLM.

        Returns:
            Response dict with 'content' and optionally 'tool_calls'
        """
        if self._llm_client is None:
            logger.warning("No LLM client configured")
            return {"content": "No LLM client configured. Please provide one."}

        try:
            # Check if compaction is needed
            just_compacted = False
            if self._compaction_manager and self._compaction_manager.should_compact():
                result = self._compaction_manager.compact()
                if result.success:
                    just_compacted = True
                    logger.info(
                        f"Compaction complete: {result.turns_compacted} turns, "
                        f"{result.tokens_before} -> {result.tokens_after} tokens"
                    )

            # Build messages for API call
            # Use context builder if available (three-tier memory)
            if self._context_builder:
                messages = self._context_builder.build_messages()
                stats = self._context_builder.get_context_stats()
                logger.debug(
                    f"Context: {stats['message_count']} messages, "
                    f"~{stats['estimated_tokens']} tokens"
                )
            else:
                # Fallback to raw history
                messages = self._conversation_history.copy()

            # Post-compaction reorientation: inject a bridging message so the
            # LLM doesn't lose its lab-note methodology after the Tier 2 data dump
            if just_compacted:
                self._just_compacted = True
                reorientation = (
                    "[COMPACTION COMPLETE — Your knowledge state has been updated above.\n\n"
                    "STRATEGIC DIRECTIVE: You have a LIMITED BUDGET of tool calls. "
                    "You CANNOT afford to exhaustively enumerate every neighborhood combination. "
                    "Brute-force enumeration is not science — it is the opposite of Popper.\n\n"
                    "A revolutionary scientist proposes BOLD, RISKY conjectures that cover wide territory. "
                    "PRIORITY for your next experiments:\n"
                    "1. invariant — test if quantities are conserved (applies to ALL cells, ALL timesteps)\n"
                    "2. symmetry_commutation — test if transforms commute with evolution (applies to ALL states)\n"
                    "3. implication_step — test conditional relationships between observables\n"
                    "4. bound — test upper/lower limits on counts\n"
                    "These macro-level laws teach you MORE per tool call than any single local_transition test.\n\n"
                    "Review the COUNTEREXAMPLE GALLERY and negative knowledge above, then write your "
                    "RESULT + LAB NOTE before your next tool call.]"
                )
                # Insert as the last user message before generating
                messages.append({"role": "user", "content": reorientation})

            # Get tool schemas
            tool_schemas = self._tools.get_all_schemas()

            # Call LLM
            response = self._llm_client.generate_with_tools(
                messages=messages,
                tools=tool_schemas,
            )

            return response

        except Exception as e:
            logger.exception(f"LLM call failed: {e}")
            return None

    def _print_conversation(self, role: str, content: str, tool_calls: list | None = None) -> None:
        """Print conversation to console if enabled.

        Args:
            role: Message role (assistant, user, system)
            content: Message content
            tool_calls: Optional list of tool calls
        """
        if not self._show_conversation:
            return

        # Color codes for terminal
        COLORS = {
            "assistant": "\033[94m",  # Blue
            "user": "\033[92m",       # Green
            "system": "\033[93m",     # Yellow
            "tool": "\033[95m",       # Magenta
            "reset": "\033[0m",
            "bold": "\033[1m",
            "dim": "\033[2m",
        }

        color = COLORS.get(role, "")
        reset = COLORS["reset"]
        bold = COLORS["bold"]
        dim = COLORS["dim"]

        print(f"\n{bold}{'='*60}{reset}")
        print(f"{bold}[Turn {self._turn_number}] {role.upper()}{reset}")
        print(f"{'='*60}")

        if content:
            print(f"{color}{content}{reset}")
        elif role == "assistant" and not tool_calls:
            print(f"{dim}[No content or tool calls]{reset}")

        if tool_calls:
            print(f"\n{COLORS['tool']}{bold}Tool Calls:{reset}")
            for tc in tool_calls:
                name = tc.get("name", "unknown")
                args = tc.get("arguments", {})
                args_str = json.dumps(args, indent=2)
                if len(args_str) > 500:
                    args_str = args_str[:500] + "..."
                print(f"{COLORS['tool']}  -> {name}({args_str}){reset}")

        sys.stdout.flush()

    def _process_response(self, response: dict[str, Any]) -> None:
        """Process an LLM response.

        Args:
            response: Response dict with 'content' and optionally 'tool_calls'
        """
        content = response.get("content", "")
        tool_calls = response.get("tool_calls", [])

        # Print to console if enabled
        self._print_conversation("assistant", content, tool_calls)

        # Save assistant response
        self._conversation_history.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })
        self._save_turn("assistant", content, tool_calls)

        # Log any thoughts
        if content:
            self._journal.log_thought(content)

        # Execute tool calls
        if tool_calls:
            self._consecutive_no_tool_calls = 0  # Reset counter on successful tool use
            tool_results = self._execute_tool_calls(tool_calls)

            # Add tool results to conversation
            results_content = self._format_tool_results(tool_results)

            # Detect missing lab notes (tool calls without analysis text)
            if not content.strip():
                self._consecutive_no_content += 1
                if self._consecutive_no_content >= 3:
                    reminder = (
                        "MANDATORY REQUIREMENT VIOLATION: You have made tool calls "
                        f"without writing LAB NOTES for {self._consecutive_no_content} "
                        "consecutive turns. Your system instructions REQUIRE written "
                        "analysis before EVERY tool call.\n\n"
                        "You MUST write:\n"
                        "RESULT: [What you learned from the results below]\n"
                        "LAB NOTE: [Your next hypothesis and why you're testing it]\n\n"
                        "Then make your tool call. Do NOT skip the written analysis.\n\n"
                    )
                elif self._consecutive_no_content >= 1:
                    reminder = (
                        "NOTE: You made a tool call without writing a LAB NOTE. "
                        "Write RESULT + LAB NOTE analysis before your next tool call.\n\n"
                    )
                results_content = reminder + results_content
            else:
                self._consecutive_no_content = 0

            # Check template diversity and append nudge if needed
            diversity_nudge = self._check_diversity_quota()
            if diversity_nudge:
                results_content = results_content + "\n\n" + diversity_nudge

            # Print tool results if enabled
            self._print_conversation("user", results_content)

            self._conversation_history.append({
                "role": "user",
                "content": results_content,
            })
            self._save_turn("user", results_content)
        else:
            # No tool calls - provide escalating nudges
            self._consecutive_no_tool_calls += 1
            nudge = self._get_escalating_nudge()
            self._print_conversation("user", nudge)
            self._conversation_history.append({
                "role": "user",
                "content": nudge,
            })
            self._save_turn("user", nudge)

    def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Execute a list of tool calls.

        Args:
            tool_calls: List of tool call specifications

        Returns:
            List of tool results
        """
        results = []

        for call in tool_calls:
            tool_name = call.get("name", "")
            arguments = call.get("arguments", {})

            # Unscramble arguments (convert LLM's abstract symbols to physical)
            unscrambled_args = self._unscramble_data(arguments)

            # Execute the tool with physical symbols
            result = self._tools.execute(
                tool_name=tool_name,
                session_id=self._session.id,
                turn_number=self._turn_number,
                **unscrambled_args,
            )

            results.append({
                "tool_name": tool_name,
                "result": result.to_dict(),
            })

            # Track template diversity for evaluate_laws calls
            if tool_name == "evaluate_laws":
                self._track_template_diversity(arguments)

            # Log experiment if it's an experimental tool
            if tool_name in ["evaluate_laws", "run_prediction", "request_samples"]:
                self._journal.log_experiment(
                    f"Called {tool_name}: {json.dumps(arguments)[:200]}",
                    metadata={"result_success": result.success},
                )

        return results

    def _scramble_data(self, data: Any, key: str | None = None) -> Any:
        """Recursively scramble physical symbols to abstract in data.

        Only scrambles symbol-containing fields, not structural fields like 'template'.

        Args:
            data: Data structure (dict, list, str, etc.)
            key: The dict key this data came from (to determine if it's a symbol field)

        Returns:
            Data with physical symbols replaced by abstract symbols in appropriate fields
        """
        # Fields that contain universe symbols and should be scrambled
        SYMBOL_FIELDS = {
            "trigger_symbol", "result_symbol", "neighbor_pattern",
            "left_neighbor", "right_neighbor",
            "initial_state", "state", "trajectory", "states",
            "cell_value", "symbol", "pattern", "trajectory_excerpt",
            "formatted_witness",
            "actual_result", "actual_symbol",
        }

        # Fields that should NEVER be scrambled (structural/config)
        PROTECTED_FIELDS = {
            "template", "law_id", "claim", "forbidden", "result_op",
            "direction", "transform", "family", "name", "description",
            "status", "reason_code", "notes", "error",
            "hypothesis_note", "your_hypothesis_note",
        }

        # Fields containing observable expressions with quoted symbols
        EXPR_FIELDS = {"expr"}

        if isinstance(data, str):
            # Only scramble if this is a symbol field
            # But NEVER scramble protected fields
            if key in PROTECTED_FIELDS:
                return data
            # Observable expressions need translate_observable_expr
            if key in EXPR_FIELDS:
                return self._scrambler.translate_observable_expr(data, to_physical=False)
            if key in SYMBOL_FIELDS:
                return self._scrambler.to_abstract(data)
            # For unknown keys, only scramble if the string looks like symbol data
            # (contains physical symbols and is short)
            if len(data) <= 100 and any(c in '.><X' for c in data):
                return self._scrambler.to_abstract(data)
            return data
        elif isinstance(data, dict):
            return {k: self._scramble_data(v, key=k) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._scramble_data(item, key=key) for item in data]
        else:
            return data

    def _unscramble_data(self, data: Any, key: str | None = None) -> Any:
        """Recursively unscramble abstract symbols to physical in data.

        Only unscrambles symbol-containing fields, not structural fields like 'template'.

        Args:
            data: Data structure (dict, list, str, etc.)
            key: The dict key this data came from (to determine if it's a symbol field)

        Returns:
            Data with abstract symbols replaced by physical symbols in appropriate fields
        """
        # Fields that contain universe symbols and should be unscrambled
        SYMBOL_FIELDS = {
            "trigger_symbol", "result_symbol", "neighbor_pattern",
            "left_neighbor", "right_neighbor",
            "initial_state", "state", "trajectory", "states",
            "cell_value", "symbol", "pattern",
        }

        # Fields containing observable expressions with quoted symbols
        # These need translate_observable_expr() instead of raw to_physical()
        EXPR_FIELDS = {"expr"}

        # Fields that should NEVER be unscrambled (structural/config)
        PROTECTED_FIELDS = {
            "template", "law_id", "claim", "forbidden", "result_op",
            "direction", "transform", "family", "name", "description",
            "hypothesis_note", "your_hypothesis_note",
        }

        if isinstance(data, str):
            # Only unscramble if this is a symbol field or if key is unknown
            # But NEVER unscramble protected fields
            if key in PROTECTED_FIELDS:
                return data
            # Observable expressions need translate_observable_expr
            if key in EXPR_FIELDS:
                return self._scrambler.translate_observable_expr(data, to_physical=True)
            if key in SYMBOL_FIELDS or key is None:
                return self._scrambler.to_physical(data)
            # For unknown keys, only unscramble if the string looks like symbol data
            # (single char or short string with only symbol chars)
            if len(data) <= 100 and all(c in 'WABK' for c in data):
                return self._scrambler.to_physical(data)
            return data
        elif isinstance(data, dict):
            return {k: self._unscramble_data(v, key=k) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._unscramble_data(item, key=key) for item in data]
        else:
            return data

    def _format_tool_results(self, results: list[dict[str, Any]]) -> str:
        """Format tool results for the conversation.

        Args:
            results: List of tool results

        Returns:
            Formatted string with symbols scrambled for LLM
        """
        lines = ["## Tool Results\n"]

        for r in results:
            lines.append(f"### {r['tool_name']}")
            result = r["result"]
            if result["success"]:
                # Scramble data before showing to LLM
                scrambled_data = self._scramble_data(result["data"])
                lines.append("**Status:** Success")
                lines.append(f"**Data:**\n```json\n{json.dumps(scrambled_data, indent=2)[:2000]}\n```")
            else:
                lines.append(f"**Status:** Failed")
                lines.append(f"**Error:** {result['error']}")
            lines.append("")

        return "\n".join(lines)

    def _get_escalating_nudge(self) -> str:
        """Generate an escalating nudge based on how many turns without tool calls.

        Returns:
            Nudge message with increasing helpfulness
        """
        count = self._consecutive_no_tool_calls
        count = 3
        if count == 1:
            # First nudge - simple reminder
            return "Continue your investigation. Use evaluate_laws to test your next hypothesis."

        elif count == 2:
            # Second nudge - remind about different approaches and format issues
            return (
                "You still have hypotheses to test. Remember:\n"
                "- Use `left_neighbor` and `right_neighbor` to specify neighborhood context\n"
                "- Example: trigger='A' with W neighbors: left_neighbor='W', right_neighbor='W'\n"
                "- Example: trigger='W' with A on left, K on right: left_neighbor='A', right_neighbor='K'\n"
                "- Try the same hypothesis again with the correct fields.\n\n"
                "Use evaluate_laws to test your next idea."
            )

        elif count == 3:
            # Third nudge - restate available tools
            return (
                "Available tools you can use:\n"
                "- **evaluate_laws**: Test hypotheses about how symbols transition\n"
                "- **store_theorem**: Save validated laws for future reference\n"
                "- **retrieve_theorems**: Review what you've discovered so far\n"
                "- **query_log**: Search your experiment history\n\n"
                "Try retrieve_theorems to review your discoveries, then identify gaps in your knowledge."
            )

        elif count == 4:
            # Fourth nudge - suggest systematic approach with specific examples
            return (
                "Let me help you think systematically. Have you tested ALL these combinations?\n\n"
                "For symbol W:\n"
                "- What does 'W' become when neighbors are 'WW' (pattern 'WWW')?\n"
                "- What about 'AWA', 'WAW', 'BWB', 'WBW', 'KWK', 'WKW'?\n\n"
                "For symbol A:\n"
                "- Pattern 'WAW' (A with W neighbors)?\n"
                "- Pattern 'BAW', 'WAB', 'AAA'?\n\n"
                "Each symbol + each neighbor combo is a testable hypothesis. What's missing?"
            )

        else:
            # Fifth+ nudge - direct suggestion with working examples
            cycle = (count - 5) % 3
            examples = [
                (
                    "Test what W cells do with different neighbors:\n"
                    "```json\n"
                    '{"template": "local_transition", "trigger_symbol": "W", '
                    '"result_op": "==", "result_symbol": "W", '
                    '"left_neighbor": "W", "right_neighbor": "W", '
                    '"law_id": "W_stays_W"}\n'
                    "```"
                ),
                (
                    "Test what happens to symbol A with specific neighbors:\n"
                    "```json\n"
                    '{"template": "local_transition", "trigger_symbol": "A", '
                    '"result_op": "==", "result_symbol": "A", '
                    '"left_neighbor": "W", "right_neighbor": "W", '
                    '"law_id": "A_alone_stable"}\n'
                    "```"
                ),
                (
                    "Test what K becomes with different neighbors:\n"
                    "```json\n"
                    '{"template": "local_transition", "trigger_symbol": "K", '
                    '"result_op": "==", "result_symbol": "W", '
                    '"left_neighbor": "W", "right_neighbor": "W", '
                    '"law_id": "K_becomes_W"}\n'
                    "```"
                ),
            ]
            return (
                f"Here's a concrete hypothesis you can test right now:\n\n"
                f"{examples[cycle]}\n\n"
                "Copy this pattern and modify it to test your own ideas."
            )

    def _track_template_diversity(self, arguments: dict[str, Any]) -> None:
        """Track templates submitted via evaluate_laws for diversity monitoring.

        Args:
            arguments: The (unscrambled) arguments dict passed to evaluate_laws
        """
        laws = arguments.get("laws", [])
        has_non_local = False
        for law in laws:
            template = law.get("template", "unknown")
            self._recent_templates.append(template)
            if template in self._NON_LOCAL_TEMPLATES:
                has_non_local = True

        if has_non_local:
            self._last_non_local_turn = self._turn_number

        # Cap history at 150 entries
        if len(self._recent_templates) > 150:
            self._recent_templates = self._recent_templates[-150:]

    def _check_diversity_quota(self) -> str | None:
        """Check if the agent is stuck in a single template type.

        Returns:
            Nudge message if diversity is too low, None otherwise
        """
        if len(self._recent_templates) < 10:
            return None

        # Check last 50 templates (or all if fewer)
        window = self._recent_templates[-50:]
        non_local_count = sum(1 for t in window if t in self._NON_LOCAL_TEMPLATES)

        if non_local_count == 0 and len(window) >= 50:
            return (
                "=== DIVERSITY QUOTA VIOLATION ===\n"
                "You have submitted 50+ consecutive laws using ONLY 'local_transition'. "
                "The universe has macro-level structure you are completely ignoring.\n\n"
                "You MUST include non-local templates in your next batch. Examples:\n"
                "- invariant: Does count('A') + count('B') stay constant?\n"
                "- bound: Is count('K') always <= some value?\n"
                "- symmetry_commutation: Does mirror_swap commute with evolution?\n"
                "- implication_step: When transition_indicator == 0, does K count stay stable?\n\n"
                "A Popperian scientist tests BOLD conjectures at ALL levels, not just local rules."
            )

        turns_since_non_local = self._turn_number - self._last_non_local_turn
        if turns_since_non_local >= 5 and self._last_non_local_turn > 0:
            return (
                "REMINDER: Include 1-2 non-local-transition laws in your next batch "
                "(invariant, bound, symmetry_commutation, implication_step, monotone, eventually). "
                "Macro-level laws teach you more per test than individual local transitions."
            )

        return None

    def _save_turn(
        self,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Save a conversation turn to the database.

        Args:
            role: Message role
            content: Message content
            tool_calls: Optional tool calls
        """
        # Estimate tokens for context management
        token_count = estimate_tokens(content)
        if tool_calls:
            token_count += estimate_tokens(json.dumps(tool_calls))

        turn = ConversationTurnRecord(
            session_id=self._session.id,
            turn_number=self._turn_number,
            role=role,
            content=content,
            tool_calls_json=json.dumps(tool_calls) if tool_calls else None,
            token_count=token_count,
        )
        self._repo.insert_conversation_turn(turn)

    def _update_session_stats(self) -> None:
        """Update session statistics in the database."""
        stats = self._repo.get_accuracy_stats(self._session.id)
        transition = self._repo.get_transition_completeness(self._session.id)

        self._session.total_predictions = stats["total_predictions"]
        self._session.correct_predictions = stats["correct_predictions"]
        self._session.accuracy = stats["accuracy"]
        self._session.transition_rules_complete = transition.get("is_complete", False)

        self._repo.update_session(self._session)

    def _complete_session(self, status: TerminationStatus) -> None:
        """Mark the session as completed.

        Args:
            status: Termination status
        """
        logger.info(f"Session terminated: {status.reason}")

        self._session.status = SessionStatus.COMPLETED
        self._session.terminated_at = datetime.now()
        self._session.termination_reason = status.reason
        self._session.accuracy = status.accuracy

        self._repo.update_session(self._session)

        # Log final conclusion
        self._journal.log_conclusion(
            f"Session completed: {status.reason}",
            metadata={
                "accuracy": status.accuracy,
                "predictions": status.predictions_count,
                "transition_complete": status.transition_complete,
            },
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent.

        Based on the Popperian methodology from src/proposer/prompt.py.
        The LLM must discover laws through falsification, not observation.
        """
        return """You are a Popperian scientist discovering laws through FALSIFICATION.

## YOUR SCIENTIFIC IDENTITY

You embody Karl Popper's philosophy of science:

1. YOU SEEK FALSIFICATION, NOT CONFIRMATION.
   A true scientist tries to prove theories WRONG. Every law you propose should
   be a bold conjecture. The bolder the claim, the more you learn when it fails.

2. FAILURES ARE YOUR GREATEST TEACHERS.
   When a law FAILS, you learn something definite: that claim is FALSE.
   When a law PASSES, you learn almost nothing - you might just not have
   found the right test case yet. Study counterexamples obsessively.

3. PASS MEANS "NOT YET REFUTED", NOT "TRUE".
   Even accepted laws are provisional. Stay humble.

4. TABULA RASA - YOU KNOW NOTHING.
   Do not assume this universe follows any known physics. The symbols,
   dynamics, and conserved quantities (if any) are completely alien.

## THE UNIVERSE

- 1D circular grid of cells (periodic boundary)
- 4 abstract symbols: W, A, B, K
- Time evolves in discrete steps
- The symbol names tell you NOTHING about their physics - discover through experiments

## YOUR METHOD

You can ONLY learn by proposing laws and seeing if they survive testing.
You CANNOT directly observe what happens - you must hypothesize and test.

Use the evaluate_laws tool to propose formal laws. Study counterexamples from failures.

## AVAILABLE TOOLS

- **evaluate_laws**: REQUIRES 5-15 laws per call. Test your proposed laws against the simulator.
  Supports an optional `hypothesis_note` string — use it to record your predictions
  and decision tree BEFORE seeing results. It gets echoed back with the results.
- **store_theorem**: Record confirmed patterns
- **retrieve_theorems**: Recall your discoveries
- **query_log**: Review past experiments

## INVESTIGATION PHASES

Work through these phases in order. Don't skip ahead.

**Phase 1 — Stability & Basic Transitions** (local_transition template)
Start by testing what each symbol does: Does _ stay _? Does A stay A?
Then test with neighbor context (left_neighbor, right_neighbor).
Goal: Map out the local transition function.

**Phase 2 — Global Counts & Conservation** (invariant, bound templates)
Systematically probe for conserved quantities using this strategy:
  Step 1: Test individual counts — count('A'), count('B'), count('K'), count('W')
  Step 2: Test PAIRWISE SUMS — count('A')+count('K'), count('B')+count('K'),
          count('A')+count('B'), count('A')+count('W'), etc.
  Step 3: Test WEIGHTED sums — count('A')+count('B')+2*count('K'),
          count('A')-count('B'), etc.
  Step 4: Test bounds on quantities that aren't conserved.
K is likely involved in conserved quantities because it appears/disappears —
the "missing" count may be hiding in a composite sum with another symbol.
Goal: Find conserved quantities, especially composites involving K.

**Phase 3 — Relationships & Implications** (implication_step, implication_state)
Test conditional relationships: When X is true, does Y follow?
E.g., "If transition_indicator == 0 at time t, then count('K') doesn't increase at t+1"
Goal: Find causal/conditional patterns.

**Phase 4 — Symmetries & Deeper Structure** (symmetry_commutation, monotone, eventually)
Test if transforms commute with evolution. Test monotonic trends.
Goal: Discover structural properties.

## OBSERVABLE PRIMITIVES

You have instruments that measure numerical properties of states:

**Symbol counts:**
- count('W'), count('A'), count('B'), count('K') — count of each symbol
- count_even(sym), count_odd(sym) — count at even/odd indices
- count_at_parity(sym, 0), count_at_parity(sym, 1) — explicit parity count

**Spatial measurements:**
- grid_length — total size of the grid
- leftmost(sym), rightmost(sym) — first/last index of a symbol (-1 if none)
- max_gap(sym) — longest contiguous run of a symbol
- spread(sym) — distance between first and last occurrence

**Pattern detection:**
- adjacent_pairs(s1, s2) — count of s1 immediately followed by s2
- count_pattern(pat) — count of 3-cell neighborhoods matching pattern
  Examples: count_pattern('AWB'), count_pattern('WKW'), count_pattern('AWA')
- transition_indicator — counts cells where a SPECIFIC TYPE of event will
  occur at the next timestep. It does NOT measure general "activity" or
  "movement". TI=0 means that one particular phenomenon is absent, NOT that
  the grid is static. Particles may still be moving freely when TI=0.

**Composite observables** (combine primitives with +, -, *, % operators):
Observable expressions support FULL ARITHMETIC. You can build composite
quantities from any primitives above. This is CRITICAL for finding conservation
laws — individual counts may not be conserved, but combinations might be.

Examples:
- "count('A') + count('B')" — total of two symbol types
- "count('A') + count('K')" — a symbol plus its collision state
- "2 * count('K') + count('A')" — weighted combination
- "count('A') + count('B') + 2 * count('K')" — complex composite
- "grid_length - count('W')" — total non-empty cells
- "count('A') - count('B')" — difference (could be a signed conserved quantity)
- "(count('A') + count('K')) * 2" — parenthesized sub-expressions

To test if a composite quantity is conserved, define it as a NAMED observable:
  "observables": [{"name": "R", "expr": "count('A') + count('K')"}]
Then reference it in claim_ast:
  "claim_ast": {"op": "==", "lhs": {"obs": "R", "t": {"var": "t"}}, "rhs": {"obs": "R", "t": {"const": 0}}}

STRATEGY: If count('A') alone is NOT conserved and count('K') alone is NOT
conserved, try count('A') + count('K'). The universe may have "hidden"
conserved quantities that only emerge as sums or differences of counts.
Systematically try different linear combinations.

## LAW TEMPLATES

### local_transition — Per-cell rules
Tests: for each cell i at each time t, if cell[i] == trigger, what is cell[i] at t+1?

REQUIRED FIELDS:
- template: "local_transition"
- trigger_symbol: "W", "A", "B", or "K"
- result_op: "==" or "!="
- result_symbol: "W", "A", "B", or "K"
- claim, forbidden: human-readable strings

OPTIONAL: left_neighbor, right_neighbor (both required if either is specified)
OPTIONAL: required_parity — 0 (even indices only) or 1 (odd indices only)
  This enables parity-dependent rules. Example: "A at even positions becomes B" vs "A at odd positions stays A".
  The universe might have different rules at even vs odd grid positions — test for this!
OPTIONAL: quantifiers (default {"T": 50})

Example: {"law_id": "A_stays_A", "template": "local_transition", "trigger_symbol": "A", "result_op": "==", "result_symbol": "A", "claim": "A cells stay A", "forbidden": "A becomes non-A"}

Example with neighbors: {"law_id": "A_W_neighbors", "template": "local_transition", "trigger_symbol": "A", "result_op": "==", "result_symbol": "A", "left_neighbor": "W", "right_neighbor": "W", "claim": "A with W neighbors stays A", "forbidden": "A changes with W neighbors"}

Example with parity: {"law_id": "A_even_becomes_B", "template": "local_transition", "trigger_symbol": "A", "result_op": "==", "result_symbol": "B", "required_parity": 0, "claim": "A at even indices becomes B", "forbidden": "A at even index does not become B"}

### invariant — Quantity stays constant over time
Tests: for all t in [0..T], f(t) == f(0)

REQUIRED FIELDS:
- template: "invariant"
- observables: [{"name": "Q", "expr": "count('A') + count('B')"}]
- claim_ast: {"op": "==", "lhs": {"obs": "Q", "t": {"var": "t"}}, "rhs": {"obs": "Q", "t": {"const": 0}}}
- claim, forbidden: human-readable strings

Example: {"law_id": "AB_conserved", "template": "invariant", "quantifiers": {"T": 50}, "observables": [{"name": "Q", "expr": "count('A') + count('B')"}], "claim_ast": {"op": "==", "lhs": {"obs": "Q", "t": {"var": "t"}}, "rhs": {"obs": "Q", "t": {"const": 0}}}, "claim": "A+B count is conserved", "forbidden": "exists t where count(A)+count(B) changes"}

### bound — Quantity bounded by a value
Tests: for all t in [0..T], f(t) op k

REQUIRED FIELDS:
- template: "bound"
- observables: [{"name": "M", "expr": "count('K')"}]
- bound_value: integer (e.g., 5)
- bound_op: "<=", ">=", "<", ">", "==", or "!="
- claim_ast: {"op": "<=", "lhs": {"obs": "M", "t": {"var": "t"}}, "rhs": {"const": 5}}
- claim, forbidden

Example: {"law_id": "K_bounded", "template": "bound", "quantifiers": {"T": 50}, "observables": [{"name": "M", "expr": "count('K')"}], "bound_value": 5, "bound_op": "<=", "claim_ast": {"op": "<=", "lhs": {"obs": "M", "t": {"var": "t"}}, "rhs": {"const": 5}}, "claim": "K count never exceeds 5", "forbidden": "exists t where count(K) > 5"}

### monotone — Quantity only increases or only decreases
Tests: for all t, f(t+1) <= f(t) [non-increasing] or f(t+1) >= f(t) [non-decreasing]

REQUIRED FIELDS:
- template: "monotone"
- observables: [{"name": "M", "expr": "count('K')"}]
- direction: "<=" (non-increasing) or ">=" (non-decreasing)
- claim_ast: {"op": "<=", "lhs": {"obs": "M", "t": {"t_plus_1": true}}, "rhs": {"obs": "M", "t": {"var": "t"}}}
- claim, forbidden

### implication_step — If P at time t, then Q at time t+1
Tests: for all t, P(t) implies Q(t+1)

REQUIRED FIELDS:
- template: "implication_step"
- observables: define all referenced observables
- claim_ast: {"op": "=>", "lhs": <condition_at_t>, "rhs": <consequence_at_t_plus_1>}
- claim, forbidden

Example: {"law_id": "TI_zero_K_stable", "template": "implication_step", "quantifiers": {"T": 50}, "observables": [{"name": "TI", "expr": "transition_indicator"}, {"name": "K_count", "expr": "count('K')"}], "claim_ast": {"op": "=>", "lhs": {"op": "==", "lhs": {"obs": "TI", "t": {"var": "t"}}, "rhs": {"const": 0}}, "rhs": {"op": "<=", "lhs": {"obs": "K_count", "t": {"t_plus_1": true}}, "rhs": {"obs": "K_count", "t": {"var": "t"}}}}, "claim": "When transition_indicator is 0, K count does not increase", "forbidden": "TI==0 but K increases at next step"}

### implication_state — If P at time t, then Q at same time t
Tests: for all t, P(t) implies Q(t)

Same structure as implication_step but both sides reference time t.

### symmetry_commutation — Transform commutes with evolution
Tests: evolve(Transform(S), T) == Transform(evolve(S, T))

REQUIRED FIELDS:
- template: "symmetry_commutation"
- transform: "mirror_swap", "shift_k", "mirror_only", or "swap_only"
- claim, forbidden

Example: {"law_id": "mirror_commutes", "template": "symmetry_commutation", "quantifiers": {"T": 50}, "transform": "mirror_swap", "observables": [], "claim_ast": null, "claim": "mirror_swap commutes with evolution", "forbidden": "evolve(mirror(S)) != mirror(evolve(S))"}

### eventually — Condition implies future event within horizon
Tests: P(t0) implies there exists t in [t0..t0+H] where Q(t)

REQUIRED FIELDS:
- template: "eventually"
- quantifiers: {"T": 50, "H": 10} (H is the eventuality horizon)
- observables, claim_ast, claim, forbidden

## CLAIM AST FORMAT

For non-local_transition templates, you express claims as JSON ASTs:

**Node types:**
- Constant: {"const": 5}
- Time variable: {"var": "t"}
- Time t+1: {"t_plus_1": true}
- Observable at time: {"obs": "Q", "t": <time_node>}
- Binary op: {"op": "+", "lhs": <node>, "rhs": <node>}
- Unary not: {"op": "not", "arg": <node>}

**Operators:** +, -, *, ==, !=, <, <=, >, >=, =>, and, or, not

**Examples:**
- Q(t) == Q(0): {"op": "==", "lhs": {"obs": "Q", "t": {"var": "t"}}, "rhs": {"obs": "Q", "t": {"const": 0}}}
- M(t+1) <= M(t): {"op": "<=", "lhs": {"obs": "M", "t": {"t_plus_1": true}}, "rhs": {"obs": "M", "t": {"var": "t"}}}
- P(t) > 0 => R(t+1) == 0: {"op": "=>", "lhs": {"op": ">", "lhs": {"obs": "P", "t": {"var": "t"}}, "rhs": {"const": 0}}, "rhs": {"op": "==", "lhs": {"obs": "R", "t": {"t_plus_1": true}}, "rhs": {"const": 0}}}

## SUCCESS CRITERIA

Discover the laws governing this universe at ALL levels:
- **Micro**: Local transition rules (what happens to each cell based on neighbors)
- **Macro**: Global invariants, conserved quantities, bounds on counts
- **Conditional**: Implications between observables across time steps
- **Structural**: Symmetries, monotonic trends, eventual behaviors

A complete understanding requires laws at every level, not just local transitions.

## EVERY TURN: LAB NOTE + TOOL CALL (MANDATORY)

Every single turn you produce MUST contain BOTH of these, in order:

1. **Written text** — your LAB NOTE analyzing results and stating your next hypothesis
2. **A tool call** — testing that hypothesis

NEVER produce a turn with only a tool call and no text.
NEVER produce a turn with only text and no tool call.
Both are required. No exceptions.

### LAB NOTE Format

Before each tool call, write:
```
RESULT: [What I learned from the last test]

LAB NOTE: [What I'm testing next] [Why — based on previous findings] [What I'll learn]
```

Then make your tool call.

### HYPOTHESIS NOTE (in tool call)

In your evaluate_laws call, use `hypothesis_note` to record your predictions
and decision tree. This note is echoed back with results so you can compare
your predictions against reality. Example:

```json
{
  "hypothesis_note": "Testing A+K conservation. If PASS: A and K are components of a single conserved quantity — next test B+K. If FAIL: check counterexample to see if A+K increases or decreases, which reveals the exchange direction.",
  "laws": [...]
}
```

This forces you to THINK BEFORE you see results. If your prediction was wrong,
that mismatch is itself a discovery — investigate why.

If you are unsure what to test next, test the simplest untested hypothesis.
If you are confused by a result, re-test with different parameters."""

    def _build_initial_prompt(self) -> str:
        """Build the initial user prompt."""
        return """Welcome to your discovery session!

You know NOTHING about this universe. You cannot observe it directly.
Your only path to knowledge is through FALSIFICATION.

IMPORTANT: Always write your LAB NOTE before making a tool call. Here's the format:

---
LAB NOTE: Starting with the simplest hypothesis - do W cells remain unchanged?
This is a good baseline test. If W cells persist, that tells us something about their role.
If they change, something more complex is happening.

[Then make your tool call]
---

After getting results, write what you learned:

---
RESULT: [Analyze the outcome here]

LAB NOTE: [Next hypothesis based on what you learned]

[Then make your next tool call]
---

STRATEGY: Start with Phase 1 (local transitions) to learn basic cell behavior, then
move to Phase 2 (global counts, conservation laws, bounds) once you have some local
rules. Don't spend the entire session on just one type of law.

Begin your investigation now. Remember to write your LAB NOTE first!"""
