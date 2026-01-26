"""Tools for managing theorems (validated laws)."""

import json
import logging
from typing import Any

from src.ahc.db.models import TheoremRecord, TheoremStatus
from src.ahc.db.repo import AHCRepository
from src.ahc.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class StoreTheoremTool(BaseTool):
    """Tool for storing a theorem (validated law or principle)."""

    def __init__(self, repo: AHCRepository | None = None):
        """Initialize the tool.

        Args:
            repo: Database repository for persistence
        """
        self._repo = repo
        self._session_id: int | None = None

    def set_context(self, session_id: int) -> None:
        """Set the current session context.

        Args:
            session_id: Database session ID
        """
        self._session_id = session_id

    @property
    def name(self) -> str:
        return "store_theorem"

    @property
    def description(self) -> str:
        return """Store a theorem (validated law or discovered principle) in your memory.

Use this to record:
- Laws that have been validated by evaluation
- Principles you've discovered about the physics
- Hypotheses you want to track
- Relationships between different laws

Theorems are named and can be retrieved later for reference."""

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="name",
                type="string",
                description="Unique name for the theorem (e.g., 'momentum_conservation', 'collision_rule_1')",
                required=True,
            ),
            ToolParameter(
                name="description",
                type="string",
                description="Human-readable description of what the theorem states",
                required=True,
            ),
            ToolParameter(
                name="law_ids",
                type="array",
                description="IDs of related laws (from evaluate_laws results)",
                required=False,
                items={"type": "string", "description": "A law ID"},
            ),
            ToolParameter(
                name="evidence",
                type="object",
                description="Supporting evidence for the theorem",
                required=False,
            ),
            ToolParameter(
                name="status",
                type="string",
                description="Status: 'proposed', 'validated', or 'refuted'",
                required=False,
                default="proposed",
                enum=["proposed", "validated", "refuted"],
            ),
        ]

    def execute(
        self,
        name: str,
        description: str,
        law_ids: list[str] | None = None,
        evidence: dict[str, Any] | None = None,
        status: str = "proposed",
        **kwargs,
    ) -> ToolResult:
        """Store a theorem.

        Args:
            name: Theorem name
            description: Description
            law_ids: Related law IDs
            evidence: Supporting evidence
            status: Theorem status

        Returns:
            ToolResult with storage confirmation
        """
        if self._repo is None or self._session_id is None:
            return ToolResult.fail("Repository not configured")

        try:
            # Check if theorem already exists
            existing = self._repo.get_theorem(self._session_id, name)
            if existing:
                return ToolResult.fail(
                    f"Theorem '{name}' already exists. Use edit_theorem to modify it."
                )

            # Parse status
            try:
                theorem_status = TheoremStatus(status)
            except ValueError:
                theorem_status = TheoremStatus.PROPOSED

            # Create theorem record
            theorem = TheoremRecord(
                session_id=self._session_id,
                name=name,
                description=description,
                law_ids_json=json.dumps(law_ids or []),
                status=theorem_status,
                evidence_json=json.dumps(evidence) if evidence else None,
            )

            self._repo.insert_theorem(theorem)

            return ToolResult.ok(
                data={
                    "stored": True,
                    "name": name,
                    "status": theorem_status.value,
                    "message": f"Theorem '{name}' stored successfully",
                }
            )

        except Exception as e:
            logger.exception(f"Failed to store theorem: {name}")
            return ToolResult.fail(f"Failed to store theorem: {str(e)}")


class RetrieveTheoremsTool(BaseTool):
    """Tool for retrieving stored theorems."""

    def __init__(self, repo: AHCRepository | None = None):
        """Initialize the tool.

        Args:
            repo: Database repository for persistence
        """
        self._repo = repo
        self._session_id: int | None = None

    def set_context(self, session_id: int) -> None:
        """Set the current session context.

        Args:
            session_id: Database session ID
        """
        self._session_id = session_id

    @property
    def name(self) -> str:
        return "retrieve_theorems"

    @property
    def description(self) -> str:
        return """Retrieve stored theorems from your memory.

Use this to:
- Look up previously stored theorems
- Review your discovered principles
- Check the status of hypotheses
- Build on existing knowledge"""

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="name",
                type="string",
                description="Specific theorem name to retrieve (optional - omit to get all)",
                required=False,
            ),
            ToolParameter(
                name="status",
                type="string",
                description="Filter by status: 'proposed', 'validated', or 'refuted'",
                required=False,
                enum=["proposed", "validated", "refuted"],
            ),
        ]

    def execute(
        self,
        name: str | None = None,
        status: str | None = None,
        **kwargs,
    ) -> ToolResult:
        """Retrieve theorems.

        Args:
            name: Optional specific theorem name
            status: Optional status filter

        Returns:
            ToolResult with matching theorems
        """
        if self._repo is None or self._session_id is None:
            return ToolResult.fail("Repository not configured")

        try:
            if name:
                # Get specific theorem
                theorem = self._repo.get_theorem(self._session_id, name)
                if theorem is None:
                    return ToolResult.ok(
                        data={
                            "found": False,
                            "message": f"Theorem '{name}' not found",
                        }
                    )
                return ToolResult.ok(
                    data={
                        "found": True,
                        "theorem": self._theorem_to_dict(theorem),
                    }
                )

            # Get all theorems with optional filter
            status_enum = TheoremStatus(status) if status else None
            theorems = self._repo.get_theorems(self._session_id, status=status_enum)

            return ToolResult.ok(
                data={
                    "count": len(theorems),
                    "theorems": [self._theorem_to_dict(t) for t in theorems],
                }
            )

        except Exception as e:
            logger.exception("Failed to retrieve theorems")
            return ToolResult.fail(f"Failed to retrieve theorems: {str(e)}")

    def _theorem_to_dict(self, theorem: TheoremRecord) -> dict[str, Any]:
        """Convert a theorem record to a dictionary."""
        return {
            "name": theorem.name,
            "description": theorem.description,
            "status": theorem.status.value,
            "law_ids": json.loads(theorem.law_ids_json) if theorem.law_ids_json else [],
            "evidence": json.loads(theorem.evidence_json) if theorem.evidence_json else None,
            "created_at": theorem.created_at.isoformat() if theorem.created_at else None,
            "updated_at": theorem.updated_at.isoformat() if theorem.updated_at else None,
        }


class EditTheoremTool(BaseTool):
    """Tool for editing an existing theorem."""

    def __init__(self, repo: AHCRepository | None = None):
        """Initialize the tool.

        Args:
            repo: Database repository for persistence
        """
        self._repo = repo
        self._session_id: int | None = None

    def set_context(self, session_id: int) -> None:
        """Set the current session context.

        Args:
            session_id: Database session ID
        """
        self._session_id = session_id

    @property
    def name(self) -> str:
        return "edit_theorem"

    @property
    def description(self) -> str:
        return """Edit an existing theorem in your memory.

Use this to:
- Update the status (e.g., mark as validated after testing)
- Add new evidence
- Refine the description
- Add related law IDs"""

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="name",
                type="string",
                description="Name of the theorem to edit",
                required=True,
            ),
            ToolParameter(
                name="description",
                type="string",
                description="New description (optional)",
                required=False,
            ),
            ToolParameter(
                name="law_ids",
                type="array",
                description="New list of related law IDs (optional)",
                required=False,
                items={"type": "string", "description": "A law ID"},
            ),
            ToolParameter(
                name="add_law_ids",
                type="array",
                description="Law IDs to add to existing list (optional)",
                required=False,
                items={"type": "string", "description": "A law ID"},
            ),
            ToolParameter(
                name="evidence",
                type="object",
                description="New evidence (optional - replaces existing)",
                required=False,
            ),
            ToolParameter(
                name="add_evidence",
                type="object",
                description="Evidence to merge with existing (optional)",
                required=False,
            ),
            ToolParameter(
                name="status",
                type="string",
                description="New status: 'proposed', 'validated', or 'refuted'",
                required=False,
                enum=["proposed", "validated", "refuted"],
            ),
        ]

    def execute(
        self,
        name: str,
        description: str | None = None,
        law_ids: list[str] | None = None,
        add_law_ids: list[str] | None = None,
        evidence: dict[str, Any] | None = None,
        add_evidence: dict[str, Any] | None = None,
        status: str | None = None,
        **kwargs,
    ) -> ToolResult:
        """Edit a theorem.

        Args:
            name: Theorem name
            description: New description
            law_ids: Replace law IDs
            add_law_ids: Add to law IDs
            evidence: Replace evidence
            add_evidence: Merge evidence
            status: New status

        Returns:
            ToolResult with update confirmation
        """
        if self._repo is None or self._session_id is None:
            return ToolResult.fail("Repository not configured")

        try:
            # Get existing theorem
            theorem = self._repo.get_theorem(self._session_id, name)
            if theorem is None:
                return ToolResult.fail(f"Theorem '{name}' not found")

            # Update fields
            if description is not None:
                theorem.description = description

            if law_ids is not None:
                theorem.law_ids_json = json.dumps(law_ids)
            elif add_law_ids:
                existing_ids = json.loads(theorem.law_ids_json) if theorem.law_ids_json else []
                combined = list(set(existing_ids + add_law_ids))
                theorem.law_ids_json = json.dumps(combined)

            if evidence is not None:
                theorem.evidence_json = json.dumps(evidence)
            elif add_evidence:
                existing_evidence = json.loads(theorem.evidence_json) if theorem.evidence_json else {}
                existing_evidence.update(add_evidence)
                theorem.evidence_json = json.dumps(existing_evidence)

            if status is not None:
                try:
                    theorem.status = TheoremStatus(status)
                except ValueError:
                    return ToolResult.fail(f"Invalid status: {status}")

            self._repo.update_theorem(theorem)

            return ToolResult.ok(
                data={
                    "updated": True,
                    "name": name,
                    "new_status": theorem.status.value,
                    "message": f"Theorem '{name}' updated successfully",
                }
            )

        except Exception as e:
            logger.exception(f"Failed to edit theorem: {name}")
            return ToolResult.fail(f"Failed to edit theorem: {str(e)}")
