"""Flask web application for browsing Popperian Discovery results.

This is a read-only web interface for viewing orchestration runs,
LLM transcripts, and discovered artifacts (laws, theorems, explanations).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Flask, g, jsonify, render_template, request

from src.db.repo import Repository

# Create Flask app with templates in web/templates
app = Flask(
    __name__,
    template_folder=Path(__file__).parent / "templates",
)

# Store the database path (set by init_app)
_db_path: str | None = None


def get_repo() -> Repository:
    """Get the repository instance for the current request.

    Creates a new connection per request to handle Flask's threading model.
    """
    if _db_path is None:
        raise RuntimeError("Database path not initialized. Call init_app() first.")

    if "repo" not in g:
        g.repo = Repository(_db_path)
        g.repo.connect()

    return g.repo


@app.teardown_appcontext
def close_repo(exception):
    """Close the repository connection at the end of each request."""
    repo = g.pop("repo", None)
    if repo is not None:
        repo.close()


def init_app(db_path: str) -> Flask:
    """Initialize the Flask app with a database path.

    Args:
        db_path: Path to the SQLite database

    Returns:
        Configured Flask app
    """
    global _db_path
    _db_path = db_path
    return app


# =============================================================================
# Template Filters
# =============================================================================


@app.template_filter("json_pretty")
def json_pretty_filter(value: str | dict | list | None) -> str:
    """Pretty-print JSON for display."""
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(value, indent=2)


@app.template_filter("truncate_text")
def truncate_text_filter(value: str | None, length: int = 100) -> str:
    """Truncate text with ellipsis."""
    if value is None:
        return ""
    if len(value) <= length:
        return value
    return value[:length] + "..."


@app.template_filter("format_tokens")
def format_tokens_filter(value: int | None) -> str:
    """Format token counts with K/M suffixes."""
    if value is None:
        return "-"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


@app.template_filter("format_duration")
def format_duration_filter(value: int | None) -> str:
    """Format milliseconds as human-readable duration."""
    if value is None:
        return "-"
    if value < 1000:
        return f"{value}ms"
    if value < 60000:
        return f"{value / 1000:.1f}s"
    return f"{value / 60000:.1f}m"


# =============================================================================
# Dashboard Routes
# =============================================================================


@app.route("/")
def index():
    """Dashboard showing recent runs and summary stats."""
    repo = get_repo()

    # Get recent orchestration runs
    runs = repo.list_orchestration_runs(limit=10)

    # Get evaluation summary
    eval_summary = repo.get_evaluation_summary()

    # Get law count
    laws = repo.list_laws(limit=1)  # Just to check if table is populated
    law_count = len(repo.list_laws(limit=10000))

    # Get theorem count
    theorems = repo.list_theorems(limit=10000)
    theorem_count = len(theorems)

    # Get LLM transcript stats
    transcript_stats = repo.get_llm_transcript_stats()

    return render_template(
        "index.html",
        runs=runs,
        eval_summary=eval_summary,
        law_count=law_count,
        theorem_count=theorem_count,
        transcript_stats=transcript_stats,
    )


# =============================================================================
# Run Routes
# =============================================================================


@app.route("/runs")
def runs_list():
    """List all orchestration runs."""
    repo = get_repo()
    status_filter = request.args.get("status")

    runs = repo.list_orchestration_runs(status=status_filter, limit=100)

    return render_template("runs/list.html", runs=runs, status_filter=status_filter)


@app.route("/runs/<run_id>")
def run_detail(run_id: str):
    """Show run detail with timeline of iterations."""
    repo = get_repo()

    run = repo.get_orchestration_run(run_id)
    if not run:
        return render_template("error.html", message=f"Run {run_id} not found"), 404

    iterations = repo.list_iterations_for_run(run_id, limit=1000)
    transitions = repo.list_phase_transitions(run_id, limit=100)
    readiness_snapshots = repo.list_readiness_snapshots_for_run(run_id, limit=1000)

    return render_template(
        "runs/detail.html",
        run=run,
        iterations=iterations,
        transitions=transitions,
        readiness_snapshots=readiness_snapshots,
    )


@app.route("/runs/<run_id>/iterations/<int:iteration_index>")
def iteration_detail(run_id: str, iteration_index: int):
    """Show single iteration detail."""
    repo = get_repo()

    run = repo.get_orchestration_run(run_id)
    if not run:
        return render_template("error.html", message=f"Run {run_id} not found"), 404

    iteration = repo.get_orchestration_iteration(run_id, iteration_index)
    if not iteration:
        return (
            render_template(
                "error.html",
                message=f"Iteration {iteration_index} not found in run {run_id}",
            ),
            404,
        )

    return render_template(
        "runs/iteration.html",
        run=run,
        iteration=iteration,
    )


# =============================================================================
# LLM Transcript Routes
# =============================================================================


@app.route("/transcripts")
def transcripts_list():
    """List and search LLM transcripts."""
    repo = get_repo()

    # Get filter parameters
    run_id = request.args.get("run_id")
    component = request.args.get("component")
    phase = request.args.get("phase")
    search = request.args.get("q")
    page = int(request.args.get("page", 1))
    per_page = 20

    offset = (page - 1) * per_page

    transcripts = repo.list_llm_transcripts(
        run_id=run_id,
        component=component,
        phase=phase,
        search_query=search,
        limit=per_page,
        offset=offset,
    )

    total_count = repo.get_llm_transcript_count(run_id=run_id, component=component)
    total_pages = (total_count + per_page - 1) // per_page

    return render_template(
        "transcripts/list.html",
        transcripts=transcripts,
        run_id=run_id,
        component=component,
        phase=phase,
        search=search,
        page=page,
        total_pages=total_pages,
        total_count=total_count,
    )


@app.route("/transcripts/<int:transcript_id>")
def transcript_detail(transcript_id: int):
    """Show full transcript detail."""
    repo = get_repo()

    transcript = repo.get_llm_transcript(transcript_id)
    if not transcript:
        return (
            render_template(
                "error.html", message=f"Transcript {transcript_id} not found"
            ),
            404,
        )

    return render_template("transcripts/detail.html", transcript=transcript)


# =============================================================================
# Law Routes
# =============================================================================


@app.route("/laws")
def laws_list():
    """Law gallery with status filtering."""
    repo = get_repo()

    status_filter = request.args.get("status")
    template_filter = request.args.get("template")

    # Get all laws
    laws = repo.list_laws(template=template_filter, limit=200)

    # Get latest evaluation for each law
    laws_with_evals = []
    for law in laws:
        eval_rec = repo.get_latest_evaluation(law.law_id)
        if status_filter and eval_rec and eval_rec.status != status_filter:
            continue
        if status_filter and not eval_rec:
            continue
        laws_with_evals.append((law, eval_rec))

    # Get evaluation summary for stats
    eval_summary = repo.get_evaluation_summary()

    return render_template(
        "laws/list.html",
        laws_with_evals=laws_with_evals,
        status_filter=status_filter,
        template_filter=template_filter,
        eval_summary=eval_summary,
    )


@app.route("/laws/<law_id>")
def law_detail(law_id: str):
    """Show law detail with evaluations and counterexamples."""
    repo = get_repo()

    law = repo.get_law(law_id)
    if not law:
        return render_template("error.html", message=f"Law {law_id} not found"), 404

    evaluations = repo.list_evaluations(limit=20)
    # Filter to this law's evaluations
    law_evaluations = [e for e in evaluations if e.law_id == law_id]

    counterexamples = repo.get_counterexamples_for_law(law_id)
    witnesses = repo.get_witnesses_for_law(law_id, limit=10)

    # Parse law JSON
    try:
        law_data = json.loads(law.law_json)
    except json.JSONDecodeError:
        law_data = {}

    return render_template(
        "laws/detail.html",
        law=law,
        law_data=law_data,
        evaluations=law_evaluations,
        counterexamples=counterexamples,
        witnesses=witnesses,
    )


# =============================================================================
# Theorem Routes
# =============================================================================


@app.route("/theorems")
def theorems_list():
    """Theorem gallery."""
    repo = get_repo()

    status_filter = request.args.get("status")

    theorems = repo.list_theorems(status=status_filter, limit=200)

    return render_template(
        "theorems/list.html",
        theorems=theorems,
        status_filter=status_filter,
    )


@app.route("/theorems/<theorem_id>")
def theorem_detail(theorem_id: str):
    """Show theorem detail."""
    repo = get_repo()

    theorem = repo.get_theorem(theorem_id)
    if not theorem:
        return (
            render_template("error.html", message=f"Theorem {theorem_id} not found"),
            404,
        )

    # Parse support JSON
    try:
        support = json.loads(theorem.support_json) if theorem.support_json else []
    except json.JSONDecodeError:
        support = []

    # Parse failure modes
    try:
        failure_modes = (
            json.loads(theorem.failure_modes_json)
            if theorem.failure_modes_json
            else []
        )
    except json.JSONDecodeError:
        failure_modes = []

    # Parse missing structure
    try:
        missing_structure = (
            json.loads(theorem.missing_structure_json)
            if theorem.missing_structure_json
            else []
        )
    except json.JSONDecodeError:
        missing_structure = []

    return render_template(
        "theorems/detail.html",
        theorem=theorem,
        support=support,
        failure_modes=failure_modes,
        missing_structure=missing_structure,
    )


# =============================================================================
# Explanation Routes
# =============================================================================


@app.route("/explanations")
def explanations_list():
    """Explanation gallery."""
    repo = get_repo()

    run_id = request.args.get("run_id")
    status_filter = request.args.get("status")

    # Get all runs for filter dropdown
    runs = repo.list_orchestration_runs(limit=100)

    # Get explanations
    if run_id:
        explanations = repo.list_explanations_for_run(run_id, limit=100)
    else:
        # Get explanations from all runs
        explanations = []
        for run in runs:
            explanations.extend(repo.list_explanations_for_run(run.run_id, limit=20))

    # Apply status filter
    if status_filter:
        explanations = [e for e in explanations if e.status == status_filter]

    return render_template(
        "explanations/list.html",
        explanations=explanations,
        runs=runs,
        run_id=run_id,
        status_filter=status_filter,
    )


@app.route("/explanations/<explanation_id>")
def explanation_detail(explanation_id: str):
    """Show explanation detail."""
    repo = get_repo()

    explanation = repo.get_explanation(explanation_id)
    if not explanation:
        return (
            render_template(
                "error.html", message=f"Explanation {explanation_id} not found"
            ),
            404,
        )

    # Parse mechanism JSON
    try:
        mechanism = (
            json.loads(explanation.mechanism_json)
            if explanation.mechanism_json
            else {}
        )
    except json.JSONDecodeError:
        mechanism = {}

    # Parse supporting theorems
    try:
        supporting_theorems = (
            json.loads(explanation.supporting_theorem_ids_json)
            if explanation.supporting_theorem_ids_json
            else []
        )
    except json.JSONDecodeError:
        supporting_theorems = []

    # Parse open questions
    try:
        open_questions = (
            json.loads(explanation.open_questions_json)
            if explanation.open_questions_json
            else []
        )
    except json.JSONDecodeError:
        open_questions = []

    # Parse criticisms
    try:
        criticisms = (
            json.loads(explanation.criticisms_json)
            if explanation.criticisms_json
            else []
        )
    except json.JSONDecodeError:
        criticisms = []

    return render_template(
        "explanations/detail.html",
        explanation=explanation,
        mechanism=mechanism,
        supporting_theorems=supporting_theorems,
        open_questions=open_questions,
        criticisms=criticisms,
    )


# =============================================================================
# API Routes
# =============================================================================


@app.route("/api/readiness/<run_id>")
def api_readiness(run_id: str):
    """JSON API for readiness chart data."""
    repo = get_repo()

    snapshots = repo.list_readiness_snapshots_for_run(run_id, limit=1000)

    data = {
        "labels": [],
        "s_pass": [],
        "s_stability": [],
        "s_novel_cex": [],
        "combined_score": [],
    }

    for snapshot in snapshots:
        data["labels"].append(snapshot.iteration_id)
        data["s_pass"].append(snapshot.s_pass or 0)
        data["s_stability"].append(snapshot.s_stability or 0)
        data["s_novel_cex"].append(snapshot.s_novel_cex or 0)
        data["combined_score"].append(snapshot.combined_score or 0)

    return jsonify(data)


# =============================================================================
# Error Handlers
# =============================================================================


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template("error.html", message="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template("error.html", message=f"Server error: {e}"), 500
