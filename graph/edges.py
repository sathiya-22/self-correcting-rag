"""
graph/edges.py — Conditional routing logic for the Self-Correcting RAG graph.
"""

from __future__ import annotations
from graph.state import GraphState
from graph.nodes import MAX_RETRIES


def route_after_grading(state: GraphState) -> str:
    """
    Decide the next node after grade_documents runs.

    Decision tree:
      - context is non-empty (relevant docs found) → go to "generate"
      - context is empty AND retry_count < MAX_RETRIES → go to "web_search"
      - context is empty AND retry_count >= MAX_RETRIES → go to "ask_for_hint"
    """
    context = state.get("context", [])
    retry_count = state.get("retry_count", 0)

    if context:
        # Relevant docs found — proceed to generation
        return "generate"

    if retry_count < MAX_RETRIES:
        # Still have retries left — search the web
        return "web_search"

    # Out of retries — ask the user for a hint
    return "ask_for_hint"
