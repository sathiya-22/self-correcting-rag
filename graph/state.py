"""
graph/state.py — Shared LangGraph state definition.
"""

from typing import Dict, List, Optional, TypedDict


class GraphState(TypedDict):
    """
    State carried through every node of the Self-Correcting RAG graph.

    Fields
    ------
    question    : The original user question.
    context     : List of text passages currently available as context
                  (from vectorstore OR from web search).
    generation  : The final synthesized answer (populated by generate node).
    retry_count : Number of times the grader has returned "irrelevant".
                  When this reaches MAX_RETRIES the graph halts and asks for a hint.
    hint        : Optional hint provided by the user after the graph halts.
    source      : Where the current context came from — "vectorstore" | "web".
    """

    question: str
    context: List[str]
    generation: str
    retry_count: int
    hint: Optional[str]
    source: str
