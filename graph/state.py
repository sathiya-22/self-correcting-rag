"""
graph/state.py — Shared LangGraph state definition.
"""

from typing import Dict, List, Optional, TypedDict, Any
from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    """
    State carried through every node of the Self-Correcting RAG graph.

    Fields
    ------
    question    : The current user question (or rewritten question).
    context     : List of text passages currently available as context.
    generation  : The final synthesized answer.
    retry_count : Number of times the grader has returned "irrelevant".
    hint        : Optional hint provided by the user.
    source      : "vectorstore" | "web".
    history     : List of past messages for conversational memory.
    """

    question: str
    context: List[Any]  # Changed to Any to support Document objects with metadata
    generation: str
    retry_count: int
    hint: Optional[str]
    source: str
    history: List[BaseMessage]
