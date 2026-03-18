"""
graph/graph.py — LangGraph StateGraph assembly.

Builds and returns the compiled Self-Correcting RAG graph.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from graph.state import GraphState
from graph.nodes import retrieve, grade_documents, web_search, generate, ask_for_hint
from graph.edges import route_after_grading


def build_graph():
    """
    Compile and return the Self-Correcting RAG StateGraph.

    Flow diagram:
        START
          │
          ▼
       retrieve
          │
          ▼
    grade_documents ──── relevant ───────────────► generate ─► END
          │                                                      ▲
          │  irrelevant + retries left                          │
          ▼                                                      │
      web_search ─► grade_documents ─── relevant ──────────────┘
                           │
                           │  irrelevant + retries exhausted
                           ▼
                     ask_for_hint ─► END
    """
    builder = StateGraph(GraphState)

    # Register nodes
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("web_search", web_search)
    builder.add_node("generate", generate)
    builder.add_node("ask_for_hint", ask_for_hint)

    # Entry point
    builder.set_entry_point("retrieve")

    # Static edges
    builder.add_edge("retrieve", "grade_documents")
    builder.add_edge("web_search", "grade_documents")  # re-grade web results
    builder.add_edge("generate", END)
    builder.add_edge("ask_for_hint", END)

    # Conditional edge — the self-correction decision point
    builder.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "web_search": "web_search",
            "ask_for_hint": "ask_for_hint",
        },
    )

    return builder.compile()
