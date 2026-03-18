"""
graph/nodes.py — All LangGraph node implementations.

Nodes:
  - retrieve          : ChromaDB similarity search
  - grade_documents   : Gemini Flash with Pydantic structured grader
  - web_search        : Tavily live web search fallback
  - generate          : Gemini Pro final answer synthesis
  - ask_for_hint      : Halts and requests user input after MAX_RETRIES
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from tavily import TavilyClient

from graph.state import GraphState

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_RETRIES = 2
TOP_K = 4
CHROMA_DB_DIR = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "rag_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ─── Shared singletons (lazy-initialised once) ────────────────────────────────
_vectorstore: Chroma | None = None
_tavily: TavilyClient | None = None
_grader_llm: ChatGoogleGenerativeAI | None = None
_generator_llm: ChatGoogleGenerativeAI | None = None


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_DIR,
        )
    return _vectorstore


def _get_tavily() -> TavilyClient:
    global _tavily
    if _tavily is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise EnvironmentError("TAVILY_API_KEY is not set in your .env file.")
        _tavily = TavilyClient(api_key=api_key)
    return _tavily


def _get_grader_llm() -> ChatGoogleGenerativeAI:
    """Gemini Flash — fast, cheap, structured grader."""
    global _grader_llm
    if _grader_llm is None:
        _grader_llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0,
        )
    return _grader_llm


def _get_generator_llm() -> ChatGoogleGenerativeAI:
    """Gemini Pro (using Flash latest as fallback) — high-quality final answer generator."""
    global _generator_llm
    if _generator_llm is None:
        _generator_llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0.3,
        )
    return _generator_llm


# ─── Pydantic schema for grader ───────────────────────────────────────────────
class GradeDocument(BaseModel):
    """Binary relevance score for a retrieved document chunk."""

    score: Literal["relevant", "irrelevant"] = Field(
        description="'relevant' if the document helps answer the question, 'irrelevant' otherwise."
    )
    reasoning: str = Field(description="One-sentence justification for the score.")


# ─── Nodes ────────────────────────────────────────────────────────────────────

def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve top-k documents from the local ChromaDB vectorstore.
    """
    print("\n🔍 [Node: retrieve] Searching ChromaDB...")
    question = state["question"]
    hint = state.get("hint")

    # Augment query with hint if provided
    query = f"{question} {hint}" if hint else question

    vs = _get_vectorstore()
    docs = vs.similarity_search(query, k=TOP_K)

    if not docs:
        print("   ⚠️  No documents found in vectorstore.")

    context = [doc.page_content for doc in docs]
    print(f"   ✅ Retrieved {len(context)} chunk(s).")

    return {**state, "context": context, "source": "vectorstore"}


def grade_documents(state: GraphState) -> GraphState:
    """
    Grade each document in state['context'] using Gemini Flash.
    Keeps only 'relevant' passages. Increments retry_count if none pass.
    """
    print("\n🧑‍⚖️  [Node: grade_documents] Grading retrieved documents...")
    question = state["question"]
    context = state["context"]
    retry_count = state.get("retry_count", 0)

    grader_llm = _get_grader_llm()
    structured_grader = grader_llm.with_structured_output(GradeDocument)

    system_prompt = (
        "You are an expert relevance grader. Given a user question and a document passage, "
        "determine if the passage is relevant to answering the question. "
        "Be strict: passages that are off-topic, too vague, or don't contribute meaningful "
        "information should be marked 'irrelevant'."
    )
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {question}\n\nDocument Passage:\n{document}"),
    ])
    chain = grader_prompt | structured_grader

    relevant_docs: list[str] = []
    for i, doc in enumerate(context):
        result: GradeDocument = chain.invoke({"question": question, "document": doc})
        status_icon = "✅" if result.score == "relevant" else "❌"
        print(f"   {status_icon} Chunk {i + 1}: {result.score} — {result.reasoning}")
        if result.score == "relevant":
            relevant_docs.append(doc)

    if not relevant_docs:
        new_retry_count = retry_count + 1
        print(f"   ⚠️  No relevant documents. Retry count: {new_retry_count}/{MAX_RETRIES}")
        return {**state, "context": [], "retry_count": new_retry_count}

    print(f"   ✅ {len(relevant_docs)} relevant chunk(s) passed grading.")
    return {**state, "context": relevant_docs, "retry_count": retry_count}


def web_search(state: GraphState) -> GraphState:
    """
    Fall back to Tavily web search when local docs are irrelevant.
    Replaces context with live web results.
    """
    print("\n🌐 [Node: web_search] Searching the web via Tavily...")
    question = state["question"]
    hint = state.get("hint")
    query = f"{question} {hint}" if hint else question

    tavily = _get_tavily()
    results = tavily.search(query=query, max_results=5)

    web_context = [
        f"[{r.get('title', 'Web Result')}]\n{r.get('content', '')}"
        for r in results.get("results", [])
    ]
    print(f"   ✅ Fetched {len(web_context)} web result(s).")
    return {**state, "context": web_context, "source": "web"}


def generate(state: GraphState) -> GraphState:
    """
    Synthesize a final answer using Gemini Pro from verified context.
    """
    print("\n✍️  [Node: generate] Generating answer with Gemini Pro...")
    question = state["question"]
    context = state["context"]
    source = state.get("source", "unknown")

    context_str = "\n\n---\n\n".join(context) if context else "No context available."

    system_prompt = (
        "You are a knowledgeable assistant. Answer the user's question using ONLY "
        "the provided context. If the context doesn't fully answer the question, say so. "
        "Be clear, concise, and cite which parts of the context support your answer. "
        f"Context source: {source}."
    )
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    chain = gen_prompt | _get_generator_llm() | StrOutputParser()

    answer = chain.invoke({"context": context_str, "question": question})
    print("   ✅ Answer generated.")
    return {**state, "generation": answer}


def ask_for_hint(state: GraphState) -> GraphState:
    """
    Called after MAX_RETRIES consecutive grading failures.
    Returns a special generation string signalling the caller to prompt the user.
    """
    print(
        f"\n🛑 [Node: ask_for_hint] Grading failed {MAX_RETRIES} time(s). "
        "Asking user for a hint."
    )
    message = (
        f"I was unable to find relevant information after {MAX_RETRIES} attempts "
        "(both from the knowledge base and the web). "
        "Could you please provide a hint or rephrase your question?"
    )
    return {**state, "generation": f"__HINT_REQUIRED__:{message}"}
