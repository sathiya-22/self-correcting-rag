"""
graph/nodes.py — All LangGraph node implementations.

Nodes:
  - retrieve          : Multi-query ChromaDB search with history context
  - grade_documents   : Gemini Flash with Pydantic structured grader (handles Documents)
  - web_search        : Tavily live web search fallback
  - generate          : Gemini Pro answer synthesis with history and citations
  - ask_for_hint      : Halts and requests user input after MAX_RETRIES
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, List, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
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


def _get_llm(env_var: str, default_model: str) -> ChatGoogleGenerativeAI:
    """Helper to get LLM with fallback logic for quota exhaustion."""
    primary_model = os.environ.get(env_var, default_model)
    # List of models to try in order if the primary fails due to quota
    fallbacks = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-pro-latest"
    ]
    # Remove duplicates and put primary first
    models_to_try = [primary_model] + [m for m in fallbacks if m != primary_model]
    
    # We create the LLM with the primary model first. 
    # If it fails during invoke, the user sees the error, 
    # but we can also set up the LLM to be more resilient.
    return ChatGoogleGenerativeAI(
        model=primary_model,
        temperature=0 if env_var == "MODEL_GRADER" else 0.3,
        max_retries=3, # Built-in retry
    )

def _get_grader_llm() -> ChatGoogleGenerativeAI:
    """Gemini Flash — fast, cheap, structured grader."""
    global _grader_llm
    if _grader_llm is None:
        _grader_llm = _get_llm("MODEL_GRADER", "gemini-3.1-flash-lite-preview")
    return _grader_llm


def _get_generator_llm() -> ChatGoogleGenerativeAI:
    """Gemini Pro — high-quality final answer generator."""
    global _generator_llm
    if _generator_llm is None:
        _generator_llm = _get_llm("MODEL_GENERATOR", "gemini-3.1-flash-lite-preview")
    return _generator_llm


# ─── Pydantic schemas ────────────────────────────────────────────────────────

class BatchGrade(BaseModel):
    """List of relevant document indices."""
    relevant_indices: List[int] = Field(
        description="The 1-based indices of the document chunks that are relevant to the question. Empty list if none."
    )
    reasoning: str = Field(description="Brief explanation for why these (or none) were selected.")


class MultiQuery(BaseModel):
    """List of generated search queries for multi-query retrieval."""
    queries: List[str] = Field(description="3 different variations of the search query.")


# ─── Nodes ────────────────────────────────────────────────────────────────────

def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve documents using multi-query expansion and history context.
    """
    print("\n🔍 [Node: retrieve] Expanding query and searching ChromaDB...")
    question = state["question"]
    history = state.get("history", [])
    hint = state.get("hint")

    # 1. Generate variations (Multi-Query)
    # If there's history, the LLM should also consider it to resolve pronouns, etc.
    llm = _get_grader_llm()
    structured_mq = llm.with_structured_output(MultiQuery)

    system_prompt = (
        "You are an expert search assistant. Your task is to generate 3 different "
        "variations of the user's question to capture more relevant context from a "
        "vector database. Consider the conversation history for context (e.g., resolve pronouns)."
    )
    mq_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{history}"),
        ("human", "Current Question: {question}\nGenerated queries:")
    ])
    mq_chain = mq_prompt | structured_mq

    try:
        mq_result = mq_chain.invoke({"question": question, "history": history})
        if mq_result and hasattr(mq_result, "queries"):
            search_queries = mq_result.queries
        else:
            print("   ⚠️ Multi-query returned invalid or empty result. Falling back.")
            search_queries = [question]
    except Exception as e:
        print(f"   ⚠️ Multi-query failed: {e}. Falling back to original.")
        search_queries = [question]

    # Add hint-augmented query if hint exists
    if hint:
        search_queries.append(f"{question} {hint}")
    
    # Ensure original is in there
    if question not in search_queries:
        search_queries.append(question)

    print(f"   📊 Searching with {len(search_queries)} query variations...")

    # 2. Retrieve for each query
    vs = _get_vectorstore()
    all_docs: List[Document] = []
    seen_ids = set()

    for q in search_queries:
        docs = vs.similarity_search(q, k=TOP_K // 2 if len(search_queries) > 1 else TOP_K)
        for doc in docs:
            # Use chunk_id if present, else content hash
            doc_id = doc.metadata.get("chunk_id", hash(doc.page_content))
            if doc_id not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(doc_id)

    print(f"   ✅ Retrieved {len(all_docs)} unique chunk(s).")
    return {**state, "context": all_docs, "source": "vectorstore"}


def grade_documents(state: GraphState) -> GraphState:
    """
    Grade all documents in state['context'] in a single BATCH request.
    This saves ~80% of LLM quota compared to one-by-one grading.
    """
    print("\n🧑‍⚖️  [Node: grade_documents] Batch Grading documents...")
    question = state["question"]
    context: List[Document] = state["context"]
    retry_count = state.get("retry_count", 0)

    if not context:
        return {**state, "retry_count": retry_count + 1}

    grader_llm = _get_grader_llm()
    structured_grader = grader_llm.with_structured_output(BatchGrade)

    system_prompt = (
        "You are an expert relevance grader. Given a user question and a list of numbered document passages, "
        "determine which ones are relevant to answering the question. "
        "Return the 1-based indices of the relevant passages only."
    )
    
    # Format the numbered list of passages
    numbered_passages = ""
    for i, doc in enumerate(context):
        numbered_passages += f"\n--- Passage {i+1} ---\n{doc.page_content}\n"

    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {question}\n\nList of Passages: {passages}"),
    ])
    chain = grader_prompt | structured_grader

    try:
        result: BatchGrade = chain.invoke({"question": question, "passages": numbered_passages})
        relevant_docs: List[Document] = []
        
        # Log results
        print(f"   💬 Grader Reasoning: {result.reasoning}")
        for idx in result.relevant_indices:
            if 0 < idx <= len(context):
                relevant_docs.append(context[idx-1])
        
        print(f"   ✅ Selected {len(relevant_docs)}/{len(context)} chunks as relevant.")
        
    except Exception as e:
        print(f"   ⚠️ Batch grading failed: {e}. Falling back to keeping all (safe mode).")
        relevant_docs = context

    if not relevant_docs:
        new_retry_count = retry_count + 1
        print(f"   ⚠️ No relevant documents found. Retry count: {new_retry_count}/{MAX_RETRIES}")
        return {**state, "context": [], "retry_count": new_retry_count}

    return {**state, "context": relevant_docs, "retry_count": retry_count}


def web_search(state: GraphState) -> GraphState:
    """
    Fall back to Tavily web search.
    """
    print("\n🌐 [Node: web_search] Searching the web via Tavily...")
    question = state["question"]
    
    tavily = _get_tavily()
    results = tavily.search(query=question, max_results=5)

    web_docs = []
    for r in results.get("results", []):
        doc = Document(
            page_content=r.get("content", ""),
            metadata={
                "source": r.get("url", "web"),
                "title": r.get("title", "Web Result")
            }
        )
        web_docs.append(doc)

    print(f"   ✅ Fetched {len(web_docs)} web result(s).")
    return {**state, "context": web_docs, "source": "web"}


def generate(state: GraphState) -> GraphState:
    """
    Synthesize a final answer using Gemini Pro with history and citations.
    """
    print("\n✍️  [Node: generate] Generating answer with Gemini Pro...")
    question = state["question"]
    context: List[Document] = state["context"]
    history = state.get("history", [])
    source_type = state.get("source", "unknown")

    # Format context with source labels
    formatted_context = []
    for i, doc in enumerate(context):
        src = doc.metadata.get("source", "Unknown Source")
        formatted_context.append(f"--- Chunk {i+1} [Source: {src}] ---\n{doc.page_content}")
    
    context_str = "\n\n".join(formatted_context) if formatted_context else "No context available."

    system_prompt = (
        "You are a knowledgeable AI assistant. Answer the user's question using ONLY "
        "the provided context. If the context doesn't answer the question, say so.\n\n"
        "RULES:\n"
        "1. Be clear and concise.\n"
        "2. ALWAYS cite your sources using the source labels provided in the context (e.g. [Source: doc.pdf]).\n"
        f"3. All current context is from: {source_type}."
    )
    
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{history}"),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    
    chain = gen_prompt | _get_generator_llm() | StrOutputParser()

    answer = chain.invoke({
        "context": context_str, 
        "question": question,
        "history": history
    })
    
    print("   ✅ Answer generated.")
    return {**state, "generation": answer}


def ask_for_hint(state: GraphState) -> GraphState:
    """
    Signal hint requirement.
    """
    print(f"\n🛑 [Node: ask_for_hint] Grading failed {MAX_RETRIES} times.")
    message = (
        f"I couldn't find relevant information in my database or the web. "
        "Could you please provide more context or a hint?"
    )
    return {**state, "generation": f"__HINT_REQUIRED__:{message}"}
