"""
main.py — CLI entry point for the Self-Correcting RAG pipeline.

Usage:
  python main.py                        # interactive prompt
  python main.py "What is LangGraph?"   # pass question as argument
"""

from __future__ import annotations

import sys
import os
import time
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env before importing any LangChain/Gemini modules
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage  # noqa: E402
from graph.graph import build_graph  # noqa: E402

HINT_SIGNAL = "__HINT_REQUIRED__:"
SEPARATOR = "─" * 60


def print_banner():
    print("\n" + SEPARATOR)
    print("  🤖  Self-Correcting RAG  |  Conversational & Multi-Query")
    print(SEPARATOR + "\n")


def print_streaming(text: str, delay: float = 0.01):
    """Prints text character-by-character for a streaming effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def run_pipeline(question: str, history: List[BaseMessage], hint: Optional[str] = None) -> dict:
    """
    Build the graph and stream a single question through it.
    Returns the final state.
    """
    graph = build_graph()

    initial_state = {
        "question": question,
        "context": [],
        "generation": "",
        "retry_count": 0,
        "hint": hint,
        "source": "",
        "history": history,
    }

    final_state = graph.invoke(initial_state)
    return final_state


def interactive_loop():
    """Run an interactive conversational loop in the terminal."""
    print_banner()
    history: List[BaseMessage] = []

    while True:
        try:
            question = input("❓ Your question (or 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("👋 Goodbye!")
            break

        hint: Optional[str] = None
        
        while True:
            print(f"\n{SEPARATOR}")
            result = run_pipeline(question, history=history, hint=hint)
            generation = result.get("generation", "")

            # Check if the system is requesting a hint
            if generation.startswith(HINT_SIGNAL):
                system_message = generation[len(HINT_SIGNAL):]
                print(f"\n💬 System: {system_message}")
                try:
                    hint = input("💡 Your hint: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 Aborting.")
                    return

                if not hint:
                    print("⚠️ No hint provided. Skipping this question.")
                    break

                # Retry with hint
                print(f"\n🔄 Retrying with your hint...")
                continue

            # Normal answer received
            print(f"\n📝 Answer:\n{SEPARATOR}")
            print_streaming(generation)
            print(SEPARATOR + "\n")
            
            # Update history
            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=generation))
            
            # Keep history manageable (last 10 messages = 5 exchanges)
            if len(history) > 10:
                history = history[-10:]
            break


def main():
    # Validate required env vars
    missing = [k for k in ("GOOGLE_API_KEY", "TAVILY_API_KEY") if not os.environ.get(k)]
    if missing:
        print(f"\n❌ Missing environment variable(s): {', '.join(missing)}")
        print("   Please update your .env file and re-run.\n")
        sys.exit(1)

    # If a question is passed as a CLI argument, run it non-interactively
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print_banner()
        print(f"❓ Question: {question}\n")
        
        result = run_pipeline(question, history=[])
        generation = result.get("generation", "")

        if generation.startswith(HINT_SIGNAL):
            print(f"💬 {generation[len(HINT_SIGNAL):]}")
        else:
            print(f"\n📝 Answer:\n{SEPARATOR}")
            print_streaming(generation)
            print(SEPARATOR)
    else:
        interactive_loop()


if __name__ == "__main__":
    load_dotenv()
    main()
