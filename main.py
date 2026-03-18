"""
main.py — CLI entry point for the Self-Correcting RAG pipeline.

Usage:
  python main.py                        # interactive prompt
  python main.py "What is LangGraph?"   # pass question as argument
"""

from __future__ import annotations

import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env before importing any LangChain/Gemini modules
load_dotenv()

from graph.graph import build_graph  # noqa: E402  (must be after load_dotenv)

HINT_SIGNAL = "__HINT_REQUIRED__:"
SEPARATOR = "─" * 60


def print_banner():
    print("\n" + SEPARATOR)
    print("  🤖  Self-Correcting RAG  |  Powered by LangGraph + Gemini")
    print(SEPARATOR + "\n")


def run_pipeline(question: str, hint: str | None = None) -> str:
    """
    Build the graph and stream a single question through it.
    Returns the final generation string.
    """
    graph = build_graph()

    initial_state = {
        "question": question,
        "context": [],
        "generation": "",
        "retry_count": 0,
        "hint": hint,
        "source": "",
    }

    final_state = graph.invoke(initial_state)
    return final_state.get("generation", "")


def interactive_loop():
    """Run an interactive question-answer loop in the terminal."""
    print_banner()

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

        hint: str | None = None
        attempts = 0

        while True:
            attempts += 1
            print(f"\n{SEPARATOR}")
            generation = run_pipeline(question, hint=hint)

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
                    print("⚠️  No hint provided. Skipping this question.")
                    break

                # Reset and try again with the hint baked in
                print(f"\n🔄 Retrying with your hint...")
                continue

            # Normal answer received
            print(f"\n📝 Answer:\n{SEPARATOR}")
            print(generation)
            print(SEPARATOR + "\n")
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
        generation = run_pipeline(question)

        if generation.startswith(HINT_SIGNAL):
            print(f"💬 {generation[len(HINT_SIGNAL):]}")
        else:
            print(f"\n📝 Answer:\n{SEPARATOR}")
            print(generation)
            print(SEPARATOR)
    else:
        interactive_loop()


if __name__ == "__main__":
    main()
