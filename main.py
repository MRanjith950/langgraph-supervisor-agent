import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Preserve package-relative imports when this file is run directly.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from .graph import build_graph
from .services import create_llm, create_retriever, create_web_search
from .state import AgentState


def main() -> None:
    # Take the question, PDF path, and PDF description from the command line.
    parser = argparse.ArgumentParser(description="Run the supervisor agent.")
    parser.add_argument("question", help="Question to ask the agent")
    parser.add_argument("--pdf", required=True, type=Path, help="Path to your PDF")
    parser.add_argument(
        "--knowledge-base",
        required=True,
        help="Short description of what the PDF covers",
    )
    args = parser.parse_args()

    # Load API keys and check the PDF before initializing the services.
    load_dotenv()
    for key in ("GROQ_API_KEY", "TAVILY_API_KEY"):
        if not os.getenv(key):
            parser.error(f"Set {key} in your .env file or environment.")

    pdf_path = args.pdf.expanduser()
    if not pdf_path.is_file():
        parser.error(f"PDF file not found: {pdf_path}")

    app = build_graph(
        llm=create_llm(),
        retriever=create_retriever(pdf_path),
        web_search=create_web_search(),
        knowledge_base_description=args.knowledge_base,
    )

    # Start with a clean state. Each node adds its fields along the way.
    initial_state: AgentState = {
        "question": args.question,
        "attempts": 0,
        "used_routes": [],
    }
    result = app.invoke(initial_state, config={"recursion_limit": 30})

    # Print routing and validation details so you can see what happened.
    print("Routes tried:", " → ".join(result["used_routes"]))
    print("Validation score:", result["validation_score"])
    print("Validation feedback:", result["validation_feedback"])
    print("\nFinal answer")
    print(result["final_answer"])


if __name__ == "__main__":
    main()
