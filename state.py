from typing import Literal, Required, TypedDict

# These values match the three worker names registered in the graph.
Route = Literal["RAG", "LLM", "WEB"]


class AgentState(TypedDict, total=False):
    # The question is required from the start.
    question: Required[str]

    # The supervisor fills these fields when it chooses a worker.
    route: Route
    route_reasoning: str
    attempts: int
    used_routes: list[Route]

    # The selected worker writes the draft and any supporting context.
    context: str
    draft_answer: str

    # The validator records its decision, score, and feedback.
    validation_passed: bool
    validation_score: int
    validation_feedback: str

    # Only a terminal node writes the user-facing answer.
    final_answer: str