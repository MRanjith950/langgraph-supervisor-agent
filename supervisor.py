from .state import AgentState


def supervisor_node(
    state: AgentState,
    *,
    supervisor_model,
    knowledge_base_description: str,
) -> dict:
    # Read the question and any feedback from a previous attempt.
    question = state["question"]
    used_routes = state.get("used_routes", [])
    previous_feedback = state.get("validation_feedback", "")
    attempts = state.get("attempts", 0)

    prompt = f"""
Choose the best worker to answer the user's question.

RAG: Use the internal PDF when the question is about its contents.
The PDF covers: {knowledge_base_description}

WEB: Use web search for current facts, recent news, or information
that may have changed since the PDF was published.

LLM: Use the model directly for greetings, general knowledge,
and conceptual questions that do not require current information.

Question: {question}
Routes already attempted: {used_routes}
Previous validation feedback: {previous_feedback}

If a draft failed validation, use the feedback to decide whether
another source would help. Return the route and your reason.
"""

    decision = supervisor_model.invoke(prompt)
    route = decision["route"]

    # Count this attempt and clear the previous draft and its context.
    return {
        "route": route,
        "route_reasoning": decision["reasoning"],
        "attempts": attempts + 1,
        "used_routes": [*used_routes, route],
        "context": "",
        "draft_answer": "",
    }