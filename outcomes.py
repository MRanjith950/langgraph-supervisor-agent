from .state import AgentState


def final_node(state: AgentState) -> dict:
    # Publish the draft that passed validation.
    return {"final_answer": state["draft_answer"]}


def failed_node(state: AgentState) -> dict:
    # Return a fallback message after the final unsuccessful attempt.
    return {
        "final_answer": (
            "The system could not generate a sufficiently "
            "validated answer after multiple attempts."
        )
    }