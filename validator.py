from .state import AgentState


def validator_node(state: AgentState, *, validator_model) -> dict:
    # Give the validator the question, draft, and evidence from the worker.
    question = state["question"]
    route = state["route"]
    draft_answer = state["draft_answer"]
    context = state.get("context", "")

    prompt = f"""
Review the draft for relevance, correctness, completeness,
and unsupported claims.

For RAG and WEB, check that the claims are supported by the provided context.
For LLM, check that the draft does not present unverified current facts.

Pass only when the score is at least 7 out of 10 and the draft has
no major factual or grounding problems.

Question: {question}
Worker: {route}

Source context:
{context}

Draft answer:
{draft_answer}

Return passed, score, and feedback.
"""

    result = validator_model.invoke(prompt)

    # Save the decision for the router and the feedback for any retry.
    return {
        "validation_passed": result["passed"],
        "validation_score": result["score"],
        "validation_feedback": result["feedback"],
    }