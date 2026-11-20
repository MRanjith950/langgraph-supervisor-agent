from typing import Literal

from .state import AgentState, Route

MAX_ATTEMPTS = 3


def router(state: AgentState) -> Route:
    # Follow the route selected by the supervisor.
    return state["route"]


def validation_router(
    state: AgentState,
) -> Literal["Final", "Failed", "Supervisor"]:
    # Accept a validated answer even if it passed on the last attempt.
    if state["validation_passed"]:
        return "Final"

    # Stop when the attempt limit has been reached.
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "Failed"

    # Send the graph back to the supervisor with the validation feedback.
    return "Supervisor"