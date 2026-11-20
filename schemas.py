from pydantic import BaseModel, Field

from .state import Route


class RouteDecision(BaseModel):
    # Restrict the supervisor to a route registered in the graph.
    route: Route = Field(description="Worker that should answer the question")
    reasoning: str = Field(description="Reason for choosing this worker")


class ValidationResult(BaseModel):
    # The validator returns a decision and feedback the supervisor can use.
    passed: bool = Field(description="Whether the draft passes validation")
    score: int = Field(ge=0, le=10, description="Answer quality from 0 to 10")
    feedback: str = Field(description="Reason for the validation decision")