from functools import partial

from langgraph.graph import END, START, StateGraph

from .outcomes import failed_node, final_node
from .routing import router, validation_router
from .schemas import RouteDecision, ValidationResult
from .state import AgentState
from .supervisor import supervisor_node
from .validator import validator_node
from .workers import llm_node, rag_node, web_node


def build_graph(llm, retriever, web_search, knowledge_base_description: str):
    # Configure the structured responses used by the supervisor and validator.
    supervisor_model = llm.with_structured_output(
        schema=RouteDecision.model_json_schema(),
        method="json_schema",
    )
    validator_model = llm.with_structured_output(
        schema=ValidationResult.model_json_schema(),
        method="json_schema",
    )

    workflow = StateGraph(AgentState)

    # Bind each node's services so LangGraph can call it with just the state.
    workflow.add_node(
        "Supervisor",
        partial(
            supervisor_node,
            supervisor_model=supervisor_model,
            knowledge_base_description=knowledge_base_description,
        ),
    )
    workflow.add_node("RAG", partial(rag_node, retriever=retriever, llm=llm))
    workflow.add_node("LLM", partial(llm_node, llm=llm))
    workflow.add_node("WEB", partial(web_node, web_search=web_search, llm=llm))
    workflow.add_node(
        "Validator",
        partial(validator_node, validator_model=validator_model),
    )
    workflow.add_node("Final", final_node)
    workflow.add_node("Failed", failed_node)

    # START → Supervisor → one selected worker.
    workflow.add_edge(START, "Supervisor")
    workflow.add_conditional_edges(
        "Supervisor",
        router,
        {"RAG": "RAG", "LLM": "LLM", "WEB": "WEB"},
    )

    # Every worker sends its draft to the same validator.
    for worker in ("RAG", "LLM", "WEB"):
        workflow.add_edge(worker, "Validator")

    # Validation can accept the draft, retry, or end with a fallback.
    workflow.add_conditional_edges(
        "Validator",
        validation_router,
        {
            "Final": "Final",
            "Supervisor": "Supervisor",
            "Failed": "Failed",
        },
    )
    workflow.add_edge("Final", END)
    workflow.add_edge("Failed", END)

    return workflow.compile()