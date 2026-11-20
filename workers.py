import json

from .state import AgentState


def rag_node(state: AgentState, *, retriever, llm) -> dict:
    # Find relevant PDF chunks and join them into one block of text.
    question = state["question"]
    documents = retriever.invoke(question)
    context = "\n\n".join(document.page_content for document in documents)

    prompt = f"""
Answer using only the retrieved PDF context.
If the context is insufficient, say so clearly.

Question: {question}

Retrieved context:
{context}
"""

    response = llm.invoke(prompt)
    return {"context": context, "draft_answer": response.content}


def llm_node(state: AgentState, *, llm) -> dict:
    # Answer from general knowledge without retrieving external context.
    question = state["question"]

    prompt = f"""
Answer the question clearly using general knowledge.
If the answer depends on current information you cannot verify,
say that it needs to be checked.

Question: {question}
"""

    response = llm.invoke(prompt)
    return {"context": "", "draft_answer": response.content}


def web_node(state: AgentState, *, web_search, llm) -> dict:
    # Search the web and keep the titles, URLs, and snippets for validation.
    question = state["question"]
    search_response = web_search.invoke({"query": question})

    if isinstance(search_response, dict):
        results = search_response.get("results", [])
        if results:
            context = "\n\n".join(
                f"Title: {result.get('title', '')}\n"
                f"URL: {result.get('url', '')}\n"
                f"Content: {result.get('content', '')}"
                for result in results
            )
        else:
            context = json.dumps(search_response, ensure_ascii=False, default=str)
    else:
        context = str(search_response)

    prompt = f"""
Answer the question using the web results below.
Include source names or URLs where useful.
If the results are insufficient, say so clearly.

Question: {question}

Web results:
{context}
"""

    response = llm.invoke(prompt)
    return {"context": context, "draft_answer": response.content}