from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_llm():
    # Use the same Groq model for routing, answering, and validation.
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        reasoning_format="hidden",
    )


def create_retriever(pdf_path: Path):
    # Load the PDF and split its text using the notebook's chunk settings.
    documents = PyMuPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("The PDF has no extractable text.")

    # Embed the chunks locally and store them in an in-memory collection.
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
    )
    vector_store = Chroma(embedding_function=embeddings)
    vector_store.add_documents(chunks)

    # Return the three most relevant chunks for each question.
    return vector_store.as_retriever(search_kwargs={"k": 3})


def create_web_search():
    # Tavily reads TAVILY_API_KEY from the environment.
    return TavilySearch(
        max_results=5,
        search_depth="advanced",
        topic="general",
    )