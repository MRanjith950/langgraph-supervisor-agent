import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")


def load_pdf_documents():
    """
    Load and parse a PDF document using PyMuPDFLoader.

    Returns:
        list[Document]: Loaded LangChain documents.
    """

    # Read PDF path from .env
    pdf_path = os.getenv("PDF_PATH")

    if not pdf_path:
        raise ValueError("PDF_PATH is missing in .env")

    # Resolve the PDF path
    pdf_file = Path(pdf_path)

    if not pdf_file.is_absolute():
        pdf_file = BASE_DIR / pdf_file

    # Check file exists
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    # Initialize PDF loader
    loader = PyMuPDFLoader(str(pdf_file))

    # Load and parse document
    documents = loader.load()

    return documents