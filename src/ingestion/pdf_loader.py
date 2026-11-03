"""
pdf_loader.py
-------------
Responsible for Step 1 of RAG: Ingestion.
Extracts raw text from each page of a PDF document along with metadata (page number).
"""

import os

import pymupdf  # PyMuPDF is fast, robust, and lightweight


def load_pdf(pdf_path: str) -> list[dict]:
    """
    Reads a PDF file page-by-page and extracts clean text.

    Parameters:
        pdf_path (str): Path to the PDF file (e.g. 'C++ BOOK.pdf').

    Returns:
        List[Dict]: A list of page dictionaries, each containing:
            - 'page_number': int (1-based index)
            - 'text': str (cleaned text of the page)
            - 'source': str (the filename)

    Why this matters for RAG:
    -------------------------
    By keeping track of page numbers here, our RAG system can later
    tell the user exactly which page of the book the answer came from!
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    documents = []
    filename = os.path.basename(pdf_path)

    # Open the PDF using PyMuPDF
    doc = pymupdf.open(pdf_path)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = str(page.get_text())

        # Clean excess whitespace while preserving line structure
        cleaned_text = text.strip()

        # Only keep pages that actually contain readable text
        if cleaned_text:
            documents.append({
                "page_number": page_idx + 1,
                "text": cleaned_text,
                "source": filename
            })

    doc.close()
    return documents

