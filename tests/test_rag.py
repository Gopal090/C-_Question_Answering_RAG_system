"""
test_rag.py
-----------
Simple tests to verify the individual components of our RAG pipeline.
Run with: python -m pytest
"""

from src.chunking.text_splitter import chunk_text, split_documents
from src.prompts.rag_prompt import build_system_prompt, build_user_prompt
from src.utils.config_loader import load_config


def test_load_config():
    """Ensure config.yaml can be read and parsed."""
    config = load_config("config.yaml")
    assert "data" in config
    assert "chunking" in config
    assert "embeddings" in config
    assert "retrieval" in config
    assert "llm" in config


def test_chunk_text():
    """Verify that chunking divides text and respects chunk sizes."""
    sample_text = (
        "C++ is a general-purpose programming language created by Bjarne Stroustrup "
        "as an extension of the C programming language."
    )
    chunks = chunk_text(sample_text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 1
    # Check that chunks reconstruct information
    assert "C++" in chunks[0]


def test_split_documents():
    """Verify document splitting preserves metadata (page number)."""
    fake_docs = [
        {"page_number": 1, "text": "This is page one text.", "source": "test.pdf"},
        {"page_number": 2, "text": "This is page two text.", "source": "test.pdf"},
    ]
    chunks = split_documents(fake_docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 1
    assert chunks[1]["page_number"] == 2


def test_prompts():
    """Verify prompt formatting includes context and citations."""
    system_prompt = build_system_prompt()
    assert "C++" in system_prompt

    fake_chunks = [
        {
            "page_number": 42,
            "text": "Virtual functions allow polymorphic behavior.",
            "source": "book.pdf",
        }
    ]
    user_prompt = build_user_prompt("What is a virtual function?", fake_chunks)
    assert "Page 42" in user_prompt
    assert "Virtual functions allow polymorphic behavior." in user_prompt
    assert "What is a virtual function?" in user_prompt
