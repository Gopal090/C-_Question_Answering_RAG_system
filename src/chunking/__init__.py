"""
Chunking module: splits documents into smaller overlapping passages.
"""
from .text_splitter import chunk_text, split_documents

__all__ = ["chunk_text", "split_documents"]

