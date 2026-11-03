"""
text_splitter.py
----------------
Responsible for Step 2 of RAG: Chunking.
Breaks long documents/pages into smaller, overlapping chunks of text.

Why chunking is needed:
1. LLM context windows and embedding models work best on focused passages.
2. If text is too long, the semantic embedding becomes a diluted average of multiple concepts.
3. Overlap (e.g. 150 characters) ensures that a key sentence or definition
   split between two chunks is preserved in full in at least one of them.
"""



def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[str]:
    """
    Splits a single string of text into overlapping chunks,
    respecting word boundaries so words are not cut in half.

    Parameters:
        text (str): The raw text to split.
        chunk_size (int): Target maximum characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[str]: List of text chunk strings.
    """
    if not text:
        return []

    # If the text is already smaller than the desired chunk size, return it as is
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # If this chunk reaches past the end of the text, take the rest
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Avoid breaking words in the middle:
        # Find the nearest space or newline before the cutoff point
        cutoff = text.rfind(" ", start, end)
        newline_cutoff = text.rfind("\n", start, end)
        best_break = max(cutoff, newline_cutoff)

        # If a reasonable word boundary was found (at least half way in), break there
        if best_break > start + (chunk_size // 2):
            end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward by (chunk_size - chunk_overlap)
        # Ensure we always advance at least 1 character to avoid infinite loops
        step = max(1, (end - start) - chunk_overlap)
        start += step

    return chunks


def split_documents(
    documents: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> list[dict]:
    """
    Splits a list of page documents into smaller chunk dictionaries
    while preserving page numbers and source metadata.

    Parameters:
        documents (List[Dict]): List of page dictionaries from load_pdf.
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Overlap in characters.

    Returns:
        List[Dict]: List of chunk dictionaries, each containing:
            - 'chunk_id': unique string id (e.g. 'page_12_chunk_1')
            - 'text': the chunk's text content
            - 'page_number': the page number it originated from
            - 'source': source PDF filename
    """
    all_chunks = []

    for doc in documents:
        page_num = doc["page_number"]
        source = doc["source"]
        raw_text = doc["text"]

        text_pieces = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for idx, piece in enumerate(text_pieces):
            chunk_dict = {
                "chunk_id": f"p{page_num}_c{idx + 1}",
                "text": piece,
                "page_number": page_num,
                "source": source
            }
            all_chunks.append(chunk_dict)

    return all_chunks

