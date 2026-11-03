"""
rag_prompt.py
-------------
Responsible for Step 6 of RAG: Prompt Engineering (Augmentation).
Combines the retrieved passages from the C++ textbook and the user's question
into a structured prompt for the Groq LLM.

Why Prompt Engineering matters in RAG:
- We explicitly instruct the LLM to ground its answer in the provided textbook context.
- We require the model to cite the specific page numbers from which it learned the facts.
- We tell it to admit when information is missing rather than hallucinating wrong C++ syntax.
"""



def build_system_prompt() -> str:
    """
    Returns the system instruction guiding the behavior of the C++ tutor.
    """
    return (
        "You are an expert, encouraging, and pedagogical C++ programming tutor.\n"
        "Your mission is to answer the student's question clearly and accurately, "
        "using ONLY the provided excerpts from the textbook 'Object Oriented Programming With C++'.\n\n"
        "Guidelines:\n"
        "1. Base your answer strictly on the provided Context excerpts.\n"
        "2. Always cite the relevant page numbers (e.g., '[Page 45]') when explaining concepts.\n"
        "3. Provide brief, correct C++ code snippets when helpful to clarify the concept.\n"
        "4. If the provided context does NOT contain enough information to answer the question, "
        "honestly state that the information is not present in the referenced textbook pages.\n"
        "5. Keep your tone polite, educational, and easy to understand."
    )


def format_context(retrieved_chunks: list[dict]) -> str:
    """
    Formats the list of retrieved chunk dictionaries into a readable string block.

    Parameters:
        retrieved_chunks (List[Dict]): Chunks returned by the Retriever.

    Returns:
        str: Cleanly formatted text block with page numbers.
    """
    if not retrieved_chunks:
        return "No relevant excerpts found in the book."

    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        page_num = chunk.get("page_number", "Unknown")
        text = chunk.get("text", "").strip()
        block = f"[Excerpt {i} - Page {page_num}]:\n{text}"
        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)


def build_user_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Constructs the final prompt sent to the Groq LLM, combining
    the retrieved book context and the student's question.

    Parameters:
        question (str): The user's question.
        retrieved_chunks (List[Dict]): Matching passages from ChromaDB.

    Returns:
        str: The complete user prompt.
    """
    context_str = format_context(retrieved_chunks)

    user_prompt = f"""Use the following textbook excerpts to answer the question.

### Book Context:
{context_str}

### Student Question:
{question}

### Answer:
"""
    return user_prompt

