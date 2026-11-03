"""
main.py
-------
Main Entrypoint for the C++ Book RAG Project.

How to run:
1. Normal run (starts Q&A loop, indexes only if not already indexed):
   python main.py

2. Force re-indexing (e.g. after changing chunk sizes):
   python main.py --reindex
"""

import io
import sys

# Configure UTF-8 encoding on Windows to prevent terminal charmap errors
if sys.platform.startswith("win"):
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.chunking.text_splitter import split_documents
from src.embeddings.embedder import EmbeddingModel
from src.ingestion.pdf_loader import load_pdf
from src.llm.groq_client import GroqClient
from src.prompts.rag_prompt import build_system_prompt, build_user_prompt
from src.retrieval.retriever import Retriever
from src.utils.config_loader import load_config
from src.utils.logger import log_error, log_info, log_success, log_warning
from src.vectordb.chroma_store import ChromaStore


def run_indexing_pipeline(config: dict, embedder: EmbeddingModel, chroma_store: ChromaStore) -> None:
    """
    Executes the one-time indexing pipeline:
    PDF Ingestion -> Chunking -> Embeddings -> ChromaDB Storage.
    """
    pdf_path = config["data"]["pdf_path"]
    chunk_size = config["chunking"]["chunk_size"]
    chunk_overlap = config["chunking"]["chunk_overlap"]
    batch_size = config["embeddings"]["batch_size"]

    log_info(f"Step 1/4: Loading PDF from '{pdf_path}'...")
    documents = load_pdf(pdf_path)
    log_success(f"Extracted text from {len(documents)} pages.")

    log_info(f"Step 2/4: Chunking text (size={chunk_size}, overlap={chunk_overlap})...")
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    log_success(f"Created {len(chunks)} text chunks.")

    log_info(f"Step 3/4: Generating vector embeddings with '{config['embeddings']['model_name']}'...")
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_texts(chunk_texts, batch_size=batch_size, show_progress_bar=True)
    log_success(f"Generated {len(embeddings)} vectors.")

    log_info("Step 4/4: Storing chunks and vectors in ChromaDB...")
    chroma_store.add_documents(chunks, embeddings)
    log_success(f"Successfully indexed {len(chunks)} chunks into ChromaDB at '{chroma_store.persist_dir}'.")


def main():
    print("=" * 65)
    print("📚  C++ Textbook Q&A Assistant (RAG Pipeline)")
    print("    Grounded in: 'Object Oriented Programming With C++'")
    print("=" * 65)

    # 1. Load project settings
    config = load_config("config.yaml")

    # 2. Initialize Embedder and ChromaDB
    embed_model_name = config["embeddings"]["model_name"]
    log_info(f"Loading embedding model '{embed_model_name}'...")
    embedder = EmbeddingModel(model_name=embed_model_name)

    chroma_dir = config["data"]["chroma_db_dir"]
    collection_name = config["data"]["collection_name"]
    chroma_store = ChromaStore(persist_dir=chroma_dir, collection_name=collection_name)

    # 3. Check if indexing is needed
    force_reindex = "--reindex" in sys.argv
    if force_reindex or not chroma_store.is_indexed():
        if force_reindex:
            log_warning("Flag '--reindex' detected. Re-indexing document...")
        else:
            log_info("Vector database is empty. Starting initial one-time indexing...")
        run_indexing_pipeline(config, embedder, chroma_store)
    else:
        log_success(f"Found existing vector database with {chroma_store.get_count()} indexed chunks.")

    # 4. Initialize Retriever and Prompts
    retriever = Retriever(
        embedder=embedder,
        chroma_store=chroma_store,
        top_k=config["retrieval"]["top_k"],
    )
    system_prompt = build_system_prompt()

    # 5. Initialize Groq LLM Client
    try:
        groq_client = GroqClient()
    except ValueError as e:
        log_error(str(e))
        print("\nPlease update '.env' with your Groq API key and rerun the program.")
        sys.exit(1)

    llm_model = config["llm"]["model_name"]
    temperature = config["llm"]["temperature"]
    max_tokens = config["llm"]["max_tokens"]

    print("\n" + "-" * 65)
    print("💡 Ready for your questions! Type 'exit' or 'quit' to stop.")
    print("-" * 65 + "\n")

    # 6. Interactive Q&A loop
    while True:
        try:
            query = input("❓ Question: ").strip()
            if not query:
                continue

            if query.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye! Happy coding in C++! 👋")
                break

            print("\n🔍 Searching textbook for relevant passages...")
            retrieved_chunks = retriever.retrieve(query)

            if not retrieved_chunks:
                print("⚠️  No relevant sections found in the textbook for your query.\n")
                continue

            # Display which pages were retrieved
            pages = sorted({chunk["page_number"] for chunk in retrieved_chunks})
            pages_str = ", ".join(f"Page {p}" for p in pages)
            print(f"📖 Context found in: {pages_str}\n")

            # Augment prompt with retrieved excerpts
            user_prompt = build_user_prompt(query, retrieved_chunks)

            # Generate and stream answer using Groq
            print("🤖 Answer:\n")
            for token in groq_client.stream(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                print(token, end="", flush=True)
            print("\n" + "-" * 65 + "\n")

        except KeyboardInterrupt:
            print("\n\nSession interrupted by user. Exiting...")
            break
        except Exception as err:  # noqa: BLE001
            log_error(f"An unexpected error occurred: {err}\n")


if __name__ == "__main__":
    main()
