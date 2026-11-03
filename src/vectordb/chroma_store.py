"""
chroma_store.py
---------------
Responsible for Step 4 of RAG: Vector Database.
Manages persistent local storage of chunks, embeddings, and metadata using ChromaDB.

Why a Vector Database:
- Standard SQL databases search for exact keyword matches.
- A Vector Database searches by *semantic similarity* (geometric distance in vector space).
- By persisting the database to disk, we only index the 900+ page book once!
"""

import os
from typing import Any

import chromadb
from tqdm import tqdm


class ChromaStore:
    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        collection_name: str = "cpp_textbook"
    ) -> None:
        """
        Initializes or loads a persistent ChromaDB instance.

        Parameters:
            persist_dir (str): Directory path on disk where database files will be stored.
            collection_name (str): The name of the collection/table.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Ensure directory exists
        os.makedirs(persist_dir, exist_ok=True)

        # Initialize the persistent ChromaDB client
        self.client: Any = chromadb.PersistentClient(path=self.persist_dir)

        # Get or create the collection with cosine distance metric
        self.collection: Any = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def is_indexed(self) -> bool:
        """
        Checks whether the collection already contains indexed documents.

        Returns:
            bool: True if already populated, False otherwise.
        """
        return bool(self.collection.count() > 0)

    def get_count(self) -> int:
        """Returns the total number of chunks currently stored."""
        return int(self.collection.count())

    def add_documents(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        batch_size: int = 500
    ) -> None:
        """
        Stores chunks and their corresponding embeddings into ChromaDB in batches.

        Parameters:
            chunks (list[dict[str, Any]]): List of chunk dictionaries containing text and metadata.
            embeddings (list[list[float]]): Corresponding list of vector embeddings.
            batch_size (int): Number of items per batch to avoid database limits.
        """
        total_items = len(chunks)

        # Insert in batches with a progress bar
        for i in tqdm(range(0, total_items, batch_size), desc="Storing in ChromaDB"):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            ids = [str(chunk["chunk_id"]) for chunk in batch_chunks]
            documents = [str(chunk["text"]) for chunk in batch_chunks]
            metadatas = [
                {
                    "page_number": int(chunk["page_number"]),
                    "source": str(chunk["source"])
                }
                for chunk in batch_chunks
            ]

            self.collection.upsert(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents,
                metadatas=metadatas
            )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4
    ) -> list[dict[str, Any]]:
        """
        Finds the top_k most similar chunks for a given query vector.

        Parameters:
            query_embedding (list[float]): Vector representation of the user's question.
            top_k (int): Number of top passages to return.

        Returns:
            list[dict[str, Any]]: List of matched chunks with their text, metadata, and distance.
        """
        results: Any = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved: list[dict[str, Any]] = []
        if not results:
            return retrieved

        docs_list = results.get("documents")
        metas_list = results.get("metadatas")
        dists_list = results.get("distances")

        if docs_list and metas_list and dists_list:
            docs = docs_list[0]
            metas = metas_list[0]
            distances = dists_list[0]

            for doc, meta, dist in zip(docs, metas, distances):
                page_val = meta.get("page_number", "Unknown") if isinstance(meta, dict) else "Unknown"
                source_val = meta.get("source", "Unknown") if isinstance(meta, dict) else "Unknown"
                retrieved.append({
                    "text": doc,
                    "page_number": page_val,
                    "source": source_val,
                    "distance": dist  # Lower distance means higher similarity in cosine space
                })

        return retrieved
