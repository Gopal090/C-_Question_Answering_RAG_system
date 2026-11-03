"""
retriever.py
------------
Responsible for Step 5 of RAG: Retrieval.
Connects the user's natural language question to the vector database:
1. Takes the user's question string.
2. Converts the question into an embedding vector.
3. Queries ChromaDB for the top-K most semantically similar passages from the C++ book.
"""


from src.embeddings.embedder import EmbeddingModel
from src.vectordb.chroma_store import ChromaStore


class Retriever:
    def __init__(self, embedder: EmbeddingModel, chroma_store: ChromaStore, top_k: int = 4):
        """
        Initializes the Retriever.

        Parameters:
            embedder (EmbeddingModel): The local embedding model instance.
            chroma_store (ChromaStore): The persistent ChromaDB store instance.
            top_k (int): Number of most relevant book passages to retrieve.
        """
        self.embedder = embedder
        self.chroma_store = chroma_store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Performs semantic search to find the most relevant book chunks for a query.

        Parameters:
            query (str): The user's question (e.g. "What is method overloading?").
            top_k (int, optional): Override the default top_k value if desired.

        Returns:
            List[Dict]: The list of top matching passages with text and page numbers.
        """
        k = top_k if top_k is not None else self.top_k

        # Step 1: Turn the user's natural language question into a vector
        query_vector = self.embedder.embed_query(query)

        # Step 2: Search ChromaDB for the closest vectors
        results = self.chroma_store.search(query_vector, top_k=k)

        return results

