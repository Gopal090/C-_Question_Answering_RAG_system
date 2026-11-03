"""
embedder.py
-----------
Responsible for Step 3 of RAG: Embeddings.
Converts human-readable text into dense numerical vectors (arrays of floats).

How Embeddings Work:
- An embedding model maps words and sentences into a mathematical coordinate space.
- Sentences with similar meanings (e.g. "What is inheritance?" and "Derived classes in OOP")
  will end up close to each other in this space, even if they use completely different words!
- We use 'all-MiniLM-L6-v2' because it runs locally on your machine for free,
  requires no API keys or internet connection once downloaded, and is very fast.
"""


from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the sentence transformer embedding model.

        Parameters:
            model_name (str): The Hugging Face model identifier.
        """
        self.model_name = model_name
        # Load the model into memory
        self.model = SentenceTransformer(model_name)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = True
    ) -> list[list[float]]:
        """
        Embeds a list of text passages into vectors in batches.

        Parameters:
            texts (List[str]): List of chunk texts to embed.
            batch_size (int): Number of passages to process together.
            show_progress_bar (bool): Whether to show the encoding progress bar.

        Returns:
            List[List[float]]: A 2D list where each element is a 384-dimensional vector.
        """
        # normalize_embeddings=True makes cosine similarity equivalent to dot product
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embeds a single search query into a vector.

        Parameters:
            query (str): The user's question or search query.

        Returns:
            List[float]: A single 384-dimensional vector.
        """
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embedding.tolist()

