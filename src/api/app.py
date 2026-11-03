"""
app.py
------
Optional REST API built with FastAPI.
Allows other applications (web frontends, mobile apps, or Postman)
to query your C++ RAG assistant over HTTP.

How to run:
uvicorn src.api.app:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.embeddings.embedder import EmbeddingModel
from src.llm.groq_client import GroqClient
from src.prompts.rag_prompt import build_system_prompt, build_user_prompt
from src.retrieval.retriever import Retriever
from src.utils.config_loader import load_config
from src.utils.logger import log_warning
from src.vectordb.chroma_store import ChromaStore

# Global state dictionary for application lifespan
app_state: dict[str, Any] = {
    "config": None,
    "embedder": None,
    "chroma_store": None,
    "retriever": None,
    "groq_client": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes models and stores on server startup,
    and cleans up resources on shutdown.
    """
    config = load_config("config.yaml")
    app_state["config"] = config

    embedder = EmbeddingModel(model_name=config["embeddings"]["model_name"])
    chroma_store = ChromaStore(
        persist_dir=config["data"]["chroma_db_dir"],
        collection_name=config["data"]["collection_name"],
    )
    retriever = Retriever(
        embedder=embedder,
        chroma_store=chroma_store,
        top_k=config["retrieval"]["top_k"],
    )

    groq_client = None
    try:
        groq_client = GroqClient()
    except (ValueError, KeyError) as e:
        log_warning(f"Groq client not initialized: {e}")

    app_state["embedder"] = embedder
    app_state["chroma_store"] = chroma_store
    app_state["retriever"] = retriever
    app_state["groq_client"] = groq_client

    yield

    app_state.clear()


app = FastAPI(
    title="C++ Textbook RAG API",
    description="Ask questions about C++ and receive answers grounded in E. Balagurusamy's textbook.",
    version="1.0.0",
    lifespan=lifespan,
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]


@app.get("/health")
def health_check():
    """Health check endpoint to verify database and service status."""
    chroma_store = app_state.get("chroma_store")
    is_ready = chroma_store is not None and chroma_store.is_indexed()
    return {
        "status": "healthy" if is_ready else "not_indexed",
        "chunks_indexed": chroma_store.get_count() if chroma_store else 0,
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Submits a C++ question and returns a grounded answer with page citations.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chroma_store = app_state.get("chroma_store")
    if not chroma_store or not chroma_store.is_indexed():
        raise HTTPException(
            status_code=503,
            detail="Vector store is empty. Please run main.py first to index.",
        )

    groq_client = app_state.get("groq_client")
    if not groq_client:
        raise HTTPException(
            status_code=500,
            detail="Groq API client is not configured. Check your GROQ_API_KEY in .env.",
        )

    retriever = app_state["retriever"]
    config = app_state["config"]

    # 1. Retrieve relevant passages
    retrieved = retriever.retrieve(request.question)

    # 2. Build prompts
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(request.question, retrieved)

    # 3. Generate answer
    answer = groq_client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=config["llm"]["model_name"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )

    # Clean sources for response
    sources = [
        {"page_number": c["page_number"], "source": c["source"]} for c in retrieved
    ]

    return AnswerResponse(
        question=request.question,
        answer=answer,
        sources=sources,
    )
