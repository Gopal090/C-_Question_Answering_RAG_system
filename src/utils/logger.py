"""
logger.py
---------
Simple, clean console logging helpers with color/tag formatting.
Helps beginners clearly see what is happening at each step of the RAG pipeline.
"""


def log_info(message: str) -> None:
    """Print an informational message."""
    print(f"[INFO] {message}")


def log_success(message: str) -> None:
    """Print a success confirmation message."""
    print(f"[SUCCESS] {message}")


def log_warning(message: str) -> None:
    """Print a warning message."""
    print(f"[WARNING] {message}")


def log_error(message: str) -> None:
    """Print an error message."""
    print(f"[ERROR] {message}")

