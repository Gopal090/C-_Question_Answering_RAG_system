"""
Prompts module: builds system and user prompts with retrieved context.
"""
from .rag_prompt import build_system_prompt, build_user_prompt, format_context

__all__ = ["build_system_prompt", "build_user_prompt", "format_context"]

