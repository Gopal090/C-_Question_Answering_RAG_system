"""
groq_client.py
--------------
Responsible for Step 7 of RAG: Generation.
Sends the augmented prompt to the Groq API and retrieves the LLM's response.

Why Groq?
- Groq runs open-source models (Llama 3.3 70B, Llama 3.1 8B) on custom LPU hardware.
- It delivers near-instantaneous token generation speeds (hundreds of tokens per second).
- It provides a free tier with generous rate limits for learning and building projects.
"""

import os
from collections.abc import Generator

from dotenv import load_dotenv
from groq import Groq


class GroqClient:
    def __init__(self, api_key: str | None = None):
        """
        Initializes the Groq API client.

        Parameters:
            api_key (str, optional): Groq API Key. If omitted, reads from the .env file.
        """
        # Load environment variables from .env file
        load_dotenv(override=True)

        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        # Validate that the user provided a real key
        if not self.api_key or "your_groq_api_key_here" in self.api_key:
            raise ValueError(
                "\n[ERROR] Missing Groq API Key!\n"
                "Please open the '.env' file in your project folder and add your key:\n"
                "  GROQ_API_KEY=gsk_your_actual_key_here\n"
                "You can get a free API key at: https://console.groq.com/keys"
            )

        self.client = Groq(api_key=self.api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generates a complete response (non-streaming).

        Returns:
            str: The LLM's generated answer text.
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content if content is not None else ""

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.2,
        max_tokens: int = 1024
    ) -> Generator[str, None, None]:
        """
        Streams response tokens one by one as they are generated.
        This allows the user in the CLI to see words appearing in real-time.

        Yields:
            str: Each chunk of text as it arrives.
        """
        stream_response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        for chunk in stream_response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
