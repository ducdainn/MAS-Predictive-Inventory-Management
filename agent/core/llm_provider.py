"""
Utility module providing cached access to LLM instances.
"""

from typing import Dict

from langchain_core.output_parsers import StrOutputParser  # noqa: F401
from langchain_core.prompts import ChatPromptTemplate  # noqa: F401
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI


class LLMProvider:
    """Manages LLM instances with caching."""

    def __init__(self):
        self._llm_cache: Dict[str, object] = {}

    def get_llm(self, model_type: str = "huggingface", temperature: float = 0.0):
        """Get LLM instance with caching."""
        cache_key = f"{model_type}_{temperature}"

        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        if model_type == "openai":
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
        elif model_type == "huggingface":
            endpoint = HuggingFaceEndpoint(
                repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
                task="text-generation",
                provider="hyperbolic"
            )
            llm = ChatHuggingFace(llm=endpoint)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self._llm_cache[cache_key] = llm
        return llm



