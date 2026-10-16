"""
RAG Chain module for AthleteIQ.
Provides prompt templates, unified LLM backends (Groq, Ollama, Offline Sports Science),
and conversational retrieval pipelines with chat memory and citation tracking.
"""

from .prompts import AthleteIQPrompts
from .llm_backend import LLMBackendFactory, BaseLLMBackend
from .conversational_chain import ConversationalRAGChain

__all__ = ["AthleteIQPrompts", "LLMBackendFactory", "BaseLLMBackend", "ConversationalRAGChain"]
