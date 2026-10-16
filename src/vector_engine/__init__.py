"""
Vector Engine package for AthleteIQ.
Provides embedding models and local ChromaDB vectorstore management.
"""

from .embeddings import AthleteIQEmbeddings
from .db_manager import VectorDBManager

__all__ = ["AthleteIQEmbeddings", "VectorDBManager"]
