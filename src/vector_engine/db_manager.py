"""
ChromaDB Database Manager for AthleteIQ.
Handles persistent vector indexing, metadata filtering, and similarity search.
"""

import os
import shutil
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from .embeddings import AthleteIQEmbeddings


class VectorDBManager:
    """
    Manages persistent local ChromaDB storage for cricket research papers,
    GPS telemetry logs, and confidential clinical notes.
    """

    def __init__(
        self,
        db_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[AthleteIQEmbeddings] = None,
    ):
        self.db_dir = db_dir or os.getenv("VECTOR_DB_DIR", "./vectorstore")
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "athlete_iq_cricket")
        self.embedder = embedding_model or AthleteIQEmbeddings()

        os.makedirs(self.db_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Initializes or accesses the Chroma collection."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "AthleteIQ Cricket Sports Science & Biomechanics Vector Index", "hnsw:space": "cosine"}
        )

    def reset_collection(self):
        """Deletes the existing collection and creates an empty one."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self._get_or_create_collection()
        print(f"[VectorDBManager] Collection '{self.collection_name}' has been reset.")

    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 64) -> int:
        """
        Embeds and stores documents in ChromaDB in batches.
        Documents must be dicts with 'page_content' and 'metadata'.
        """
        if not documents:
            return 0

        total_added = 0
        texts = [doc["page_content"] for doc in documents]
        
        # Compute embeddings
        print(f"[VectorDBManager] Generating embeddings for {len(texts)} document chunks...")
        embeddings = self.embedder.embed_documents(texts)

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            batch_ids = []
            batch_metadatas = []

            for idx, doc in enumerate(batch_docs):
                meta = doc.get("metadata", {}).copy()
                doc_id = meta.get("chunk_id", f"chunk_{i + idx}")
                
                # ChromaDB requires metadata values to be str, int, float, or bool
                sanitized_meta = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        sanitized_meta[k] = v
                    elif v is None:
                        sanitized_meta[k] = ""
                    else:
                        sanitized_meta[k] = str(v)

                batch_ids.append(str(doc_id))
                batch_metadatas.append(sanitized_meta)

            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )
            total_added += len(batch_docs)

        print(f"[VectorDBManager] Successfully indexed {total_added} chunks into '{self.collection_name}'.")
        return total_added

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs vector similarity search against the query and returns
        tuples of (document_dict, similarity_score).
        """
        if self.count() == 0:
            return []

        query_emb = self.embedder.embed_query(query)
        kwargs = {
            "query_embeddings": [query_emb],
            "n_results": min(k, self.count()),
            "include": ["documents", "metadatas", "distances"]
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        items = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for d, m, dist in zip(docs, metas, distances):
                # Cosine distance: similarity = 1 - distance
                similarity = max(0.0, min(1.0, 1.0 - dist))
                items.append(({"page_content": d, "metadata": m}, round(similarity, 4)))

        return items

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Returns just the top-k document dicts matching the query."""
        results_with_scores = self.similarity_search_with_score(query, k=k, where=where)
        return [doc for doc, _ in results_with_scores]

    def count(self) -> int:
        """Returns total number of chunks stored in the collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Provides high-level collection stats for UI display."""
        total = self.count()
        doc_types = {}
        if total > 0:
            try:
                # Sample collection metadatas
                data = self.collection.get(limit=min(total, 500), include=["metadatas"])
                if data and "metadatas" in data:
                    for m in data["metadatas"]:
                        dt = m.get("document_type", "unknown")
                        doc_types[dt] = doc_types.get(dt, 0) + 1
            except Exception:
                pass

        return {
            "collection_name": self.collection_name,
            "total_chunks": total,
            "document_types": doc_types,
            "storage_path": os.path.abspath(self.db_dir)
        }
