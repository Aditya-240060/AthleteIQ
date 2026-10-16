"""
Hybrid Ensemble Retriever for AthleteIQ.
Blends Dense Semantic Search (ChromaDB) with Sparse Keyword Search (BM25)
using Reciprocal Rank Fusion (RRF) and Weighted Score Ensemble.
"""

from typing import List, Dict, Any, Optional, Tuple
from ..vector_engine.db_manager import VectorDBManager
from .bm25_retriever import AthleteIQBM25Retriever


class HybridRetriever:
    """
    Ensemble retriever combining dense vector retrieval with BM25 sparse search.
    Guarantees both deep conceptual understanding and exact keyword precision.
    """

    def __init__(
        self,
        db_manager: VectorDBManager,
        bm25_retriever: Optional[AthleteIQBM25Retriever] = None,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ):
        """
        Args:
            db_manager: Instance of VectorDBManager for dense retrieval.
            bm25_retriever: Instance of AthleteIQBM25Retriever for sparse retrieval.
            alpha: Weight for dense vector search (1.0 = purely dense, 0.0 = purely BM25).
            rrf_k: Smoothing constant for Reciprocal Rank Fusion (default 60).
        """
        self.db_manager = db_manager
        self.bm25_retriever = bm25_retriever or AthleteIQBM25Retriever()
        self.alpha = alpha
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        where: Optional[Dict[str, Any]] = None,
        fusion_method: str = "rrf",
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Executes hybrid search and returns scored results.
        fusion_method: 'rrf' (Reciprocal Rank Fusion) or 'linear' (weighted scores).
        """
        # Fetch dense candidates
        candidate_k = max(top_k * 2, 10)
        dense_results = self.db_manager.similarity_search_with_score(query, k=candidate_k, where=where)
        
        # Fetch sparse BM25 candidates
        sparse_results = self.bm25_retriever.search(query, top_k=candidate_k, where=where)

        if not dense_results and not sparse_results:
            return []
        if not dense_results:
            return sparse_results[:top_k]
        if not sparse_results:
            return dense_results[:top_k]

        if fusion_method == "linear":
            return self._linear_score_fusion(dense_results, sparse_results, top_k)
        else:
            return self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Dict[str, Any], float]],
        sparse_results: List[Tuple[Dict[str, Any], float]],
        top_k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Combines ranked lists using Reciprocal Rank Fusion."""
        doc_pool = {}  # key -> (doc, rrf_score, dense_score, sparse_score)

        def get_doc_key(doc: Dict[str, Any]) -> str:
            meta = doc.get("metadata", {})
            return str(meta.get("chunk_id", doc.get("page_content", "")[:60]))

        # Process dense ranks
        for rank, (doc, score) in enumerate(dense_results):
            key = get_doc_key(doc)
            rrf_val = self.alpha * (1.0 / (self.rrf_k + rank + 1))
            if key not in doc_pool:
                doc_pool[key] = {
                    "doc": doc,
                    "rrf_score": rrf_val,
                    "dense_score": score,
                    "sparse_score": 0.0
                }
            else:
                doc_pool[key]["rrf_score"] += rrf_val
                doc_pool[key]["dense_score"] = score

        # Process sparse ranks
        for rank, (doc, score) in enumerate(sparse_results):
            key = get_doc_key(doc)
            rrf_val = (1.0 - self.alpha) * (1.0 / (self.rrf_k + rank + 1))
            if key not in doc_pool:
                doc_pool[key] = {
                    "doc": doc,
                    "rrf_score": rrf_val,
                    "dense_score": 0.0,
                    "sparse_score": score
                }
            else:
                doc_pool[key]["rrf_score"] += rrf_val
                doc_pool[key]["sparse_score"] = score

        # Sort descending by rrf_score
        sorted_items = sorted(doc_pool.values(), key=lambda x: x["rrf_score"], reverse=True)

        results = []
        max_rrf = sorted_items[0]["rrf_score"] if sorted_items else 1.0
        for item in sorted_items[:top_k]:
            normalized_score = round(item["rrf_score"] / max_rrf, 4)
            # Attach retrieval debugging scores into metadata
            item["doc"]["metadata"]["dense_score"] = item["dense_score"]
            item["doc"]["metadata"]["sparse_score"] = item["sparse_score"]
            item["doc"]["metadata"]["hybrid_score"] = normalized_score
            results.append((item["doc"], normalized_score))

        return results

    def _linear_score_fusion(
        self,
        dense_results: List[Tuple[Dict[str, Any], float]],
        sparse_results: List[Tuple[Dict[str, Any], float]],
        top_k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Combines scores using linear weighting."""
        doc_pool = {}

        def get_doc_key(doc: Dict[str, Any]) -> str:
            meta = doc.get("metadata", {})
            return str(meta.get("chunk_id", doc.get("page_content", "")[:60]))

        for doc, score in dense_results:
            key = get_doc_key(doc)
            doc_pool[key] = {"doc": doc, "dense_score": score, "sparse_score": 0.0}

        for doc, score in sparse_results:
            key = get_doc_key(doc)
            if key in doc_pool:
                doc_pool[key]["sparse_score"] = score
            else:
                doc_pool[key] = {"doc": doc, "dense_score": 0.0, "sparse_score": score}

        scored_items = []
        for item in doc_pool.values():
            comb = self.alpha * item["dense_score"] + (1.0 - self.alpha) * item["sparse_score"]
            comb = round(comb, 4)
            item["doc"]["metadata"]["dense_score"] = item["dense_score"]
            item["doc"]["metadata"]["sparse_score"] = item["sparse_score"]
            item["doc"]["metadata"]["hybrid_score"] = comb
            scored_items.append((item["doc"], comb))

        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items[:top_k]
