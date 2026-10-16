"""
BM25 Sparse Keyword Retriever for AthleteIQ.
Optimized for domain-specific medical, biomechanical, and cricket terminology.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi


class AthleteIQBM25Retriever:
    """
    Sparse keyword search engine using BM25Okapi with cricket and sports-science tokenization.
    Crucial for finding exact clinical identifiers (e.g., 'L5 pars', 'ACWR > 1.5', 'Player 9', 'Stork test').
    """

    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None):
        self.documents: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        if documents:
            self.index_documents(documents)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenizes text while preserving clinical acronyms, hyphenated terms,
        and numeric codes (e.g. 'l5-s1', 'counter-rotation', '3t mri').
        """
        text_clean = text.lower()
        # Replace punctuation except hyphens and numbers
        text_clean = re.sub(r"[^\w\s\-]", " ", text_clean)
        raw_tokens = text_clean.split()

        processed_tokens = []
        for t in raw_tokens:
            t = t.strip("-")
            if not t or len(t) < 2:
                continue
            processed_tokens.append(t)
            # If token is hyphenated (e.g., counter-rotation), add sub-parts too
            if "-" in t:
                parts = t.split("-")
                for p in parts:
                    if len(p) >= 2:
                        processed_tokens.append(p)

        return processed_tokens

    def index_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Builds the BM25 index over the provided documents."""
        if not documents:
            return 0

        self.documents = documents
        self.corpus_tokens = [self.tokenize(doc.get("page_content", "")) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print(f"[BM25Retriever] Indexed {len(self.documents)} documents for sparse retrieval.")
        return len(self.documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieves top_k documents scored by BM25, optionally filtered by metadata.
        Returns list of (doc, normalized_score).
        """
        if not self.bm25 or not self.documents:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

        scored_docs = []
        for doc, score in zip(self.documents, scores):
            if score <= 0:
                continue

            # Apply metadata filter if provided
            if where:
                meta = doc.get("metadata", {})
                match = True
                for k, v in where.items():
                    if meta.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            # Normalize BM25 score to [0, 1]
            norm_score = round(score / max_score, 4)
            scored_docs.append((doc, norm_score))

        # Sort descending by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]
