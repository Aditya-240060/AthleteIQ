"""
Reranker and Context Relevance Filter for AthleteIQ.
Applies cross-scoring, entity overlap, and clinical relevance filtering to candidate chunks.
"""

import re
from typing import List, Dict, Any, Tuple


class AthleteIQReranker:
    """
    Reranks candidate document chunks retrieved by the hybrid search engine.
    Applies query keyword coverage, named entity alignment, and sports-science salience scoring.
    """

    CRITICAL_ENTITIES = [
        "player 9", "arjun patel", "acwr", "acute:chronic", "danger zone", "sweet spot",
        "pars interarticularis", "spondylolysis", "bone marrow edema", "stork test",
        "counter-rotation", "mixed action", "front foot contact", "ground reaction",
        "lateral flexion", "l4", "l5", "s1", "mri", "stir", "playerload"
    ]

    def __init__(self, top_n: int = 3):
        self.top_n = top_n

    def compute_relevance_score(self, query: str, document_text: str, initial_score: float) -> float:
        """Calculates a composite reranking score."""
        q_lower = query.lower()
        d_lower = document_text.lower()

        # 1. Query token overlap
        q_tokens = set(re.findall(r"\b\w{3,}\b", q_lower))
        if not q_tokens:
            return initial_score

        d_tokens = set(re.findall(r"\b\w{3,}\b", d_lower))
        overlap_count = len(q_tokens.intersection(d_tokens))
        token_overlap_ratio = overlap_count / len(q_tokens)

        # 2. Critical sports science entity matching
        entity_hits = 0
        for entity in self.CRITICAL_ENTITIES:
            if entity in q_lower and entity in d_lower:
                entity_hits += 1.5
            elif entity in d_lower:
                entity_hits += 0.2

        # 3. Exact phrase matching bonus
        phrase_bonus = 0.0
        for q_token in q_tokens:
            if len(q_token) > 4 and q_token in d_lower:
                phrase_bonus += 0.05

        # Composite score calculation
        composite = (
            0.40 * initial_score
            + 0.35 * token_overlap_ratio
            + 0.15 * min(1.0, entity_hits / 3.0)
            + 0.10 * min(1.0, phrase_bonus)
        )

        return round(min(1.0, max(0.0, composite)), 4)

    def rerank(
        self,
        query: str,
        scored_candidates: List[Tuple[Dict[str, Any], float]],
        top_n: int = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Reranks scored candidate tuples and returns the top_n highest scoring documents.
        """
        n = top_n or self.top_n
        if not scored_candidates:
            return []

        reranked = []
        for doc, score in scored_candidates:
            text = doc.get("page_content", "")
            new_score = self.compute_relevance_score(query, text, score)
            doc["metadata"]["rerank_score"] = new_score
            reranked.append((doc, new_score))

        # Sort descending by rerank_score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:n]
