"""
Conversational RAG Chain with Memory and Citation Tracking for AthleteIQ.
Orchestrates hybrid retrieval, contextual reranking, LLM generation, and citation extraction.
"""

from typing import List, Dict, Any, Optional, Tuple
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.reranker import AthleteIQReranker
from .prompts import AthleteIQPrompts
from .llm_backend import BaseLLMBackend, LLMBackendFactory


class ConversationalRAGChain:
    """
    End-to-end Conversational RAG pipeline with multi-turn chat memory,
    confidence scoring, and transparent citations.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Optional[AthleteIQReranker] = None,
        llm_backend: Optional[BaseLLMBackend] = None,
        max_history_turns: int = 5,
    ):
        self.retriever = hybrid_retriever
        self.reranker = reranker or AthleteIQReranker(top_n=3)
        self.llm = llm_backend or LLMBackendFactory.create_backend()
        self.max_history_turns = max_history_turns
        self.history: List[Dict[str, str]] = []

    def set_llm_backend(self, backend: BaseLLMBackend):
        """Allows dynamically switching LLM backend (e.g. from Streamlit UI)."""
        self.llm = backend

    def clear_history(self):
        """Resets conversational memory."""
        self.history = []

    def _format_history_string(self) -> str:
        """Formats the sliding window of past messages for prompt inclusion."""
        if not self.history:
            return ""
        recent = self.history[-(self.max_history_turns * 2):]
        formatted = []
        for turn in recent:
            role = "Coach/User" if turn["role"] == "user" else "AthleteIQ"
            formatted.append(f"{role}: {turn['content']}")
        return "\n".join(formatted)

    def _format_context_and_citations(
        self, scored_docs: List[Tuple[Dict[str, Any], float]]
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """Formats retrieved chunks into prompt context and extracts citation objects."""
        if not scored_docs:
            return "No relevant documents found in knowledge base.", [], 0.0

        context_blocks = []
        citations = []
        scores = []

        for idx, (doc, score) in enumerate(scored_docs):
            content = doc.get("page_content", "").strip()
            meta = doc.get("metadata", {})
            file_name = meta.get("file_name", "Document")
            doc_type = meta.get("document_type", "General")
            page_num = meta.get("page_number", "")
            chunk_id = meta.get("chunk_id", f"chunk_{idx}")
            page_info = f", Page {page_num}" if page_num else ""

            block_header = f"--- Document [{idx + 1}] | {file_name}{page_info} | Type: {doc_type} | Relevance: {score:.2f} ---"
            context_blocks.append(f"{block_header}\n{content}")

            citations.append({
                "source": file_name,
                "document_type": doc_type,
                "page": page_num if page_num else "N/A",
                "chunk_id": chunk_id,
                "relevance_score": score,
                "snippet": content[:240] + ("..." if len(content) > 240 else "")
            })
            scores.append(score)

        avg_confidence = round(sum(scores) / len(scores), 2) if scores else 0.0
        combined_context = "\n\n".join(context_blocks)
        return combined_context, citations, avg_confidence

    def query(
        self,
        user_query: str,
        where_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 6,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG query:
        1. Hybrid search (Dense + BM25)
        2. Relevance reranking
        3. Prompt engineering with citations
        4. LLM inference
        5. Updates chat memory
        """
        # Step 1: Hybrid retrieval
        candidates = self.retriever.retrieve(user_query, top_k=top_k, where=where_filter)

        # Step 2: Contextual reranking
        reranked_docs = self.reranker.rerank(user_query, candidates)

        # Step 3: Context formatting and citation collection
        context_str, citations, confidence = self._format_context_and_citations(reranked_docs)
        history_str = self._format_history_string()

        # Step 4: Prompt compilation
        prompt = AthleteIQPrompts.format_rag_prompt(
            query=user_query,
            context=context_str,
            chat_history=history_str
        )

        # Step 5: LLM generation
        try:
            answer = self.llm.generate(
                prompt=prompt,
                system_prompt=AthleteIQPrompts.SYSTEM_PROMPT
            )
        except Exception as e:
            answer = f"Error during generation: {e}"

        # Update conversational memory
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": answer})

        unique_sources = list({c["source"] for c in citations})

        return {
            "answer": answer,
            "citations": citations,
            "confidence_score": confidence,
            "sources": unique_sources,
            "llm_provider": self.llm.provider_name,
            "retrieved_count": len(reranked_docs)
        }
