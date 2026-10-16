"""
Unit and integration tests for AthleteIQ Vector Engine and Hybrid Retrieval (Phases 2 & 3).
"""

import os
import pytest
from src.vector_engine.embeddings import AthleteIQEmbeddings
from src.vector_engine.db_manager import VectorDBManager
from src.retrieval.bm25_retriever import AthleteIQBM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import AthleteIQReranker
from src.rag_chain.conversational_chain import ConversationalRAGChain


@pytest.fixture
def sample_test_documents():
    return [
        {
            "page_content": "Player 9 (Arjun Patel) acute workload spiked to 36.0 overs with an ACWR of 1.72 in the danger zone.",
            "metadata": {"chunk_id": "test_1", "player_id": "Player 9", "document_type": "telemetry_summary"}
        },
        {
            "page_content": "Shoulder counter-rotation exceeding 30 degrees in a mixed action fast bowler increases lumbar pars shear stress.",
            "metadata": {"chunk_id": "test_2", "document_type": "research_paper"}
        },
        {
            "page_content": "Gabbett's ACWR sweet spot between 0.8 and 1.3 minimizes cricket bowling injury hazard.",
            "metadata": {"chunk_id": "test_3", "document_type": "research_paper"}
        },
        {
            "page_content": "Clinical examination revealed positive Stork test and Grade 1 bone marrow edema at right L5 pars.",
            "metadata": {"chunk_id": "test_4", "player_id": "Player 9", "document_type": "medical_report"}
        }
    ]


def test_embeddings():
    embedder = AthleteIQEmbeddings()
    texts = ["Fast bowling lumbar stress fractures", "Acute chronic workload ratio"]
    embeddings = embedder.embed_documents(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384, "Embeddings should have 384 dimensions"
    
    # Check L2 unit normalization (sum of squares approx 1.0)
    norm = sum(x * x for x in embeddings[0])
    assert 0.99 <= norm <= 1.01, "Embedding should be unit normalized"


def test_bm25_retriever(sample_test_documents):
    bm25 = AthleteIQBM25Retriever()
    bm25.index_documents(sample_test_documents)

    results = bm25.search("ACWR sweet spot", top_k=2)
    assert len(results) > 0
    top_doc, score = results[0]
    assert "sweet spot" in top_doc["page_content"].lower()
    assert score > 0.0


def test_hybrid_and_reranker(sample_test_documents):
    embedder = AthleteIQEmbeddings()
    db = VectorDBManager(collection_name="test_athleteiq_pytest", embedding_model=embedder)
    db.reset_collection()
    db.add_documents(sample_test_documents)

    bm25 = AthleteIQBM25Retriever(sample_test_documents)
    hybrid = HybridRetriever(db_manager=db, bm25_retriever=bm25, alpha=0.5)

    hybrid_results = hybrid.retrieve("Player 9 ACWR spike lumbar", top_k=3)
    assert len(hybrid_results) > 0

    reranker = AthleteIQReranker(top_n=2)
    reranked = reranker.rerank("Player 9 ACWR spike lumbar", hybrid_results)
    assert len(reranked) <= 2

    # Clean up test collection
    db.reset_collection()


def test_conversational_rag_chain(sample_test_documents):
    embedder = AthleteIQEmbeddings()
    db = VectorDBManager(collection_name="test_chain_pytest", embedding_model=embedder)
    db.reset_collection()
    db.add_documents(sample_test_documents)

    bm25 = AthleteIQBM25Retriever(sample_test_documents)
    hybrid = HybridRetriever(db_manager=db, bm25_retriever=bm25, alpha=0.5)
    reranker = AthleteIQReranker(top_n=2)

    chain = ConversationalRAGChain(hybrid_retriever=hybrid, reranker=reranker)
    response = chain.query("What was Player 9's ACWR and what did the MRI show?")

    assert "answer" in response
    assert len(response["answer"]) > 50
    assert "citations" in response
    assert len(response["citations"]) > 0

    # Clean up
    db.reset_collection()
