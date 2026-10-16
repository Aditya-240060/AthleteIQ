"""
AthleteIQ CLI Entry Point.
Run ingestion, query the sports science hybrid RAG pipeline, check stats, or launch Streamlit.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Ensure root directory is in sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

load_dotenv()

# Ensure utf-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_ingestion import PDFPaperLoader, TelemetryParser, AthleteIQTextSplitter
from src.vector_engine import AthleteIQEmbeddings, VectorDBManager
from src.retrieval import AthleteIQBM25Retriever, HybridRetriever, AthleteIQReranker
from src.rag_chain import LLMBackendFactory, ConversationalRAGChain


def run_ingestion():
    """Runs data extraction, natural language translation, chunking, and ChromaDB vector indexing."""
    print("=" * 70)
    print("🏏 ATHLETEIQ: RUNNING DATA INGESTION PIPELINE")
    print("=" * 70)

    # 1. Load PDFs
    print("\n[Step 1/4] Extracting Sports Science Research Literature (PDFs)...")
    pdf_loader = PDFPaperLoader()
    pdf_docs = pdf_loader.load_all_papers()
    print(f"-> Extracted {len(pdf_docs)} pages from research papers.")

    # 2. Parse Telemetry CSV and Physio Reports
    print("\n[Step 2/4] Parsing GPS Telemetry and Clinical Medical Reports...")
    telemetry_parser = TelemetryParser()
    telemetry_docs = telemetry_parser.load_all_telemetry_and_reports()
    print(f"-> Converted {len(telemetry_docs)} telemetry records & clinical notes.")

    # 3. Semantic Chunking
    print("\n[Step 3/4] Performing Recursive Semantic Chunking...")
    all_raw_docs = pdf_docs + telemetry_docs
    splitter = AthleteIQTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = splitter.split_documents(all_raw_docs)
    print(f"-> Generated {len(chunks)} contextual chunks.")

    # 4. Vectorstore & BM25 Indexing
    print("\n[Step 4/4] Indexing into ChromaDB Persistent Vector Store & BM25...")
    embedder = AthleteIQEmbeddings()
    db_manager = VectorDBManager(embedding_model=embedder)
    db_manager.reset_collection()
    added_count = db_manager.add_documents(chunks)

    # Save BM25 tokens verification
    bm25 = AthleteIQBM25Retriever()
    bm25.index_documents(chunks)

    print("\n" + "=" * 70)
    print(f"✅ INGESTION COMPLETE! Indexed {added_count} chunks into '{db_manager.collection_name}'.")
    print("=" * 70)


def run_query(query_text: str, athlete_id: str = None):
    """Executes a hybrid RAG query directly from CLI."""
    print("=" * 70)
    print(f"🏏 ATHLETEIQ QUERY: '{query_text}'")
    if athlete_id:
        print(f"   Filtering by athlete: {athlete_id}")
    print("=" * 70)

    embedder = AthleteIQEmbeddings()
    db_manager = VectorDBManager(embedding_model=embedder)
    
    if db_manager.count() == 0:
        print("[Notice] Vectorstore empty. Running automatic ingestion first...")
        run_ingestion()

    # Load stored documents into BM25
    bm25 = AthleteIQBM25Retriever()
    stored = db_manager.collection.get(include=["documents", "metadatas"])
    docs = []
    if stored and "documents" in stored:
        for doc_text, meta in zip(stored["documents"], stored["metadatas"]):
            docs.append({"page_content": doc_text, "metadata": meta})
        bm25.index_documents(docs)

    hybrid_retriever = HybridRetriever(db_manager=db_manager, bm25_retriever=bm25, alpha=0.5)
    reranker = AthleteIQReranker(top_n=3)
    llm = LLMBackendFactory.create_backend()
    rag_chain = ConversationalRAGChain(hybrid_retriever=hybrid_retriever, reranker=reranker, llm_backend=llm)

    where_filter = {"player_id": athlete_id} if athlete_id else None
    result = rag_chain.query(user_query=query_text, where_filter=where_filter)

    print("\n" + result["answer"])
    print("\n" + "-" * 70)
    print(f"📑 CITATIONS ({len(result['citations'])} sources, Confidence: {int(result['confidence_score'] * 100)}%):")
    for idx, c in enumerate(result["citations"]):
        print(f"  [{idx + 1}] {c['source']} (Page {c['page']}) [{c['document_type']}] - Match: {int(c['relevance_score'] * 100)}%")
        print(f"      Snippet: \"{c['snippet'][:120]}...\"")
    print("-" * 70)


def show_stats():
    """Prints vectorstore statistics."""
    embedder = AthleteIQEmbeddings()
    db_manager = VectorDBManager(embedding_model=embedder)
    stats = db_manager.get_stats()

    print("=" * 70)
    print("🏏 ATHLETEIQ KNOWLEDGE BASE STATUS")
    print("=" * 70)
    print(f"Collection Name : {stats['collection_name']}")
    print(f"Total Chunks    : {stats['total_chunks']}")
    print(f"Storage Path    : {stats['storage_path']}")
    print("Document Types Breakdown:")
    for dt, count in stats["document_types"].items():
        print(f"  - {dt}: {count} chunks")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="AthleteIQ Cricket Sports Science & Biomechanics Hybrid RAG CLI")
    parser.add_argument("--ingest", action="store_true", help="Run document ingestion & vector indexing")
    parser.add_argument("--query", type=str, help="Ask AthleteIQ a sports science query")
    parser.add_argument("--athlete", type=str, default=None, help="Optional athlete ID filter (e.g. 'Player 9')")
    parser.add_argument("--stats", action="store_true", help="Display vectorstore chunk statistics")
    parser.add_argument("--run-app", action="store_true", help="Launch the Streamlit web dashboard")

    args = parser.parse_args()

    if args.ingest:
        run_ingestion()
    elif args.query:
        run_query(args.query, athlete_id=args.athlete)
    elif args.stats:
        show_stats()
    elif args.run_app:
        print("Launching Streamlit application...")
        os.system(".venv\\Scripts\\streamlit run app\\app.py")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
