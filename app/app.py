"""
Main Streamlit Application for AthleteIQ.
Elite Cricket Sports Science, Biomechanics & Workload Management Hybrid RAG System.
"""

import os
import sys

# Ensure root directory is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.data_ingestion import PDFPaperLoader, TelemetryParser, AthleteIQTextSplitter
from src.vector_engine import AthleteIQEmbeddings, VectorDBManager
from src.retrieval import AthleteIQBM25Retriever, HybridRetriever, AthleteIQReranker
from src.rag_chain import LLMBackendFactory, ConversationalRAGChain
from app.components import render_sidebar, render_telemetry_view, render_citations


st.set_page_config(
    page_title="AthleteIQ | Cricket Sports Science RAG",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "static", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing AthleteIQ Knowledge Engine...")
def initialize_system():
    """Initializes embeddings, database manager, retrievers, and RAG chain."""
    embedder = AthleteIQEmbeddings()
    db_manager = VectorDBManager(embedding_model=embedder)
    bm25 = AthleteIQBM25Retriever()

    # If ChromaDB is empty or missing, run initial ingestion
    if db_manager.count() == 0:
        print("[AthleteIQ App] Performing initial document ingestion...")
        pdf_loader = PDFPaperLoader()
        telemetry_parser = TelemetryParser()
        splitter = AthleteIQTextSplitter()

        pdf_docs = pdf_loader.load_all_papers()
        telemetry_docs = telemetry_parser.load_all_telemetry_and_reports()
        all_raw_docs = pdf_docs + telemetry_docs

        chunks = splitter.split_documents(all_raw_docs)
        db_manager.add_documents(chunks)
        bm25.index_documents(chunks)
    else:
        # Load documents into BM25 from Chroma
        try:
            stored = db_manager.collection.get(include=["documents", "metadatas"])
            docs = []
            if stored and "documents" in stored:
                for doc_text, meta in zip(stored["documents"], stored["metadatas"]):
                    docs.append({"page_content": doc_text, "metadata": meta})
                bm25.index_documents(docs)
        except Exception as e:
            print(f"[AthleteIQ App] Error loading existing docs to BM25: {e}")

    hybrid_retriever = HybridRetriever(db_manager=db_manager, bm25_retriever=bm25, alpha=0.5)
    reranker = AthleteIQReranker(top_n=3)
    llm = LLMBackendFactory.create_backend()
    rag_chain = ConversationalRAGChain(hybrid_retriever=hybrid_retriever, reranker=reranker, llm_backend=llm)

    return db_manager, bm25, hybrid_retriever, rag_chain


db_manager, bm25_retriever, hybrid_retriever, rag_chain = initialize_system()


def reindex_knowledge_base():
    """Wipes vectorstore and re-indexes all papers and telemetry."""
    with st.spinner("Rebuilding sports science index from source documents..."):
        pdf_loader = PDFPaperLoader()
        telemetry_parser = TelemetryParser()
        splitter = AthleteIQTextSplitter()

        pdf_docs = pdf_loader.load_all_papers()
        telemetry_docs = telemetry_parser.load_all_telemetry_and_reports()
        all_raw_docs = pdf_docs + telemetry_docs

        chunks = splitter.split_documents(all_raw_docs)
        db_manager.reset_collection()
        db_manager.add_documents(chunks)
        bm25_retriever.index_documents(chunks)
        st.session_state["messages"] = []
        st.success(f"Successfully re-indexed {len(chunks)} chunks into vectorstore!")
        st.rerun()


# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to AthleteIQ.** I am your Lead Cricket Sports Scientist and Biomechanical Consultant.\n\n"
                "I can analyze fast bowling biomechanics, quantify Acute:Chronic Workload Ratios (ACWR), "
                "evaluate lumbar pars stress fractures, and review athlete GPS telemetry.\n\n"
                "**How can I assist your high-performance pace unit today?**"
            ),
            "citations": [],
            "confidence_score": 1.0
        }
    ]

# Render Sidebar
sidebar_params = render_sidebar(db_stats=db_manager.get_stats(), on_reindex=reindex_knowledge_base)

# Update RAG chain with sidebar configuration
updated_llm = LLMBackendFactory.create_backend(
    provider=sidebar_params["provider_code"],
    groq_api_key=sidebar_params["groq_api_key"],
    groq_model=sidebar_params["groq_model"],
    ollama_url=sidebar_params["ollama_url"],
    ollama_model=sidebar_params["ollama_model"]
)
rag_chain.set_llm_backend(updated_llm)
hybrid_retriever.alpha = sidebar_params["alpha"]


# Main Page Header
st.markdown(
    f"""
    <div class="athlete-header-container">
        <div>
            <h1 class="athlete-title">AthleteIQ</h1>
            <p class="athlete-subtitle">Cricket Sports Science & Biomechanics Hybrid RAG Platform</p>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 11px; color: #94a3b8;">ACTIVE BACKEND</div>
            <div style="font-size: 13px; font-weight: 700; color: #38bdf8;">{updated_llm.provider_name}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

tab_chat, tab_telemetry, tab_kb = st.tabs([
    "💬 AthleteIQ AI Consultant",
    "📊 Telemetry & Workload Dashboard",
    "📑 Knowledge Base Explorer"
])

# -------------------------------------------------------------
# TAB 1: Conversational Chat
# -------------------------------------------------------------
with tab_chat:
    st.markdown("#### Quick Consultation Prompts:")
    chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)

    prompt_to_submit = None

    with chip_col1:
        if st.button("🚨 Analyze Player 9 Lumbar Stress Risk", use_container_width=True):
            prompt_to_submit = "Analyze Player 9 (Arjun Patel). Cross-reference his recent workload spike with his lumbar physio notes and fast bowling biomechanics."

    with chip_col2:
        if st.button("⚖️ Explain ACWR Sweet Spot vs Danger Zone", use_container_width=True):
            prompt_to_submit = "Explain Tim Gabbett's Acute:Chronic Workload Ratio (ACWR) framework for cricket fast bowlers. What is the sweet spot vs danger zone?"

    with chip_col3:
        if st.button("🔄 Biomechanics of Counter-Rotation (>30°)", use_container_width=True):
            prompt_to_submit = "How does shoulder counter-rotation exceeding 30 degrees in a mixed bowling action lead to pars interarticularis bone stress injuries?"

    with chip_col4:
        if st.button("📋 6-Phase Return-to-Play Protocol", use_container_width=True):
            prompt_to_submit = "What is the recommended 6-phase criteria-based return-to-play protocol for a fast bowler recovering from lumbar bone marrow edema?"

    st.markdown("---")

    # Display chat message history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                render_citations(msg["citations"], msg.get("confidence_score", 0.85))

    # Handle chat input
    user_input = st.chat_input("Ask a clinical, biomechanical, or workload question...")
    effective_query = prompt_to_submit or user_input

    if effective_query:
        # Append user message
        st.session_state["messages"].append({"role": "user", "content": effective_query})
        with st.chat_message("user"):
            st.markdown(effective_query)

        # Build metadata filter if specified in sidebar
        where_filter = None
        if sidebar_params["athlete_filter"] != "All Athletes":
            if "Player 9" in sidebar_params["athlete_filter"]:
                where_filter = {"player_id": "Player 9"}
            elif "Player 4" in sidebar_params["athlete_filter"]:
                where_filter = {"player_id": "Player 4"}
            elif "Player 12" in sidebar_params["athlete_filter"]:
                where_filter = {"player_id": "Player 12"}

        with st.chat_message("assistant"):
            with st.spinner("Analyzing sports science literature, telemetry, and biomechanics..."):
                result = rag_chain.query(
                    user_query=effective_query,
                    where_filter=where_filter,
                    top_k=sidebar_params["top_k"]
                )
                answer = result["answer"]
                citations = result["citations"]
                confidence = result["confidence_score"]

                st.markdown(answer)
                if citations:
                    render_citations(citations, confidence)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "confidence_score": confidence
                })

    if len(st.session_state["messages"]) > 1:
        if st.button("🗑️ Clear Chat History"):
            st.session_state["messages"] = [st.session_state["messages"][0]]
            rag_chain.clear_history()
            st.rerun()

# -------------------------------------------------------------
# TAB 2: Telemetry & Workload Dashboard
# -------------------------------------------------------------
with tab_telemetry:
    render_telemetry_view()

# -------------------------------------------------------------
# TAB 3: Knowledge Base Explorer
# -------------------------------------------------------------
with tab_kb:
    st.markdown("### 📚 Indexed Cricket Sports Science Knowledge Base")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Research Papers</div>
                <div class="kpi-value" style="color: #10b981;">3 Peer-Reviewed</div>
                <div class="kpi-subtext">Lumbar Stress • ACWR • Biomechanics</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_stat2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Private Telemetry Logs</div>
                <div class="kpi-value" style="color: #38bdf8;">120 GPS Sessions</div>
                <div class="kpi-subtext">4 Athletes • 30 Consecutive Days</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_stat3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Clinical Physio Notes</div>
                <div class="kpi-value" style="color: #f43f5e;">1 MRI Clinical Log</div>
                <div class="kpi-subtext">Player 9 Confidential Case Report</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### Document Index Overview")
    doc_type_choice = st.radio(
        "Select Domain to Inspect",
        options=["Research Literature", "Physiotherapy & Medical Notes", "Telemetry Natural Summaries"],
        horizontal=True
    )

    if doc_type_choice == "Research Literature":
        st.markdown("##### 📄 Indexed Scientific Papers (`data/public_papers/`)")
        papers = [
            ("fast_bowling_lumbar_stress.pdf", "Lumbar Bone Stress Injuries and Spondylolysis in Cricket Fast Bowlers: Biomechanical Etiology and Clinical Management", "Dr. Marcus Vance et al."),
            ("acute_chronic_workload_cricket.pdf", "The Acute:Chronic Workload Ratio (ACWR) and Bowling Injury Prevention in Elite Cricket", "Dr. Timothy J. Gabbett et al."),
            ("shoulder_counter_rotation_biomechanics.pdf", "Shoulder Counter-Rotation and Spinal Torsional Torque in Cricket Fast Bowling Delivery Actions", "Prof. Bruce Elliott et al.")
        ]
        for f, t, a in papers:
            with st.expander(f"📑 {f} — {t}"):
                st.write(f"**Authors:** {a}")
                st.write(f"**Domain:** Biomechanics, Kinematics, Spinal Stress, Workload Ratios")
                st.write(f"**Storage Path:** `data/public_papers/{f}`")

    elif doc_type_choice == "Physiotherapy & Medical Notes":
        st.markdown("##### 🏥 Confidential Clinical Report (`data/private_telemetry/medical_reports/player_9_physio_notes.txt`)")
        med_path = os.path.join("data", "private_telemetry", "medical_reports", "player_9_physio_notes.txt")
        if os.path.exists(med_path):
            with open(med_path, "r", encoding="utf-8") as f:
                st.text_area("Full Clinical & MRI Report", f.read(), height=360)

    else:
        st.markdown("##### 🏃 Natural Language Summaries (`data/private_telemetry/parsed_txt/player_daily_summaries.txt`)")
        sum_path = os.path.join("data", "private_telemetry", "parsed_txt", "player_daily_summaries.txt")
        if os.path.exists(sum_path):
            with open(sum_path, "r", encoding="utf-8") as f:
                content = f.read()
                st.text_area("Sample Generated Narratives", content[:2500] + "...", height=360)
