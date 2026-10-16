"""
Sidebar Component for AthleteIQ Streamlit Application.
Provides LLM provider switching, parameter controls, knowledge base re-indexing, and athlete filtering.
"""

import os
import streamlit as st
from typing import Dict, Any, Callable


def render_sidebar(db_stats: Dict[str, Any], on_reindex: Callable[[], None]) -> Dict[str, Any]:
    """Renders the AthleteIQ navigation and configuration sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                <div style="font-size: 26px;">🏏</div>
                <div>
                    <h2 style="margin: 0; font-size: 20px; font-weight: 700; color: #38bdf8;">AthleteIQ</h2>
                    <p style="margin: 0; font-size: 11px; color: #94a3b8;">High-Performance Cricket RAG</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("### ⚙️ Intelligence Engine")

        provider_options = [
            "Offline Sports Science Consultant (Built-in)",
            "Groq Cloud API (Llama 3.3 70B)",
            "Ollama Local Server (Llama 3)"
        ]

        selected_provider_label = st.selectbox(
            "Select LLM Provider",
            options=provider_options,
            index=0,
            help="Choose between ultra-fast Groq Cloud, local Ollama, or deterministic Offline Sports Science engine."
        )

        groq_api_key = None
        groq_model = "llama-3.3-70b-versatile"
        ollama_url = "http://localhost:11434"
        ollama_model = "llama3"
        provider_code = "offline_sports_science"

        if "Groq" in selected_provider_label:
            provider_code = "groq"
            groq_api_key = st.text_input(
                "Groq API Key",
                type="password",
                value=os.getenv("GROQ_API_KEY", ""),
                help="Get a free API key at console.groq.com"
            )
            groq_model = st.selectbox(
                "Groq Model",
                options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
                index=0
            )
        elif "Ollama" in selected_provider_label:
            provider_code = "ollama"
            ollama_url = st.text_input("Ollama Base URL", value="http://localhost:11434")
            ollama_model = st.text_input("Ollama Model Name", value="llama3")
        else:
            provider_code = "offline_sports_science"
            st.info("💡 Offline Sports Science Engine active. Zero API keys required.")

        st.markdown("---")
        st.markdown("### 🎯 Retrieval & Filtering")

        athlete_filter = st.selectbox(
            "Filter by Athlete",
            options=["All Athletes", "Player 9 (Arjun Patel)", "Player 4 (Liam Smith)", "Player 12 (Rashid Khan)"],
            index=0
        )

        doc_type_filter = st.selectbox(
            "Document Domain Filter",
            options=["All Domains", "Research Papers", "Telemetry Summaries", "Medical Reports"],
            index=0
        )

        alpha = st.slider(
            "Hybrid Weight (Dense vs BM25)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="0.0 = pure BM25 keyword search, 1.0 = pure dense vector embedding search."
        )

        top_k = st.slider("Top Retrieval Chunks", min_value=2, max_value=8, value=4)

        st.markdown("---")
        st.markdown("### 📚 Knowledge Base Index")

        total_chunks = db_stats.get("total_chunks", 0)
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 8px; border: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 10px;">
                <span class="status-indicator"></span>
                <span style="font-size: 13px; font-weight: 600;">Indexed Chunks: <b>{total_chunks}</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("🔄 Re-index Knowledge Base", use_container_width=True):
            on_reindex()

        st.markdown(
            """
            <div style="font-size: 11px; color: #64748b; margin-top: 20px; text-align: center;">
                AthleteIQ v1.0.0 • Sports Science AI
            </div>
            """,
            unsafe_allow_html=True
        )

        return {
            "provider_code": provider_code,
            "groq_api_key": groq_api_key,
            "groq_model": groq_model,
            "ollama_url": ollama_url,
            "ollama_model": ollama_model,
            "athlete_filter": athlete_filter,
            "doc_type_filter": doc_type_filter,
            "alpha": alpha,
            "top_k": top_k
        }
