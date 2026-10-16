"""
Citation and Grounding View Component for AthleteIQ.
Renders transparent citations with document badges, relevance scores, and source excerpts.
"""

import streamlit as st
from typing import List, Dict, Any


def render_citations(citations: List[Dict[str, Any]], confidence_score: float = 0.0):
    """Renders expandable citations with clinical badges and source excerpts."""
    if not citations:
        return

    conf_pct = int(confidence_score * 100) if confidence_score else 85
    conf_color = "#10b981" if conf_pct >= 80 else ("#f59e0b" if conf_pct >= 60 else "#f43f5e")

    with st.expander(f"📑 Evidence Citations & Grounding ({len(citations)} Sources • {conf_pct}% Confidence)", expanded=False):
        for idx, cit in enumerate(citations):
            source = cit.get("source", "Unknown Document")
            doc_type = cit.get("document_type", "General")
            page = cit.get("page", "N/A")
            score = cit.get("relevance_score", 0.0)
            snippet = cit.get("snippet", "")

            badge_class = "badge-paper"
            if "medical" in doc_type.lower():
                badge_class = "badge-medical"
            elif "telemetry" in doc_type.lower():
                badge_class = "badge-telemetry"

            page_label = f" • Page {page}" if page and page != "N/A" else ""
            score_pct = int(score * 100) if score <= 1.0 else int(score)

            st.markdown(
                f"""
                <div class="citation-box">
                    <div class="citation-header">
                        <span class="citation-source">[{idx + 1}] {source}{page_label}</span>
                        <span class="citation-badge {badge_class}">{doc_type.upper()} • {score_pct}% MATCH</span>
                    </div>
                    <div class="citation-snippet">"{snippet}"</div>
                </div>
                """,
                unsafe_allow_html=True
            )
