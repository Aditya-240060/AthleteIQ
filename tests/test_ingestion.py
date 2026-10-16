"""
Unit and integration tests for AthleteIQ Data Ingestion (Phase 1).
"""

import os
import pytest
from src.data_ingestion.pdf_loader import PDFPaperLoader
from src.data_ingestion.telemetry_parser import TelemetryParser
from src.data_ingestion.text_splitter import AthleteIQTextSplitter


def test_pdf_paper_loader():
    loader = PDFPaperLoader()
    docs = loader.load_all_papers()
    
    assert len(docs) > 0, "PDF loader should extract pages from public_papers"
    sample_doc = docs[0]
    assert "page_content" in sample_doc
    assert "metadata" in sample_doc
    assert sample_doc["metadata"]["document_type"] == "research_paper"
    assert sample_doc["metadata"]["page_number"] >= 1
    assert len(sample_doc["page_content"]) > 50


def test_telemetry_parser():
    parser = TelemetryParser()
    acwr_under = parser.get_acwr_interpretation(0.70)
    acwr_sweet = parser.get_acwr_interpretation(1.10)
    acwr_danger = parser.get_acwr_interpretation(1.65)

    assert "sub-optimal" in acwr_under.lower()
    assert "sweet spot" in acwr_sweet.lower()
    assert "danger zone" in acwr_danger.lower()

    telemetry_docs = parser.parse_csv_telemetry()
    assert len(telemetry_docs) > 0, "Telemetry parser should extract rows from CSV"
    
    sample = telemetry_docs[0]
    assert "page_content" in sample
    assert "TELEMETRY LOG:" in sample["page_content"]
    assert sample["metadata"]["document_type"] == "telemetry_summary"

    medical_docs = parser.parse_medical_reports()
    assert len(medical_docs) > 0, "Medical notes parser should load physio reports"
    assert "player 9" in medical_docs[0]["metadata"]["player_id"].lower()


def test_text_splitter():
    splitter = AthleteIQTextSplitter(chunk_size=300, chunk_overlap=50)
    test_docs = [
        {
            "page_content": (
                "Lumbar Bone Stress Injuries and Spondylolysis in Cricket Fast Bowlers. " * 15
            ),
            "metadata": {"file_name": "test.pdf", "page_number": 1}
        }
    ]

    chunks = splitter.split_documents(test_docs)
    assert len(chunks) > 1, "Long document should be split into multiple chunks"
    for chunk in chunks:
        assert len(chunk["page_content"]) <= 400
        assert "chunk_id" in chunk["metadata"]
        assert "chunk_index" in chunk["metadata"]
