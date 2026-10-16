"""
Data ingestion module for AthleteIQ.
Extracts research papers, parses GPS telemetry CSVs into natural language,
and splits documents into semantically coherent chunks.
"""

from .pdf_loader import PDFPaperLoader
from .telemetry_parser import TelemetryParser
from .text_splitter import AthleteIQTextSplitter

__all__ = ["PDFPaperLoader", "TelemetryParser", "AthleteIQTextSplitter"]
