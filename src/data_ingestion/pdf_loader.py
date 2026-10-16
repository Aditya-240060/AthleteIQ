"""
PDF Research Paper Loader for AthleteIQ.
Extracts sports science & biomechanics literature with metadata extraction and text cleaning.
"""

import os
import re
from typing import List, Dict, Any, Optional
from pypdf import PdfReader


class PDFPaperLoader:
    """
    Loads sports science research PDFs (e.g. PubMed/arXiv studies on lumbar stress,
    workload ratios, and bowling counter-rotation) and converts them into structured documents.
    """

    def __init__(self, papers_dir: Optional[str] = None):
        self.papers_dir = papers_dir or os.path.join("data", "public_papers")

    def clean_text(self, text: str) -> str:
        """Cleans extracted text by fixing hyphenation and normalizing whitespaces."""
        if not text:
            return ""
        # Fix line breaks within hyphenated words (e.g., 'bio-\nmechanics' -> 'biomechanics')
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Normalize multiple spaces, tabs, and duplicate blank lines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def load_single_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts pages and metadata from an individual PDF file.
        Returns a list of dicts with 'page_content' and 'metadata'.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        file_name = os.path.basename(pdf_path)
        documents = []

        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            
            # Derive paper title from file name or first lines
            title_guess = file_name.replace(".pdf", "").replace("_", " ").title()

            for page_idx, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                cleaned = self.clean_text(raw_text)
                
                if not cleaned:
                    continue

                metadata = {
                    "source": pdf_path,
                    "file_name": file_name,
                    "document_type": "research_paper",
                    "page_number": page_idx + 1,
                    "total_pages": total_pages,
                    "title": title_guess,
                    "domain": "sports_science_biomechanics"
                }

                documents.append({
                    "page_content": cleaned,
                    "metadata": metadata
                })

        except Exception as e:
            print(f"[PDFPaperLoader] Error loading {pdf_path}: {e}")

        return documents

    def load_all_papers(self) -> List[Dict[str, Any]]:
        """Scans the papers directory and extracts all PDF documents."""
        if not os.path.exists(self.papers_dir):
            print(f"[PDFPaperLoader] Directory does not exist: {self.papers_dir}")
            return []

        all_docs = []
        for file in os.listdir(self.papers_dir):
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(self.papers_dir, file)
                docs = self.load_single_pdf(full_path)
                all_docs.extend(docs)

        print(f"[PDFPaperLoader] Loaded {len(all_docs)} pages across PDF literature.")
        return all_docs
