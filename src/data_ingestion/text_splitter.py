"""
Semantic Text Splitter for AthleteIQ.
Splits research literature, telemetry logs, and medical notes into coherent chunks
while preserving domain boundaries and clinical context.
"""

from typing import List, Dict, Any, Optional
import copy


class AthleteIQTextSplitter:
    """
    Splits text recursively using domain-aware separators (sports science headings,
    medical report sections, and clinical paragraphs).
    """

    DEFAULT_SEPARATORS = [
        "\n--------------------------------------------------------------------------------\n",
        "\n## ",
        "\n### ",
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        " ",
        ""
    ]

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursive chunking core algorithm."""
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._split_text(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.append(s)

        # Merge splits up to chunk_size with chunk_overlap
        current_chunk = []
        current_length = 0

        for piece in good_splits:
            piece_len = len(piece)
            if current_length + piece_len + len(separator) > self.chunk_size and current_chunk:
                combined = separator.join(current_chunk).strip()
                if combined:
                    final_chunks.append(combined)

                # Keep overlap pieces from the tail of current_chunk
                while current_length > self.chunk_overlap and len(current_chunk) > 1:
                    popped = current_chunk.pop(0)
                    current_length -= (len(popped) + len(separator))

            current_chunk.append(piece)
            current_length += piece_len + len(separator)

        if current_chunk:
            combined = separator.join(current_chunk).strip()
            if combined:
                final_chunks.append(combined)

        return final_chunks

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes a list of document dicts with 'page_content' and 'metadata',
        splits each document into chunks, and attaches chunk metadata.
        """
        chunked_docs = []

        for doc_idx, doc in enumerate(documents):
            content = doc.get("page_content", "")
            base_meta = copy.deepcopy(doc.get("metadata", {}))

            if not content.strip():
                continue

            # Short telemetry logs that already fit comfortably within chunk_size
            # are retained as unified semantic units
            if len(content) <= self.chunk_size:
                chunks = [content.strip()]
            else:
                chunks = self._split_text(content, self.separators)

            total_chunks = len(chunks)
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_meta = copy.deepcopy(base_meta)
                clean_name = chunk_meta.get("file_name", "doc").replace(".csv", "").replace(".pdf", "").replace(".txt", "")
                page_part = f"_p{chunk_meta['page_number']}" if "page_number" in chunk_meta else ""
                date_part = f"_{chunk_meta['date']}" if "date" in chunk_meta else ""
                player_part = f"_{chunk_meta['player_id'].replace(' ', '')}" if "player_id" in chunk_meta else ""
                
                chunk_id = f"{clean_name}{player_part}{date_part}{page_part}_d{doc_idx}_c{chunk_idx + 1}"
                chunk_meta.update({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx + 1,
                    "total_chunks": total_chunks,
                    "character_count": len(chunk_text)
                })

                chunked_docs.append({
                    "page_content": chunk_text,
                    "metadata": chunk_meta
                })

        return chunked_docs
