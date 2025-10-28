import os
import re
import hashlib
from typing import List, Dict, Any, Iterable, Tuple
from langchain_community.document_loaders import PyPDFLoader
from app.document_ingestion.vanilla.tagger import Tagger


def _stable_doc_id(file_path: str) -> str:
    """Stable ID per PDF (based on file path)."""
    return hashlib.sha1(file_path.encode("utf-8")).hexdigest()  # 40-char hex


def _is_heading(line: str) -> bool:
    """
    Heuristic heading detector:
    - short line
    - Title Case or ALL CAPS
    - optional numbering prefix like '1.' or '2.3.4'
    """
    line = line.strip()
    if not line:
        return False
    if len(line) > 100:
        return False

    if re.match(r"^\d+(\.\d+)*[\.\)]?\s+[A-Za-z]", line):
        return True

    if len(line) >= 3 and line.upper() == line and re.search(r"[A-Z]", line):
        return True

    words = [w for w in re.split(r"\s+", line) if w]
    if words:
        cap_words = sum(1 for w in words if re.match(r"^[A-Z][A-Za-z0-9\-\(\)/]*$", w))
        if cap_words / max(1, len(words)) > 0.6:
            return True

    return False


def _first_non_empty_line(text: str) -> str:
    for l in text.splitlines():
        ls = l.strip()
        if ls:
            return ls
    return ""


class SectionChunker:
    """
    Recursively loads PDFs and produces section-aware chunks.
    Each chunk has:
      - content
      - metadata: {doc_id, file_name, file_path, section_title, page_start, page_end, tags, summary}
    """

    def __init__(
        self,
        root_dir: str,
        min_section_chars: int = 400,
        max_section_chars: int = 2000,
        overlap_chars: int = 200,
        max_docs: int = 20,
    ):
        self.root_dir = root_dir
        self.min_section_chars = min_section_chars
        self.max_section_chars = max_section_chars
        self.overlap_chars = overlap_chars
        self.max_docs = max_docs
        self.tagger = Tagger()

    def _iter_pdf_paths(self) -> Iterable[str]:
        files = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for f in filenames:
                if f.lower().endswith(".pdf"):
                    files.append(os.path.join(dirpath, f))
        return files[: self.max_docs]

    def _load_pdf_pages(self, file_path: str) -> List[Tuple[int, str]]:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return [(i + 1, d.page_content or "") for i, d in enumerate(docs)]

    def _pages_to_sections(self, pages: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
        sections = []
        i = 0
        while i < len(pages):
            start_page, start_text = pages[i]
            candidate = _first_non_empty_line(start_text)
            section_title = candidate if _is_heading(candidate) else ""

            buf = start_text.strip()
            end_page = start_page

            j = i + 1
            while len(buf) < self.min_section_chars and j < len(pages):
                end_page, next_text = pages[j]
                buf += "\n\n" + next_text.strip()
                j += 1

            if len(buf) > self.max_section_chars:
                buf = buf[: self.max_section_chars]
                cut = max(buf.rfind("\n\n"), buf.rfind(". "))
                if cut > 0 and cut >= self.max_section_chars - 400:
                    buf = buf[:cut]

            i = j
            sections.append(
                {
                    "content": buf,
                    "page_start": start_page,
                    "page_end": end_page,
                    "section_title": section_title,
                }
            )
        return sections

    def chunk_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        pages = self._load_pdf_pages(file_path)
        if not pages:
            return []

        file_name = os.path.basename(file_path)
        doc_id = _stable_doc_id(file_path)

        sections = self._pages_to_sections(pages)
        chunks = []
        for sec in sections:
            chunks.append(
                {
                    "content": sec["content"],
                    "metadata": {
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "file_path": file_path,
                        "section_title": sec["section_title"],
                        "page_start": sec["page_start"],
                        "page_end": sec["page_end"]
                    },
                }
            )


        return chunks

    def chunk_corpus(self) -> List[Dict[str, Any]]:
        all_chunks: List[Dict[str, Any]] = []
        print(f"[SectionChunker] Scanning: {self.root_dir}")
        for pdf_path in self._iter_pdf_paths():
            print(f"[SectionChunker] Reading: {pdf_path}")
            try:
                chunks = self.chunk_pdf(pdf_path)
                print(f"[SectionChunker] -> {len(chunks)} chunks")
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"[SectionChunker] ERROR {pdf_path}: {e}")
        print(f"[SectionChunker] DONE. Total chunks: {len(all_chunks)}")
        return all_chunks
