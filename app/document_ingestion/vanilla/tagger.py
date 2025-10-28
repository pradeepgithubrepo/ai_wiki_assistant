# app/document_ingestion/vanilla/tagger.py

from app.utils.llm_utils import Helperclass
from app.document_ingestion.vanilla.vectordb import VectorStoreHandler
from collections import Counter
import json
import uuid

class Tagger:
    def __init__(self):
        self.llm = Helperclass().openai_client()
        self.vectordb_handler = VectorStoreHandler()

 
    def generate_tags_and_summary(self, text: str, max_tags: int = 10, word_limit: int = 50) -> tuple[list[str], str]:
        """Generate tags (with frequency counts) and summary in one LLM call."""
        if not text.strip():
            return [], ""

        prompt = f"""
        You are an assistant that extracts metadata from documents.

        From the following text, do two things:
        1. Generate {max_tags} short, high-level tags (1–4 words each).
        2. Summarize the text in a single sentence of about {word_limit} words.

        Return ONLY valid JSON in the format:
        {{
            "tags": ["tag1", "tag2", "..."],
            "summary": "short summary here"
        }}

        Text:
        {text[:2500]}
        """

        response = self.llm.predict(prompt).strip()

        try:
            data = json.loads(response)
            tags = [t.strip().lower() for t in data.get("tags", []) if t.strip()]

            # Count frequencies
            counts = Counter(tags)

            # Rebuild tags with counts
            tags_with_counts = [f"{tag} ({count})" for tag, count in counts.items()]

            # Keep order of first appearance & limit
            unique_tags = []
            for tag in tags_with_counts:
                if len(unique_tags) < max_tags:
                    unique_tags.append(tag)

            summary = data.get("summary", "").strip()
            words = summary.split()
            if len(words) > word_limit + 5:
                summary = " ".join(words[:word_limit]) + "..."
            return unique_tags, summary

        except Exception as e:
            print(f"⚠️ JSON parse failed, falling back. Raw response: {response}")
            return [], ""



    def enrich_chunks_with_metadata(self, chunks: list[dict]) -> list[dict]:
        """Enrich chunks with tags + summary per doc_id and persist all chunks into DB."""
        if not chunks:
            return chunks

        enriched_chunks = []
        docs_text = {}

        # group by doc_id
        for ch in chunks:
            doc_id = ch["metadata"]["doc_id"]
            docs_text.setdefault(doc_id, []).append(ch)

        for doc_id, doc_chunks in docs_text.items():

            # Check if *any* chunk for this doc already exists in DB
            existing = self.vectordb_handler.vectordb.get(where={"doc_id": doc_id})
            if existing and len(existing["ids"]) > 0:
                print(f"⚡ Skipping metadata enrichment for {doc_id} (already exists in DB)")
                enriched_chunks.extend(doc_chunks)
                continue

            # ✅ Generate metadata ONCE per doc (using first chunk content)
            first_chunk_text = doc_chunks[0]["content"]
            tags, summary = self.generate_tags_and_summary(first_chunk_text, max_tags=10, word_limit=20)

            chunk_docs = []
            for idx, ch in enumerate(doc_chunks):
                ch_metadata = {
                    "chunk_id": f"{doc_id}_chunk_{idx}_{uuid.uuid4().hex[:8]}",
                    "doc_id": doc_id,
                    "file_name": ch["metadata"].get("file_name"),
                    "file_path": ch["metadata"].get("file_path"),
                    "section_title": ch["metadata"].get("section_title"),
                    "page_start": ch["metadata"].get("page_start"),
                    "page_end": ch["metadata"].get("page_end"),
                    # ✅ Apply same tags/summary to every chunk
                    "tags": tags,
                    "summary": summary,
                }

                # update original chunk
                ch["metadata"].update(ch_metadata)
                enriched_chunks.append(ch)

                # prepare for DB insert (all chunks go in!)
                chunk_docs.append({
                    "content": ch["content"],
                    "metadata": ch_metadata
                })

            # ✅ Persist ALL chunks for this doc
            if chunk_docs:
                self.persist_to_vectordb(chunk_docs)

        return enriched_chunks



    def persist_to_vectordb(self, docs: list[dict]):
        """
        Send docs to ChromaDB (handles deduplication).
        """
        if not docs:
            return
        self.vectordb_handler.add_documents(docs)
        print(f"✅ Persisted {len(docs)} documents into VectorDB.")