# app/retrieval/retriever.py

from typing import List, Tuple
from langchain.schema import Document
from app.document_ingestion.vanilla.vectordb import VectorStoreHandler
from app.utils.llm_utils import Helperclass
import numpy as np
from typing import List, Dict, Any, Tuple


class WeightedRetriever:
    def __init__(self, k: int = 20):
        self.vectordb = VectorStoreHandler()
        self.llm = Helperclass().openai_client()
        self.embedding_model = Helperclass().gemini_client()
        self.k = k

    def _cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    def _jaccard(self, set1, set2):
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    def search(self, query: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        """
        Perform weighted retrieval with frequency-aware tag weighting:
        - Default: 0.2 * embedding similarity, 0.4 * tag overlap, 0.4 * summary similarity
        - If max tag frequency > 1.5x next tag frequency:
            - Increase tag weight to 0.6
            - Reduce embedding + summary weights to 0.2 each
        Filters out documents with final_score < 0.5

        Returns:
            final_summary: str (concise overall summary from top docs)
            ranked_docs: List[Dict] with {link, summary, tags}
            unique_tags: List[str] (deduplicated tags from top docs)
        """
        results: List[Document] = self.vectordb.similarity_search(query, k=self.k)
        if not results:
            return "No results found.", [], []

        print("\n=== Raw Retrieved Chunks ===")
        for i, doc in enumerate(results, start=1):
            link = doc.metadata.get("file_path") or doc.metadata.get("source") or doc.metadata.get("file_name")
            chunk_preview = doc.page_content[:200].replace("\n", " ")
            print(f"{i}. [Doc: {link}] Chunk Preview: {chunk_preview}...")

        query_vec = self.embedding_model.embed_query(query)
        query_tokens = set([w.lower() for w in query.split() if len(w) > 2])

        # --- Count tag frequencies across retrieved docs ---
        tag_freq = {}
        for doc in results:
            tags = doc.metadata.get("tags", "")
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            for t in tags:
                tag_freq[t.lower()] = tag_freq.get(t.lower(), 0) + 1

        # Determine dominant tag condition
        sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_tags) >= 2 and sorted_tags[0][1] > 1.5 * sorted_tags[1][1]:
            emb_w, tag_w, sum_w = 0.2, 0.6, 0.2
            print(f"⚡ Frequency boost: '{sorted_tags[0][0]}' dominates, using tag_w={tag_w}")
        else:
            emb_w, tag_w, sum_w = 0.2, 0.4, 0.4

        scored = []
        for doc in results:
            meta = doc.metadata
            tags = meta.get("tags", "")
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            summary = meta.get("summary", "")

            # --- embedding score ---
            doc_vec = self.embedding_model.embed_query(doc.page_content[:500])
            emb_score = self._cosine_similarity(query_vec, doc_vec)

            # --- tag overlap score ---
            tag_tokens = set([t.lower() for t in tags])
            tag_score = self._jaccard(query_tokens, tag_tokens)

            # --- summary similarity score ---
            if summary:
                sum_vec = self.embedding_model.embed_query(summary)
                sum_score = self._cosine_similarity(query_vec, sum_vec)
            else:
                sum_score = 0.0

            # --- weighted final score ---
            final_score = emb_w * emb_score + tag_w * tag_score + sum_w * sum_score

            scored.append((final_score, doc))

        # Sort and dedup by link
        scored.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        top_docs = []
        for _, doc in scored:
            link = doc.metadata.get("file_path") or doc.metadata.get("source") or doc.metadata.get("file_name")
            if link and link not in seen:
                seen.add(link)
                top_docs.append(doc)
            if len(top_docs) >= 10:
                break

        # --- Build snippets for final summary ---
        collected_text = []
        for doc in top_docs:
            meta = doc.metadata
            snippet_tags = meta.get("tags", [])
            snippet = f"Summary: {meta.get('summary','')}\nTags: {', '.join(snippet_tags)}\nContent: {doc.page_content[:300]}"
            collected_text.append(snippet)

        prompt = f"""
        You are an assistant. Based on the following retrieved document snippets,
        write one concise summary (3–4 sentences) relevant to the query.

        Query: {query}

        Retrieved Snippets:
        {chr(10).join(collected_text)}
        """
        final_summary = self.llm.invoke(prompt).content.strip()

        # --- Build ranked output ---
        ranked_docs = []
        unique_tags = set()
        for doc in top_docs:
            meta = doc.metadata
            tags = meta.get("tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]

            ranked_docs.append({
                "link": meta.get("file_path") or meta.get("source") or meta.get("file_name"),
                "summary": meta.get("summary", ""),
                "tags": tags
            })
            unique_tags.update(tags)

        return final_summary, ranked_docs, sorted(unique_tags)


