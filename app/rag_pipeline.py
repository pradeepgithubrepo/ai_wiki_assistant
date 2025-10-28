from typing import List, Dict, Any
from app.document_ingestion.loader import load_documents_from_dir
from app.document_ingestion.embedder import Embedder
from app.utils.llm_utils import Helperclass


class RAGPipeline:
    def __init__(self, data_dir: str = "data/wiki_pdfs", k: int = 5):
        """
        RAG Pipeline with 2 master methods:
        - index_documents(): Build and persist Chroma + BM25
        - query_documents(): Query hybrid retriever + LLM
        """
        self.data_dir = data_dir
        self.k = k
        self.embedder = Embedder()
        self.helper = Helperclass()
        self.llm = self.helper.openai_client()  # Azure OpenAI client

    async def index_documents(self) -> None:
        """One-time indexing of PDFs into Chroma + BM25."""
        docs = await load_documents_from_dir(self.data_dir)
        print(f"📂 Loaded {len(docs)} documents")

        # Split into chunks
        chunks = self.embedder.chunk_documents(docs)
        print(f"🔗 Created {len(chunks)} chunks")

        # Persist into Chroma
        self.embedder.embed_chunks(chunks)
        print("✅ Indexed & persisted into Chroma DB")

        # Persist BM25
        self.embedder.save_bm25_index(docs)
        print("✅ Persisted BM25 index")

    def query_documents(self, query: str) -> Dict[str, Any]:
        """Run full RAG pipeline: retrieve → construct prompt → LLM answer."""
        # Load both indexes
        self.embedder.load_vectorstore()
        self.embedder.load_bm25_index()
        self.embedder.build_hybrid_retriever(k=self.k)

        # Step 1: Hybrid retrieval
        results = self.embedder.hybrid_search(query, top_k=self.k)

        # Deduplicate by source
        seen = set()
        unique_results = []
        for r in results:
            src = r["metadata"]["source"]
            if src not in seen:
                seen.add(src)
                unique_results.append(r)

        # Step 2: Build context
        context = "\n\n".join([r["text"] for r in unique_results[:self.k]])

        prompt = f"""
You are an expert assistant specializing in Microsoft Azure documentation.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don’t know".

Context:
{context}

Question:
{query}

Answer (summarize clearly, step-by-step if applicable):
"""

        # Step 3: Query LLM
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        # Step 4: Collect sources
        sources = [r["metadata"]["source"] for r in unique_results[:self.k]]

        return {
            "answer": answer.strip(),
            "sources": sources
        }
