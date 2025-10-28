import os
import pickle
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import CrossEncoder
from collections import defaultdict
from app.utils.llm_utils import Helperclass



class Embedder:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, persist_directory: str = "chroma_db"):
        """
        Initialize Embedder with Gemini embeddings, Chroma persistence, and BM25 persistence.
        """
        helper = Helperclass()
        self.embedding_model = helper.gemini_client()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.persist_directory = persist_directory
        self.bm25_index_path = os.path.join(persist_directory, "bm25.pkl")
        self.db = None
        self.hybrid_retriever = None
        self.bm25_retriever = None
        self.semantic_retriever = None
        self.k = 5

        # Predefined query expansion dictionary
        self.query_expansion_map = {
            "observability": ["monitoring", "logging", "telemetry", "tracing", "metrics"],
            "capability hub": ["capability model", "competency hub", "skills hub"],
            "metadata": ["data catalog", "data dictionary", "data lineage"],
            "devops": ["ci/cd", "continuous integration", "continuous deployment", "infrastructure as code"],
        }

    # -----------------------
    # INDEXING PHASE
    # -----------------------
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Splits documents into smaller chunks."""
        chunks = []
        for doc in documents:
            if not doc["content"]:
                continue
            cleaned_content = doc["content"].replace("\x00", "")
            splits = self.splitter.split_text(cleaned_content)
            for idx, chunk in enumerate(splits):
                chunks.append({
                    "text": chunk,
                    "metadata": {"source": doc["source"], "chunk": idx}
                })
        return chunks

    def embed_chunks(self, chunks: List[Dict[str, Any]]):
        """Generate embeddings for chunks and persist into Chroma."""
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        self.db = Chroma.from_texts(
            texts=texts,
            embedding=self.embedding_model,
            metadatas=metadatas,
            persist_directory=self.persist_directory
        )
        return self.db

    def save_bm25_index(self, documents: List[Dict[str, Any]], k: int = 5):
        """Build and persist BM25 index to disk."""
        doc_objs = [
            Document(page_content=doc["content"], metadata={"source": doc["source"]})
            for doc in documents if doc.get("content")
        ]
        self.bm25_retriever = BM25Retriever.from_documents(doc_objs)
        self.bm25_retriever.k = k

        with open(self.bm25_index_path, "wb") as f:
            pickle.dump(self.bm25_retriever, f)

    # -----------------------
    # QUERY PHASE
    # -----------------------
    def load_vectorstore(self):
        """Load existing Chroma DB without re-embedding."""
        self.db = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory
        )
        return self.db

    def load_bm25_index(self):
        """Load persisted BM25 retriever from disk."""
        if os.path.exists(self.bm25_index_path):
            with open(self.bm25_index_path, "rb") as f:
                self.bm25_retriever = pickle.load(f)
            return self.bm25_retriever
        else:
            raise FileNotFoundError("BM25 index not found. Run save_bm25_index() first.")

    def build_hybrid_retriever(self, k: int = 5):
        """Prepare retrievers (semantic + BM25) for RRF."""
        if not self.db:
            raise RuntimeError("You must call embed_chunks() or load_vectorstore() first.")
        if not self.bm25_retriever:
            raise RuntimeError("You must call save_bm25_index() or load_bm25_index() first.")

        self.semantic_retriever = self.db.as_retriever(search_kwargs={"k": k})
        self.k = k

    def _expand_query(self, query: str) -> List[str]:
        """Expand query using predefined synonyms."""
        expanded_queries = [query]
        for keyword, synonyms in self.query_expansion_map.items():
            if keyword.lower() in query.lower():
                expanded_queries.extend(synonyms)
        return list(set(expanded_queries))

    def hybrid_search(self, query: str, top_k: int = 10, rrf_k: int = 50) -> List[Dict[str, Any]]:
        """
        Run hybrid search with Reciprocal Rank Fusion (RRF).
        Expands queries and merges results.
        """
        if not self.semantic_retriever or not self.bm25_retriever:
            raise ValueError("Hybrid retrievers not built. Call build_hybrid_retriever().")

        expanded_queries = self._expand_query(query)
        print(f"\n🔎 Original Query: {query}")
        print(f"📌 Expanded Queries: {expanded_queries}\n")

        scores = defaultdict(float)
        docs_map = {}

        for q in expanded_queries:
            print(f"➡️ Running sub-query: {q}")

            semantic_results = self.semantic_retriever.invoke(q)
            bm25_results = self.bm25_retriever.invoke(q)

            print(f"   🔹 Semantic hits: {[d.metadata.get('source') for d in semantic_results]}")
            print(f"   🔸 BM25 hits: {[d.metadata.get('source') for d in bm25_results]}")

            for rank, doc in enumerate(semantic_results, start=1):
                key = (doc.page_content.strip(), str(doc.metadata))
                scores[key] += 1 / (rrf_k + rank)
                docs_map[key] = doc

            for rank, doc in enumerate(bm25_results, start=1):
                key = (doc.page_content.strip(), str(doc.metadata))
                scores[key] += 1 / (rrf_k + rank)
                docs_map[key] = doc

        ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        ranked_docs = [docs_map[k] for k in ranked_keys[:top_k]]

        print("\n🏆 Final Top Sources:", [d.metadata.get("source") for d in ranked_docs])

        return [{"text": d.page_content, "metadata": d.metadata} for d in ranked_docs]


#     def hybrid_search_v1(self, query: str, top_k: int = 10, rrf_k: int = 50) -> Dict[str, Any]:
#         """
#         Hybrid search with query expansion, RRF fusion, cross-encoder reranking,
#         and LLM answer with references.
#         Returns best match, top_k docs, and final LLM response.
#         """
#         if not self.semantic_retriever or not self.bm25_retriever:
#             raise ValueError("Hybrid retrievers not built. Call build_hybrid_retriever().")

#         # ✅ Query Expansion (multi-query, step-back, etc.)
#         expanded_queries = self._expand_query(query)
#         print(f"\n🔎 Original Query: {query}")
#         print(f"📌 Expanded Queries: {expanded_queries}\n")

#         scores = defaultdict(float)
#         docs_map = {}

#         # ✅ Hybrid Retrieval with Reciprocal Rank Fusion
#         for q in expanded_queries:
#             semantic_results = self.semantic_retriever.invoke(q)
#             bm25_results = self.bm25_retriever.invoke(q)

#             for rank, doc in enumerate(semantic_results, start=1):
#                 key = (doc.page_content.strip(), str(doc.metadata))
#                 scores[key] += 1 / (rrf_k + rank)
#                 docs_map[key] = doc

#             for rank, doc in enumerate(bm25_results, start=1):
#                 key = (doc.page_content.strip(), str(doc.metadata))
#                 scores[key] += 1 / (rrf_k + rank)
#                 docs_map[key] = doc

#         # Initial ranking with RRF
#         ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
#         candidate_docs = [docs_map[k] for k in ranked_keys[: rrf_k]]

#         # ✅ Cross-Encoder Reranking
#         reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#         pairs = [(query, d.page_content) for d in candidate_docs]
#         rerank_scores = reranker.predict(pairs)
#         reranked_docs = [doc for _, doc in sorted(zip(rerank_scores, candidate_docs), reverse=True)]

#         # Top-k final docs
#         top_docs = reranked_docs[:top_k]
#         best_match = top_docs[0]

#         return {
#             "query": query,
#             "best_match": {
#                 "text": best_match.page_content,
#                 "metadata": best_match.metadata
#             },
#             "top_k_docs": [
#                 {"text": d.page_content, "metadata": d.metadata} for d in top_docs
#             ]
#         }
# def generate_answer_with_citations(llm, query: str, top_docs: List[Dict[str, Any]]) -> str:
#     """
#     Generate final answer from retrieved documents with citations.
#     """
#     context = "\n\n".join([
#         f"[{i+1}] Source: {d['metadata'].get('source', 'unknown')}\n{d['text']}"
#         for i, d in enumerate(top_docs)
#     ])

#     prompt_template = PromptTemplate(
#         input_variables=["query", "context"],
#         template=(
#             "You are a helpful assistant. Use only the context below to answer the query.\n"
#             "Always cite sources in square brackets like [1], [2].\n\n"
#             "Context:\n{context}\n\n"
#             "Query: {query}\n\n"
#             "Answer clearly, with references."
#         )
#     )

#     llm_chain = LLMChain(llm=llm, prompt=prompt_template)
#     return llm_chain.run(query=query, context=context)
