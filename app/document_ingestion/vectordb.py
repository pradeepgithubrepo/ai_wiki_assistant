import os
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from app.utils.llm_utils import Helperclass


class VectorDB:
    def __init__(self, persist_dir: str = "data/chroma_db"):
        """
        Initialize Chroma vector database.
        """
        helper = Helperclass()
        self.embedding_model = helper.gemini_client()
        self.persist_dir = persist_dir

        # Create or load Chroma DB
        self.db = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.persist_dir
        )

    def add_documents(self, embedded_docs: List[Dict[str, Any]]):
        """
        Add embedded documents to Chroma.
        """
        docs = [
            Document(
                page_content=doc["text"],
                metadata=doc["metadata"]
            )
            for doc in embedded_docs
        ]
        self.db.add_documents(docs)
        self.db.persist()

    def similarity_search(self, query: str, top_k: int = 5, score_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Run similarity search on Chroma DB with relevance scoring.
        """
        results = self.db.similarity_search_with_relevance_scores(query, k=top_k)

        filtered = [
            {"text": r[0].page_content, "metadata": r[0].metadata, "score": r[1]}
            for r in results if r[1] >= score_threshold
        ]

        return filtered

