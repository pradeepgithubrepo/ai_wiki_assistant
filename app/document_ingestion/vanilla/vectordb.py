import os
from langchain_community.vectorstores import Chroma
from app.utils.llm_utils import Helperclass
from langchain.schema import Document
import numpy as np
from typing import List
from langchain_community.vectorstores.utils import filter_complex_metadata

class VectorStoreHandler:
    def __init__(self, persist_directory="data/chroma_db_v2"):
        self.persist_directory = persist_directory
        self.helper = Helperclass()
        self.embedding = self.helper.gemini_client()

        # Initialize Chroma with persistence
        self.vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding
        )


    def similarity_search(self, query, k=10):
        """
        Run semantic search in Chroma.
        """
        return self.vectordb.similarity_search(query, k=k)
    


    def add_documents(self, docs: list[dict]):
        """
        Adds new document chunks to ChromaDB.
        Assumes enrichment (tags, summary, chunk_id) is already done.
        Converts complex metadata (lists) to strings for Chroma.
        """
        if not docs:
            return

        added = 0
        for doc in docs:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            chunk_id = metadata.get("chunk_id") or str(hash(content))

            # Skip if already exists
            existing = self.vectordb.get(ids=[chunk_id])
            if existing and len(existing["ids"]) > 0:
                print(f"⚡ Skipping {chunk_id} (already exists in DB)")
                continue

            # Convert lists in metadata to comma-separated strings
            metadata_clean = {}
            for k, v in metadata.items():
                if isinstance(v, list):
                    metadata_clean[k] = ", ".join(map(str, v))
                else:
                    metadata_clean[k] = v

            # Add chunk
            self.vectordb.add_texts([content], ids=[chunk_id], metadatas=[metadata_clean])
            added += 1

        if added > 0:
            self.vectordb.persist()
            print(f"✅ Added {added} new chunks with tags + summary metadata to ChromaDB")
        else:
            print("⚡ No new chunks were added (all exist already).")





