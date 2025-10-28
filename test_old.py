from app.utils.llm_utils import Helperclass
from langchain_core.prompts import ChatPromptTemplate

import asyncio
from app.rag_pipeline import RAGPipeline

if __name__ == "__main__":
    rag = RAGPipeline()

    # Uncomment the first run to build indexes
    asyncio.run(rag.index_documents())

    # Test queries
    queries = [
        "observablity component?",
    ]

    for q in queries:
        print(f"\n❓ Query: {q}")
        result = rag.query_documents(q)
        print(f"💡 Answer: {result['answer']}")
        print(f"📌 Sources: {result['sources']}")




