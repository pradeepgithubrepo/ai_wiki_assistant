from fastapi import FastAPI, Query
from pydantic import BaseModel
import uvicorn
from typing import List, Dict
from app.document_ingestion.vanilla.retreiver import WeightedRetriever

# Initialize FastAPI app
app = FastAPI(title="AI Wiki Assistant Search API", version="1.0")

# Initialize your retriever once
retriever = WeightedRetriever(k=20)

class SearchResponse(BaseModel):
    final_summary: str
    ranked_docs: List[Dict[str, str]]  # link, summary, tags
    unique_tags: List[str]

@app.get("/search", response_model=SearchResponse)
def search(query: str = Query(..., min_length=3, description="Query string for RAG search")):
    """
    Weighted retrieval of documents based on query.
    Returns final summary, top documents with tags, and unique tags.
    """
    final_summary, ranked_docs, unique_tags = retriever.search(query)
    return SearchResponse(
        final_summary=final_summary,
        ranked_docs=ranked_docs,
        unique_tags=unique_tags
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
