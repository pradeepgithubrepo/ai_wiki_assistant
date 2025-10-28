from app.utils.llm_utils import Helperclass
from app.document_ingestion.graph.neo4j_client import Neo4jClient

class QueryEngine:
    def __init__(self):
        self.llm = Helperclass().openai_client()
        self.neo4j_client = Neo4jClient()

    def query(self, user_query: str, top_k: int = 10, min_similarity: float = 0.35):
        # Fetch top similar documents above threshold
        docs = self.neo4j_client.find_similar_documents(
            query=user_query,
            top_k=top_k,
            min_similarity=min_similarity
        )

        if not docs:
            return {"summary": "No relevant documents found.", "links": []}

        # Prepare context for LLM summarization (limit per doc to avoid long prompts)
        context_texts = [d["content"][:1500] for d in docs]
        context = "\n\n".join(context_texts)

        prompt = f"""
        You are an AI assistant. Summarize the following documents concisely in response to the query.
        Provide:
        1. A clear and concise summary.
        2. Do not hallucinate or add info outside the provided context.
        Query: {user_query}
        Context:
        {context}
        """

        response = self.llm.predict(prompt)

        # Extract unique file paths and include weights + similarity
        links = []
        seen_paths = set()
        for d in docs:
            path = d.get("file_path")
            if path and path not in seen_paths:
                links.append({
                    "file_path": path,
                    "file_name": d.get("file_name"),
                    "similarity": d.get("similarity"),
                    "weights": d.get("weights")
                })
                seen_paths.add(path)

        return {"summary": response, "links": links}