from neo4j import GraphDatabase
import numpy as np
from app.utils.llm_utils import Helperclass
from app.document_ingestion.graph import config

class Neo4jClient:
    def __init__(self, uri=None, user=None, password=None, database=None):
        self.driver = GraphDatabase.driver(
            uri or config.NEO4J_URI,
            auth=(user or config.NEO4J_USER, password or config.NEO4J_PASSWORD)
        )
        self.embeddings = Helperclass().gemini_client()
        self.database = database or config.NEO4J_DATABASE

    def close(self):
        self.driver.close()

    def test_connection(self):
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 'Neo4j connection successful!' AS message")
                return result.single()["message"]
        except Exception as e:
            return f"Connection failed: {str(e)}"

    def create_document_node(self, doc):
        text = doc.page_content
        if not isinstance(text, str):
            text = str(text)
        text = text[:3000]  # truncate if too long

        metadata = doc.metadata
        embedding = self.embeddings.embed_query(text)

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                CREATE (d:Document {
                    content: $content,
                    embedding: $embedding,
                    file_name: $file_name,
                    file_path: $file_path,
                    first_paragraph: $first_paragraph,
                    heading: $heading
                })
                RETURN d
                """,
                content=text,
                embedding=embedding,
                file_name=metadata.get("file_name", "unknown"),
                file_path=metadata.get("file_path", "unknown"),
                first_paragraph=metadata.get("first_paragraph", text.split("\n\n")[0] if "\n\n" in text else text[:200]),
                heading=metadata.get("heading", text.split("\n")[0] if "\n" in text else "")
            )
            return result.single()[0]

    def find_similar_documents(self, query: str, top_k: int = 10, min_similarity: float = 0.35,
                            weights: dict = None):
        if weights is None:
            weights = {"embedding": 0.6, "file_name": 0.1, "heading": 0.15, "first_paragraph": 0.15}

        query_embedding = self.embeddings.embed_query(query)
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (d:Document)
                WHERE d.embedding IS NOT NULL
                RETURN d.content AS content, d.file_path AS file_path, d.file_name AS file_name, 
                    d.embedding AS embedding, d.first_paragraph AS first_paragraph, d.heading AS heading
                """
            )

            docs = []
            seen_paths = set()
            for record in result:
                file_path = record["file_path"]
                if file_path in seen_paths:
                    continue

                sim_embedding = self._cosine_similarity(query_embedding, record["embedding"])
                sim_file_name = self._cosine_similarity(
                    self.embeddings.embed_query(query), self.embeddings.embed_query(record["file_name"])
                )
                sim_heading = self._cosine_similarity(
                    self.embeddings.embed_query(query), self.embeddings.embed_query(record.get("heading", ""))
                )
                sim_first_para = self._cosine_similarity(
                    self.embeddings.embed_query(query), self.embeddings.embed_query(record.get("first_paragraph", ""))
                )

                combined_score = (
                    weights["embedding"] * sim_embedding +
                    weights["file_name"] * sim_file_name +
                    weights["heading"] * sim_heading +
                    weights["first_paragraph"] * sim_first_para
                )

                if combined_score >= min_similarity:
                    docs.append({
                        "content": record["content"],
                        "file_path": file_path,
                        "file_name": record["file_name"],
                        "similarity": combined_score,
                        "weights": {
                            "embedding": sim_embedding * weights["embedding"],
                            "file_name": sim_file_name * weights["file_name"],
                            "heading": sim_heading * weights["heading"],
                            "first_paragraph": sim_first_para * weights["first_paragraph"]
                        }
                    })
                    seen_paths.add(file_path)

        docs = sorted(docs, key=lambda x: x["similarity"], reverse=True)
        return docs[:top_k]

    @staticmethod
    def _cosine_similarity(vec1, vec2):
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    
    def cleanup_documents(self):
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (d:Document) DETACH DELETE d")
        print("All existing Document nodes deleted from Neo4j.")