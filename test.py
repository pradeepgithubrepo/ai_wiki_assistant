from app.utils.llm_utils import Helperclass
from langchain_core.prompts import ChatPromptTemplate

# def test_openai_connection():
#     try:
#         helper = Helperclass()
#         llm = helper.openai_client()

#         # Basic test prompt
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", "You are a helpful assistant."),
#             ("user", "Talk me about dinosaurs.")
#         ])

#         chain = prompt | llm
#         response = chain.invoke({"input": ""})
#         print("✅ OpenAI is accessible. Response:")
#         print(response.content)

#     except Exception as e:
#         print("❌ Failed to connect to OpenAI:")
#         print(e)

# if __name__ == "__main__":
#     test_openai_connection()

# tests/test_loader.py

# import asyncio
# from app.document_ingestion.loader import load_file, load_documents_from_dir
# async def main():
#     docs = await load_documents_from_dir("./data/wiki_pdfs")  # adjust path as needed
#     print(f"Loaded {len(docs)} documents")
#     for doc in docs:
#         print("=" * 80)
#         print(f"Source: {doc['source']}")
#         print(f"Path: {doc['file_path']}")
#         print(f"Preview: {doc['content'][:200]}...")  # show first 200 chars

# if __name__ == "__main__":
#     asyncio.run(main())

# import asyncio
# from app.rag_pipeline import RAGPipeline

# if __name__ == "__main__":
#     rag = RAGPipeline()

#     # Uncomment the first run to build indexes
#     asyncio.run(rag.index_documents())

#     # Test queries
#     queries = [
#         "setting up unity catalog?",
#     ]

#     for q in queries:
#         print(f"\n❓ Query: {q}")
#         result = rag.query_documents(q)
#         print(f"💡 Answer: {result['answer']}")
#         print(f"📌 Sources: {result['sources']}")

# from app.document_ingestion.graph.neo4j_client import Neo4jClient

# if __name__ == "__main__":
#     # Step 1: Create client
#     client = Neo4jClient()

#     # Step 2: Test connection
#     if client.test_connection():
#         print("✅ Connection successful!")

#         # Step 3: Create a sample node
#         node = client.create_node(
#             label="Document",
#             properties={"title": "Azure Setup Guide", "type": "pdf"}
#         )
#         print("✅ Created node:", node)

#     else:
#         print("❌ Connection failed!")

#     # Step 4: Close connection
#     client.close()

# test.py

# test.py (project root)

# from app.document_ingestion.graph.pdf_loader import PDFLoader
# from app.document_ingestion.graph.neo4j_client import Neo4jClient
# from app.document_ingestion.graph.query_engine import QueryEngine
# import os

# if __name__ == "__main__":
#     # --- Load PDFs ---
#     neo_client = Neo4jClient()
#     neo_client.cleanup_documents()  # Clean up existing documents before loading new ones
#     print("Cleaned up existing documents in Neo4j.")

#     loader = PDFLoader("data/wiki_pdfs/")
#     documents = loader.load_pdfs()
#     print(f"Loaded {len(documents)} PDFs.")

#     # Store all documents in Neo4j
#     for doc in documents:
#         neo_client.create_document_chunk_node(doc)
#     print("All documents stored in Neo4j.")

#     # --- Query ---
    # engine = QueryEngine()
    # query = "setting up unity catalog"
    # result = engine.query(user_query=query, top_k=10, min_similarity=0.35)

    # # --- Display summary ---
    # print("\n=== Summary ===")
    # print(result["summary"])

    # # --- Display top links with similarity and weights ---
    # print("\n=== Top Links with Weights ===")
    # for doc in result["links"]:
    #     print(f"File: {doc['file_name']}")
    #     print(f"Path: {doc['file_path']}")
    #     print(f"Similarity: {doc['similarity']:.3f}")
    #     print(f"Weights Contribution: {doc['weights']}")
    #     print("-" * 60)


#     neo_client.close()


# Vanilla 

# test.py
# app/document_ingestion/vanilla/test.py

# from app.document_ingestion.vanilla.section_chunker import SectionChunker
# from app.document_ingestion.vanilla.tagger import Tagger

# if __name__ == "__main__":
#     # Step 1: Chunk documents
#     chunker = SectionChunker(root_dir="data/wiki_pdfs")
#     chunks = chunker.chunk_corpus()

#     print(f"Total chunks before tagging: {len(chunks)}")

#     # Step 2: Enrich chunks with tags + summary
#     tagger = Tagger()
#     enriched_chunks = tagger.enrich_chunks_with_metadata(chunks)

#     print("\n=== Sample Enriched Chunk Metadata ===")
#     for ch in enriched_chunks[:16]:
#         meta = ch["metadata"]
    
#         print(f"\nFile Nam  : {meta['file_name']}")
#         print(f"\nFile Path : {meta['file_path']}")
#         print(f"Pages     : {meta['page_start']}–{meta['page_end']}")
#         print(f"Tags      : {meta['tags']}")
#         print(f"Summary   : {meta['summary']}")
#         preview = ch['content'][:200].replace("\n", " ")
#         print(f"Content   : {preview}...")


# test.py

from app.document_ingestion.vanilla.retreiver import WeightedRetriever

def run_test():
    # Instantiate retriever
    retriever = WeightedRetriever(k=20)

    # Example query
    query = "Late Arriving dimensions"

    print(f"\n=== Running Weighted Search for Query: '{query}' ===\n")

    # Perform search
    summary, ranked_docs, unique_tags = retriever.search(query)

    # Print results
    print("\n--- Final Summary ---")
    print(summary)

    print("\n--- Top Retrieved Documents ---")
    for i, doc in enumerate(ranked_docs, start=1):
        print(f"{i}. Link: {doc['link']}")
        print(f"   Summary: {doc['summary']}")
        print(f"   Tags: {', '.join(doc['tags']) if doc['tags'] else 'None'}\n")

    print("\n--- Unique Tags Across Results ---")
    print(", ".join(unique_tags) if unique_tags else "None")

if __name__ == "__main__":
    run_test()




# test_list_vectordb.py

# from app.document_ingestion.vanilla.vectordb import VectorStoreHandler

# def list_all_contents():
#     # Initialize vector DB handler
#     vectordb_handler = VectorStoreHandler()

#     # Fetch everything
#     results = vectordb_handler.vectordb.get()

#     print("\n=== Vector DB Contents ===\n")
#     for i, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"]), start=1):
#         print(f"\n--- Chunk {i} ---")
#         print(f"Content Preview: {doc[:200]}...")  # Truncated preview
#         print("\nAttributes:")
#         for key, value in metadata.items():
#             print(f"  {key}: {value}")
#         print("-" * 120)

#     print(f"\n✅ Total Chunks in VectorDB: {len(results['documents'])}")

# if __name__ == "__main__":
#     list_all_contents()











