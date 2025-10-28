# AI Wiki Assistant

A lightweight RAG (Retrieval-Augmented Generation) assistant focused on ingesting PDF/DOCX documentation, building hybrid vector and BM25 indexes, and providing an API + developer utilities to query the knowledge base with citations and summarized answers.

This repository combines document ingestion, vector storage (Chroma), BM25, optional Neo4j graph indexing, reranking, and a small retrieval API. It uses Google Generative embeddings (Gemini) for semantic vectors and Azure OpenAI (chat) for LLM responses (configurable through environment variables).

## High-level overview

- Ingestion: load PDF/DOCX files, clean text, split into chunks, enrich with tags/summaries.
- Indexing: create/persist Chroma embeddings and a BM25 index for lexical retrieval.
- Hybrid retrieval: semantic (embedding) + lexical (BM25) fused via Reciprocal Rank Fusion (RRF), optional cross-encoder reranking.
- RAG pipeline: assemble retrieved context, call LLM with explicit instruction to only use provided context, return answer with sources.
- API: small FastAPI endpoint exposing a /search route backed by a WeightedRetriever for production-like usage.

## Repo structure (important files / folders)

- `main.py` — small entry example script.
- `test.py` — local test runner that exercises the vanilla WeightedRetriever.
- `app/` — core application code
	- `rag_pipeline.py` — main RAGPipeline class (index_documents, query_documents)
	- `rag_pipeline_v1.py` — variant of the pipeline (alternate hybrid flow)
	- `api/searchapi.py` — FastAPI application exposing `/search` backed by `WeightedRetriever`
	- `document_ingestion/` — ingestion and indexing modules
		- `loader.py` — async loaders for PDF/DOCX files
		- `embedder.py` — chunking, embedding, Chroma + BM25 wiring, hybrid search (RRF) and helper methods
		- `vectordb.py` — Chroma wrapper(s) and helpers (different flavors exist in vanilla and top-level folder)
		- `reranker.py` — cross-encoder reranker (transformers based)
		- `vanilla/` — simpler, self-contained chunker/tagger/retriever flow
	- `graph/` — Neo4j related code (connector, client, pdf loader, query engine, splitter)
	- `utils/` — small helpers
		- `llm_utils.py` — LLM & embedding clients (AzureChatOpenAI + Google Gemini embeddings)
		- `file_utils.py` — placeholder for file helper functions

## Contracts / Key classes (quick reference)

- Embedder (app.document_ingestion.embedder.Embedder)
	- Inputs: list of document dicts {file_path, content, source}
	- Primary methods:
		- chunk_documents(documents) -> List[chunks]
		- embed_chunks(chunks) -> persists to Chroma
		- save_bm25_index(documents) -> persists BM25 to disk
		- load_vectorstore(), load_bm25_index(), build_hybrid_retriever(k)
		- hybrid_search(query, top_k) -> List[{'text','metadata'}]

- VectorStoreHandler / VectorDB (vanilla vectordb handlers)
	- Wrap Chroma instantiation, add_documents, similarity_search

- WeightedRetriever (app.document_ingestion.vanilla.retreiver.WeightedRetriever)
	- Combines embedding similarity, tag overlap, and summary similarity with frequency-boosted weighting
	- search(query) -> (final_summary, ranked_docs, unique_tags)

- RAGPipeline (app.rag_pipeline.RAGPipeline)
	- index_documents() -> runs ingest -> chunk -> embed -> save BM25
	- query_documents(query) -> hybrid retrieval -> prompt construction -> LLM invoke -> returns answer + sources

- Helperclass (app.utils.llm_utils.Helperclass)
	- openai_client() -> AzureChatOpenAI wrapper
	- gemini_client() -> GoogleGenerativeAIEmbeddings wrapper

## Data layout / persistence

- `data/wiki_pdfs/` — raw PDFs and DOCX used as source corpus.
- `data/chroma_db/` or `data/chroma_db_v2/` — Chroma persistent DB (embedding store). The repo also includes an on-disk `chroma_db` under `chroma_db/`.
- `app/document_ingestion/embedder.py` persists BM25 to `<persist_directory>/bm25.pkl` by default.

Note: Some modules use slightly different default persist paths (e.g., `chroma_db` vs `data/chroma_db_v2`). When running, confirm the paths or pass explicit arguments.

## Environment variables (required)

Set these (via `.env` or environment) before running pipelines or the API:

- `GEMINI_API_KEY` — Google Gemini / embedding API key
- `AZURE_API_KEY` — Azure OpenAI API key
- `AZURE_ENDPOINT` — Azure OpenAI endpoint
- `AZURE_DEPLOYMENT` — Azure deployment name
- `AZURE_MODEL` — model name
- `AZURE_API_VERSION` — API version

The project uses `python-dotenv` in `llm_utils.py` to load a `.env` at runtime.

## Quickstart — Setup (Linux)

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Add credentials to `.env` in the repo root (example):

```env
GEMINI_API_KEY=your_gemini_key
AZURE_API_KEY=your_azure_key
AZURE_ENDPOINT=https://your-azure-endpoint
AZURE_DEPLOYMENT=your-deployment-name
AZURE_MODEL=gpt-4o-mini
AZURE_API_VERSION=2023-05-15
```

3. Prepare data: place PDFs/DOCX under `data/wiki_pdfs/` (subfolders allowed).

4. Index documents (one-time):

```bash
# Python interactive / script example
python - <<'PY'
from app.rag_pipeline import RAGPipeline
import asyncio

pipeline = RAGPipeline(data_dir='data/wiki_pdfs')
asyncio.run(pipeline.index_documents())
PY
```

This will: load documents async, chunk them, embed chunks into Chroma, and persist a BM25 index.

## Run the API (search endpoint)

The FastAPI app is at `app/api/searchapi.py`. Run:

```bash
# Option A: uvicorn directly
uvicorn app.api.searchapi:app --host 0.0.0.0 --port 8000

# Option B: run the file (it calls uvicorn in __main__)
python app/api/searchapi.py
```

Endpoint:
- GET /search?query=your+query

Response shape (pydantic model `SearchResponse`):
{
	"final_summary": "...",
	"ranked_docs": [{"link": "...", "summary": "...", "tags": "..."}],
	"unique_tags": ["tag1", "tag2"]
}

## Local testing / dev

- `python test.py` — runs `WeightedRetriever` test with an example query (uses persisted Chroma DB and embedding client).
- The repository includes commented test harnesses in `test.py` and other sample flows; uncomment and adapt them to run different pipelines.

## Development notes & architecture details

- The project uses two main retrieval approaches:
	1. `Embedder` + Chroma + BM25 hybrid with RRF fusion (in `embedder.py`).
	2. `Vanilla` flow: chunker/tagger -> VectorStoreHandler (a simpler Chroma wrapper) -> WeightedRetriever (custom scoring and LLM summarization).

- Reranking: optional Cross-Encoder reranking is available in `reranker.py` (using `cross-encoder/ms-marco-MiniLM-L-6-v2`). It's currently wired in comments as a higher-quality rerank step.

- Neo4j graph: the `graph/` folder contains connector / client utilities to model documents as graph nodes and run similarity queries via stored embeddings or metadata. Use `QueryEngine` for Neo4j-backed summarization flows.

- LLM & Embeddings clients are encapsulated in `Helperclass` (`app.utils.llm_utils`). Swap implementations there to change providers.

Edge cases & recommendations:
- Contentless documents are filtered out by `loader.py`.
- Chroma/BM25 persistence paths differ across modules — pass explicit paths or align them before running multiple modules.
- Keep an eye on token length when building prompts; the pipeline currently concatenates top-k chunks into context — consider trimming or chunk-prioritizing for long contexts.

## Suggested next steps / improvements

- Add CLI wrappers for common operations (index, search, dump-index).
- Add unit tests (pytest) for loader, chunker, embedder, and retriever components.
- Add a simple Dockerfile / docker-compose to run the API + (optional) Neo4j locally.
- Add robust input sanitization and error handling when external services fail.

## Troubleshooting

- File not found for BM25: ensure `embedder.save_bm25_index()` was run and `bm25.pkl` exists in the `persist_directory`.
- Authentication: confirm `.env` keys are loaded and valid. Use small test in `test.py` to validate LLM/embedding clients.
- Chroma issues: inspect `persist_directory` and confirm you have write permissions.

## Where to look in code for a specific task

- Add documents / change chunking: `app/document_ingestion/loader.py` and `embedder.chunk_documents`
- Change embedding provider: `app/utils/llm_utils.py` (Helperclass.gemini_client)
- Change ranking weights: `app/document_ingestion/vanilla/retreiver.py` (WeightedRetriever.search)
- Expose a new endpoint: `app/api/searchapi.py` (FastAPI app)

