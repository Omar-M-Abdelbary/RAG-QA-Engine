# RAG Question Answering System

A production-ready Retrieval-Augmented Generation (RAG) system for answering questions using
semantic search and large language models (LLMs). Designed with modular architecture,
multilingual support, and extensive analytics, it demonstrates a scalable foundation for
AI-driven Q&A platforms.

---

## Features

* **Dataset Processing**: Load and preprocess Natural Questions dataset with chunking and
  metadata enrichment.
* **Multilingual Embeddings**: SentenceTransformers for cross-lingual semantic vectors.
* **Efficient Retrieval**: FAISS vector index with metadata file lookup.
* **LLM Integration**: Async Groq client powering answer generation with prompt templates.
* **Advanced Query Handling**: Cleaning, expansion, multi-turn context, and relevance
  ranking.
* **Validation & Caching**: Response quality checks, uncertainty detection, and file-based
  caching.
* **Performance Monitoring**: Track latency, throughput, cache stats, and custom metrics.
* **REST API**: FastAPI endpoints for Q&A, health, stats, admin utilities, and evaluation.
* **Database Logging**: SQLAlchemy models store queries, feedback, and analytics (SQLite).
* **Evaluation Framework**: Run dataset evaluations with similarity scoring and reporting.

---

## Repository Structure

```
app/
├── clients/          # External service wrappers (LLM)
├── controller/       # Request handlers bridging routes and services
├── core/             # Configuration, constants
├── data/             # Dataset loader & preprocessing
├── infrastructure/   # Database and vector storage logic
├── routes/           # FastAPI routers
├── schemas/          # Pydantic models for requests/responses
├── services/         # Business logic (RAG, retrieval, evaluation...)
├── utils/            # Helpers: logging, caching, monitoring
scripts/              # Standalone scripts (create embeddings, preprocess)
Natural-Questions-Base/ # Local dataset files and indexes
requirements.txt
README.md            # ← this file
```

---

## Setup & Installation

1. **Clone repository**
   ```bash
   git clone <repo-url> rag-project
   cd rag-project
   ```

2. **Create Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   .venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Environment variables**
   Create a `.env` file at the project root with:
   ```ini
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Initialize database** (SQLite by default)
   ```bash
   python -c "from app.infrastructure.database.session import init_db; init_db()"
   ```

6. **Prepare dataset and embeddings**
   ```bash
   # adjust paths in app/core/config.py if needed
   python app/scripts/preprocess_data.py
   python app/scripts/create_embeddings.py
   ```

7. **Run server**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

Open Swagger UI at http://localhost:8000/docs for interactive API exploration.

---

## Available API Endpoints

### Public

| Method | Path                  | Description                         |
|--------|-----------------------|-------------------------------------|
| POST   | `/api/v1/ask-question` | Ask a question (returns answer)     |
| GET    | `/api/v1/health`      | System health check                 |
| GET    | `/api/v1/stats`       | Query analytics and performance     |

### Admin

| Method | Path                          | Description                         |
|--------|-------------------------------|-------------------------------------|
| POST   | `/api/v1/admin/embeddings/create` | Generate embeddings from text    |
| GET    | `/api/v1/admin/embeddings/test`   | Verify embedding model           |
| POST   | `/api/v1/admin/llm/test`         | Test LLM connectivity            |
| GET    | `/api/v1/admin/llm/health`       | LLM health                      |
| POST   | `/api/v1/admin/preprocess/text`  | Preprocess a single text        |
| POST   | `/api/v1/admin/preprocess/batch` | Batch text preprocessing        |
| GET    | `/api/v1/admin/system/info`      | System component info           |
| POST   | `/api/v1/admin/evaluation/run`   | Run dataset evaluation          |

Responses are validated with Pydantic models defined in `app/schemas`.

---

## Core Components

* **Dataset Loader** (`app/data/dataset_loader.py`) – reads CSV and verifies columns.
* **Preprocessor** (`app/data/preprocessing.py`) – selects long/short answers, chunks text, adds basic metadata.
* **Chunker** (`app/data/chunking.py`) – configurable text splitter using LangChain.
* **EmbeddingService** – uses SentenceTransformer for vectorization.
* **RetrievalService** – loads FAISS index and corresponding metadata; retrieval with top-K.
* **RAGService** – orchestrates query processing, retrieval, prompting, LLM generation, validation, caching, and monitoring.
* **EvaluationService** – runs question-by-question evaluation and produces JSON reports.
* **LLMClient** – async wrapper around Groq API (now using AsyncGroq).
* **QueryProcessor** – cleans, normalizes, and expands queries; supports multi-turn.
* **ResponseValidator** – checks relevance, detects uncertainty/hallucination, scores quality.
* **Cache** – simple file-based cache with TTL.
* **PerformanceMonitor** – metrics collector with counters and latency trackers.
* **Database (SQLite)** – models for queries and feedback, CRUD utilities, stats computation.

---

## Development Notes

* Python 3.12 compatible.
* Uses FastAPI’s dependency injection for database sessions.
* Async/await throughout to avoid blocking on I/O (LLM calls, DB writes).
* Logging with contextual information; logs stored under `logs/`.

---

## Known Limitations & Future Work

* Metadata enrichment limited to answer type; domain/difficulty absent.
* Evaluation metrics basic (SequenceMatcher); BLEU/ROUGE and precision@K pending.
* Conversation history kept in memory; persisting per-session would improve UX.
* Cache is file-based; migrate to Redis for scaling.
* Detailed visual reports and dashboards not yet implemented.

Contributions and enhancements are welcome!

