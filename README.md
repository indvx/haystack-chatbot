# Haystack Chatbot REST API

A production-ready FastAPI backend for the **Haystack Chatbot**, supporting document indexing and retrieval-augmented search (RAG) with Web Search fallback.

---

## Features

- **Document Ingestion API (`POST /api/upload`)**: Upload PDF, TXT, or CSV files to automatically clean, chunk, embed, and index them into Weaviate Vector Store using Haystack pipelines.
- **Search & AI Query API (`POST /api/search`)**: Submit queries to retrieve context-aware answers using Haystack RAG with real-time Web Search fallback.
- **Interactive Swagger Documentation**: Built-in interactive API docs accessible via `/docs` or `/redoc`.
- **Production Architecture**: Strict Pydantic data schemas, structured logging, CORS middleware, and global exception handlers.

---

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **AI Framework**: Haystack (haystack-ai)
- **Generative & Embedding Models**: Google Gemini (`gemini-2.5-flash` & `text-embedding-004`)
- **Vector Database**: Weaviate (`weaviate-haystack`)

---

## API Endpoints Overview

| Method | Endpoint | Description | Payload / Query |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/upload` | Upload & Index Documents | `multipart/form-data` with `files` (`.pdf`, `.txt`, `.csv`) |
| `POST` | `/api/search` | Search & Ask AI Assistant | `application/json` `{ "query": "Your question..." }` |
| `GET`  | `/health` | API Health & Storage Check | None |
| `GET`  | `/docs` | Interactive Swagger OpenAPI Docs | None |

---

## Running the Server

### 1. Environment Setup
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
DOC_DIR=./documents
PNG_DIR=./pngs
WEAVIATE_URL=http://localhost:8080
```

### 2. Start Weaviate Vector DB
Ensure Weaviate is running locally via Docker:
```bash
docker compose up -d
```

### 3. Run FastAPI Application
Start the Uvicorn development server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Access Swagger API documentation at: `http://localhost:8000/docs`

---

## Sample Request Usage

### 1. Upload Document API
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "files=@/path/to/sample.pdf"
```

### 2. Search API
```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings in the uploaded document?"
  }'
```
