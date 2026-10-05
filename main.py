import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from decouple import config

from schemas import (
    SearchRequest,
    SearchResponse,
    FileUploadResponse,
    ErrorResponse,
)
from services.file_upload_service import FileUploadService
from services.search_service import SearchService
from utils.logger import logger

DOC_DIR = Path(config("DOC_DIR", default="./documents"))
os.makedirs(DOC_DIR, exist_ok=True)

app = FastAPI(
    title="Haystack Chatbot API",
    version="1.0.0",
    description="FastAPI service for document indexing and RAG-based search.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_upload_service = FileUploadService()
search_service = SearchService()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error handling {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal Server Error",
            "detail": str(exc),
        },
    )


@app.get("/", summary="Root status", tags=["Health"])
async def root():
    return {
        "status": "online",
        "service": "Haystack Chatbot API",
        "version": "1.0.0",
    }


@app.get("/health", summary="Health check", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "doc_dir": str(DOC_DIR.resolve()),
        "doc_dir_exists": DOC_DIR.exists(),
    }


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv"}


@app.post(
    "/api/upload",
    response_model=FileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload & Index Documents",
    tags=["Documents"],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def upload_documents(file: UploadFile = File(...)):
    """Upload PDF, TXT, or CSV files to index into the vector store."""
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided.",
        )

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Unsupported extension rejected: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{file_ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    target_path = DOC_DIR / file.filename
    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved upload to {target_path}")
    except Exception as e:
        logger.error(f"Failed writing file {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed writing file '{file.filename}': {str(e)}",
        )
    finally:
        await file.close()

    try:
        index_result = file_upload_service.upload_data(target_files=[target_path])
        logger.info(f"Indexing completed: {index_result}")
    except Exception as e:
        logger.error(f"Indexing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error indexing documents: {str(e)}",
        )

    return FileUploadResponse(
        message=f"Uploaded and indexed {file.filename}.",
        filename=file.filename if file.filename else "",
        file_path=str(target_path.resolve()) if target_path else "",
    )


@app.post(
    "/api/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search & Query Assistant",
    tags=["Search"],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def search(request: SearchRequest):
    """Query the Haystack assistant using RAG search with web fallback."""
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    try:
        result = search_service.search(query=request.query)
        return SearchResponse(
            query=result["query"],
            answer=result["answer"],
            tool_used=result["tool_used"],
            success=True,
        )
    except Exception as e:
        logger.error(f"Search request failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search service error: {str(e)}",
        )
