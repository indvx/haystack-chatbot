from pydantic import BaseModel, Field
from typing import List, Optional


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="The search query or prompt for the AI assistant",
        example="What is the summary of the document?",
    )


class SearchResponse(BaseModel):
    query: str
    answer: str
    tool_used: str
    success: bool = True


class FileUploadResponse(BaseModel):
    message: str
    filename: Optional[str] = None
    file_path: Optional[str] = None
    indexed: bool = True


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
