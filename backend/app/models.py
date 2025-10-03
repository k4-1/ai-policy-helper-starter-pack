from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum

class ErrorType(str, Enum):
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_ERROR = "internal_error"
    NOT_FOUND = "not_found"
    TIMEOUT_ERROR = "timeout_error"

class ErrorDetail(BaseModel):
    type: ErrorType
    message: str
    user_message: str
    suggestions: Optional[List[str]] = None

class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: Optional[str] = None

class IngestResponse(BaseModel):
    indexed_docs: int
    indexed_chunks: int

class AskRequest(BaseModel):
    query: str
    k: int | None = 4
    stream: bool | None = False

class Citation(BaseModel):
    title: str
    section: str | None = None

class Chunk(BaseModel):
    title: str
    section: str | None = None
    text: str

class AggregatedChunk(BaseModel):
    text: str
    sources: List[Citation]
    source_count: int

class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    chunks: List[AggregatedChunk]
    metrics: Dict[str, Any]

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    helpful: bool
    comment: Optional[str] = None
    rating: Optional[int] = None

class FeedbackResponse(BaseModel):
    id: str
    status: str

class MetricsResponse(BaseModel):
    total_docs: int
    total_chunks: int
    avg_retrieval_latency_ms: float
    avg_generation_latency_ms: float
    embedding_model: str
    llm_model: str
