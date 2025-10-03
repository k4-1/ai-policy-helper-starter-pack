from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.requests import Request
from typing import List, Dict
import traceback
import json
import uuid
import sqlite3
import os
import time
from contextlib import asynccontextmanager
from .models import IngestResponse, AskRequest, AskResponse, MetricsResponse, Citation, AggregatedChunk, ErrorResponse, ErrorDetail, ErrorType, FeedbackRequest, FeedbackResponse
from .settings import settings
from .ingest import load_documents
from .rag import RAGEngine, build_chunks_from_docs
from .policy_logic import PolicyInterpreter

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_feedback_db()
    yield

app = FastAPI(title="AI Policy & Product Helper", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine()
policy_interpreter = PolicyInterpreter()

def init_feedback_db():
    p = settings.feedback_db_path
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(p)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            helpful INTEGER NOT NULL,
            comment TEXT,
            rating INTEGER,
            created_at INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

def insert_feedback(fid: str, query: str, answer: str, helpful: bool, comment: str | None, rating: int | None):
    # Ensure DB and table exist
    init_feedback_db()
    p = settings.feedback_db_path
    conn = sqlite3.connect(p)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO feedback (id, query, answer, helpful, comment, rating, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (fid, query, answer, 1 if helpful else 0, comment, rating, int(time.time()))
    )
    conn.commit()
    conn.close()

# Startup handled via lifespan above
#Error Helper

def create_error_response(error_type: ErrorType, message: str, user_message: str, suggestions: List[str] = None) -> ErrorResponse:
    """Create a standardized error response"""
    return ErrorResponse(
        error=ErrorDetail(
            type=error_type,
            message=message,
            user_message=user_message,
            suggestions=suggestions or []
        ),
        request_id=str(uuid.uuid4())
    )

#Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    request_id = str(uuid.uuid4())
    
    # Log the full error for debugging
    print(f"Request ID {request_id}: Unhandled exception: {str(exc)}")
    print(f"Traceback: {traceback.format_exc()}")
    
    # Check for specific error types
    if "API key" in str(exc) or "authentication" in str(exc).lower():
        error_response = create_error_response(
            ErrorType.AUTHENTICATION_ERROR,
            str(exc),
            "There's an issue with the AI service configuration. Please contact support.",
            ["Check if the OpenAI API key is properly configured", "Contact your administrator"]
        )
    elif "rate limit" in str(exc).lower() or "429" in str(exc):
        error_response = create_error_response(
            ErrorType.RATE_LIMIT_ERROR,
            str(exc),
            "The AI service is currently busy. Please try again in a moment.",
            ["Wait a few seconds and try again", "Try asking a simpler question"]
        )
    elif "timeout" in str(exc).lower() or "connection" in str(exc).lower():
        error_response = create_error_response(
            ErrorType.TIMEOUT_ERROR,
            str(exc),
            "The request took too long to process. Please try again.",
            ["Try asking a shorter question", "Check your internet connection"]
        )
    else:
        error_response = create_error_response(
            ErrorType.INTERNAL_ERROR,
            str(exc),
            "Something went wrong while processing your request. Please try again.",
            ["Try rephrasing your question", "Contact support if the problem persists"]
        )
    
    error_response.request_id = request_id
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
    )

#Health API
@app.get("/api/health")
def health():
    # Check configuration errors
    config_errors = settings.validate_configuration()
    if config_errors:
        # Return the first configuration error as a health check failure
        error = config_errors[0]
        raise HTTPException(
            status_code=503,
            detail=create_error_response(
                ErrorType.SERVICE_UNAVAILABLE,
                error["message"],
                error["user_message"],
                error["suggestions"]
            ).model_dump()
        )
    
    return {"status": "healthy", "llm_provider": settings.llm_provider}

#Metrics API
@app.get("/api/metrics", response_model=MetricsResponse)
def metrics():
    s = engine.stats()
    return MetricsResponse(**s)

#Ingest API
@app.post("/api/ingest", response_model=IngestResponse)
def ingest():
    try:
        docs = load_documents(settings.data_dir)
        if not docs:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    ErrorType.NOT_FOUND,
                    "No documents found in data directory",
                    "No policy documents were found to process.",
                    ["Check if documents are placed in the correct directory", "Contact support for help"]
                ).model_dump()
            )
        
        chunks = build_chunks_from_docs(docs, settings.chunk_size, settings.chunk_overlap)
        new_docs, new_chunks = engine.ingest_chunks(chunks)
        
        if new_chunks == 0:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    ErrorType.VALIDATION_ERROR,
                    "No new content to process",
                    "The documents have already been processed or contain no valid content.",
                    ["Try uploading different documents", "Check if documents contain readable text"]
                ).model_dump()
            )
        
        return IngestResponse(indexed_docs=new_docs, indexed_chunks=new_chunks)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                ErrorType.INTERNAL_ERROR,
                str(e),
                "Failed to process the policy documents. Please try again.",
                ["Check if documents are in a supported format", "Contact support if the problem persists"]
            ).model_dump()
        )

#Ask API (Stream)
@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        # Validate input
        if not req.query or not req.query.strip():
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    ErrorType.VALIDATION_ERROR,
                    "Empty query provided",
                    "Please enter a question to get help with your policy inquiry.",
                    ["Type your question in the chat box", "Ask about returns, warranties, or product information"]
                ).model_dump()
            )
        
        # Retrieve context
        ctx = engine.retrieve(req.query, k=req.k or 4)
        
        if not ctx:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    ErrorType.NOT_FOUND,
                    "No relevant information found",
                    "I couldn't find any relevant policy information for your question.",
                    ["Try rephrasing your question", "Ask about specific products or policies", "Contact customer service for specialized help"]
                ).model_dump()
            )
        
        # Generate response
        base_answer = engine.generate(req.query, ctx)
        
        if not base_answer or base_answer.strip() == "":
            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    ErrorType.SERVICE_UNAVAILABLE,
                    "Failed to generate response",
                    "I'm having trouble generating a response right now. Please try again.",
                    ["Try asking a simpler question", "Wait a moment and try again", "Contact support if the issue persists"]
                ).model_dump()
            )
        
        # Apply policy-specific business logic
        enhanced_response = policy_interpreter.enhance_rag_response(req.query, ctx, base_answer)
        
        display_ctx = [c for c in ctx if float(c.get("score", 0.0)) >= settings.relevance_threshold]
        if not enhanced_response.get("is_policy_query", False):
            display_ctx = []
        citations = [Citation(title=c.get("title"), section=c.get("section")) for c in display_ctx]

        # Aggregate duplicate chunk texts into a single item with sources
        def _normalize_text(s: str) -> str:
            return " ".join((s or "").strip().split()).lower()

        agg_map: Dict[str, Dict] = {}
        for c in display_ctx:
            t = c.get("text") or ""
            key = _normalize_text(t)
            src = Citation(title=c.get("title"), section=c.get("section"))
            if key in agg_map:
                # Append if not duplicate source
                existing_sources = agg_map[key]["sources"]
                if not any(s.title == src.title and s.section == src.section for s in existing_sources):
                    existing_sources.append(src)
            else:
                agg_map[key] = {"text": c.get("text"), "sources": [src]}

        chunks = [
            AggregatedChunk(text=v["text"], sources=v["sources"], source_count=len(v["sources"]))
            for v in agg_map.values()
        ]
        stats = engine.stats()
        
        return AskResponse(
            query=req.query,
            answer=enhanced_response["answer"],
            citations=citations,
            chunks=chunks,
            metrics={
                "retrieval_ms": stats["avg_retrieval_latency_ms"],
                "generation_ms": stats["avg_generation_latency_ms"],
                "confidence": enhanced_response["confidence"],
                "requires_human_review": enhanced_response["requires_human_review"],
                "policy_analysis": enhanced_response.get("policy_analysis")
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise e

@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    try:
        if not settings.streaming_enabled:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    ErrorType.SERVICE_UNAVAILABLE,
                    "Streaming disabled",
                    "Streaming responses are disabled in the current configuration.",
                    ["Enable STREAMING_ENABLED in environment"]
                ).model_dump()
            )
        if not req.query or not req.query.strip():
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    ErrorType.VALIDATION_ERROR,
                    "Empty query provided",
                    "Please enter a question to get help with your policy inquiry.",
                    ["Type your question in the chat box"]
                ).model_dump()
            )
        ctx = engine.retrieve(req.query, k=req.k or 4)
        if not ctx:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    ErrorType.NOT_FOUND,
                    "No relevant information found",
                    "I couldn't find any relevant policy information for your question.",
                    ["Try rephrasing your question"]
                ).model_dump()
            )
        base_answer = engine.generate(req.query, ctx)
        enhanced_response = policy_interpreter.enhance_rag_response(req.query, ctx, base_answer)
        def gen():
            rid = str(uuid.uuid4())
            yield f"data: {json.dumps({'type':'start','request_id':rid}, separators=(',',':'))}\n\n"
            text = enhanced_response["answer"]
            parts = [p for p in text.split("\n") if p.strip()]
            for p in parts:
                yield f"data: {json.dumps({'type':'chunk','text':p}, separators=(',',':'))}\n\n"
            stats = engine.stats()
            payload = {
                "retrieval_ms": stats["avg_retrieval_latency_ms"],
                "generation_ms": stats["avg_generation_latency_ms"],
                "confidence": enhanced_response["confidence"],
                "requires_human_review": enhanced_response["requires_human_review"]
            }
            yield f"data: {json.dumps({'type':'end','metrics':payload}, separators=(',',':'))}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        raise e

# Feedback API
@app.post("/api/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    try:
        fid = str(uuid.uuid4())
        insert_feedback(fid, req.query, req.answer, req.helpful, req.comment, req.rating)
        return FeedbackResponse(id=fid, status="ok")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                ErrorType.INTERNAL_ERROR,
                str(e),
                "Failed to store feedback.",
                ["Try again later"]
            ).model_dump()
        )
