"""
BizAssist AI - Main FastAPI Application.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from bizassist_ai.core.config import settings
from bizassist_ai.core.extractors import ExtractionError, extract_text
from bizassist_ai.core.validators import (
    ValidationError,
    safe_filename,
    validate_extension,
    validate_size,
)

from bizassist_ai.core.chunking import chunk_text

# ---------- Logging Setup ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("bizassist")


# ---------- Pydantic Models ----------

class QuestionRequest(BaseModel):
    """User ka sawal."""
    question: str = Field(..., min_length=1, max_length=1000)


class AnswerResponse(BaseModel):
    """AI ka jawab."""
    answer: str
    sources: list[str] = Field(default_factory=list)
    status: str = "success"


class UploadResponse(BaseModel):
    """File upload ka response."""
    status: str
    filename: str
    size_bytes: int
    text_length: int
    preview: str
    message: str


# ---------- FastAPI App ----------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG-based AI Business Knowledge & Customer Support Assistant",
)


# ---------- Routes ----------

@app.get("/")
async def root():
    """Welcome message."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Server health check."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "environment": settings.app_env,
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """User ka sawal le kar jawab deta hai (placeholder)."""
    fake_answer = (
        f"Aap ne poocha: '{request.question}'. "
        "Yeh abhi placeholder hai. Phase 8 mein real RAG lagega."
    )
    return AnswerResponse(
        answer=fake_answer,
        sources=["placeholder.txt"],
        status="success",
    )


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Document upload karta hai aur text extract karta hai.
    
    Supported: PDF, TXT, DOCX, MD.
    Max size: 10 MB.
    """
    logger.info(f"Upload request: {file.filename}")

    # Step 1: Filename validate karein
    try:
        extension = validate_extension(file.filename or "")
        clean_name = safe_filename(file.filename or "")
    except ValidationError as e:
        logger.warning(f"Validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Step 2: File content read karein (memory mein)
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"File read error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File read karne mein error aaya.",
        )

    # Step 3: Size validate karein
    try:
        validate_size(len(content))
    except ValidationError as e:
        logger.warning(f"Size validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(e),
        )

    # Step 4: File save karein
    save_path = settings.upload_dir / clean_name
    try:
        save_path.write_bytes(content)
        logger.info(f"File saved: {save_path}")
    except Exception as e:
        logger.error(f"File save error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File save karne mein error aaya.",
        )

    # Step 5: Text extract karein
    try:
        text = extract_text(save_path)
    except ExtractionError as e:
        logger.error(f"Extraction failed: {e}")
        # Agar extraction fail ho, file delete kar dein
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    # Step 6: Preview banayein (pehle 200 characters)
    preview = text[:200] + ("..." if len(text) > 200 else "")

    logger.info(
        f"Upload success: {clean_name} ({len(content)} bytes, {len(text)} chars)"
    )

    return UploadResponse(
        status="success",
        filename=clean_name,
        size_bytes=len(content),
        text_length=len(text),
        preview=preview,
        message=f"File '{clean_name}' upload aur extract ho gayi.",
    )
    
     
# ---------- Chunking Test Models ----------

class ChunkRequest(BaseModel):
    """Chunking request."""
    text: str = Field(..., min_length=1)
    chunk_size: int = Field(default=800, ge=100, le=5000)
    overlap: int = Field(default=100, ge=0, le=500)


class ChunkInfo(BaseModel):
    """Ek chunk ka info."""
    index: int
    text: str
    length: int
    start_char: int
    end_char: int


class ChunkResponse(BaseModel):
    """Chunking response."""
    status: str
    total_chunks: int
    original_length: int
    chunks: list[ChunkInfo]


# ---------- Chunking Test Endpoint ----------

@app.post("/chunk", response_model=ChunkResponse)
async def chunk_endpoint(request: ChunkRequest):
    """
    Text ko chunks mein todta hai (testing ke liye).
    Phase 7 mein yeh internally use hoga.
    """
    try:
        chunks = chunk_text(
            text=request.text,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return ChunkResponse(
        status="success",
        total_chunks=len(chunks),
        original_length=len(request.text),
        chunks=[
            ChunkInfo(
                index=c.index,
                text=c.text,
                length=len(c.text),
                start_char=c.start_char,
                end_char=c.end_char,
            )
            for c in chunks
        ],
    )