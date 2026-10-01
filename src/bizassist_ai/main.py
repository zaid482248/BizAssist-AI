"""
BizAssist AI - Main FastAPI Application.

Yeh file FastAPI app banati hai aur saare routes (endpoints) define karti hai.
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from bizassist_ai.core.config import settings


# ---------- Pydantic Models (Request/Response) ----------

class QuestionRequest(BaseModel):
    """User ka sawal."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User ka sawal (1 se 1000 characters tak)",
    )


class AnswerResponse(BaseModel):
    """AI ka jawab."""
    answer: str = Field(..., description="AI ka jawab")
    sources: list[str] = Field(
        default_factory=list,
        description="Jin documents se jawab liya gaya",
    )
    status: str = Field(default="success", description="success ya error")


# ---------- FastAPI App ----------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG-based AI Business Knowledge & Customer Support Assistant",
)


# ---------- Routes (Endpoints) ----------

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
    """Server health check. Docker aur monitoring ke liye zaroori."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "environment": settings.app_env,
    }
@app.get("/about")
async def about_the_app(): 
    """App ke baare mein basic info."""
    return {
        "project": "BizAssist AI",
        "phase": "Phase 3",
        "author": "Zaid Rehman"
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    User ka sawal le kar jawab deta hai.
    
    ABHI YEH FAKE ANSWER DEGA — Phase 8 mein real RAG lagega.
    """
    # Phase 8 mein yahan RAG pipeline aayega
    fake_answer = (
        f"Aap ne poocha: '{request.question}'. "
        "Yeh abhi ek placeholder jawab hai. "
        "Phase 8 mein real RAG pipeline yahan lagegi."
    )
    
    return AnswerResponse(
        answer=fake_answer,
        sources=["placeholder.txt"],
        status="success",
    )