# BizAssist AI

**RAG-Based AI Business Knowledge & Customer Support Assistant**

Ek AI assistant jo small businesses ke documents (PDF, TXT, DOCX) se information nikaal kar customer questions ka jawab deta hai — bina koi information banaye.

## Features (Planned)

- 📄 Document upload (PDF, TXT, DOCX)
- 🔍 Semantic search via FAISS
- 🤖 RAG pipeline with free LLM APIs
- 🔐 JWT authentication
- 💬 Streamlit chat interface
- 🐳 Docker deployment

## Tech Stack

- **Backend:** FastAPI, Python 3.11+
- **AI:** Groq API (LLM), Hugging Face (embeddings)
- **Vector DB:** FAISS
- **Database:** PostgreSQL
- **Frontend:** Streamlit
- **Package Manager:** uv
- **Testing:** pytest
- **Deployment:** Docker

## Status

🚧 Phase 2 — Project structure complete

## Setup

```bash
# Install uv from https://docs.astral.sh/uv/
uv sync

# Copy environment template
cp .env.example .env
# Edit .env with your API keys