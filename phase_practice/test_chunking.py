"""Chunking test - Phase 5."""

from pathlib import Path

from bizassist_ai.core.chunking import chunk_text


def main():
    
    docs_dir = Path("data/sample_docs")
    for doc in sorted(docs_dir.glob("*.md")):
        text = doc.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        print(f"{doc.name}: {len(text)} chars → {len(chunks)} chunks")
        
    
if __name__ == "__main__":
    main()