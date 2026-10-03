"""
Text chunking module.
Yeh module text ko chhote chunks mein todta hai (RAG ke liye).
"""

import re
from dataclasses import dataclass


# ---------- Chunk Dataclass ----------

@dataclass
class Chunk:
    """Ek text chunk ka representation."""
    text: str
    index: int          # Chunk number (0-based)
    start_char: int     # Original text mein kahan se shuru
    end_char: int       # Original text mein kahan tak


# ---------- Sentence Splitter ----------

def split_into_sentences(text: str) -> list[str]:
    """
    Text ko sentences mein todta hai.
    
    Simple rule: '.', '!', '?' par split karein, lekin decimal numbers 
    (3.5) aur abbreviations (Mr.) ko protect karein.
    """
    # Common abbreviations jo '.' ke saath aate hain
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.", r"\1<DOT>", text)
    
    # Decimal numbers protect karein (3.5, 99.9)
    text = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    
    # Sentence endings par split karein
    sentences = re.split(r"(?<=[.!?])\s+", text)
    
    # Placeholders wapas '.' bana dein
    sentences = [s.replace("<DOT>", ".") for s in sentences]
    
    # Khali sentences hata dein
    return [s.strip() for s in sentences if s.strip()]


# ---------- Main Chunking Function ----------

def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[Chunk]:
    """
    Text ko overlapping chunks mein todta hai.
    
    Strategy:
    1. Text ko sentences mein toden.
    2. Sentences ko jama kar ke chunk banayein jab tak chunk_size tak na pohanch jaye.
    3. Next chunk overlap characters peeche se shuru ho.
    
    Args:
        text: Input text.
        chunk_size: Har chunk ka max characters (default 800).
        overlap: Chunks ke darmiyan overlap characters (default 100).
    
    Returns:
        List of Chunk objects.
    """
    # Validation
    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("Overlap chunk_size se chhota hona chahiye.")

    text = text.strip()
    sentences = split_into_sentences(text)

    chunks: list[Chunk] = []
    current_chunk: list[str] = []
    current_length = 0
    chunk_start_char = 0
    running_char = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # Agar ek hi sentence chunk_size se bara hai, to usay forcefully todna parega
        if sentence_length > chunk_size:
            # Pehle current chunk save karein
            if current_chunk:
                chunk_content = " ".join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_content,
                    index=len(chunks),
                    start_char=chunk_start_char,
                    end_char=chunk_start_char + len(chunk_content),
                ))
                current_chunk = []
                current_length = 0

            # Bade sentence ko char-based chunks mein toden
            for i in range(0, sentence_length, chunk_size - overlap):
                sub = sentence[i:i + chunk_size]
                if sub.strip():
                    chunks.append(Chunk(
                        text=sub,
                        index=len(chunks),
                        start_char=running_char + i,
                        end_char=running_char + i + len(sub),
                    ))
            running_char += sentence_length + 1
            chunk_start_char = running_char
            continue

        # Agar sentence add karne se chunk overflow ho jaye
        if current_length + sentence_length + 1 > chunk_size and current_chunk:
            # Current chunk save karein
            chunk_content = " ".join(current_chunk)
            chunks.append(Chunk(
                text=chunk_content,
                index=len(chunks),
                start_char=chunk_start_char,
                end_char=chunk_start_char + len(chunk_content),
            ))

            # Overlap ke liye, last kuch sentences rakhein
            overlap_chars = 0
            overlap_sentences: list[str] = []
            for s in reversed(current_chunk):
                if overlap_chars + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_chars += len(s) + 1

            current_chunk = overlap_sentences
            current_length = overlap_chars
            chunk_start_char = running_char - overlap_chars

        # Sentence add karein
        current_chunk.append(sentence)
        current_length += sentence_length + 1
        running_char += sentence_length + 1

    # Last chunk save karein
    if current_chunk:
        chunk_content = " ".join(current_chunk)
        chunks.append(Chunk(
            text=chunk_content,
            index=len(chunks),
            start_char=chunk_start_char,
            end_char=chunk_start_char + len(chunk_content),
        ))

    return chunks