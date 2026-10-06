"""
Embeddings module.
Yeh module text ko vectors (numbers) mein convert karta hai Hugging Face API se.
"""

import logging
import os

from huggingface_hub import InferenceClient

from bizassist_ai.core.config import settings


logger = logging.getLogger("bizassist.embeddings")


# ---------- Configuration ----------

# Recommended embedding model (free, fast, good quality)
# 384-dimensional embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Batch size — ek baar mein kitne texts bhejne hain
BATCH_SIZE = 16


# ---------- Custom Exceptions ----------

class EmbeddingError(Exception):
    """Jab embedding banane mein error aaye."""
    pass


# ---------- Client Setup ----------

def _get_client() -> InferenceClient:
    """Hugging Face client banata hai. Har call par naya banate hain (thread-safe)."""
    token = os.getenv("HF_API_TOKEN") or getattr(settings, "hf_api_token", None)

    if not token or token.startswith("hf_xxx"):
        raise EmbeddingError(
            "HF_API_TOKEN set nahi hai. .env file mein daalein."
        )

    return InferenceClient(
        provider="hf-inference",
        api_key=token,
    )


# ---------- Single Text Embedding ----------

def embed_text(text: str) -> list[float]:
    """
    Ek text ka embedding banata hai.
    
    Args:
        text: Input text (chunk).
    
    Returns:
        List of floats (vector). For all-MiniLM-L6-v2, length is 384.
    
    Raises:
        EmbeddingError: Agar API call fail ho.
    """
    if not text or not text.strip():
        raise EmbeddingError("Khali text ka embedding nahi ban sakta.")

    try:
        client = _get_client()
        vector = client.feature_extraction(
            text.strip(),
            model=EMBEDDING_MODEL,
        )

        # `feature_extraction` sometimes returns nested list — flatten if needed
        if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
            vector = vector[0]

        return [float(x) for x in vector]

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise EmbeddingError(f"Embedding API error: {e}") from e


# ---------- Batch Embedding ----------

# def embed_batch(texts: list[str], batch_size: int = BATCH_SIZE) -> list[list[float]]:
#     """
#     Kai texts ke embeddings ek saath banata hai (faster).
    
#     Args:
#         texts: List of texts (chunks).
#         batch_size: Ek API call mein kitne texts.
    
#     Returns:
#         List of vectors (same order as input).
#     """
#     if not texts:
#         return []

#     all_vectors: list[list[float]] = []

#     for i in range(0, len(texts), batch_size):
#         batch = texts[i:i + batch_size]
#         logger.info(f"Embedding batch {i // batch_size + 1} ({len(batch)} texts)")

#         try:
#             client = _get_client()
#             vectors = client.feature_extraction(
#                 batch,
#                 model=EMBEDDING_MODEL,
#             )

#             # Hugging Face returns 2D array for batch
#             if isinstance(vectors, list) and len(vectors) > 0:
#                 # Check karein ke nested hai ya flat
#                 if isinstance(vectors[0], list) and len(vectors[0]) > 1:
#                     # 2D — already correct
#                     all_vectors.extend([[float(x) for x in v] for v in vectors])
#                 else:
#                     # Flat — single vector returned for a batch? Fallback to individual
#                     logger.warning("Unexpected flat response for batch. Falling back to individual.")
#                     for text in batch:
#                         all_vectors.append(embed_text(text))
#             else:
#                 raise EmbeddingError("Invalid response from API.")

#         except Exception as e:
#             logger.error(f"Batch embedding failed: {e}")
#             raise EmbeddingError(f"Batch embedding error: {e}") from e

#     return all_vectors

# ---------- Batch Embedding ----------

def embed_batch(texts: list[str], batch_size: int = BATCH_SIZE) -> list[list[float]]:
    """
    Kai texts ke embeddings banata hai.
    
    Simple aur reliable approach: har text ka alag embedding banate hain.
    Hugging Face ka batch response format unpredictable hai, isliye hum
    individual calls use karte hain.
    
    Args:
        texts: List of texts (chunks).
        batch_size: Yeh parameter abhi use nahi ho raha, future optimization ke liye.
    
    Returns:
        List of vectors (same order as input).
    """
    if not texts:
        return []

    all_vectors: list[list[float]] = []

    for i, text in enumerate(texts):
        logger.info(f"Embedding {i + 1}/{len(texts)}")
        try:
            vector = embed_text(text)
            all_vectors.append(vector)
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Batch embedding failed at index {i}: {e}")
            raise EmbeddingError(f"Batch embedding error at index {i}: {e}") from e

    return all_vectors
# ---------- Utility ----------

def get_embedding_dimension() -> int:
    """Embedding vector ki dimension return karta hai."""
    return 384  # all-MiniLM-L6-v2
