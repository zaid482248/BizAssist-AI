"""
FAISS vector store.
Yeh module embeddings ko store karta hai aur similarity search karta hai.
"""

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from bizassist_ai.core.config import settings


logger = logging.getLogger("bizassist.vector_store")


# ---------- Configuration ----------

DEFAULT_INDEX_NAME = "bizassist_index"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


# ---------- Custom Exceptions ----------

class VectorStoreError(Exception):
    """FAISS operations mein error."""
    pass


# ---------- Metadata Dataclass ----------

@dataclass
class ChunkMetadata:
    """Ek chunk ke baare mein metadata."""
    chunk_index: int          # Chunk number in document
    document_name: str        # e.g., "shipping_policy.md"
    text: str                 # Actual chunk text
    start_char: int = 0
    end_char: int = 0


# ---------- Vector Store Class ----------

class VectorStore:
    """
    FAISS-based vector store with metadata mapping.
    
    Usage:
        store = VectorStore()
        store.add(vectors, metadatas)
        store.save()
        
        results = store.search(query_vector, top_k=5)
    """

    def __init__(
        self,
        index_name: str = DEFAULT_INDEX_NAME,
        dimension: int = EMBEDDING_DIM,
    ):
        self.index_name = index_name
        self.dimension = dimension
        self.index: faiss.IndexFlatL2 | None = None
        self.metadata: dict[int, ChunkMetadata] = {}

        # Paths
        self.index_path = settings.data_dir / f"{index_name}.faiss"
        self.metadata_path = settings.data_dir / f"{index_name}_metadata.json"

    # ---------- Initialization ----------

    def _ensure_index(self) -> None:
        """Index ko initialize karein agar abhi tak nahi hua."""
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"Created new FAISS index (dim={self.dimension})")

    # ---------- Add Vectors ----------

    def add(
        self,
        vectors: list[list[float]],
        metadatas: list[ChunkMetadata],
    ) -> list[int]:
        """
        Vectors aur unke metadata add karein.
        
        Args:
            vectors: List of embedding vectors (each length = dimension).
            metadatas: Parallel list of ChunkMetadata.
        
        Returns:
            List of assigned IDs (FAISS internal IDs).
        
        Raises:
            VectorStoreError: Agar lengths match na karein ya validation fail ho.
        """
        if len(vectors) != len(metadatas):
            raise VectorStoreError(
                f"Vectors ({len(vectors)}) aur metadata ({len(metadatas)}) "
                "ki length match nahi karti."
            )

        if not vectors:
            logger.warning("Empty vectors list — kuch add nahi hua.")
            return []

        # Validate dimensions
        for i, v in enumerate(vectors):
            if len(v) != self.dimension:
                raise VectorStoreError(
                    f"Vector {i} ki dimension {len(v)} hai, expected {self.dimension}."
                )

        self._ensure_index()

        # Convert to numpy float32 (FAISS requirement)
        vectors_np = np.array(vectors, dtype=np.float32)

        # Starting ID (FAISS auto-increments)
        start_id = self.index.ntotal

        # Add to FAISS
        self.index.add(vectors_np)
        logger.info(f"Added {len(vectors)} vectors. Total now: {self.index.ntotal}")

        # Save metadata
        assigned_ids = []
        for offset, meta in enumerate(metadatas):
            new_id = start_id + offset
            self.metadata[new_id] = meta
            assigned_ids.append(new_id)

        return assigned_ids

    # ---------- Search ----------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Query vector ke sab se qareebi k chunks dhundhein.
        
        Args:
            query_vector: Query ka embedding (length = dimension).
            top_k: Kitne results chahiye.
        
        Returns:
            List of dicts: [{"chunk": ChunkMetadata, "distance": float, "id": int}]
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index khali hai — kuch search nahi hua.")
            return []

        if len(query_vector) != self.dimension:
            raise VectorStoreError(
                f"Query vector dimension {len(query_vector)} hai, expected {self.dimension}."
            )

        query_np = np.array([query_vector], dtype=np.float32)

        # FAISS search
        distances, ids = self.index.search(query_np, top_k)

        results = []
        for dist, idx in zip(distances[0], ids[0]):
            if idx == -1:  # FAISS returned "no result"
                continue

            meta = self.metadata.get(int(idx))
            if meta is None:
                logger.warning(f"Metadata missing for ID {idx}")
                continue

            results.append({
                "id": int(idx),
                "distance": float(dist),
                "chunk": meta,
            })

        return results

    # ---------- Persistence ----------

    def save(self) -> None:
        """Index aur metadata ko disk par save karein."""
        if self.index is None:
            raise VectorStoreError("Index banaya nahi gaya — save nahi ho sakta.")

        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"Index saved: {self.index_path}")

            # Save metadata as JSON (human-readable)
            metadata_serializable = {
                str(k): asdict(v) for k, v in self.metadata.items()
            }
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_serializable, f, ensure_ascii=False, indent=2)
            logger.info(f"Metadata saved: {self.metadata_path}")

        except Exception as e:
            logger.error(f"Save failed: {e}")
            raise VectorStoreError(f"Save karne mein error: {e}") from e

    def load(self) -> bool:
        """
        Disk se index aur metadata load karein.
        
        Returns:
            True agar load hua, False agar files nahi mili.
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.warning(f"Index files nahi mili: {self.index_path}")
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"Index loaded: {self.index.ntotal} vectors")

            # Load metadata
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            self.metadata = {
                int(k): ChunkMetadata(**v) for k, v in raw.items()
            }
            logger.info(f"Metadata loaded: {len(self.metadata)} entries")

            return True

        except Exception as e:
            logger.error(f"Load failed: {e}")
            raise VectorStoreError(f"Load karne mein error: {e}") from e

    # ---------- Stats ----------

    def stats(self) -> dict:
        """Index ki information."""
        return {
            "index_name": self.index_name,
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal if self.index else 0,
            "metadata_entries": len(self.metadata),
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
        }

    # ---------- Clear ----------

    def clear(self) -> None:
        """Sab kuch reset karein (test ke liye)."""
        self.index = None
        self.metadata = {}
        logger.info("Vector store cleared.")