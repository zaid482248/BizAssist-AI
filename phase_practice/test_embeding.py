"""Embeddings test - Phase 6."""

import os
from pathlib import Path

from dotenv import load_dotenv

# .env file load karein
load_dotenv()

from bizassist_ai.core.chunking import chunk_text
from bizassist_ai.core.embeddings import embed_text, embed_batch, get_embedding_dimension
from bizassist_ai.core.similarity import cosine_similarity


def main():
    print("=" * 60)
    print("Phase 6: Embeddings Test")
    print("=" * 60)

    # Token check
    token = os.getenv("HF_API_TOKEN", "")
    if not token or token.startswith("hf_xxx"):
        print(" HF_API_TOKEN .env file mein set nahi hai.")
        print("   Hugging Face se token lein aur .env mein daalein.")
        return

    print(f" Token mila: {token[:8]}...")
    print(f" Expected dimension: {get_embedding_dimension()}")
    print("=" * 60)

    # Test 1: Single text embedding
    print("\n Test 1: Single text embedding")
    text = "ShopEase provides free shipping on orders above 2000 PKR."
    vector = embed_text(text)
    print(f"   Text: {text[:60]}...")
    print(f"   Vector length: {len(vector)}")
    print(f"   First 5 values: {[round(v, 4) for v in vector[:5]]}")

    # Test 2: Similarity check (magic of embeddings!)
    print("\n Test 2: Similarity check")
    v1 = embed_text("I love pizza and pasta")
    v2 = embed_text("Pizza and pasta are my favorite food")
    v3 = embed_text("My car engine is broken")

    sim_12 = cosine_similarity(v1, v2)
    sim_13 = cosine_similarity(v1, v3)


    print(f"   'pizza' vs 'pizza pasta' similarity: {sim_12:.4f}  (should be HIGH)")
    print(f"   'pizza' vs 'car broken' similarity:  {sim_13:.4f}  (should be LOW)")

    # Test 3: Real document chunks
    print("\n Test 3: Real document chunks")
    doc_path = Path("data/sample_docs/shipping_policy.md")
    if doc_path.exists():
        text = doc_path.read_text(encoding="utf-8")
        chunks = chunk_text(text, chunk_size=800, overlap=100)
        print(f"   Document: {doc_path.name}")
        print(f"   Chunks: {len(chunks)}")

        # Pehle 3 chunks ka embedding banayein
        chunk_texts = [c.text for c in chunks[:3]]
        vectors = embed_batch(chunk_texts)
        print(f"   Embeddings banaye: {len(vectors)}")
        for i, v in enumerate(vectors):
            print(f"   Chunk {i}: {len(v)} dimensions")
    else:
        print(f"     Document nahi mila: {doc_path}")

    print("\n" + "=" * 60)
    print(" Embeddings test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()