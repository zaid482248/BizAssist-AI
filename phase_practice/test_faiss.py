"""FAISS test - Phase 7."""

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from bizassist_ai.core.chunking import chunk_text
from bizassist_ai.core.embeddings import embed_text, embed_batch
from bizassist_ai.core.vector_store import ChunkMetadata, VectorStore


def main():
    print("=" * 60)
    print("Phase 7: FAISS Vector Store Test")
    print("=" * 60)

    # ---------- Step 1: Document Load + Chunk ----------
    doc_path = Path("data/sample_docs/shipping_policy.md")
    text = doc_path.read_text(encoding="utf-8")
    chunks = chunk_text(text, chunk_size=800, overlap=100)

    print(f"\n📄 Document: {doc_path.name}")
    print(f"   Characters: {len(text)}")
    print(f"   Chunks: {len(chunks)}")

    # ---------- Step 2: Embed ----------
    # Sirf pehle 4 chunks (rate limit safe)
    test_chunks = chunks[:4]
    print(f"\n🔢 Embedding {len(test_chunks)} chunks...")
    vectors = embed_batch([c.text for c in test_chunks])
    print(f"   Done: {len(vectors)} vectors ({len(vectors[0])} dims each)")

    # ---------- Step 3: Add to FAISS ----------
    print("\n📦 Adding to FAISS...")
    store = VectorStore(index_name="test_index")

    metadatas = [
        ChunkMetadata(
            chunk_index=c.index,
            document_name=doc_path.name,
            text=c.text,
            start_char=c.start_char,
            end_char=c.end_char,
        )
        for c in test_chunks
    ]

    ids = store.add(vectors, metadatas)
    print(f"   Added IDs: {ids}")
    print(f"   Stats: {store.stats()}")

    # ---------- Step 4: Save ----------
    print("\n💾 Saving to disk...")
    store.save()
    print(f"   Index: {store.index_path}")
    print(f"   Metadata: {store.metadata_path}")

    # ---------- Step 5: Search ----------
    print("\n🔍 Search test")
    query = "What is the shipping cost for remote areas?"
    print(f"   Query: '{query}'")

    query_vec = embed_text(query)
    results = store.search(query_vec, top_k=3)

    print(f"   Found {len(results)} results:")
    for i, r in enumerate(results, start=1):
        print(f"\n   --- Result {i} ---")
        print(f"   ID: {r['id']}")
        print(f"   Distance: {r['distance']:.4f}  (chhota = zyada similar)")
        print(f"   Document: {r['chunk'].document_name}")
        print(f"   Preview: {r['chunk'].text[:120]}...")

    # ---------- Step 6: Reload Test ----------
    print("\n🔄 Reload test (fresh store)")
    store2 = VectorStore(index_name="test_index")
    loaded = store2.load()
    print(f"   Loaded: {loaded}")
    print(f"   Stats: {store2.stats()}")

    results2 = store2.search(query_vec, top_k=3)
    print(f"   Same results after reload? {len(results2) == len(results)}")

    print("\n" + "=" * 60)
    print("🎉 FAISS test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
    