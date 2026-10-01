"""
Phase 1 Practice - BizAssist AI
Yeh code Python ke basics demonstrate karta hai.
"""

import json
from pathlib import Path


# ---------- 1. FUNCTION ----------

def clean_text(text: str) -> str:
    """Extra spaces hataata hai aur text ko strip karta hai."""
    return " ".join(text.split())


# ---------- 2. CLASS ----------
class DocumentProcessor:
    """Ek simple document processor jo file padh kar usay process karta hai."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.text = ""

    def load(self) -> None:
        """File se text load karta hai."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                raw = f.read()
            self.text = clean_text(raw)
            print(f"✅ File loaded: {self.file_path}")
        except FileNotFoundError:
            print(f"❌ File nahi mili: {self.file_path}")
            raise
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

    def stats(self) -> dict:
        """Text ki statistics return karta hai."""
        words = self.text.split()
        return {
            "characters": len(self.text),
            "words": len(words),
            "preview": self.text[:50] + "..." if len(self.text) > 50 else self.text,
        }


# ---------- 3. FILE HANDLING ----------
def create_sample_file(path: str) -> None:
    """Ek sample text file banata hai test ke liye."""
    sample_content = """
    BizAssist AI ek business knowledge assistant hai.
    Yeh documents se information nikalta hai.
    Phir user ke sawalon ka jawab deta hai.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(sample_content)
    print(f"✅ Sample file banayi: {path}")


# ---------- 4. MAIN PROGRAM ----------
def main():
    print("=" * 50)
    print("Phase 1 Practice - BizAssist AI")
    print("=" * 50)

    # Sample file banao
    sample_path = "sample.txt"
    create_sample_file(sample_path)

    # DocumentProcessor use karo
    processor = DocumentProcessor(sample_path)
    processor.load()

    # Stats print karo
    stats = processor.stats()
    print("\n📊 Document Stats:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # Exception test karo (juz se missing file)
    print("\n🧪 Testing exception handling with missing file:")
    try:
        bad_processor = DocumentProcessor("missing.txt")
        bad_processor.load()
    except FileNotFoundError:
        print("✅ Exception sahi handle hua!")

    print("\n🎉 Phase 1 practice complete!")


if __name__ == "__main__":
    main()