"""
NUST Chatbot - Data Ingestion Script
=====================================
Reads all .txt files from nust_data/ folder,
splits them into chunks, embeds them using Ollama,
and stores them in a local ChromaDB vector database.

Run ONCE after scraping:
    python ingest.py

Re-run whenever you update the scraped data.
"""

import os
import glob
import time
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = "./nust_data"          # folder with scraped .txt files
DB_DIR      = "./nust_db"            # where ChromaDB will be saved
EMBED_MODEL = "nomic-embed-text"     # Ollama embedding model
CHUNK_SIZE  = 500                    # characters per chunk
CHUNK_OVERLAP = 80                   # overlap between chunks


def load_documents():
    """Load all .txt files from nust_data/ recursively."""
    txt_files = glob.glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True)

    if not txt_files:
        print(f"[ERROR] No .txt files found in '{DATA_DIR}'")
        print("  → Run the scraper first: python nust_scraper.py")
        exit(1)

    print(f"Found {len(txt_files)} text files...")
    docs = []
    for filepath in txt_files:
        try:
            loader = TextLoader(filepath, encoding="utf-8")
            loaded = loader.load()
            # Add source metadata to each document
            for doc in loaded:
                doc.metadata["source_file"] = os.path.basename(filepath)
            docs.extend(loaded)
        except Exception as e:
            print(f"  [SKIP] {os.path.basename(filepath)}: {e}")

    print(f"Loaded {len(docs)} documents successfully.")
    return docs


def split_documents(docs):
    """Split documents into smaller chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_vectorstore(chunks):
    """Embed chunks and store in ChromaDB."""
    print(f"\nConnecting to Ollama embedding model: '{EMBED_MODEL}'...")
    print("Make sure Ollama is running: ollama serve")
    print("And model is pulled:         ollama pull nomic-embed-text\n")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Test connection
    try:
        _ = embeddings.embed_query("test")
        print("Ollama connection OK ✓")
    except Exception as e:
        print(f"[ERROR] Cannot connect to Ollama: {e}")
        print("  → Start Ollama with: ollama serve")
        exit(1)

    # Delete old DB if exists
    if os.path.exists(DB_DIR):
        import shutil
        shutil.rmtree(DB_DIR)
        print(f"Cleared old database at '{DB_DIR}'")

    print(f"\nEmbedding {len(chunks)} chunks... (this may take a few minutes)")
    start = time.time()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    vectorstore.persist()

    elapsed = time.time() - start
    print(f"\nDone! Embedded {len(chunks)} chunks in {elapsed:.1f}s")
    print(f"Vector database saved to: {os.path.abspath(DB_DIR)}")
    return vectorstore


def verify_vectorstore():
    """Quick test to confirm the DB works."""
    print("\nVerifying database with a test query...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    results = db.similarity_search("What are the eligibility criteria for undergraduate admission?", k=3)
    print(f"Test search returned {len(results)} results ✓")
    if results:
        print(f"Sample: {results[0].page_content[:200]}...\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  NUST Chatbot - Ingestion Pipeline")
    print("=" * 60 + "\n")

    docs   = load_documents()
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    verify_vectorstore()

    print("=" * 60)
    print("  Ingestion complete! Now run: streamlit run chatbot.py")
    print("=" * 60)