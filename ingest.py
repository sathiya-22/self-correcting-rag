"""
ingest.py — PDF Ingestion Script for Self-Correcting RAG
Drop PDFs into knowledge_base/ and run: python ingest.py
"""

import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPDFLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"
CHROMA_DB_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "rag_docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_files() -> list[Path]:
    """Recursively find all supported files in the knowledge_base directory."""
    extensions = [".pdf", ".docx"]
    files = []
    for ext in extensions:
        files.extend(list(KNOWLEDGE_BASE_DIR.rglob(f"*{ext}")))
    return files


def compute_doc_id(file_path: Path, chunk_index: int) -> str:
    """Stable unique ID for a chunk: SHA256 of file path + index."""
    raw = f"{file_path.resolve()}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def load_and_split(file_path: Path) -> list:
    """Load a file and split it into overlapping text chunks."""
    ext = file_path.suffix.lower()
    docs = []
    
    try:
        if ext == ".pdf":
            try:
                # Try PyPDF first
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
            except Exception as e:
                print(f"   ⚠️  PyPDFLoader failed for {file_path.name}: {e}. Trying PDFMiner...")
                try:
                    loader = PDFMinerLoader(str(file_path))
                    docs = loader.load()
                except Exception as e2:
                    print(f"   ⚠️  PDFMinerLoader failed: {e2}. Falling back to Unstructured (last resort)...")
                    loader = UnstructuredPDFLoader(str(file_path))
                    docs = loader.load()
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(str(file_path))
            docs = loader.load()
    except Exception as e:
        print(f"   ❌ Failed to load {file_path.name}: {e}")
        return []

    if not docs:
        print(f"   ❌ No content extracted from {file_path.name}")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # Attach stable IDs
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = str(file_path.name)
        chunk.metadata["chunk_id"] = compute_doc_id(file_path, i)
    return chunks


def ingest():
    """Main ingestion pipeline."""
    files = get_files()
    if not files:
        print("⚠️  No supported files (.pdf, .docx) found in knowledge_base/. Drop your files there and re-run.")
        return

    print(f"📂 Found {len(files)} supported file(s): {[f.name for f in files]}")

    # Initialize embedding model (downloads on first run)
    print(f"🔄 Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Initialize Chromadb (ensure dir exists)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    # Load existing vectorstore or create new one
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )

    # Get already-ingested IDs to avoid duplicates
    existing_ids: set[str] = set()
    try:
        existing = vectorstore.get()
        if existing and existing.get("ids"):
            existing_ids = set(existing["ids"])
        print(f"ℹ️  Existing chunks in store: {len(existing_ids)}")
    except Exception as e:
        print(f"ℹ️  Starting fresh vector store. (Error reading: {e})")

    total_new_chunks = 0
    for file_path in files:
        print(f"\n📄 Processing: {file_path.name}")
        chunks = load_and_split(file_path)
        
        if not chunks:
            continue

        new_chunks = [c for c in chunks if c.metadata["chunk_id"] not in existing_ids]
        if not new_chunks:
            print(f"   ✅ Already ingested — skipping.")
            continue

        ids = [c.metadata["chunk_id"] for c in new_chunks]
        vectorstore.add_documents(new_chunks, ids=ids)
        total_new_chunks += len(new_chunks)
        print(f"   ✅ Ingested {len(new_chunks)} new chunks.")

    print(f"\n🎉 Done! Added {total_new_chunks} new chunks to ChromaDB at '{CHROMA_DB_DIR}'.")


if __name__ == "__main__":
    ingest()
