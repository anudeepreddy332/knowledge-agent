# import config data
import os.path
import re

import chromadb
from sentence_transformers import SentenceTransformer

from src.knowledge_agent.config import DATA_DIR,DB_DIR, EMBEDDING_MODEL

# Config

COLLECTION = "knowledge"
CHUNK_SIZE = 400
OVERLAP = 80

# 1. Load

def load_data(data_dir: str) -> list[dict]:
    """
    Scan data_dir and read every .txt and .md file.
    Returns a list of dicts: {filename, text}
    """
    data = []
    for fname in os.listdir(data_dir):
        if not fname.endswith((".txt", ".md")):
            continue
        path = os.path.join(DATA_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        data.append({
            "filename": fname,
            "text": text
        })
        print(f" Loaded: {fname} ({len(text)} chars)")
    return data


# 2. Chunk

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """
    Paragraph-first chunking with fixed-size fallback.

    Strategy:
        1. Split the document into paragraphs (by looking for blank lines).
        2. For each paragraph:
            If it's short enough to fit in the current chunk, add it.
            If adding it would make the chunk too big, finish that chunk, start a new one, and carry over the last OVERLAP characters from the previous chunk as context.
            If a single paragraph is longer than CHUNK_SIZE, break it down into sentences and do the same thing.
        3. At the end, add any leftover text as the final chunk.

    This gives us the best of both worlds: semantic boundaries where they exist,
    graceful degradation where they don't.

    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            sentences = re.split(r"(?<=[.!])\s+", para)
            for sent in sentences:
                if len(current) + len(sent) + 1 <= chunk_size:
                    current = (current + " " + sent).strip()
                else:
                    if current:
                        chunks.append(current)
                        # carry overlap tail into next chunk
                    current = (current[-overlap:] + " " + sent).strip() if overlap else sent
        else:
            if len(current) + len(para) + 2 <= chunk_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                # overlap: take the tail of the last chunk as context seed
                tail = current[-overlap:] if overlap else ""
                current = (tail + "\n\n" + para).strip()

    if current:
        chunks.append(current)

    return chunks


# 3. Embed + Store

def ingest(data_dir: str = DATA_DIR, db_dir: str = DB_DIR):
    print("=== Ingestion Pipeline ===\n")

    # load
    print("[1/3] Loading data...")
    data = load_data(data_dir)
    if not data:
        print(f"  No .txt or .md files found in '{data_dir}'. Dropping out.")
        return

    # chunk
    print(f"\n[2/3] Chunking (size={CHUNK_SIZE}, overlap={OVERLAP})...")
    all_chunks = []
    for d in data:
        chunks = chunk_text(d["text"])
        print(f"  {d['filename']} → {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "filename": d["filename"],
                "chunk_index": i,
            })
    print(f"  Total chunks: {len(all_chunks)}")


    # Embed
    print(f"\n[3/3] Embedding with '{EMBEDDING_MODEL}' and storing in ChromaDB...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in all_chunks]
    print(f"  Embedding {len(texts)} chunks...")
    embeddings_np = model.encode(texts, show_progress_bar=True)
    embeddings = embeddings_np.tolist()
    # Store
    client = chromadb.PersistentClient(path=DB_DIR)

    # Wipe and recreate so that re-runs don't double insert
    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print(f"  Dropped existing '{COLLECTION}' collection (clean rebuild).")

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"} # use cosine similarity
    )

    ids = [f"{c['filename']}__chunk_{c['chunk_index']}" for c in all_chunks]
    metadatas = [{
        "filename": c["filename"],
        "chunk_index": c["chunk_index"]
    } for c in all_chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"\n  Stored {len(ids)} chunks in '{db_dir}/{COLLECTION}'.")
    print("\n=== Ingestion complete. DB is ready for Day 10. ===")


# Smoke test - query the db right after ingestion

def smoke_test(query: str = "What is RAG and why does it matter?"):
    """Quick retrieval test to verify the pipeline works end to end"""
    print(f"\n--- Smoke test: '{query}' ---")

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION)

    q_embedding_np = model.encode([query])
    q_embedding = q_embedding_np[0].tolist()
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        print(f"\n  Result {i+1} | {meta['filename']} chunk {meta['chunk_index']} | distance {dist:.4f}")
        print(f"  {doc[:200]}{'...' if len(doc) > 200 else ''}")



if __name__ == "__main__":
    ingest()
    smoke_test()

