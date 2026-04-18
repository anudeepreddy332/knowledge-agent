"""
Each tool has two parts:
    1. A SCHEMA - the JSON dict that gets sent to LLM so it knows the tool exists
        and what arguments to pass.
    2. An EXECUTOR - a plain Python function that actually runs when the LLM calls the tool.

The agent loop in main.py handles the routing: it reads the tool_name the LLM emits,
looks it up here, and calls the right executor.
"""
import os
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.knowledge_agent.config import (
    EMBEDDING_MODEL, DB_DIR, RERANKER_MODEL,
    TOP_K_CANDIDATES, TOP_K, BM25_WEIGHT, CHUNK_SIZE, CHUNK_OVERLAP)
from tavily import TavilyClient
from rank_bm25 import BM25Okapi
import re


# Shared singletons
# Both the model and the ChromaDB client are expensive to initialize.
# We load them once at module import time and reuse across calls.


_model = None
_reranker = None
_chroma = None
_collection = None
_bm25 = None  # BM25 index built from all chunks in ChromaDB
_bm25_docs = None  # parallel list of (doc_text, metadata) for BM25 results

COLLECTION = "knowledge"


def _get_retriever():
    """
    Lazy-initialize bi-encoder, cross-encoder, ChromaDB, and BM25 index.

    Two models, one function:
    - _model (SentenceTransformer): same bi-encoder used during ingestion.
      Encodes query → vector → cosine search in ChromaDB.
    - _reranker (CrossEncoder): takes (query, chunk) concatenated, outputs
      a relevance float. Higher = more relevant. No vector space involved.

    Why cache both here: both are expensive to load (~1s, ~80MB each).
    Lazy init means a session that never calls KB pays nothing.

    BM25 index is built by loading all chunks from ChromaDB at startup.
    Why rebuild from ChromaDB rather than from disk? ChromaDB is the source
    of truth — it includes web-saved chunks from save_to_knowledge_base.
    Building BM25 from ChromaDB ensures hybrid search sees everything.

    BM25 index is in-memory and rebuilt each session.
    For 42 chunks this is instantaneous. At 10k chunks it's still fast (<1s).
    At 1M chunks you'd want to persist the index — not a concern here.

    """
    global _model, _chroma, _collection, _reranker, _bm25, _bm25_docs

    # Lazy init of bi-encoder, cross-encoder, ChromaDB
    if _collection is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
        _chroma = chromadb.PersistentClient(path=DB_DIR)
        _collection = _chroma.get_collection(COLLECTION)

    # Build BM25 index from ChromaDB if missing (first load or after invalidation)
    if _bm25 is None:
        all_results = _collection.get(include=["documents", "metadatas"])
        all_docs = all_results["documents"]
        all_metas = all_results["metadatas"]

        # Tokenize by whitespace
        tokenized = [doc.lower().split() for doc in all_docs]
        _bm25 = BM25Okapi(tokenized)
        _bm25_docs = list(zip(all_docs, all_metas))


    return _model, _collection, _reranker, _bm25, _bm25_docs


# Tool 1: query_knowledge_base

# Schema - A dict used by the model to reason and decide *when* to call the tool and *what to pass*

QUERY_KB_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_knowledge_base",
        "description": (
            "Search the local knowledge base for information from ingested documents. "
            "Use this tool when the question is about topics covered in your private notes "
            "or documents — for example, concepts, explanations, or details that would "
            "be found in a personal knowledge base rather than live on the web. "
            "Do NOT use this for real-time information, current events, or anything "
            "that requires up-to-date data. "
            "SINGLE USE PER TURN: call this tool exactly once. "
            "After it returns, synthesize from what you have — do not call again "
            "with a rephrased or narrower query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language question or search phrase to look up in the knowledge base.",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve. Defaults to 3. Increase to 5 for broad questions.",
                    "default": 3,
                },
            },
        "required": ["query"]
        },
    },
}


# Executor
# Called by the agent loop with the arguments the LLM passed.

def execute_query_knowledge_base(query: str, n_results: int = TOP_K) -> str:
    """
    Three-stage retrieval: BM25 + cosine -> RRF merge -> cross-encoder rerank.

    Stage 1a - BM25 (keyword search):
        Tokenize query, score all chunks by term frequency. Returns ranked list.
        Good at: exact term matches ("HNSW", "MiniLM", "context poisoning").
        Bad at: semantic similarity, synonyms, paraphrasing.

    Stage 1b - Dense cosine search (semantic search):
        Embed query, retrieve TOP_K_CANDIDATES from ChromaDB by cosine distance.
        Good at: semantic similarity, paraphrasing, concept matching.
        Bad at: exact rare terms that weren't in training data.

    Stage 2 - Reciprocal Rank Fusion (RRF):
        For each chunk, score = 1/(rank_in_bm25 + k) + 1/(rank_in_dense + k).
        Chunks appearing high in both lists get the highest RRF scores.
        Chunks appearing in only one list get partial credit.
        k=60 is the standard default - dampens the advantage of rank-1 vs rank-2.
        Take top TOP_K_CANDIDATES from merged list as candidate set.

    Stage 3 - Cross-encoder rerank:
        Feed (query, chunk) pairs to cross-encoder. Keep top n_results.

    Why this order matters?
        BM25 alone misses semantic matches.
        Dense alone misses exact rare terms.
        RRF combines both, giving the cross-encoder the best candidate set.
        Cross-encoder then applies precision scoring on that rich candidate set.

    Vulnerabilities:
        - BM25 index is built at startup from ChromaDB — stale if ingest runs
          mid-session. Acceptable: ingest is a one-time batch job.
        - RRF assumes both lists have the same chunks available. BM25 scores
          all chunks; dense only returns TOP_K_CANDIDATES. Handled by using
          BM25 rank over all chunks but capping the merge at TOP_K_CANDIDATES.

    """

    if not query.strip():
        return "[query_knowledge_base] Empty query — nothing to search."

    try:
        model, collection, reranker, bm25, bm25_docs = _get_retriever()
    except Exception as e:
        return (
            f"[query_knowledge_base] Failed to load retriever: {e}\n"
            f"Make sure you have run scripts/ingest.py before querying."

        )

    # Clamp TOP_K_CANDIDATES to collections
    total = collection.count()
    if total == 0:
        return "[query_knowledge_base] Knowledge base is empty. Run ingest.py first."

    candidates_count = min(TOP_K_CANDIDATES, total)

    # Stage 1a: BM25 — rank all chunks by keyword match
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)  # shape: (total_chunks,)

    # Build rank map: chunk index in bm25_docs → BM25 rank (0 = best)
    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_rank = {idx: rank for rank, idx in enumerate(bm25_ranked_indices)}

    # Stage 1b: Dense cosine search — retrieve top candidates from ChromaDB
    q_vec = model.encode([query])[0].tolist()

    dense_results = collection.query(
        query_embeddings=[q_vec],
        n_results=candidates_count,
        include=["documents", "metadatas", "distances"],
    )

    dense_docs = dense_results["documents"][0]
    dense_metas = dense_results["metadatas"][0]

    # Build a lookup: chunk_index → (doc, meta) for dense results
    dense_lookup = {
        meta["chunk_index"]: (doc, meta) for doc, meta in zip(dense_docs, dense_metas)
    }
    dense_rank = {
        meta["chunk_index"]: rank for rank, meta in enumerate(dense_metas)}


    # Stage 2: RRF merge
    # We need a unified candidate set. Start with all dense results, then boost using BM25 ranks from the full corpus.
    k = BM25_WEIGHT
    rrf_scores = {}

    for chunk_idx, (doc, meta) in dense_lookup.items():
        d_rank = dense_rank[chunk_idx]
        # Find this chunk's position in bm25_docs to get its BM25 rank
        bm25_position = next(
            (i for i, (_, m) in enumerate(bm25_docs) if m["chunk_index"] == chunk_idx),
            len(bm25_docs)  # if not found, treat as last rank
        )
        b_rank = bm25_rank.get(bm25_position, len(bm25_docs))
        rrf_scores[chunk_idx] = (1 / (d_rank + k)) + (1 / (b_rank + k))

    # Also add any BM25-only top results not in dense results
    for bm25_position in bm25_ranked_indices[:candidates_count]:
        doc, meta = bm25_docs[bm25_position]
        chunk_idx = meta["chunk_index"]
        if chunk_idx not in rrf_scores:
            b_rank = bm25_rank[bm25_position]
            rrf_scores[chunk_idx] = 1 / (b_rank + k)

    # Sort by RRF score, take top candidates
    top_chunk_indices = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:candidates_count]

    # Resolve chunk_index back to (doc, meta)
    chunk_lookup = {
        meta["chunk_index"]: (doc, meta) for doc, meta in bm25_docs
    }
    candidates = [chunk_lookup[idx] for idx in top_chunk_indices if idx in chunk_lookup]

    if not candidates:
        return "[query_knowledge_base] No candidates found after merge."

    # Stage 3: cross-encoder rerank
    # CrossEncoder.predict takes a list of [query, chunk] pairs.
    # Returns a numpy array of floats — higher = more relevant.
    pairs = [[query, doc] for doc, _ in candidates]
    scores = reranker.predict(pairs)  # shape: (candidates,)

    # Sort by score descending, keep top n_results
    ranked = sorted(
        zip(scores, [doc for doc, _ in candidates], [meta for _, meta in candidates]),
        key=lambda x: x[0],
        reverse=True
    )[:n_results]

    # Format output — note: score replaces distance, higher is better

    output_lines = []
    for score, doc, meta in ranked:
        output_lines.append(
            f"[Source: {meta['filename']} | chunk {meta['chunk_index']} | relevance: {score:.4f}]\n{doc}"
        )
    return "\n---\n".join(output_lines)

# Tool 2: web_search

WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current, real-time, or general information. "
            "Use this when the question involves recent events, news, live data, "
            "or anything that would NOT be covered in a personal knowledge base. "
            "Also use this as a fallback if query_knowledge_base returns no useful results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string. Be specific — treat this like a Google search.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return. Default 5. Max 10.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


def execute_web_search(query: str, max_results: int = 5) -> str:
    if not query.strip():
        return "[web_search] Empty query — nothing to search."
    try:
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = tavily.search(query=query, max_results=max_results)
    except Exception as e:
        return f"[web_search] Search failed: {e}"

    results = response.get("results", [])
    if not results:
        return f"[web_search] No results for: '{query}'"

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "No snippet")
        lines.append(f"[{i}] {title}\nURL: {url}\nSnippet: {content}")

    return "\n---\n".join(lines)


SAVE_TO_KB_SCHEMA = {
    "type": "function",
    "function": {
        "name": "save_to_knowledge_base",
        "description": (
            "Save a piece of text to the local knowledge base for future retrieval. "
            "Use this after web_search returns useful information that you want to "
            "persist for future sessions. The text will be chunked, embedded, and "
            "stored in ChromaDB with source metadata. "
            "Only save factual, high-quality content — not search result noise."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text content to save. Should be clean prose, not raw HTML.",
                },
                "source_url": {
                    "type": "string",
                    "description": "The URL or source identifier this content came from.",
                },
                "topic": {
                    "type": "string",
                    "description": "A short label for what this content is about (e.g. 'HNSW paper 2025').",
                },
            },
            "required": ["text", "source_url", "topic"],
        },
    },
}


def execute_save_to_kb(text: str, source_url: str, topic: str) -> str:
    """
    Chunk, embed, and store text in ChromaDB with web source metadata.

    How it works:
        1. Split text into chunks using the same para-first strategy as ingest.py.
        2. Embed each chunk with the same bi-encoder used during ingestion.
        3. Store in the same ChromaDB collection with metadata flagging source as "web".
        4. Invalidate the BM25 index so next session rebuilds it with the new chunks.

    Why invalidate BM25: _bm25 is built at startup from all chunks. New chunks
    added mid-session won't be in the index. Setting _bm25 = None forces a rebuild
    on the next _get_retriever() call. Dense search (ChromaDB) is updated immediately
    since we write directly to it.

    Vulnerabilities:
        - No deduplication: saving the same URL twice creates duplicate chunks.
          Guard: could check metadata for existing source_url before saving.
          Not implemented — keeping it simple for now.
        - Chunk IDs must be globally unique. Using source_url + chunk_index as ID.
          If source_url contains special characters, sanitize first.
        - text quality is not checked. Saving noise degrades retrieval.

    """
    if not text.strip():
        return "[save_to_knowledge_base] Empty text — nothing to save."

    try:
        model, collection, _, _, _ = _get_retriever()
    except Exception as e:
        return f"[save_to_knowledge_base] Failed to load retriever: {e}"

    # Chunk using the same strategy as ingest.py
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para.split()) <= CHUNK_SIZE:
            chunks.append(para)
        else:
            # Fixed-size fallback with overlap
            words = para.split()
            for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk = " ".join(words[i:i + CHUNK_SIZE])
                if chunk:
                    chunks.append(chunk)

    if not chunks:
        return "[save_to_knowledge_base] Text produced no chunks after splitting."

    # Sanitize source_url for use as ID prefix
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', source_url)[:50]

    # Check for existing chunks from this source to avoid duplicates
    existing = collection.get(where={"source_url": source_url})
    if existing["documents"]:
        return (
            f"[save_to_knowledge_base] Source already in KB: '{source_url}' "
            f"({len(existing['documents'])} chunks). Skipping to avoid duplicates."
        )

    # Embed ad store
    embeddings = model.encode(chunks).tolist()
    ids = [f"{safe_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "filename": topic,
            "chunk_index": i,
            "source": source_url,
        }
        for i in range(len(chunks))
    ]

    try:
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    except Exception as e:
        return f"[save_to_knowledge_base] Failed to write to ChromaDB: {e}"

    # Invalidate BM25 index so it rebuilds next call with new chunks
    global _bm25, _bm25_docs
    _bm25 = None
    _bm25_docs = None

    return (
        f"[save_to_knowledge_base] Saved {len(chunks)} chunks from '{source_url}' "
        f"to knowledge base under topic '{topic}'."
    )


# Tool 3: verify_claim

VERIFY_CLAIM_SCHEMA = {
    "type": "function",
    "function": {
        "name": "verify_claim",
        "description": (
            "Verify whether a specific claim is supported by a specific chunk "
            "in the knowledge base. Use this after query_knowledge_base returns results "
            "to confirm that a statement you are about to make is actually grounded "
            "in the retrieved source. Pass the exact claim and the chunk index "
            "from the source metadata."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of specific factual claims to verify against the chunk. Pass all claims from one source chunk in a single call.",
                },
                "chunk_index": {
                    "type": "integer",
                    "description": "The chunk_index value from the [Source: ... | chunk N | ...] line.",
                },
                "filename": {
                    "type": "string",
                    "description": "The filename from the [Source: filename | ...] line.",
                },
            },
            "required": ["claims", "chunk_index", "filename"],
        },
    },
}


def execute_verify_claim(claims: list[str], chunk_index: int, filename: str) -> str:
    """
        Verify a list of claims against a single chunk in one call.

        Why a list: prevents the model from calling verify_claim once per claim,
        which burns tool rounds and hits MAX_TOOL_ROUNDS before answering.
        All claims for a given source chunk are verified in one shot.

        Vulnerabilities:
            - Same chunk_index/filename constraints as before.
            - Similarity threshold 0.75 is a tuning knob.
            - Empty claims list: returns early with a clear error.
    """
    if not claims:
        return "[verify_claim] No claims provided."

    try:
        model, collection, _, _, _ = _get_retriever()
    except Exception as e:
        return f"[verify_claim] Failed to load retriever: {e}"

    try:
        results = collection.get(
            where={"$and": [{"filename": filename}, {"chunk_index": chunk_index}]},
            include=["documents", "metadatas"],
        )
    except Exception as e:
        return f"[verify_claim] ChromaDB query failed: {e}"

    docs = results.get("documents", [])
    if not docs:
        return (
            f"[verify_claim] Chunk not found: filename='{filename}', "
            f"chunk_index={chunk_index}."
        )

    chunk_text = docs[0]

    import numpy as np
    chunk_vec = model.encode([chunk_text])[0]  # encode chunk once, reuse for all claims

    SUPPORT_THRESHOLD = 0.65
    lines = [f"Source: {filename} | chunk {chunk_index}"]
    lines.append(f"Chunk preview: {chunk_text[:200]}{'...' if len(chunk_text) > 200 else ''}\n")

    for claim in claims:
        if not claim.strip():
            continue
        claim_vec = model.encode([claim])[0]
        similarity = float(
            np.dot(claim_vec, chunk_vec) /
            (np.linalg.norm(claim_vec) * np.linalg.norm(chunk_vec))
        )
        verdict = "SUPPORTED" if similarity >= SUPPORT_THRESHOLD else "NOT SUPPORTED"
        lines.append(f"[{verdict}] (sim={similarity:.4f}) {claim}")

    return "\n".join(lines)


# Tool registry
# main.py imports two objects.
# TOOL_SCHEMAS -> list of dicts sent to the LLM in every API call
# TOOL_EXECUTORS -> name -> callable, used to dispatch tool_use responses

TOOL_SCHEMAS = [
    QUERY_KB_SCHEMA,
    WEB_SEARCH_SCHEMA,
    VERIFY_CLAIM_SCHEMA,
    SAVE_TO_KB_SCHEMA,
]

TOOL_EXECUTORS = {
    "query_knowledge_base": execute_query_knowledge_base,
    "web_search": execute_web_search,
    "verify_claim": execute_verify_claim,
    "save_to_kb": execute_save_to_kb,
}
