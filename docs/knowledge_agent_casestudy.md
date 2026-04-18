# Phase 2 Case Study: Building a Persistent Knowledge Agent with Hybrid RAG

## 1. Overview & Context

Phase 2 upgrades the agent into a persistent knowledge system using Retrieval-Augmented Generation (RAG).

Instead of relying only on web search, the agent:
- stores knowledge  
- retrieves relevant chunks  
- verifies claims  
- reuses past information  

This directly fixes Phase 1 limitations.

### Architecture

User Query  
→ Routing Decision (KB vs Web)  
→ query_knowledge_base / web_search  
→ Dense Retrieval + BM25  
→ Reciprocal Rank Fusion  
→ Cross-Encoder Reranking  
→ verify_claim  
→ Answer Synthesis  
→ Optional save_to_knowledge_base  

---

## 2. Core Mechanics

Problem:
**How do we make knowledge reusable and reliable?**

Pipeline:
store → retrieve → rerank → verify → generate  

Plain English:  
Instead of asking the internet every time, the agent builds its own memory.

---

## 3. Technical Deep Dive

### Embeddings (Dense Retrieval)

**Math:**
```
cosine_similarity = (A · B) / (||A|| ||B||)
```
**Plain English:**  
Two texts are similar if they point in the same direction.

---

### Ingestion Pipeline

**What it does:**  
Splits documents into chunks.

Strategy:
- Paragraph-first  
- fallback chunking  

**Why:**  
Preserve semantic meaning while enabling retrieval.

---

### query_knowledge_base

**Flow:**
query → embed → search → retrieve chunks  

**Example:**  
Query: “What is RAG?” → retrieves chunk containing “context window”  

---

### Hybrid Search (BM25 + Dense)

**Formula:**
```
score = 1/(rank_dense + 60) + 1/(rank_bm25 + 60)
```
**Plain English:**  
Combine keyword matching with semantic similarity.

---

### Cross-Encoder Reranking

**Difference:**
- Bi-encoder → compare independently  
- Cross-encoder → compare jointly  

**Plain English:**  
Read query and document together instead of separately.

---

### Concrete Retrieval Example

Query: “What is context poisoning in RAG?”

Step 1: BM25 finds chunk with exact keyword  
Step 2: Dense retrieval finds semantically related chunk  
Step 3: RRF merges rankings  
Step 4: Cross-encoder scores relevance  
→ Best chunk selected  

---

### verify_claim

**What it does:**  
Checks if a claim is supported.

**Logic:**
cosine_similarity ≥ 0.65 → SUPPORTED  

**Example:**  
Claim compared with chunk → similarity 0.71 → supported  

---

### Dual-Tool Routing

Rule:
- Strong KB match → use KB  
- Weak match → use web  

---

### save_to_knowledge_base

**What it does:**  
Stores useful web results for reuse.

**Example flow:**  
Web search → save → next query uses KB  

---

### Evaluation Harness

Dataset: eval_queries.json  

Measures:
- routing accuracy  
- content correctness  
- relevance score  

---

## 4. Challenges, Failures, and Pivots

### Routing Failures
- Fix: explicit prompt rules  

---

### Context Poisoning
- Fix: cross-encoder reranking  

---

### BM25 Invalidation Bug
- Fix: rebuild BM25 independently  

---

### Eval Mismatch
- Fix: shared SYSTEM_PROMPT  

---

## 5. Evaluation & Results

| Metric | Result |
|------|--------|
| Tool routing accuracy | 100% |
| Content hit rate | 92% |
| Overall pass rate | 92% |
| Mean relevance | -4.0948 |

**Interpretation:**  
- Routing solved  
- Retrieval mostly correct  
- One failure due to keyword mismatch  

---

## 6. Key Learnings & Takeaways

- Principle: Retrieval quality > model quality  
  Example: reranking fixes wrong context  

- Principle: Hybrid search improves recall  
  Example: BM25 + dense outperform single method  

- Principle: Verification reduces hallucination  
  Example: verify_claim filters weak matches  

- Principle: Evaluation must match production  
  Example: prompt mismatch broke eval  

- Principle: Systems fail silently  
  Example: BM25 bug surfaced at runtime  

---

## 7. What’s Next (Phase 3)

Remaining gaps:
- No self-correction loop  
- No execution feedback  
- Cost tracking not fully wired  

Phase 3 introduces:
- LangGraph state machine  
- execution → diagnose → patch loop  

---

## 8. Appendix: How to Run

python -m scripts.ingest  
python -m scripts.evaluate  
python main.py  

Code reference: src/knowledge_agent/tools.py, main.py  