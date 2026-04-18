# Knowledge Agent – Persistent RAG with Hybrid Search

A local, memory-augmented agent that builds its own knowledge base.

Instead of starting blank every time, it retrieves, verifies, and reuses information from your documents and past web searches.

**Stack**: DeepSeek API · ChromaDB · Sentence Transformers · Cross-Encoder Reranking · BM25 + Dense Retrieval · Tavily

---

## What it does

- **Persistent memory** – JSON state file remembers past questions and facts  
- **Hybrid RAG** – BM25 keyword search + dense vector retrieval + reciprocal rank fusion  
- **Cross-encoder reranking** – re-scores the best candidates for higher precision  
- **Claim verification** – checks if a statement is grounded in a specific source chunk  
- **Save to KB** – persists useful web results for future sessions  
- **Dual routing** – decides between local KB and web search based on question type  

---

## Architecture

    User Query  
    → Routing (KB vs Web)  
    → query_knowledge_base / web_search  
    → Hybrid Retrieval (BM25 + Dense + RRF)  
    → Cross-Encoder Rerank  
    → verify_claim  
    → Answer with citations  
    → (optional) save_to_knowledge_base  

---

## Getting started

### 1. Clone & install dependencies

git clone git@github.com:anudeepreddy332/knowledge-agent.git  
cd knowledge-agent  
uv sync  

### 2. Set environment variables

Create a `.env` file:

DEEPSEEK_API_KEY=your_deepseek_key  
TAVILY_API_KEY=your_tavily_key  

### 3. Ingest your documents

Place `.txt` or `.md` files in `data/` (see `data/example.md` for format).  
Then run:

python -m scripts.ingest  

This chunks, embeds, and stores them in ChromaDB.

### 4. Run the agent

python -m main  

Example session:

    You: What is context poisoning in RAG?  
    Agent: [cites chunk from my_notes.md]  
    You: Save what you just found about OpenAI to the knowledge base  
    Agent: Saved 5 chunks from help.openai.com  
    You: What did OpenAI release in early 2025?  
    Agent: [retrieves from saved KB, verifies claims, answers with inline citations]  

### 5. Evaluate

python -m scripts.evaluate  

Expected output (13 queries):

| Metric | Result |
|--------|--------|
| Tool routing accuracy | 100% |
| Content hit rate | 92% |
| Overall pass rate | 92% |

---

## Project structure (what you need to know)
```
knowledge-agent/  
├── main.py                 # ReAct loop, tool dispatch  
├── scripts/  
│   ├── ingest.py           # Chunk + embed + store to ChromaDB  
│   └── evaluate.py         # Run eval suite  
├── src/knowledge_agent/  
│   ├── tools.py            # All tool schemas + executors  
│   └── config.py           # Constants (chunk size, model names, etc.)  
├── tests/  
│   └── eval_queries.json   # 13 ground-truth Q&A pairs  
├── data/  
│   └── example.md          # Sample document (excluded from git)  
├── docs/  
│   └── memory_rag_agent_casestudy.md   # Full case study  
└── pyproject.toml          # Dependencies (uv)  
```

> db/, outputs/, __pycache__/, .env are ignored.

---

## Key design choices

- **Hybrid retrieval** (BM25 + dense) catches both exact terms and semantic meaning.  
- **Cross-encoder reranking** fixes “wrong chunk” issues from bi-encoder alone.  
- **verify_claim** prevents hallucination – every claim is checked against its source.  
- **save_to_knowledge_base** closes the loop: web → KB → reusable memory.  
- **No Anthropic / OpenAI billing** – only DeepSeek API.  

---

## What’s next (Phase 3)

- Self-correction loop (LangGraph)  
- Execution feedback and automatic repair  
- Full cost tracking and latency budgets  

---

## License

MIT  