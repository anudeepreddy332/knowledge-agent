import os

from dotenv import load_dotenv
from pathlib import Path


load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"
REPORTS_DIR = BASE_DIR / "reports"
STATE_FILE = BASE_DIR / "state.json"
TESTS_DIR = BASE_DIR / "tests"
EVAL_PATH = TESTS_DIR / "eval_queries.json"

# LLM
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 5
TOP_K_CANDIDATES = 20
BM25_WEIGHT = 60    # k constant in RRF formula — standard default, higher = less steep rank penalty

# Cost tracking (deepseek's latest pricing)
COST_PER_1M_INPUT_TOKENS = 0.27
COST_PER_1M_OUTPUT_TOKENS = 1.10
MAX_COST_PER_RUN = 0.10