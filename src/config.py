"""
config.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_RAW        = BASE_DIR / "data" / "raw"
DATA_PROCESSED  = BASE_DIR / "data" / "processed"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "rufimelo/bert-large-portuguese-cased-sts"
)

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL         = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS    = 1024

TOP_K = 5
