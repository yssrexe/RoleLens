import os
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import quote

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DB_CONNECTION = (
    f"postgresql+psycopg://{quote(os.getenv('DB_USER', 'rolelens'), safe='')}:"
    f"{quote(os.getenv('DB_PASSWORD', ''), safe='')}@"
    f"{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/"
    f"{quote(os.getenv('DB_NAME', 'rolelens'), safe='')}"
)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL    = "llama3.2:latest"
TOP_K_RETRIEVE  = 4   # how many pgvector returns
TOP_K_RERANK    = 2
