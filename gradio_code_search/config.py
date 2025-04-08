# gradio_code_search/config.py
import os
from dotenv import load_dotenv
from loguru import logger # Using loguru, ensure it's installed

# --- Load Environment Variables ---
load_dotenv() # Load .env file from the current directory or parent directories

EMBEDDING_ENDPOINT_URL = os.getenv("EMBEDDING_ENDPOINT_URL")
try:
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 0))
except ValueError:
    EMBEDDING_DIMENSION = 0

QDRANT_STORAGE_LOCATION = os.getenv("QDRANT_STORAGE_LOCATION", "./qdrant_data") # Default to local dir

# --- Validate Critical Configuration ---
if not EMBEDDING_ENDPOINT_URL:
    logger.error("FATAL: EMBEDDING_ENDPOINT_URL not set in environment or .env file.")
    exit(1) # Exit if critical config is missing
if EMBEDDING_DIMENSION <= 0:
    logger.error("FATAL: EMBEDDING_DIMENSION not set or invalid in environment or .env file.")
    exit(1)

# --- Constants ---
CHUNK_SIZE = 50 # Lines per chunk
EMBEDDING_BATCH_SIZE = 64 # Batch size for getting embeddings
UPSERT_BATCH_SIZE = 500 # Batch size for upserting to Qdrant (adjust as needed)
COLLECTION_NAME_PREFIX = "code_index_"
CODE_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
    '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.md', '.sh',
    '.yaml', '.yml', '.json', '.html', '.css'
}
# Max concurrent requests to the embedding service
EMBEDDING_CONCURRENCY_LIMIT = int(os.getenv("EMBEDDING_CONCURRENCY_LIMIT", "10"))
# Max workers for parallel file processing (adjust based on CPU/IO)
FILE_PROCESSING_WORKERS = int(os.getenv("FILE_PROCESSING_WORKERS", os.cpu_count() or 4))

# Timeout for embedding requests (seconds)
EMBEDDING_REQUEST_TIMEOUT = 180.0

# --- Log Configuration ---
# Configure Loguru or standard logging here if needed
# Example Loguru setup:
# logger.add("file_{time}.log", rotation="1 week") # Log to file

logger.info("Configuration loaded:")
logger.info(f"  Embedding Endpoint: {EMBEDDING_ENDPOINT_URL}")
logger.info(f"  Embedding Dimension: {EMBEDDING_DIMENSION}")
logger.info(f"  Qdrant Storage: {QDRANT_STORAGE_LOCATION}")
logger.info(f"  Embedding Batch Size: {EMBEDDING_BATCH_SIZE}")
logger.info(f"  Upsert Batch Size: {UPSERT_BATCH_SIZE}")
logger.info(f"  Embedding Concurrency Limit: {EMBEDDING_CONCURRENCY_LIMIT}")
logger.info(f"  File Processing Workers: {FILE_PROCESSING_WORKERS}")