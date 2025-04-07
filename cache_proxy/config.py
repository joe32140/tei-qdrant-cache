# cache_proxy/config.py
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal

# Load .env file from the parent directory relative to this script
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

class Settings(BaseSettings):
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", 6333))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "text_embedding_cache")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", 0)) # Require explicit setting
    nginx_upstream_url: str = os.getenv("NGINX_UPSTREAM_URL", "http://localhost:8081") # Default if running proxy standalone
    cache_hash_function: Literal['sha256', 'md5'] = os.getenv("CACHE_HASH_FUNCTION", "sha256")

    # Tell Pydantic to ignore extra fields found in the environment
    model_config = {
        "extra": "ignore"
    }

settings = Settings()

# Explicitly check required settings AFTER loading
if settings.embedding_dimension <= 0:
    raise ValueError("EMBEDDING_DIMENSION must be set in the .env file and be > 0.")
if not settings.nginx_upstream_url:
     raise ValueError("NGINX_UPSTREAM_URL must be set in the .env file.")
if settings.cache_hash_function not in ['sha256', 'md5']:
    raise ValueError("CACHE_HASH_FUNCTION must be 'sha256' or 'md5'.")