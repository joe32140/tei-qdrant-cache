# gradio_code_search/embedding_client.py
import httpx
import asyncio
from typing import List, Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import config
from config import EMBEDDING_ENDPOINT_URL, EMBEDDING_REQUEST_TIMEOUT

# Define specific retryable HTTP errors
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

def _should_retry_embedding_request(exception: BaseException) -> bool:
    """Determine if we should retry based on the exception."""
    if isinstance(exception, httpx.TimeoutException):
        logger.warning("Embedding request timed out, retrying...")
        return True
    if isinstance(exception, httpx.RequestError):
        # Don't retry on client-side errors like invalid URL immediately
        # unless it's a ConnectError or ReadError which might be transient
        if isinstance(exception, (httpx.ConnectError, httpx.ReadError, httpx.PoolTimeout)):
             logger.warning(f"Embedding request connection/read error, retrying: {exception}")
             return True
        logger.error(f"Non-retryable embedding request error: {exception}")
        return False
    if isinstance(exception, httpx.HTTPStatusError):
        if exception.response.status_code in RETRYABLE_STATUS_CODES:
            logger.warning(f"Embedding service returned {exception.response.status_code}, retrying...")
            return True
        else:
            logger.error(f"Embedding service returned non-retryable status {exception.response.status_code}")
            return False
    # Don't retry other unexpected errors by default
    logger.error(f"Unexpected error during embedding request: {exception}")
    return False


# Apply retry decorator
@retry(
    stop=stop_after_attempt(3), # Retry up to 3 times
    wait=wait_exponential(multiplier=1, min=2, max=10), # Wait 2s, 4s, 8s... up to 10s
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError, httpx.TimeoutException))
    # Use the custom function for more fine-grained control if needed:
    # retry=_should_retry_embedding_request
)
async def get_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    """
    Calls the TEI endpoint to get embeddings for a batch of texts with retries.
    Returns None if embeddings cannot be obtained after retries.
    """
    if not texts:
        return []

    payload = {
        "inputs": texts,
        # "normalize": True, # Assuming normalization is desired
        "prompt_name": None,
        "prompt": None
    }
    try:
        async with httpx.AsyncClient(timeout=EMBEDDING_REQUEST_TIMEOUT) as client:
            response = await client.post(EMBEDDING_ENDPOINT_URL, json=payload)
            response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx
            result = response.json()

            # --- Response Validation ---
            if not isinstance(result, list):
                 logger.error(f"Embedding endpoint returned non-list response: {type(result)}")
                 return None
            if len(result) != len(texts):
                 logger.error(f"Embedding endpoint returned {len(result)} embeddings for {len(texts)} inputs.")
                 # Attempt to pad with None or handle based on service behavior? For now, fail.
                 return None
            if not all(isinstance(emb, list) and all(isinstance(f, float) for f in emb) for emb in result):
                 logger.error(f"Embedding endpoint returned invalid embedding format.")
                 # Log a sample of the bad data if possible (be careful with large data)
                 # logger.debug(f"Sample invalid embedding data: {result[0] if result else 'N/A'}")
                 return None
            # --- End Validation ---

            # logger.debug(f"Successfully received {len(result)} embeddings.")
            return result

    except httpx.TimeoutException as e:
        logger.error(f"Embedding request timed out after {EMBEDDING_REQUEST_TIMEOUT}s: {e}")
        raise # Re-raise TimeoutException to trigger tenacity retry
    except httpx.RequestError as e:
        logger.error(f"HTTP request error calling embedding endpoint {EMBEDDING_ENDPOINT_URL}: {e}")
        raise # Re-raise RequestError to potentially trigger retry
    except httpx.HTTPStatusError as e:
        logger.error(f"Embedding endpoint returned error {e.response.status_code}: {e.response.text[:500]}") # Log more of the error
        raise # Re-raise HTTPStatusError to potentially trigger retry
    except Exception as e: # Catch other potential errors like JSON decoding
        logger.error(f"Unexpected error getting embeddings: {e}", exc_info=True)
        return None # Return None for unexpected errors after retries fail