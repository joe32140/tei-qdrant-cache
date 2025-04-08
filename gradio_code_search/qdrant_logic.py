# gradio_code_search/qdrant_logic.py
import os
import shutil
import uuid
import asyncio
from asyncio import Queue
from typing import List, Optional
from loguru import logger
import concurrent
import concurrent.futures
import functools # <--- Import functools

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    logger.error("FATAL: qdrant-client not found. Please install it: pip install qdrant-client")
    exit(1)

# Import config
from config import (
    QDRANT_STORAGE_LOCATION,
    EMBEDDING_DIMENSION,
    UPSERT_BATCH_SIZE
)

# --- Qdrant Client Initialization ---
qdrant_client: Optional[QdrantClient] = None
qdrant_storage_info: str = ":memory:"

def initialize_qdrant_client():
    """Initializes the global Qdrant client."""
    global qdrant_client, qdrant_storage_info

    if QDRANT_STORAGE_LOCATION == ":memory:":
        logger.info("Initializing Qdrant client in :memory: mode.")
        qdrant_client = QdrantClient(":memory:")
        qdrant_storage_info = ":memory:"
    else:
        # Clean up existing directory if it's not :memory:
        # Be cautious with this in production - ensure it's intended behavior
        if os.path.exists(QDRANT_STORAGE_LOCATION):
            logger.warning(f"Removing existing Qdrant storage at {QDRANT_STORAGE_LOCATION}")
            try:
                shutil.rmtree(QDRANT_STORAGE_LOCATION)
            except OSError as e:
                logger.error(f"Error removing existing Qdrant directory: {e}. Proceeding might use old data or fail.")
        
        try:
            os.makedirs(QDRANT_STORAGE_LOCATION, exist_ok=True)
            qdrant_storage_info = os.path.abspath(QDRANT_STORAGE_LOCATION)
            logger.info(f"Initializing Qdrant client with storage path: {qdrant_storage_info}")
            qdrant_client = QdrantClient(path=qdrant_storage_info)
        except Exception as e:
             logger.error(f"FATAL: Failed to initialize Qdrant client at path {QDRANT_STORAGE_LOCATION}: {e}")
             exit(1)

    # Perform a quick health check
    try:
         # Qdrant client doesn't have a direct health check, listing collections is a simple way
         qdrant_client.get_collections()
         logger.info("Qdrant client initialized and connection verified.")
    except Exception as e:
         logger.error(f"FATAL: Qdrant client initialized but failed connection check: {e}")
         exit(1)

# Call initialization when the module is loaded
initialize_qdrant_client()

# --- Qdrant Functions ---

def setup_qdrant_collection(collection_name: str):
    """Creates or confirms existence of the Qdrant collection."""
    if not qdrant_client:
        logger.error("Qdrant client not initialized. Cannot setup collection.")
        raise RuntimeError("Qdrant client not available.")
    try:
        qdrant_client.get_collection(collection_name=collection_name)
        logger.info(f"Using existing Qdrant collection: {collection_name}")
    except (UnexpectedResponse, ValueError) as e:
        # Handle Qdrant client's ValueError for not found and potential 404 via UnexpectedResponse
        is_not_found_error = False
        if isinstance(e, ValueError) and "not found" in str(e).lower():
             is_not_found_error = True
        if isinstance(e, UnexpectedResponse) and e.status_code == 404:
             is_not_found_error = True
             
        if is_not_found_error:
            logger.info(f"Creating Qdrant collection: {collection_name} with dim {EMBEDDING_DIMENSION}")
            try:
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
                    # Consider adding optimizations if needed later:
                    # optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
                    # hnsw_config=models.HnswConfigDiff(on_disk=True, m=16, ef_construct=100)
                )
                logger.info(f"Collection '{collection_name}' created successfully.")
            except Exception as create_e:
                 logger.error(f"Failed to create Qdrant collection '{collection_name}': {create_e}", exc_info=True)
                 raise # Re-raise creation error
        else:
            logger.error(f"Unexpected error checking/creating Qdrant collection '{collection_name}': {e}", exc_info=True)
            raise # Re-raise other unexpected errors

async def upsert_points_queue(
    points_queue: Queue,
    collection_name: str,
    executor: concurrent.futures.Executor
):

    """
    Consumer task: Reads points from a queue and upserts them into Qdrant in batches
    using a thread pool executor to avoid blocking the event loop.
    """
    if not qdrant_client:
        logger.error("Qdrant client not initialized. Upsert consumer cannot start.")
        return

    points_buffer: List[PointStruct] = []
    processed_count = 0
    loop = asyncio.get_running_loop()

    logger.info(f"Upsert consumer started for collection '{collection_name}'. Batch size: {UPSERT_BATCH_SIZE}")

    while True:
        try:
            points_batch = await asyncio.wait_for(points_queue.get(), timeout=1.0)
            points_buffer.extend(points_batch)
            points_queue.task_done()

            if len(points_buffer) >= UPSERT_BATCH_SIZE:
                # --- Execute Upsert in Thread Pool ---
                current_batch_to_upsert = points_buffer[:] # Create a copy for the thread
                points_buffer = [] # Clear buffer immediately

                logger.debug(f"Submitting upsert batch of {len(current_batch_to_upsert)} points to executor...")

                # Use run_in_executor for the potentially blocking client call
                upsert_func = functools.partial(
                    qdrant_client.upsert,
                    collection_name=collection_name,
                    points=current_batch_to_upsert,
                    wait=False # Pass wait=False to upsert via partial
                )
                await loop.run_in_executor(executor, upsert_func)
                # Note: wait=False is passed implicitly if it's the default or
                # explicitly like: qdrant_client.upsert(collection_name=collection_name, points=current_batch_to_upsert, wait=False)

                processed_count += len(current_batch_to_upsert)
                logger.debug(f"Upsert task submitted. Total points processed count (client-side): {processed_count}")
                # The actual upsert happens in the background thread now.
                # --- End Execution in Thread Pool ---

        except asyncio.TimeoutError:
            continue

        except asyncio.CancelledError:
             logger.info("Upsert consumer cancellation requested.")
             # Handle final batch - also needs executor if large
             if points_buffer:
                 logger.info(f"Submitting final upsert batch of {len(points_buffer)} points to executor...")
                 try:
                     # Wait for the final upsert in the executor
                    final_upsert_func = functools.partial(
                        qdrant_client.upsert,
                        collection_name=collection_name,
                        points=points_buffer,
                        wait=True # Pass wait=True to upsert via partial
                    )
                    # Run the partial function in the executor
                    await loop.run_in_executor(
                        executor,
                        final_upsert_func # Pass the callable partial object
                    )

                    logger.info("Final upsert batch completed.")
                 except Exception as final_e:
                     logger.error(f"Error during final upsert in executor: {final_e}")
             logger.info(f"Upsert consumer finished. Total points processed count (client-side): {processed_count}")
             raise

        except Exception as e:
            logger.error(f"Error in upsert consumer main loop: {e}", exc_info=True)
            points_buffer = [] # Clear buffer on error


def search_qdrant(collection_name: str, query_vector: List[float], top_k: int) -> List[models.ScoredPoint]:
    """Performs a search query against the Qdrant collection."""
    if not qdrant_client:
        logger.error("Qdrant client not initialized. Cannot perform search.")
        raise RuntimeError("Qdrant client not available.")
    if not query_vector:
        logger.error("Search called with empty query vector.")
        return []

    try:
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=int(top_k),
            with_payload=True # Ensure payload is retrieved
        )
        logger.info(f"Qdrant search returned {len(search_result)} results.")
        return search_result
    except (UnexpectedResponse, ValueError) as e:
        # Check if it's a "collection not found" error
        is_not_found_error = False
        if isinstance(e, ValueError) and "not found" in str(e).lower():
             is_not_found_error = True
        if isinstance(e, UnexpectedResponse) and e.status_code == 404:
             is_not_found_error = True

        if is_not_found_error:
             logger.error(f"Search failed: Collection '{collection_name}' not found.")
             # Re-raise a specific error or return empty list? Re-raising is clearer.
             raise ValueError(f"Collection '{collection_name}' not found.") from e
        else:
             logger.error(f"Error searching Qdrant collection '{collection_name}': {e}", exc_info=True)
             raise # Re-raise other search errors
    except Exception as e:
        logger.error(f"Unexpected error during Qdrant search: {e}", exc_info=True)
        raise

def get_collection_point_count(collection_name: str) -> Optional[int]:
    """Gets the number of points in a collection. Returns None on error."""
    if not qdrant_client: return None
    try:
        collection_info = qdrant_client.get_collection(collection_name=collection_name)
        return collection_info.points_count
    except Exception as e:
        logger.error(f"Error getting point count for collection '{collection_name}': {e}")
        return None