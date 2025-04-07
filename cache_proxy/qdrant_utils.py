# cache_proxy/qdrant_utils.py (Corrected - Convert UUID to str for client)
import logging
import uuid # Keep import
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
from config import settings
from typing import List, Dict

logger = logging.getLogger(__name__)

async_client = None

def get_qdrant_client() -> AsyncQdrantClient:
    """Gets the singleton async Qdrant client instance."""
    global async_client
    if async_client is None:
        logger.info(f"Initializing Async Qdrant client for host: {settings.qdrant_host}:{settings.qdrant_port}")
        async_client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return async_client

async def ensure_collection_exists():
    """Ensures the configured Qdrant collection exists, creating it if necessary."""
    # ... (This function remains the same as the previous Async version) ...
    qdrant = get_qdrant_client()
    collection_name = settings.qdrant_collection
    try:
        await qdrant.get_collection(collection_name=collection_name)
        logger.info(f"Qdrant collection '{collection_name}' already exists.")
    except (UnexpectedResponse, ValueError) as e:
        if isinstance(e, UnexpectedResponse) and e.status_code == 404 or "not found" in str(e).lower():
            logger.warning(f"Qdrant collection '{collection_name}' not found. Creating...")
            try:
                await qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Successfully created Qdrant collection '{collection_name}'.")
            except Exception as create_e:
                logger.error(f"Failed to create Qdrant collection '{collection_name}': {create_e}", exc_info=True)
                raise
        else:
             logger.error(f"Unexpected error checking Qdrant collection '{collection_name}': {e}", exc_info=True)
             raise


# Accept list of UUIDs, return dict with UUID keys
async def retrieve_embeddings(point_ids: List[uuid.UUID]) -> Dict[uuid.UUID, List[float]]:
    """Retrieves points by ID asynchronously and returns a dict {UUID: vector}."""
    if not point_ids:
        return {}
    qdrant = get_qdrant_client()
    collection_name = settings.qdrant_collection
    try:
        # *** Convert UUIDs to strings for the client call ***
        ids_as_strings = [str(pid) for pid in point_ids]
        logger.debug(f"Retrieving points with string IDs: {ids_as_strings}")

        results = await qdrant.retrieve(
            collection_name=collection_name,
            ids=ids_as_strings, # Pass list of strings
            with_payload=False,
            with_vectors=True
        )

        # *** Convert string IDs received back to UUIDs for the return dict ***
        found_embeddings = {}
        for point in results:
            if point.vector:
                try:
                    # Qdrant returns string IDs, convert back to UUID for consistency
                    point_uuid = uuid.UUID(hex=point.id)
                    found_embeddings[point_uuid] = point.vector
                except ValueError:
                    logger.warning(f"Received non-UUID string ID from Qdrant retrieve: {point.id}. Skipping.")

        logger.debug(f"Qdrant retrieve found {len(found_embeddings)} embeddings for {len(point_ids)} requested UUIDs.")
        return found_embeddings
    except Exception as e:
        # Log the specific error, including potentially the IDs that caused it if possible
        logger.error(f"Failed to retrieve embeddings from Qdrant for IDs {point_ids}: {e}", exc_info=True)
        return {}

# Accept list of UUIDs
async def store_embeddings(texts: List[str], vectors: List[List[float]], point_ids: List[uuid.UUID]):
    """Stores embeddings in Qdrant asynchronously using UUIDs (converted to str) as IDs."""
    if not texts or not vectors or not point_ids or len(texts) != len(vectors) or len(texts) != len(point_ids):
        logger.warning("Mismatch in lengths or empty lists provided to store_embeddings. Skipping.")
        return

    qdrant = get_qdrant_client()
    collection_name = settings.qdrant_collection

    points_to_upsert = []
    for text, vector, point_id_uuid in zip(texts, vectors, point_ids):
        try:
            # *** Convert UUID to string when creating PointStruct ***
            point_struct = models.PointStruct(
                id=str(point_id_uuid), # Pass ID as string
                vector=vector,
                payload={"text": text}
            )
            points_to_upsert.append(point_struct)
        except Exception as e: # Catch potential errors during PointStruct creation
             logger.error(f"Failed to create PointStruct for ID {point_id_uuid}: {e}", exc_info=True)
             # Decide whether to skip this point or fail the batch
             continue # Skipping problematic point for now

    if not points_to_upsert:
        logger.warning("No valid points constructed for upsert after potential errors.")
        return

    try:
        logger.debug(f"Upserting {len(points_to_upsert)} points with string IDs.")
        response = await qdrant.upsert(
            collection_name=collection_name,
            points=points_to_upsert,
            wait=False
        )
        logger.debug(f"Qdrant upsert response status: {response.status}")
    except Exception as e:
        # Log the specific error
        logger.error(f"Failed to store embeddings in Qdrant: {e}", exc_info=True)


async def close_qdrant_client():
    """Closes the async Qdrant client connection."""
    global async_client
    if async_client:
        logger.info("Closing Async Qdrant client.")
        await async_client.close()
        async_client = None