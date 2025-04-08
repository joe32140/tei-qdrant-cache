# cache_proxy/main.py (Updated with TEI Batching)
import logging
import hashlib
import time
import orjson # Faster JSON
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse
import httpx
from contextlib import asynccontextmanager
from typing import List
import uuid # <--- IMPORT uuid library
import asyncio # For potential sleep between batches

# Use absolute imports
import schemas
import qdrant_utils
import config

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Define batch size for requests sent TO the TEI endpoint
# Adjust this based on model token limits and average chunk size
# Start low (e.g., 8 or 16) and increase if needed.
TEI_REQUEST_BATCH_SIZE = 32 # Example value, tune this!

# --- Hashing Function ---
def get_text_hash(text: str) -> uuid.UUID:
    """Calculates SHA256 hash and returns it as a UUID object.
    
    # A UUID is 128 bits (16 bytes) which requires 32 hex characters to represent.
    # The hash_hex string contains more characters than needed for a UUID, so we only
    # take the first 32 hex chars (16 bytes) to create a valid UUID object.
    # The remaining hash characters are truncated since UUIDs have a fixed size.
    """
    hasher = hashlib.new(config.settings.cache_hash_function)
    hasher.update(text.encode('utf-8'))
    hash_hex = hasher.hexdigest()
    # Create a UUID from the first 32 hex characters (16 bytes) of the hash
    return uuid.UUID(hex=hash_hex[:32])

# --- FastAPI Lifespan for Qdrant Init/Cleanup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI app starting up...")
    try:
        qdrant_utils.get_qdrant_client() # Initialize client connection pool
        await qdrant_utils.ensure_collection_exists() # Create collection if needed
        logger.info("Qdrant connection initialized and collection checked.")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant during startup: {e}", exc_info=True)
        # raise # Optional: prevent startup if Qdrant is essential
    yield
    # Clean up resources on shutdown
    logger.info("FastAPI app shutting down...")
    await qdrant_utils.close_qdrant_client() # Close the async client


app = FastAPI(lifespan=lifespan, default_response_class=ORJSONResponse)
# Use a persistent httpx client for connection pooling
http_client = httpx.AsyncClient(timeout=60.0) # Adjust timeout as needed

@app.post("/embed", response_model=List[List[float]])
async def create_embeddings(request: schemas.EmbedRequest):
    start_time = time.monotonic()
    original_inputs = [request.inputs] if isinstance(request.inputs, str) else request.inputs
    num_inputs = len(original_inputs)
    logger.info(f"Received request to embed {num_inputs} texts.")

    if not original_inputs:
        return []

    # 1. Calculate Hashes (returns UUIDs)
    input_hashes: List[uuid.UUID] = [get_text_hash(text) for text in original_inputs]
    logger.debug(f"Calculated UUIDs: {input_hashes}")

    # 2. Check Cache (Qdrant)
    cache_check_start = time.monotonic()
    cached_embeddings_map = await qdrant_utils.retrieve_embeddings(input_hashes)
    cache_check_duration = time.monotonic() - cache_check_start
    logger.debug(f"Qdrant cache check took {cache_check_duration:.4f}s. Found {len(cached_embeddings_map)} items.")

    # 3. Identify Hits and Misses
    final_embeddings = [None] * num_inputs
    missed_indices = [] # Store original indices of missed items
    missed_texts = []   # Store text of missed items
    missed_hashes: List[uuid.UUID] = [] # Store UUID of missed items

    for i, text_hash_uuid in enumerate(input_hashes):
        if text_hash_uuid in cached_embeddings_map:
            final_embeddings[i] = cached_embeddings_map[text_hash_uuid]
        else:
            missed_indices.append(i) # Track original index
            missed_texts.append(original_inputs[i])
            missed_hashes.append(text_hash_uuid)

    cache_hits = num_inputs - len(missed_texts)
    logger.info(f"Cache Hits: {cache_hits}, Cache Misses: {len(missed_texts)}")

    # 4. Handle Misses (Call Nginx -> TEI) - WITH BATCHING
    if missed_texts:
        logger.info(f"Processing {len(missed_texts)} cache misses in batches of {TEI_REQUEST_BATCH_SIZE}...")
        all_batches_successful = True # Flag to track if any batch failed

        # Iterate through missed items in smaller batches
        for i in range(0, len(missed_texts), TEI_REQUEST_BATCH_SIZE):
            # Slice the data for the current batch
            batch_texts = missed_texts[i : i + TEI_REQUEST_BATCH_SIZE]
            batch_original_indices = missed_indices[i : i + TEI_REQUEST_BATCH_SIZE]
            batch_hashes_to_store = missed_hashes[i : i + TEI_REQUEST_BATCH_SIZE]

            if not batch_texts: continue # Should not happen, but safe check

            inference_start = time.monotonic()
            # Prepare payload for this specific batch
            tei_payload = {
                "inputs": batch_texts,
                "normalize": request.normalize, # Pass through relevant parameters
                "prompt_name": request.prompt_name,
                "prompt": request.prompt,
            }
            tei_payload = {k: v for k, v in tei_payload.items() if v is not None}

            logger.debug(f"Forwarding batch of {len(batch_texts)} texts to TEI (Indices: {batch_original_indices})")
            try:
                # Send the smaller batch to the TEI endpoint
                response = await http_client.post(config.settings.nginx_upstream_url + "/embed", json=tei_payload)
                response.raise_for_status() # Raise exception for 4xx/5xx errors
                tei_result = response.json()

                # Validate and extract embeddings from the response
                if isinstance(tei_result, list):
                     batch_new_embeddings = tei_result
                elif isinstance(tei_result, dict) and 'embeddings' in tei_result:
                     batch_new_embeddings = tei_result['embeddings']
                else:
                     logger.error(f"Unexpected response structure from TEI for batch: {tei_result}")
                     all_batches_successful = False
                     continue # Skip storing for this failed batch

                inference_duration = time.monotonic() - inference_start
                logger.info(f"TEI inference took {inference_duration:.4f}s for batch of {len(batch_texts)} texts.")

                # Verify the number of embeddings received matches the number sent
                if len(batch_new_embeddings) != len(batch_texts):
                    logger.error(f"Mismatch in batch: requested {len(batch_texts)}, received {len(batch_new_embeddings)}.")
                    all_batches_successful = False
                    continue # Skip storing for this failed batch

                # 5. Populate Cache & Combine Results for this batch
                # Store embeddings for this successful batch in Qdrant
                await qdrant_utils.store_embeddings(batch_texts, batch_new_embeddings, batch_hashes_to_store)

                # Place the received embeddings into the correct positions in the final list
                for j, original_index in enumerate(batch_original_indices):
                    final_embeddings[original_index] = batch_new_embeddings[j]

            except httpx.RequestError as e:
                logger.error(f"HTTP request error for TEI batch (Indices: {batch_original_indices}): {e}", exc_info=True)
                all_batches_successful = False
                # Decide how to handle failed batches - here we just log and continue
                # You might want to mark corresponding final_embeddings as None or raise an error later
            except httpx.HTTPStatusError as e:
                 logger.error(f"TEI service error {e.response.status_code} for batch (Indices: {batch_original_indices}): {e.response.text[:200]}")
                 all_batches_successful = False
            except Exception as e:
                 logger.error(f"Error processing TEI response for batch (Indices: {batch_original_indices}): {e}", exc_info=True)
                 all_batches_successful = False

            # Optional small delay between batches if hitting rate limits or for smoother load
            # await asyncio.sleep(0.05)

        # After processing all batches, check if any failed
        if not all_batches_successful:
            logger.warning("One or more batches failed during TEI processing. Final results might be incomplete.")
            # You could raise an HTTPException here if partial results are unacceptable
            # raise HTTPException(status_code=502, detail="Failed to process all embedding batches.")


    # 6. Return Combined Results
    total_duration = time.monotonic() - start_time
    logger.info(f"Total request processing time: {total_duration:.4f}s")

    # Final check: ensure all slots are filled (unless batches failed and we allowed partial results)
    if None in final_embeddings and all_batches_successful: # Check only if all batches were expected to succeed
         logger.error("Failed to populate all final embeddings despite successful batches. This indicates a logic error.")
         # Only raise 500 if we expected completion but didn't get it
         raise HTTPException(status_code=500, detail="Internal error: Failed to assemble all embeddings")
    elif None in final_embeddings and not all_batches_successful:
         logger.warning("Returning potentially incomplete results due to batch processing errors.")
         # Return partial results (items that failed will be None)
         # Or filter out None values if the client expects only successful results
         # return [emb for emb in final_embeddings if emb is not None] # Option: Return only non-None

    # If returning partial results is okay, just return the list which might contain None
    # If partial results are NOT okay, the exception should have been raised earlier.
    # Assuming partial results are okay for now if batches fail:
    # Replace None with a default vector or handle appropriately if needed before returning
    # For now, just return the list possibly containing None
    # A robust implementation might replace None with zeros or raise a specific error code.
    # Let's filter out None for now, assuming client wants only successful embeddings.
    successful_embeddings = [emb for emb in final_embeddings if emb is not None]
    if len(successful_embeddings) != num_inputs and all_batches_successful:
         # This case should ideally not happen if all_batches_successful is true
         logger.error("Logic error: All batches reported success, but final embeddings are incomplete.")
         raise HTTPException(status_code=500, detail="Internal error assembling embeddings.")
    elif len(successful_embeddings) != num_inputs and not all_batches_successful:
         logger.warning(f"Returning {len(successful_embeddings)} embeddings out of {num_inputs} requested due to errors.")
         # Decide on API contract: return partial list or error? Returning partial list for now.
         return successful_embeddings # Return only the ones that succeeded

    return final_embeddings # Return the full list if all succeeded


@app.get("/health")
async def health_check():
    # Could add an async Qdrant health check here if needed
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Text Embedding Cache Proxy running (Async Qdrant)"}