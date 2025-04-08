# gradio_code_search/indexing.py
import asyncio
import time
import tempfile
import shutil
import os
import uuid
import concurrent.futures
from asyncio import Queue, Semaphore
from typing import List, Tuple, Dict, Optional
from loguru import logger
import concurrent
try:
    import git
    from git.exc import GitCommandError
except ImportError:
    logger.warning("GitPython not found. GitHub URL cloning disabled. `pip install GitPython`")
    git = None

# Import from other modules
from config import (
    COLLECTION_NAME_PREFIX,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CONCURRENCY_LIMIT,
    FILE_PROCESSING_WORKERS,
    CHUNK_SIZE # Needed for process_file if chunk_file isn't imported directly
)
from utils import (
    is_github_url,
    get_code_files,
    chunk_file, # Import the blocking chunk_file function
    log_memory_usage
)
from embedding_client import get_embeddings
from qdrant_logic import (
    setup_qdrant_collection,
    upsert_points_queue,
    get_collection_point_count,
    PointStruct # Import PointStruct type
)

# --- Helper for Parallel File Processing ---

async def process_file_task(
    file_path: str,
    repo_base_path: str,
    embedding_queue: Queue,
    executor: concurrent.futures.Executor
):
    """
    Task to read, chunk a single file in a thread pool, and put chunks into the queue.
    Returns the number of chunks generated for the file.
    """
    loop = asyncio.get_running_loop()
    try:
        # Run blocking I/O (open, read) and CPU-bound (chunking) code in executor
        file_chunks: List[Tuple[str, int, int]] = await loop.run_in_executor(
            executor, chunk_file, file_path # Pass chunk_file and its args
        )

        if not file_chunks:
            return 0 # No chunks generated (empty file or error during chunking)

        relative_path = os.path.relpath(file_path, repo_base_path)
        chunks_added_to_queue = 0
        for chunk_text, start_line, end_line in file_chunks:
            metadata = {
                "file_path": relative_path,
                "start_line": start_line,
                "end_line": end_line,
                "text": chunk_text # Include text in metadata for Qdrant payload
            }
            # Put the tuple (text_for_embedding, metadata_payload) into the queue
            await embedding_queue.put((chunk_text, metadata))
            chunks_added_to_queue += 1

        return chunks_added_to_queue
    except FileNotFoundError:
        logger.warning(f"File disappeared before processing: {file_path}")
        return 0
    except Exception as e:
        logger.error(f"Error processing file {file_path} in task: {e}", exc_info=True)
        return 0

# --- Embedding Consumer ---

async def process_embeddings_queue(
    embedding_queue: Queue,
    points_queue: Queue,
    semaphore: Semaphore
):
    """
    Consumer task: Reads chunks from queue, gets embeddings in batches, puts points into points_queue.
    Uses a semaphore to limit concurrent calls to the embedding service.
    """
    batch_texts: List[str] = []
    batch_metadata: List[Dict] = []
    processed_chunks_count = 0
    logger.info(f"Embedding consumer started. Batch size: {EMBEDDING_BATCH_SIZE}, Concurrency: {semaphore._value}")

    while True:
        try:
            # Wait for an item with a timeout
            chunk_text, metadata = await asyncio.wait_for(embedding_queue.get(), timeout=1.0)
            batch_texts.append(chunk_text)
            batch_metadata.append(metadata)
            embedding_queue.task_done()

            # Process batch if full
            if len(batch_texts) >= EMBEDDING_BATCH_SIZE:
                async with semaphore: # Acquire semaphore before calling embedding service
                    logger.debug(f"Acquired semaphore. Getting embeddings for batch of {len(batch_texts)}...")
                    embeddings = await get_embeddings(batch_texts) # Call the robust get_embeddings

                if embeddings:
                    points = [
                        PointStruct(id=str(uuid.uuid4()), vector=emb, payload=meta)
                        for emb, meta in zip(embeddings, batch_metadata) if emb # Ensure embedding is not None
                    ]
                    if points:
                        await points_queue.put(points)
                        processed_chunks_count += len(points)
                        logger.debug(f"Put {len(points)} points into upsert queue. Total processed: {processed_chunks_count}")
                    else:
                         logger.warning(f"Embedding service returned valid list but potentially with None values for batch starting with: {batch_texts[0][:50]}...")

                else:
                    # Handle failure to get embeddings for the batch (already logged in get_embeddings)
                    logger.error(f"Failed to get embeddings for batch starting with: {batch_texts[0][:50]}... Skipping batch.")
                    # Optionally implement dead-letter queue or other error handling

                # Clear the batch regardless of success/failure
                batch_texts = []
                batch_metadata = []

        except asyncio.TimeoutError:
            # Queue was empty, check if we should process remaining items and/or exit
            # Similar logic needed as in upsert consumer regarding exit signals
            # For now, just continue waiting
            continue

        except asyncio.CancelledError:
            logger.info("Embedding consumer cancellation requested.")
            # Process any remaining items in the current batch before exiting
            if batch_texts:
                logger.info(f"Processing final batch of {len(batch_texts)} chunks...")
                try:
                    async with semaphore:
                         embeddings = await get_embeddings(batch_texts)
                    if embeddings:
                        points = [
                            PointStruct(id=str(uuid.uuid4()), vector=emb, payload=meta)
                            for emb, meta in zip(embeddings, batch_metadata) if emb
                        ]
                        if points:
                            await points_queue.put(points)
                            processed_chunks_count += len(points)
                            logger.info(f"Put final {len(points)} points into upsert queue.")
                except Exception as final_e:
                    logger.error(f"Error processing final embedding batch: {final_e}")
            logger.info(f"Embedding consumer finished. Total chunks processed into points: {processed_chunks_count}")
            raise # Re-raise CancelledError

        except Exception as e:
            logger.error(f"Error in embedding consumer: {e}", exc_info=True)
            # Clear current batch on error and continue
            batch_texts = []
            batch_metadata = []


# --- Main Indexing Function ---

async def index_repository(source: str, source_type: str) -> Tuple[str, Optional[str], str, str, str, str]:
    """
    Orchestrates the indexing process: clone/access repo, parallel file processing,
    embedding pipeline, and Qdrant upserting.
    Returns: (status_message, collection_name, total_time_str, files_found_str, chunks_gen_str, chunks_idx_str)
    """
    start_time = time.monotonic()
    repo_path: Optional[str] = None
    temp_dir: Optional[str] = None
    collection_name: Optional[str] = None
    task_exception: Optional[Exception] = None # To store exceptions from tasks

    # Statistics
    num_files_found = 0
    total_chunks_generated = 0 # Chunks created by file processing
    # Final indexed count will be read from Qdrant

    # Default return values on failure
    status_message = "[Error] Indexing did not complete."
    final_collection_name = None
    total_time_str = "N/A"
    files_found_str = "0"
    chunks_gen_str = "0"
    chunks_idx_str = "0"

    log_memory_usage("Start Indexing")

    try:
        # 1. Determine Repository Path and Collection Name
        if source_type == "GitHub URL":
            if not git:
                status_message = "[Error] GitPython not installed. Cannot clone repository. `pip install GitPython`"
                return status_message, None, "0.00s", "0", "0", "0"

            cleaned_source = source.strip()
            if "/tree/" in cleaned_source: # Handle URLs pointing to specific branches/folders
                cleaned_source = cleaned_source.split('/tree/')[0]
            if not cleaned_source.endswith('.git'):
                cleaned_source = cleaned_source.rstrip('/') + '.git'

            if not is_github_url(cleaned_source.replace('.git', '')):
                status_message = f"[Error] Invalid GitHub URL: {source}"
                return status_message, None, "0.00s", "0", "0", "0"

            repo_name = cleaned_source.split('/')[-1].replace('.git', '')
            collection_name = f"{COLLECTION_NAME_PREFIX}{repo_name}_{int(time.time())}"
            temp_dir = tempfile.mkdtemp(prefix="git_clone_")

            logger.info(f"Attempting to clone {cleaned_source} into {temp_dir}...")
            try:
                git.Repo.clone_from(cleaned_source, temp_dir, depth=1) # Shallow clone
                repo_path = temp_dir
                logger.info(f"Cloning '{repo_name}' complete.")
            except GitCommandError as e:
                logger.error(f"Git clone failed: {e}")
                status_message = f"[Error] Failed to clone repository: {e}"
                # Clean up temp dir even on clone failure
                if temp_dir and os.path.exists(temp_dir): shutil.rmtree(temp_dir, ignore_errors=True)
                return status_message, None, "0.00s", "0", "0", "0"
            except Exception as e:
                logger.error(f"Unexpected error during cloning: {e}", exc_info=True)
                status_message = f"[Error] Cloning failed unexpectedly: {e}"
                if temp_dir and os.path.exists(temp_dir): shutil.rmtree(temp_dir, ignore_errors=True)
                return status_message, None, "0.00s", "0", "0", "0"

        elif source_type == "Local Directory":
            if not os.path.isdir(source):
                status_message = f"[Error] Invalid directory path: {source}"
                return status_message, None, "0.00s", "0", "0", "0"
            repo_path = source
            dir_name = os.path.basename(os.path.normpath(source))
            collection_name = f"{COLLECTION_NAME_PREFIX}{dir_name}_{int(time.time())}"
        else:
            status_message = "[Error] Invalid source type selected."
            return status_message, None, "0.00s", "0", "0", "0"

        if not repo_path or not collection_name: # Safety check
             raise ValueError("Repository path or collection name could not be determined.")

        # 2. Setup Qdrant Collection
        logger.info(f"Preparing Qdrant collection: {collection_name}")
        setup_qdrant_collection(collection_name)
        final_collection_name = collection_name # Store the name for return value

        # 3. Find Code Files
        code_files = get_code_files(repo_path)
        num_files_found = len(code_files)
        files_found_str = str(num_files_found)
        if not code_files:
            status_message = f"[Warning] No indexable code files found in {source}."
            total_time = time.monotonic() - start_time
            return status_message, collection_name, f"{total_time:.2f}s", files_found_str, "0", "0"

        # 4. Setup Async Pipeline Components
        # Queues with buffer sizes (adjust if memory becomes an issue)
        embedding_queue = Queue(maxsize=2000) # Queue for (chunk_text, metadata)
        points_queue = Queue(maxsize=1000) # Queue for List[PointStruct]

        # Semaphore to limit concurrent embedding requests
        embedding_semaphore = Semaphore(EMBEDDING_CONCURRENCY_LIMIT)

        # Thread pool executor for parallel file processing
        # Using 'with' ensures shutdown
        with concurrent.futures.ThreadPoolExecutor(max_workers=FILE_PROCESSING_WORKERS) as executor:

            # Start consumer tasks
            embedding_consumer_task = asyncio.create_task(
                process_embeddings_queue(embedding_queue, points_queue, embedding_semaphore),
                name="EmbeddingConsumer"
            )
            upsert_consumer_task = asyncio.create_task(
                upsert_points_queue(points_queue, collection_name, executor),
                name="UpsertConsumer"
            )
            consumer_tasks = [embedding_consumer_task, upsert_consumer_task]

            # 5. Start File Processing Tasks (Producer)
            logger.info(f"Starting parallel processing of {num_files_found} files using {FILE_PROCESSING_WORKERS} workers...")
            file_processing_tasks = [
                process_file_task(f_path, repo_path, embedding_queue, executor)
                for f_path in code_files
            ]

            processed_files_count = 0
            # Process results as they complete
            for future in asyncio.as_completed(file_processing_tasks):
                try:
                    chunks_generated_for_file = await future
                    total_chunks_generated += chunks_generated_for_file
                    processed_files_count += 1
                    if processed_files_count % 100 == 0 or processed_files_count == num_files_found:
                        logger.info(f"Files processed: {processed_files_count}/{num_files_found}. Chunks generated so far: {total_chunks_generated}. Queue size: ~{embedding_queue.qsize()}")
                        log_memory_usage(f"After {processed_files_count} files")
                except Exception as e:
                    logger.error(f"Error retrieving result from file processing task: {e}", exc_info=True)
                    # Continue processing other files

            logger.info(f"All {processed_files_count} file processing tasks submitted. Total chunks generated: {total_chunks_generated}.")
            chunks_gen_str = str(total_chunks_generated)

            # 6. Wait for Queues to be Emptied by Consumers
            logger.info("Waiting for embedding queue to be processed...")
            await embedding_queue.join() # Wait until all items are gotten and marked done
            logger.info("Embedding queue processed. Waiting for points queue...")
            await points_queue.join() # Wait until all points are gotten and marked done
            logger.info("Points queue processed.")

            # 7. Signal Consumers to Finish and Wait for them
            logger.info("Signaling consumer tasks to finish...")
            for task in consumer_tasks:
                task.cancel()

            # Wait for tasks to finish (and handle potential exceptions)
            results = await asyncio.gather(*consumer_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                 task_name = consumer_tasks[i].get_name()
                 if isinstance(result, asyncio.CancelledError):
                     logger.info(f"{task_name} successfully cancelled.")
                 elif isinstance(result, Exception):
                     logger.error(f"{task_name} raised an exception: {result}", exc_info=result)
                     task_exception = result # Store the first exception encountered
                 else:
                      logger.info(f"{task_name} finished.") # Should ideally be cancelled

            # If a consumer task failed, report it
            if task_exception:
                 raise task_exception # Propagate the error to the main try-except block

        # 8. Final Reporting
        end_time = time.monotonic()
        total_time = end_time - start_time
        total_time_str = f"{total_time:.2f}s"

        # Get final count from Qdrant
        final_point_count = get_collection_point_count(collection_name)
        if final_point_count is not None:
             chunks_idx_str = str(final_point_count)
             logger.info(f"Verified {final_point_count} points in Qdrant collection '{collection_name}'.")
        else:
             chunks_idx_str = "Unknown (Error fetching count)"
             logger.warning(f"Could not verify final point count in Qdrant collection '{collection_name}'.")


        status_message = (
            f"Indexing complete for '{source}'.\n"
            f"Collection: '{collection_name}'.\n"
            f"Total time: {total_time_str}."
        )
        logger.info(status_message)
        log_memory_usage("End Indexing")

        return status_message, final_collection_name, total_time_str, files_found_str, chunks_gen_str, chunks_idx_str

    except Exception as e:
        logger.error(f"Error during indexing process: {e}", exc_info=True)
        status_message = f"[Error] Indexing failed: {e}"
        # Ensure stats reflect what was processed before the error
        files_found_str = str(num_files_found) if num_files_found > 0 else "N/A"
        chunks_gen_str = str(total_chunks_generated) if total_chunks_generated > 0 else "N/A"
        # Indexed chunks might be partially complete or unknown
        final_point_count = get_collection_point_count(final_collection_name) if final_collection_name else None
        chunks_idx_str = str(final_point_count) if final_point_count is not None else "Error/Partial"

        total_time = time.monotonic() - start_time
        total_time_str = f"{total_time:.2f}s (Failed)"

        return status_message, final_collection_name, total_time_str, files_found_str, chunks_gen_str, chunks_idx_str

    finally:
        # Cleanup temporary directory if it was created
        if temp_dir and os.path.exists(temp_dir):
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            try:
                shutil.rmtree(temp_dir, ignore_errors=True) # Use ignore_errors for robustness
            except Exception as e:
                logger.error(f"Error removing temporary directory {temp_dir}: {e}")
        log_memory_usage("After Cleanup")