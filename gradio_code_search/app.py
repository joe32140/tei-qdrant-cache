# gradio_code_search/app.py (Removed gr.Progress)

import gradio as gr
import os
import time
import tempfile
import shutil
import uuid
from typing import List, Tuple, Dict, Optional
from urllib.parse import urlparse

# --- Dependencies ---
try:
    import git
    from git.exc import GitCommandError
except ImportError:
    print("GitPython not found. Please install it: pip install GitPython")
    git = None # Flag that git operations are unavailable

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    print("qdrant-client not found. Please install it: pip install qdrant-client")
    exit()

try:
    import httpx
except ImportError:
    print("httpx not found. Please install it: pip install httpx")
    exit()

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not found. Install it: pip install python-dotenv")
    exit()

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# --- Configuration ---
load_dotenv() # Load .env file from the current directory

# Get endpoint URL and dimension from environment variables
EMBEDDING_ENDPOINT_URL = os.getenv("EMBEDDING_ENDPOINT_URL")
try:
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 0))
except ValueError:
    EMBEDDING_DIMENSION = 0

if not EMBEDDING_ENDPOINT_URL:
    logger.error("EMBEDDING_ENDPOINT_URL not set in environment or .env file.")
    exit()
if EMBEDDING_DIMENSION <= 0:
    logger.error("EMBEDDING_DIMENSION not set or invalid in environment or .env file.")
    exit()
# Constants
CHUNK_SIZE = 50
BATCH_SIZE = 128 # Batch size for getting embeddings
UPSERT_BATCH_SIZE = 256 # Batch size for upserting to Qdrant
COLLECTION_NAME_PREFIX = "code_index_"
CODE_EXTENSIONS = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.md', '.sh', '.yaml', '.yml', '.json', '.html', '.css'}

# --- Qdrant client Initialization ---
QDRANT_STORAGE_LOCATION = "./qdrant_data"
if QDRANT_STORAGE_LOCATION == ":memory:":
    qdrant_client = QdrantClient(":memory:")
    qdrant_storage_info = ":memory:"
else:
    os.makedirs(QDRANT_STORAGE_LOCATION, exist_ok=True)
    qdrant_client = QdrantClient(path=QDRANT_STORAGE_LOCATION)
    qdrant_storage_info = os.path.abspath(QDRANT_STORAGE_LOCATION)

logger.info(f"Initializing Qdrant client. Data directory: {qdrant_storage_info}")

# --- Helper Functions ---
def is_github_url(url: str) -> bool:
    """Check if a string is a valid GitHub repository URL."""
    if not url: return False
    try:
        parsed = urlparse(url)
        return parsed.scheme in ('http', 'https') and parsed.netloc in ('github.com', 'www.github.com')
    except ValueError: return False

def get_code_files(repo_path: str) -> List[str]:
    """Recursively find code files in a directory."""
    code_files = []
    logger.info(f"Scanning for code files in: {repo_path}")
    for root, _, files in os.walk(repo_path):
        if '.git' in root.split(os.sep): continue
        for file in files:
            if os.path.splitext(file)[1].lower() in CODE_EXTENSIONS:
                full_path = os.path.join(root, file)
                try:
                    if os.path.getsize(full_path) > 5 * 1024 * 1024:
                        logger.warning(f"Skipping large file: {full_path}")
                        continue
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                         f.read(1024)
                    code_files.append(full_path)
                except UnicodeDecodeError: logger.warning(f"Skipping file with potential non-UTF8 content: {full_path}")
                except OSError as e: logger.warning(f"Skipping file {full_path} due to OS error: {e}")
                except Exception as e: logger.warning(f"Skipping file {full_path} due to generic read error: {e}")
    logger.info(f"Found {len(code_files)} potentially indexable code files.")
    return code_files

def chunk_file(file_path: str, chunk_size: int) -> List[Tuple[str, int, int]]:
    """Chunks a file into segments of specified line size."""
    chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: lines = f.readlines()
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            if not chunk_lines: continue
            chunk_text = "".join(chunk_lines).strip()
            if not chunk_text: continue
            start_line = i + 1
            end_line = i + len(chunk_lines)
            chunks.append((chunk_text, start_line, end_line))
    except Exception as e: logger.error(f"Error chunking file {file_path}: {e}")
    return chunks

async def get_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    """Calls the TEI endpoint to get embeddings for a batch of texts."""
    if not texts: return []
    payload = {"inputs": texts}
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(EMBEDDING_ENDPOINT_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and all(isinstance(emb, list) for emb in result):
                if len(result) == len(texts): return result
                else: logger.error(f"Embedding endpoint returned {len(result)} embeddings for {len(texts)} inputs."); return None
            else: logger.error(f"Unexpected response format from embedding endpoint: {type(result)}"); return None
    except httpx.RequestError as e: logger.error(f"HTTP request error calling embedding endpoint {EMBEDDING_ENDPOINT_URL}: {e}"); return None
    except httpx.HTTPStatusError as e: logger.error(f"Embedding endpoint returned error {e.response.status_code}: {e.response.text[:200]}"); return None
    except Exception as e: logger.error(f"Error getting embeddings: {e}", exc_info=True); return None

def setup_qdrant_collection(collection_name: str):
    """Creates or confirms existence of the Qdrant collection."""
    try:
        qdrant_client.get_collection(collection_name=collection_name)
        logger.info(f"Using existing Qdrant collection: {collection_name}")
    except (UnexpectedResponse, ValueError) as e:
         if isinstance(e, UnexpectedResponse) and e.status_code == 404 or "not found" in str(e).lower():
            logger.info(f"Creating Qdrant collection: {collection_name}")
            qdrant_client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE))
         else: logger.error(f"Unexpected error checking Qdrant collection: {e}", exc_info=True); raise
# --- End Helper Functions ---


# --- Gradio Interface Functions ---

# REMOVED the 'progress' parameter from the function definition
async def index_repository(source: str, source_type: str):
    """Clones/accesses repo, chunks files, gets embeddings, and indexes in Qdrant."""
    start_time = time.monotonic()
    repo_path = None
    temp_dir = None
    collection_name = None
    upsert_time = 0.0

    num_files_found = 0
    total_chunks_generated = 0
    num_chunks_indexed = 0
    status_message = "[Error] Indexing did not complete."
    upsert_time_str = "N/A"
    files_found_str = "N/A"
    chunks_gen_str = "N/A"
    chunks_idx_str = "N/A"

    try:
        # 1. Get Repository Files
        if source_type == "GitHub URL":
            if not git:
                status_message = "[Error] GitPython not installed. Cannot clone repository."
                return status_message, None, upsert_time_str, files_found_str, chunks_gen_str, chunks_idx_str
            # --- URL Cleaning ---
            cleaned_source = source.strip()
            if "/tree/" in cleaned_source: cleaned_source = cleaned_source.split('/tree/')[0]
            if not cleaned_source.endswith('.git'): cleaned_source = cleaned_source.rstrip('/') + '.git'
            if not is_github_url(cleaned_source.replace('.git', '')):
                status_message = f"[Error] Invalid or cleaned GitHub URL: {source}"
                return status_message, None, upsert_time_str, files_found_str, chunks_gen_str, chunks_idx_str
            # --- End URL Cleaning ---
            repo_name = cleaned_source.split('/')[-1].replace('.git', '')
            collection_name = f"{COLLECTION_NAME_PREFIX}{repo_name}_{int(time.time())}"
            temp_dir = tempfile.mkdtemp(prefix="git_clone_")
            logger.info(f"Attempting to clone {cleaned_source} into {temp_dir}...")
            # REMOVED progress(0, ...) call
            logger.info(f"Cloning {repo_name}...") # Log instead
            try:
                git.Repo.clone_from(cleaned_source, temp_dir, depth=1)
                repo_path = temp_dir
                logger.info("Cloning complete.")
            except GitCommandError as e:
                logger.error(f"Git clone failed: {e}")
                status_message = f"[Error] Failed to clone repository: {e}"
                return status_message, None, upsert_time_str, files_found_str, chunks_gen_str, chunks_idx_str
            except Exception as e:
                 logger.error(f"Unexpected error during cloning: {e}")
                 status_message = f"[Error] Cloning failed: {e}"
                 return status_message, None, upsert_time_str, files_found_str, chunks_gen_str, chunks_idx_str

        elif source_type == "Local Directory":
            if not os.path.isdir(source):
                status_message = f"[Error] Invalid directory path: {source}"
                return status_message, None, upsert_time_str, files_found_str, chunks_gen_str, chunks_idx_str
            repo_path = source
            dir_name = os.path.basename(os.path.normpath(source))
            collection_name = f"{COLLECTION_NAME_PREFIX}{dir_name}_{int(time.time())}"
        else:
            status_message = "[Error] Invalid source type selected."
            return status_message, None, upsert_time_str, files_found_str, chunks_gen_str, chunks_idx_str

        # 2. Setup Qdrant Collection
        logger.info(f"Preparing Qdrant collection: {collection_name}")
        # REMOVED progress(0.05, ...) call
        logger.info("Setting up vector index...") # Log instead
        setup_qdrant_collection(collection_name)

        # 3. Find and Process Files
        code_files = get_code_files(repo_path)
        num_files_found = len(code_files)
        files_found_str = str(num_files_found)

        if not code_files:
            status_message = f"[Warning] No code files with recognized extensions found in {source}."
            chunks_gen_str = "0"
            chunks_idx_str = "0"
            upsert_time_str = "0.00 seconds (No data)"
            return status_message, collection_name, upsert_time_str, files_found_str, chunks_gen_str, chunks_idx_str

        processed_files = 0
        all_points: List[PointStruct] = []
        chunks_batch: List[str] = []
        metadata_batch: List[Dict] = []
        # total_chunks_generated = 0 # Re-init moved up

        logger.info("Starting file chunking and embedding...") # Log start
        # REMOVED progress.tqdm - iterate normally or use console tqdm if installed
        for file_path in code_files: # Iterate directly
            # Log progress manually every N files if desired
            if (processed_files + 1) % 20 == 0: # Log every 20 files
                 logger.info(f"Processing file {processed_files + 1}/{len(code_files)}...")

            relative_path = os.path.relpath(file_path, repo_path)
            logger.debug(f"Processing: {relative_path}")
            file_chunks = chunk_file(file_path, CHUNK_SIZE)
            total_chunks_generated += len(file_chunks)

            for chunk_text, start_line, end_line in file_chunks:
                chunks_batch.append(chunk_text)
                metadata_batch.append({
                    "file_path": relative_path, "start_line": start_line,
                    "end_line": end_line, "text": chunk_text
                })

                if len(chunks_batch) >= BATCH_SIZE:
                    logger.debug(f"Embedding batch of {len(chunks_batch)} chunks...")
                    embeddings = await get_embeddings(chunks_batch)
                    if embeddings:
                        points = [ PointStruct(id=str(uuid.uuid4()), vector=emb, payload=meta)
                                   for emb, meta in zip(embeddings, metadata_batch) ]
                        all_points.extend(points)
                    else: logger.warning(f"Failed embedding batch from {relative_path}. Skipping.")
                    chunks_batch = []
                    metadata_batch = []
            processed_files += 1

        # Process any remaining chunks
        if chunks_batch:
            logger.debug(f"Embedding final batch of {len(chunks_batch)} chunks...")
            embeddings = await get_embeddings(chunks_batch)
            if embeddings:
                points = [ PointStruct(id=str(uuid.uuid4()), vector=emb, payload=meta)
                           for emb, meta in zip(embeddings, metadata_batch) ]
                all_points.extend(points)
            else: logger.warning("Failed embedding final batch. Skipping.")

        # Update chunk count strings
        chunks_gen_str = str(total_chunks_generated)
        num_chunks_indexed = len(all_points)
        chunks_idx_str = str(num_chunks_indexed)

        # 4. Upsert points to Qdrant and TIME IT
        if all_points:
            logger.info(f"Upserting {num_chunks_indexed} points into Qdrant collection {collection_name}...")
            # REMOVED progress(0.95, ...) call
            logger.info("Storing vectors...") # Log instead
            upsert_start_time = time.monotonic()
            try:
                for i in range(0, num_chunks_indexed, UPSERT_BATCH_SIZE):
                     batch_to_upsert = all_points[i:i + UPSERT_BATCH_SIZE]
                     qdrant_client.upsert(collection_name=collection_name, points=batch_to_upsert, wait=True)
                     logger.debug(f"Upserted Qdrant batch {i//UPSERT_BATCH_SIZE + 1}")
                upsert_end_time = time.monotonic()
                upsert_time = upsert_end_time - upsert_start_time
                upsert_time_str = f"{upsert_time:.2f} seconds"
                logger.info(f"Upsert complete in {upsert_time_str}.")
            except Exception as upsert_e:
                 logger.error(f"Failed during Qdrant upsert: {upsert_e}", exc_info=True)
                 status_message = f"[Error] Indexing partially complete ({num_chunks_indexed} chunks prepared). Failed during storage: {upsert_e}"
                 upsert_time_str = "Error during upsert"
                 return status_message, collection_name, upsert_time_str, files_found_str, chunks_gen_str, chunks_idx_str
        else:
            logger.warning("No points generated to upsert.")
            status_message = f"[Warning] No embeddings could be generated or stored for {source}."
            upsert_time_str = "0.00 seconds (No data)"
            return status_message, collection_name, upsert_time_str, files_found_str, chunks_gen_str, chunks_idx_str

        end_time = time.monotonic()
        total_time = end_time - start_time
        status_message = (
            f"Indexing complete for '{source}'.\n"
            f"Collection: '{collection_name}'.\n"
            f"Total time: {total_time:.2f} seconds."
        )
        logger.info(status_message)
        return status_message, collection_name, upsert_time_str, files_found_str, chunks_gen_str, chunks_idx_str

    except Exception as e:
        logger.error(f"Error during indexing: {e}", exc_info=True)
        status_message = f"[Error] Indexing failed: {e}"
        upsert_time_str = "N/A (Indexing Failed)"
        files_found_str = files_found_str if num_files_found > 0 else "N/A"
        chunks_gen_str = chunks_gen_str if total_chunks_generated > 0 else "N/A"
        chunks_idx_str = chunks_idx_str if num_chunks_indexed > 0 else "N/A"
        return status_message, None, upsert_time_str, files_found_str, chunks_gen_str, chunks_idx_str
    finally:
        if temp_dir and os.path.exists(temp_dir):
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)


async def search_repository(query: str, collection_name: Optional[str], top_k: int):
    """Gets embedding for query and searches the specified Qdrant collection."""
    # ... (Search logic remains the same) ...
    if not query: return "Please enter a query."
    if not collection_name: return "Please index a repository first."
    if top_k <= 0: return "Please set Top K to a positive number."

    logger.info(f"Searching collection '{collection_name}' for query: '{query}' (top_k={top_k})")
    start_time = time.monotonic()

    query_embedding_list = await get_embeddings([query])
    if not query_embedding_list or not query_embedding_list[0]: return "[Error] Failed to get embedding for the query."
    query_vector = query_embedding_list[0]

    try:
        search_result = qdrant_client.search(collection_name=collection_name, query_vector=query_vector, limit=int(top_k), with_payload=True)
    except Exception as e: logger.error(f"Error searching Qdrant collection '{collection_name}': {e}", exc_info=True); return f"[Error] Failed to search index: {e}"

    if not search_result: return "No relevant chunks found."

    markdown_output = f"Found {len(search_result)} results:\n\n"
    for i, hit in enumerate(search_result):
        payload = hit.payload or {}; file_path = payload.get("file_path", "N/A"); start_line = payload.get("start_line", "?"); end_line = payload.get("end_line", "?"); text = payload.get("text", ""); score = hit.score
        markdown_output += f"### Result {i+1} (Score: {score:.4f})\n"
        markdown_output += f"**File:** `{file_path}` (Lines: {start_line}-{end_line})\n"
        markdown_output += f"```\n{text.strip()}\n```\n\n---\n\n"

    end_time = time.monotonic(); search_time = end_time - start_time
    logger.info(f"Search completed in {search_time:.2f} seconds, found {len(search_result)} results.")
    final_output = f"Search completed in **{search_time:.2f}s**.\n\n{markdown_output}"
    return final_output


# --- Gradio App Definition ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Code Repository Search Engine")
    gr.Markdown(f"Uses a local Qdrant index and calls embedding endpoint: `{EMBEDDING_ENDPOINT_URL}`")

    active_collection = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Index Repository")
            source_type = gr.Radio(["GitHub URL", "Local Directory"], label="Source Type", value="GitHub URL")
            source_input = gr.Textbox(label="Repository URL or Local Path", placeholder="e.g., https://github.com/user/repo or /path/to/local/repo")
            index_button = gr.Button("Start Indexing", variant="primary")

            with gr.Group():
                index_status = gr.Textbox(label="Indexing Status", interactive=False, lines=3)
                files_found_output = gr.Textbox(label="Files Found", interactive=False)
                chunks_gen_output = gr.Textbox(label="Chunks Generated", interactive=False)
                chunks_idx_output = gr.Textbox(label="Chunks Indexed", interactive=False)
                upsert_time_output = gr.Textbox(label="Qdrant Upsert Time", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("## 2. Search Code")
            query_input = gr.Textbox(label="Search Query", placeholder="e.g., function to calculate distance")
            top_k_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Top K Results")
            search_button = gr.Button("Search", variant="primary")
            search_results_output = gr.Markdown(label="Search Results")


    # --- Event Handlers ---
    index_button.click(
        fn=index_repository,
        inputs=[source_input, source_type],
        outputs=[
            index_status,
            active_collection,
            upsert_time_output,
            files_found_output,
            chunks_gen_output,
            chunks_idx_output
        ],
        # show_progress="full" # Keep default basic progress indicator
    )

    search_button.click(
        fn=search_repository,
        inputs=[query_input, active_collection, top_k_slider],
        outputs=[search_results_output]
        # show_progress="full" # Keep default basic progress indicator
    )

# --- Launch App ---
if __name__ == "__main__":
    logger.info("Starting Gradio App...")
    logger.info(f"Using Embedding Endpoint: {EMBEDDING_ENDPOINT_URL}")
    logger.info(f"Required Embedding Dimension: {EMBEDDING_DIMENSION}")
    logger.info(f"Qdrant data will be stored at: {qdrant_storage_info}")
    demo.launch()