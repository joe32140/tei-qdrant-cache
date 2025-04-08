# gradio_code_search/app.py
import gradio as gr
import time
from typing import Optional
from loguru import logger

# Import necessary functions and variables from other modules
from config import EMBEDDING_ENDPOINT_URL, EMBEDDING_DIMENSION
from qdrant_logic import qdrant_storage_info, search_qdrant # Import search_qdrant
from embedding_client import get_embeddings
from indexing import index_repository
from utils import escape_html_tags # Import HTML escaper

# --- Search Function (UI Facing) ---

async def search_code_handler(query: str, collection_name: Optional[str], top_k: int) -> str:
    """
    Handles the search request from the Gradio UI.
    Gets query embedding, searches Qdrant, and formats results.
    """
    if not query:
        return "⚠️ Please enter a search query."
    if not collection_name:
        # Check if any collections exist? Maybe later.
        return "⚠️ Please index a repository first or ensure a collection exists."
    if not top_k or top_k <= 0:
        return "⚠️ Please set 'Number of Results' to a positive value."

    logger.info(f"Received search request: query='{query[:50]}...', collection='{collection_name}', k={top_k}")
    search_start_time = time.monotonic()

    # 1. Get Query Embedding
    try:
        query_embedding_list = await get_embeddings([query]) # Use the robust client
        if not query_embedding_list or not query_embedding_list[0]:
            logger.error("Failed to get embedding for the search query.")
            return "[Error] Failed to get embedding for the query. Check the embedding service and logs."
        query_vector = query_embedding_list[0]
    except Exception as e:
         logger.error(f"Error getting query embedding: {e}", exc_info=True)
         return f"[Error] Could not get query embedding: {e}"

    # 2. Search Qdrant
    try:
        search_results = search_qdrant(collection_name, query_vector, int(top_k))
    except ValueError as e: # Catch specific "Collection not found" error from search_qdrant
         logger.warning(f"Search failed because collection '{collection_name}' was not found.")
         return f"[Error] Collection '{collection_name}' not found. Please index it again or select a valid one."
    except Exception as e:
        logger.error(f"Error during Qdrant search: {e}", exc_info=True)
        return f"[Error] Failed to search index '{collection_name}': {e}"

    search_time = time.monotonic() - search_start_time

    # 3. Format Results
    if not search_results:
        return f"✅ Search completed in **{search_time:.2f}s**. No relevant chunks found in '{collection_name}' for your query."

    markdown_output = f"✅ Search completed in **{search_time:.2f}s**. Found **{len(search_results)}** results in '{collection_name}':\n\n---\n\n"
    for i, hit in enumerate(search_results):
        payload = hit.payload or {}
        file_path = payload.get("file_path", "N/A")
        start_line = payload.get("start_line", "?")
        end_line = payload.get("end_line", "?")
        # Text should be in payload if saved correctly during indexing
        text_content = payload.get("text", "[Error: Text not found in payload]")
        score = hit.score

        # Escape HTML within the code block for safe rendering
        escaped_text = escape_html_tags(text_content.strip())

        markdown_output += f"### Result {i+1} (Score: {score:.4f})\n"
        markdown_output += f"**File:** `{file_path}` (Lines: {start_line}-{end_line})\n\n"
        # Use Markdown code block with language hint (optional, 'python' is a guess)
        # Or use <pre><code> for guaranteed literal display
        markdown_output += f"```\n{escaped_text}\n```\n\n"
        # markdown_output += f"<pre><code>{escaped_text}</code></pre>\n\n" # Alternative using HTML
        markdown_output += "---\n\n"

    logger.info(f"Search successful. Returning {len(search_results)} results. Time: {search_time:.2f}s")
    return markdown_output

# --- Gradio App Definition ---

def build_gradio_app():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Code Repository Search Engine")
        gr.Markdown(
            f"Using Qdrant (Storage: `{qdrant_storage_info}`) | "
            f"Embedding Service: `{EMBEDDING_ENDPOINT_URL}` (Dim: {EMBEDDING_DIMENSION})"
        )

        # State to hold the name of the currently indexed/active collection
        # Initialize with None, updated after successful indexing
        active_collection = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Index Repository")
                source_type = gr.Radio(["GitHub URL", "Local Directory"], label="Source Type", value="GitHub URL")
                source_input = gr.Textbox(
                    label="Repository URL or Local Path",
                    placeholder="e.g., https://github.com/user/repo OR /path/to/local/repo",
                    lines=1
                )
                index_button = gr.Button("Start Indexing", variant="primary")

                with gr.Accordion("Indexing Status & Details", open=False):
                    index_status_output = gr.Textbox(label="Overall Status", interactive=False, lines=3)
                    files_found_output = gr.Textbox(label="Files Found", interactive=False)
                    chunks_gen_output = gr.Textbox(label="Chunks Generated", interactive=False)
                    chunks_idx_output = gr.Textbox(label="Chunks Indexed (in DB)", interactive=False)
                    index_time_output = gr.Textbox(label="Total Indexing Time", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("## 2. Search Code")
                # Display the active collection for clarity
                active_collection_display = gr.Textbox(label="Currently Searching Collection", interactive=False)
                query_input = gr.Textbox(label="Search Query", placeholder="e.g., function to calculate distance matrix", lines=2)
                top_k_slider = gr.Slider(minimum=1, maximum=30, value=5, step=1, label="Number of Results (Top K)")
                search_button = gr.Button("Search", variant="primary")
                search_results_output = gr.Markdown(label="Search Results", elem_id="search-results-markdown")

        # --- Event Handlers ---

        # Function to update the display box when active_collection state changes
        def update_active_collection_display(collection_name):
            return collection_name if collection_name else "None (Index a repository first)"

        # When indexing finishes, update status fields and the active_collection state
        index_button.click(
            fn=index_repository, # Call the function from indexing.py
            inputs=[source_input, source_type],
            outputs=[
                index_status_output,
                active_collection, # Update the state with the new collection name
                index_time_output,
                files_found_output,
                chunks_gen_output,
                chunks_idx_output
            ],
            # show_progress="full" # Gradio's default progress indicator
        ).then(
            # After indexing finishes, update the display box showing the active collection
            fn=update_active_collection_display,
            inputs=[active_collection],
            outputs=[active_collection_display]
        )

        # When search button is clicked, use the query, active_collection state, and top_k
        search_button.click(
            fn=search_code_handler, # Call the UI handler function
            inputs=[query_input, active_collection, top_k_slider], # Pass active_collection state
            outputs=[search_results_output]
            # show_progress="full"
        )

        # Also update the display box if the active_collection state is somehow changed elsewhere (e.g., loaded)
        # This might be useful if you add features to load existing collections later
        active_collection.change(
             fn=update_active_collection_display,
             inputs=active_collection,
             outputs=active_collection_display
        )

    return demo

# --- Launch App ---
if __name__ == "__main__":
    logger.info("Starting Gradio Application...")
    # Ensure Qdrant client is ready (initialization happens on import of qdrant_logic)
    # Build and launch the Gradio app
    app = build_gradio_app()
    app.launch()
    logger.info("Gradio Application Stopped.")