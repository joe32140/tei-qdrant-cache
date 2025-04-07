## Gradio Code Search App

This repository also includes a simple Gradio application located in the `gradio_code_search/` directory. This app allows you to:

1.  Index code files from a GitHub repository or a local directory.
2.  Chunk the code into 50-line segments.
3.  Generate embeddings for these chunks using the deployed TEI service endpoint.
4.  Store the embeddings and metadata in a local Qdrant instance (separate from the main cache).
5.  Search the indexed code chunks using natural language queries.

**To run the Gradio app:**

1.  Navigate to the `gradio_code_search` directory:
    ```bash
    cd gradio_code_search
    ```
2.  Install its specific requirements:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ensure your main embedding service (TEI + Nginx + Cache Proxy + Qdrant Cache) is running.
4.  Create a `.env` file inside `gradio_code_search` (or set environment variables) specifying `EMBEDDING_ENDPOINT_URL` (e.g., `http://localhost:8080/embed`) and the correct `EMBEDDING_DIMENSION`.
5.  Run the app:
    ```bash
    python app.py
    ```
6.  Open the displayed local URL (usually `http://127.0.0.1:7860`) in your browser.

Refer to `gradio_code_search/README.md` (if you create one) for more details.

