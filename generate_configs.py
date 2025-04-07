# generate_configs.py (Corrected Formatting)

import os
import sys
import re
import shutil # Used for directory checks potentially, although not strictly needed now

# --- Configuration ---
ENV_FILE = ".env"
DOCKER_COMPOSE_TPL = "docker-compose.yml"
NGINX_CONF_TPL = os.path.join("nginx", "nginx.conf")
NGINX_DIR = "nginx"
CACHE_PROXY_DIR = "cache_proxy" # Directory containing cache proxy code and Dockerfile
DEFAULT_TEI_REPO = "ghcr.io/huggingface/text-embeddings-inference"

# --- Helper Functions ---

def parse_env(filename):
    """Parses a simple .env file."""
    config = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                # Ignore empty lines and comments
                if line and not line.startswith('#'):
                    # Handle potential comments after the value
                    line = line.split('#', 1)[0].strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Remove potential surrounding quotes (optional, but common)
                        value = value.strip().strip("'\"")
                        config[key.strip()] = value
    except FileNotFoundError:
        print(f"Error: Environment file '{filename}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing environment file '{filename}': {e}", file=sys.stderr)
        sys.exit(1)
    return config

def get_full_image_name(config):
    """Determines the full TEI docker image name."""
    image_value = config.get("TEI_IMAGE", "latest") # Default to 'latest' tag
    if ':' in image_value and '/' in image_value:
        # Assumes a full image path is provided (e.g., repo/image:tag)
        return image_value
    else:
        # Assumes only a tag is provided, use the default repo
        return f"{DEFAULT_TEI_REPO}:{image_value}"

def generate_tei_service(index, config, full_image_name):
    """Generates the YAML block for a single TEI service."""
    service_name = f"tei-{index}"
    model_id = config.get("MODEL_ID") # Already validated in main
    revision = config.get("REVISION")
    auto_truncate = config.get("AUTO_TRUNCATE", "False") # Default if not set
    max_batch_tokens = config.get("MAX_BATCH_TOKENS", "16384") # Default if not set
    dtype = config.get("DTYPE")
    hf_token = config.get("HUGGING_FACE_HUB_TOKEN")

    # Prepare environment variables section
    environment_section = ""
    if hf_token:
        # Pass the token if it exists in the config
        environment_section = f"""
    environment:
      - HUGGING_FACE_HUB_TOKEN={hf_token}"""
    else:
        # Include a placeholder comment or leave empty if no token
        environment_section = """
    # environment:
    #   - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-} # Example if using compose env interpolation"""


    # Prepare command arguments
    command_parts = [
        "      - --model-id",
        f"      - {model_id}",
    ]
    if revision:
        command_parts.extend([f"      - --revision", f"      - {revision}"])

    command_parts.extend([
        "      - --port",
        "      - '80'", # TEI listens on port 80 inside the container
        "      - --max-batch-tokens",
        f"      - '{max_batch_tokens}'",
    ])

    if dtype:
         command_parts.extend([f"      - --dtype", f"      - {dtype}"])

    if auto_truncate.lower() == "true":
        command_parts.append("      - --auto-truncate")

    command_str = "\n".join(command_parts)

    # Use triple quotes for cleaner multi-line YAML string
    # Indentation is crucial here!
    # Note: Using f-string interpolation directly for simplicity here.
    # For very complex YAML, using a library like PyYAML might be safer.
    return f"""
  {service_name}:
    # Documentation: https://github.com/huggingface/text-embeddings-inference
    image: {full_image_name} # Use the image specified in .env or default
    container_name: {service_name}
    {environment_section}
    volumes:
      - tei-model-cache:/data # Mount named volume for model data persistence/sharing
    command: # Command arguments for TEI launcher
{command_str}
    networks:
      - tei-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['{index}'] # *** Assign GPU {index} ***
              capabilities: [gpu]
    restart: unless-stopped
    # Optional: Add healthcheck if needed (check TEI image documentation for health endpoint)
    # healthcheck:
    #   test: ["CMD", "curl", "-f", "http://localhost:80/health"] # Adjust port/path if needed
    #   interval: 30s
    #   timeout: 10s
    #   retries: 3
    #   start_period: 60s # Give time for model download/load
"""

# --- Main Script ---

if __name__ == "__main__":
    print(f"--- Running Configuration Generator ---")
    print(f"Reading configuration from: {ENV_FILE}")
    env_config = parse_env(ENV_FILE)

    # --- Validate Essential Configuration ---
    try:
        num_replicas = int(env_config.get("NUM_REPLICAS", 0))
        if num_replicas <= 0:
            raise ValueError("NUM_REPLICAS must be a positive integer.")
        print(f"Found NUM_REPLICAS = {num_replicas}")
    except (ValueError, TypeError):
        print(f"Error: NUM_REPLICAS not found or invalid in '{ENV_FILE}'. Please set it to the desired number of GPUs (e.g., NUM_REPLICAS=8).", file=sys.stderr)
        sys.exit(1)

    model_id = env_config.get("MODEL_ID")
    if not model_id:
        print(f"Error: MODEL_ID not found in '{ENV_FILE}'. Please set it (e.g., MODEL_ID=BAAI/bge-large-en-v1.5).", file=sys.stderr)
        sys.exit(1)
    print(f"Found MODEL_ID = {model_id}")

    embedding_dimension = env_config.get('EMBEDDING_DIMENSION')
    try:
        if not embedding_dimension or int(embedding_dimension) <= 0:
             raise ValueError("EMBEDDING_DIMENSION must be a positive integer.")
        print(f"Found EMBEDDING_DIMENSION = {embedding_dimension}")
    except (ValueError, TypeError):
        print(f"Error: EMBEDDING_DIMENSION not found or invalid in '{ENV_FILE}'. Please set it (e.g., EMBEDDING_DIMENSION=1024).", file=sys.stderr)
        sys.exit(1)


    host_port = env_config.get("HOST_PORT", "8080") # Port for Cache Proxy
    print(f"Using HOST_PORT = {host_port} for Cache Proxy")

    tei_full_image = get_full_image_name(env_config) # Get the full image name
    print(f"Using TEI Image = {tei_full_image}")

    # Read other necessary configs
    qdrant_collection = env_config.get('QDRANT_COLLECTION', 'text_embedding_cache')
    nginx_upstream_url = env_config.get('NGINX_UPSTREAM_URL', 'http://nginx:80') # Internal URL
    qdrant_host = env_config.get('QDRANT_HOST', 'qdrant') # Service name
    qdrant_port = env_config.get('QDRANT_PORT', '6333') # Internal gRPC port

    # --- Generate docker-compose.yml ---
    print(f"\nGenerating {DOCKER_COMPOSE_TPL}...")

    # --- Qdrant Service Definition ---
    qdrant_service = f"""
  qdrant:
    image: qdrant/qdrant:latest # Consider pinning to a specific version e.g. v1.7.4
    container_name: qdrant-db
    ports:
      - "{qdrant_port}:{qdrant_port}" # Expose Qdrant gRPC port (optional, for external access/debug)
      - "6334:6334" # Default REST port (optional)
    volumes:
      - qdrant-storage:/qdrant/storage # Persistent storage for vectors
    networks:
      - tei-net
    restart: unless-stopped
    environment:
      # Optional: Configure Qdrant settings if needed via environment variables
      QDRANT__SERVICE__GRPC_PORT: "{qdrant_port}"
      # QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS: "..." # Example tuning
"""

    # --- Cache Proxy Service Definition ---
    # Builds the image from the cache_proxy directory
    cache_proxy_service = f"""
  cache-proxy:
    build:
      context: ./{CACHE_PROXY_DIR} # Specifies the build context directory
      dockerfile: Dockerfile # Specifies the Dockerfile within that context
    container_name: cache-proxy-service
    ports:
      - "{host_port}:8000" # Map HOST_PORT from .env to internal port 8000 (uvicorn default)
    volumes:
      # Mount code for development (optional, remove for production image build)
      # - ./cache_proxy:/app
      - ./.env:/app/.env:ro # Mount .env file read-only so proxy can read it
    networks:
      - tei-net
    depends_on:
      qdrant: # Depends on Qdrant being started
        condition: service_started # Basic check, use service_healthy if Qdrant adds a healthcheck
      nginx: # Depends on Nginx being started (which depends on TEI)
        condition: service_started
    environment:
      # Pass necessary env vars directly (uvicorn/FastAPI reads from env)
      # These are also available via the mounted .env file for the config.py loader
      QDRANT_HOST: "{qdrant_host}"
      QDRANT_PORT: "{qdrant_port}"
      QDRANT_COLLECTION: "{qdrant_collection}"
      EMBEDDING_DIMENSION: "{embedding_dimension}"
      NGINX_UPSTREAM_URL: "{nginx_upstream_url}"
      PYTHONUNBUFFERED: "1" # Ensures print statements and logs appear without delay
      LOG_LEVEL: "INFO" # Control cache proxy log level (e.g., DEBUG, INFO, WARNING)
      PORT: "8000" # Internal port uvicorn should listen on inside the container
    restart: unless-stopped
"""

    # --- Nginx Service Definition (Internal Load Balancer) ---
    # No external port mapping needed now, accessed internally by cache-proxy
    nginx_service = f"""
  nginx:
    image: nginx:latest
    container_name: nginx-internal-lb # Renamed for clarity
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/conf.d/default.conf:ro # Mount generated nginx config
    networks:
      - tei-net
    depends_on: # Nginx depends on all TEI instances being started
"""
    # Add depends_on entries dynamically for Nginx
    for i in range(num_replicas):
        nginx_service += f"      - tei-{i}\n"
    nginx_service += "    restart: unless-stopped\n" # Add restart policy for nginx


    # --- Assemble docker-compose.yml ---
    # Using f-string for cleaner structure
    compose_content = f"""# docker-compose.yml (Generated by generate_configs.py)
# DO NOT EDIT THIS FILE DIRECTLY - Modify .env and re-run generate_configs.py
services:
  # --- Cache Proxy Service (Acts as the main entrypoint) ---
{cache_proxy_service}

  # --- Qdrant Cache Store (Vector Database) ---
{qdrant_service}

  # --- Nginx Load Balancer (Internal routing to TEI instances) ---
{nginx_service}

  # --- TEI Instances (Auto-generated based on NUM_REPLICAS) ---"""

    # Add TEI service blocks dynamically
    for i in range(num_replicas):
        compose_content += generate_tei_service(i, env_config, tei_full_image)

    # --- Shared Resources Section ---
    compose_content += """

# --- Shared Resources ---
networks:
  tei-net: # Custom bridge network for inter-container communication
    driver: bridge
    name: tei-net # Explicitly name the network

volumes:
  tei-model-cache: # Named volume for downloaded Hugging Face models used by TEI
    driver: local
    name: tei-model-cache # Explicitly name the volume
  qdrant-storage: # Named volume for persistent Qdrant data
    driver: local
    name: qdrant-storage # Explicitly name the volume
"""
    # --- Write docker-compose.yml ---
    try:
        with open(DOCKER_COMPOSE_TPL, 'w') as f:
            f.write(compose_content)
        print(f"Successfully generated: {DOCKER_COMPOSE_TPL}")
    except IOError as e:
        print(f"Error writing {DOCKER_COMPOSE_TPL}: {e}", file=sys.stderr)
        sys.exit(1)


    # --- Generate nginx/nginx.conf ---
    # (This part remains unchanged as it doesn't depend on the other services directly,
    # only on the names of the TEI containers)
    print(f"\nGenerating {NGINX_CONF_TPL}...")
    nginx_upstream_servers = ""
    for i in range(num_replicas):
        # Nginx uses container names for resolution on the shared Docker network
        # TEI containers listen on port 80 internally by default
        nginx_upstream_servers += f"    server tei-{i}:80;\n"

    nginx_content = f"""# nginx/nginx.conf (Generated by generate_configs.py)
# DO NOT EDIT THIS FILE DIRECTLY - Modify .env and re-run generate_configs.py

# Define the group of upstream TEI servers that Nginx will balance load across
upstream tei_servers {{
    # Load balancing strategy:
    # round-robin; # Default strategy, usually sufficient for stateless requests.
    # least_conn;  # Consider if request processing times vary significantly. Sends request to server with fewest active connections.
    # ip_hash;     # Ensures requests from the same client IP go to the same server (less useful here).

{nginx_upstream_servers}
    # Optional: Keepalive connections to upstream servers can reduce latency for frequent requests.
    # keepalive 32;
}}

server {{
    # Nginx listens on port 80 *inside* its container. It's not exposed externally.
    listen 80;
    server_name localhost; # Internal server name

    # Default location block to proxy all requests coming from the cache-proxy service
    location / {{
        # Pass requests to the upstream group defined above
        proxy_pass http://tei_servers;

        # Set standard proxy headers to pass client information (though less critical here as client is cache-proxy)
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Optional: Adjust proxy timeouts if needed
        # proxy_connect_timeout 60s;
        # proxy_send_timeout 60s;
        # proxy_read_timeout 120s; # Increase if embedding large batches takes time
    }}

    # Optional: Nginx status endpoint (accessible only within the docker network)
    location /nginx_status {{
        stub_status;      # Nginx module provides basic status info
        allow 127.0.0.1;  # Allow access only from localhost within the container/network
        allow 172.16.0.0/12; # Allow access from default Docker bridge networks (adjust if needed)
        allow 192.168.0.0/16; # Allow access from common private networks (adjust if needed)
        deny all;         # Deny all other access
    }}
}}
"""

    # --- Write nginx/nginx.conf ---
    try:
        # Ensure the nginx directory exists before writing the file
        os.makedirs(NGINX_DIR, exist_ok=True)
        with open(NGINX_CONF_TPL, 'w') as f:
            f.write(nginx_content)
        print(f"Successfully generated: {NGINX_CONF_TPL}")
    except IOError as e:
        print(f"Error writing {NGINX_CONF_TPL}: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Configuration generation complete ---")
    # Reminder about the cache_proxy directory structure
    print(f"Ensure the '{CACHE_PROXY_DIR}' directory exists and contains:")
    print(f"  Dockerfile, main.py, requirements.txt, schemas.py, qdrant_utils.py, config.py")
    print(f"\nYou can now run 'docker compose up -d --build'") # Add --build flag reminder