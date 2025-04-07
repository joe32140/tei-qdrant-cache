FROM python:3.10-slim

WORKDIR /app

# Install build dependencies for potential C extensions (like in qdrant-client or orjson)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port uvicorn will run on
EXPOSE 8000

# Command to run the application using uvicorn
# Use environment variable PORT if available, otherwise default to 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
# Consider increasing workers based on CPU cores if needed, e.g., using Gunicorn:
# CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:${PORT:-8000}"]