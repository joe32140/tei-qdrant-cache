# load_test_embed.py
import asyncio
import httpx
import time
import random
import argparse
import statistics
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Sample Data ---
# Add more diverse and longer texts for realistic testing
DEFAULT_SAMPLE_TEXTS = [
    "This is the first test sentence.",
    "Here is another sentence for embedding.",
    "Deep learning has revolutionized many fields.",
    "What is the capital of France?",
    "Natural language processing enables computers to understand text.",
    "This is the first test sentence.", # Include duplicates to test caching
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming industries.",
    "How does the caching mechanism perform under load?",
    "This is the first test sentence.",
]

# --- Helper Functions ---

def calculate_percentiles(latencies: List[float], percentiles: List[int]) -> Dict[int, float]:
    """Calculates specified percentiles from a list of latencies."""
    if not latencies:
        return {p: 0.0 for p in percentiles}
    
    latencies.sort()
    results = {}
    for p in percentiles:
        if p < 0 or p > 100:
            continue
        index = int(len(latencies) * p / 100)
        # Ensure index is within bounds
        index = min(index, len(latencies) - 1)
        results[p] = latencies[index]
    return results

async def send_request(
    client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
    request_data: Dict[str, Any],
) -> Tuple[Optional[float], Optional[int], bool]:
    """Sends a single request, measures latency, and handles errors."""
    latency: Optional[float] = None
    status_code: Optional[int] = None
    success = False
    async with semaphore: # Acquire semaphore before making request
        start_time = time.monotonic()
        try:
            response = await client.post(url, json=request_data, timeout=30.0) # Adjust timeout
            latency = time.monotonic() - start_time
            status_code = response.status_code
            if 200 <= status_code < 300:
                # Optionally check response content basic structure
                # response.json() # This would parse JSON, adding overhead
                success = True
                logger.debug(f"Request successful ({status_code}) - Latency: {latency:.4f}s")
            else:
                logger.warning(
                    f"Request failed with status {status_code} - Latency: {latency:.4f}s - Response: {response.text[:100]}"
                )

        except httpx.TimeoutException:
            latency = time.monotonic() - start_time
            logger.warning(f"Request timed out after {latency:.4f}s")
        except httpx.RequestError as e:
            latency = time.monotonic() - start_time
            logger.error(f"Request failed: {e.__class__.__name__} - {e}")
        except Exception as e:
            latency = time.monotonic() - start_time
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        finally:
            # Return latency (even on failure), status code, and success flag
            return latency, status_code, success

# --- Main Load Test Function ---

async def run_load_test(
    url: str, total_requests: int, concurrency: int, data_file: Optional[str]
):
    """Runs the load test with specified parameters."""
    if data_file:
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                sample_texts = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(sample_texts)} samples from {data_file}")
            if not sample_texts:
                 logger.warning("Data file is empty, falling back to default samples.")
                 sample_texts = DEFAULT_SAMPLE_TEXTS
        except FileNotFoundError:
            logger.error(f"Data file not found: {data_file}. Using default samples.")
            sample_texts = DEFAULT_SAMPLE_TEXTS
        except Exception as e:
            logger.error(f"Error reading data file {data_file}: {e}. Using default samples.")
            sample_texts = DEFAULT_SAMPLE_TEXTS
    else:
        sample_texts = DEFAULT_SAMPLE_TEXTS
        logger.info(f"Using default {len(sample_texts)} sample texts.")


    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    results = [] # List to store (latency, status_code, success) tuples

    logger.info(
        f"Starting load test: URL={url}, Total Requests={total_requests}, Concurrency={concurrency}"
    )
    overall_start_time = time.monotonic()

    async with httpx.AsyncClient() as client:
        for i in range(total_requests):
            # Prepare request data - select random text
            text_input = random.choice(sample_texts)
            payload = {"inputs": text_input} # Send single text per request

            # Create and schedule the task
            task = asyncio.create_task(
                send_request(client, url, semaphore, payload)
            )
            tasks.append(task)

        # Wait for all tasks to complete and gather results
        raw_results = await asyncio.gather(*tasks)
        results.extend(raw_results) # Add results from gather

    overall_end_time = time.monotonic()
    total_duration = overall_end_time - overall_start_time

    # --- Process Results ---
    successful_requests = 0
    failed_requests = 0
    latencies = []
    status_codes: Dict[int, int] = {}

    for latency, status, success in results:
        if success and latency is not None:
            successful_requests += 1
            latencies.append(latency)
        else:
            failed_requests += 1

        if status is not None:
            status_codes[status] = status_codes.get(status, 0) + 1
        else:
             # Count timeouts/connection errors under a generic 'None' status
             status_codes[0] = status_codes.get(0, 0) + 1 # Using 0 for errors without status


    # --- Calculate Statistics ---
    rps = successful_requests / total_duration if total_duration > 0 else 0
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    median_latency = statistics.median(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    percentile_values = calculate_percentiles(latencies, [90, 95, 99])
    success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0

    # --- Print Report ---
    print("\n--- Load Test Summary ---")
    print(f"Target URL:          {url}")
    print(f"Total Requests:      {total_requests}")
    print(f"Concurrency Level:   {concurrency}")
    print(f"Total Duration:      {total_duration:.2f} s")
    print("-" * 25)
    print(f"Successful Requests: {successful_requests}")
    print(f"Failed Requests:     {failed_requests}")
    print(f"Success Rate:        {success_rate:.2f}%")
    print(f"Requests Per Second: {rps:.2f} RPS")
    print("-" * 25)
    print("Latencies (Successful Requests):")
    print(f"  Average:           {avg_latency:.4f} s")
    print(f"  Median:            {median_latency:.4f} s")
    print(f"  Min:               {min_latency:.4f} s")
    print(f"  Max:               {max_latency:.4f} s")
    print(f"  90th Percentile:   {percentile_values.get(90, 0.0):.4f} s")
    print(f"  95th Percentile:   {percentile_values.get(95, 0.0):.4f} s")
    print(f"  99th Percentile:   {percentile_values.get(99, 0.0):.4f} s")
    print("-" * 25)
    print("Status Code Distribution:")
    # Sort status codes for consistent output
    for code in sorted(status_codes.keys()):
        count = status_codes[code]
        status_label = f"{code}" if code > 0 else "Errors (No Status)"
        print(f"  {status_label}:             {count}")
    print("--- End Summary ---\n")


# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test the /embed endpoint.")
    parser.add_argument(
        "url",
        help="Target URL for the /embed endpoint (e.g., http://localhost:8080/embed)",
    )
    parser.add_argument(
        "-r",
        "--requests",
        type=int,
        default=100,
        help="Total number of requests to send.",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests.",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=None,
        help="Optional path to a file containing sample texts (one per line).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging for requests.",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.DEBUG) # Enable httpx debug logs

    # Run the async main function
    asyncio.run(run_load_test(args.url, args.requests, args.concurrency, args.data))