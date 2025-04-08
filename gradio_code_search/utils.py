# gradio_code_search/utils.py
import os
import html
import psutil
from urllib.parse import urlparse
from typing import List, Tuple
from loguru import logger

# Import constants from config
from config import CODE_EXTENSIONS, CHUNK_SIZE

def is_github_url(url: str) -> bool:
    """Check if a string is a valid GitHub repository URL."""
    if not url: return False
    try:
        parsed = urlparse(url)
        # Allow github.com and www.github.com
        return parsed.scheme in ('http', 'https') and parsed.netloc.replace('www.', '') == 'github.com'
    except ValueError:
        return False

def get_code_files(repo_path: str) -> List[str]:
    """Recursively find code files in a directory, skipping git and large files."""
    code_files = []
    logger.info(f"Scanning for code files in: {repo_path}")
    files_scanned = 0
    skipped_large = 0
    skipped_encoding = 0
    skipped_os_error = 0
    skipped_other = 0

    for root, dirs, files in os.walk(repo_path, topdown=True):
        # Skip .git directory efficiently
        if '.git' in dirs:
            dirs.remove('.git')
        if '.git' in root.split(os.sep):
             continue # Should be caught by dirs.remove, but belt-and-suspenders

        for file in files:
            files_scanned += 1
            if os.path.splitext(file)[1].lower() in CODE_EXTENSIONS:
                full_path = os.path.join(root, file)
                try:
                    # Check size before opening
                    file_size = os.path.getsize(full_path)
                    if file_size > 5 * 1024 * 1024: # 5 MB limit
                        # logger.warning(f"Skipping large file (>5MB): {full_path}")
                        skipped_large += 1
                        continue
                    if file_size == 0: # Skip empty files
                        continue

                    # Try reading a small part to catch encoding issues early
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                         f.read(1024) # Read only a bit to test encoding
                    code_files.append(full_path)
                except UnicodeDecodeError:
                    # logger.warning(f"Skipping file with potential non-UTF8 content: {full_path}")
                    skipped_encoding += 1
                except OSError as e:
                    # logger.warning(f"Skipping file {full_path} due to OS error: {e}")
                    skipped_os_error += 1
                except Exception as e:
                    # logger.warning(f"Skipping file {full_path} due to generic read error: {e}")
                    skipped_other += 1

    logger.info(f"File scan complete. Found {len(code_files)} potential code files out of {files_scanned} total files.")
    if skipped_large: logger.warning(f"Skipped {skipped_large} large files (>5MB).")
    if skipped_encoding: logger.warning(f"Skipped {skipped_encoding} files due to encoding issues.")
    if skipped_os_error: logger.warning(f"Skipped {skipped_os_error} files due to OS errors.")
    if skipped_other: logger.warning(f"Skipped {skipped_other} files due to other read errors.")
    return code_files

def chunk_file(file_path: str) -> List[Tuple[str, int, int]]:
    """
    Chunks a file into segments of configured line size.
    Returns list of (chunk_text, start_line, end_line).
    Runs as a blocking function, intended to be called via run_in_executor.
    """
    chunks = []
    try:
        # Use errors='ignore' which is generally safer for diverse codebases
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for i in range(0, len(lines), CHUNK_SIZE):
            chunk_lines = lines[i:i + CHUNK_SIZE]
            if not chunk_lines: continue # Should not happen with valid range, but safety check

            chunk_text = "".join(chunk_lines).strip()
            # Only add chunk if it contains non-whitespace characters
            if chunk_text:
                start_line = i + 1
                end_line = i + len(chunk_lines)
                chunks.append((chunk_text, start_line, end_line))
    except FileNotFoundError:
        logger.error(f"File not found during chunking: {file_path}")
    except Exception as e:
        # Log error but allow indexing to continue with other files
        logger.error(f"Error chunking file {file_path}: {e}")
    return chunks


def log_memory_usage(context: str = "Current"):
    """Log current memory usage of the process."""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"{context} memory usage: {memory_info.rss / 1024 / 1024:.2f} MB RSS")
    except ImportError:
        # Only warn once if psutil is missing
        if not hasattr(log_memory_usage, "psutil_warned"):
            logger.warning("psutil not installed, skipping memory logging. `pip install psutil`")
            log_memory_usage.psutil_warned = True
    except Exception as e:
        logger.warning(f"Could not get memory usage: {e}")

def escape_html_tags(text: str) -> str:
    """Escapes HTML characters for safe rendering in Markdown/HTML."""
    return html.escape(text)