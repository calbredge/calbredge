import logging
import time
from pathlib import Path

import requests
from calbredge_protocol import BenchmarkResult

logger = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.0  # seconds; doubles each attempt: 1s, 2s, 4s
_AUTH_STATUSES = {401, 403}


def upload_result(
    result: BenchmarkResult,
    api_url: str,
    api_key: str | None = None,
    debug: bool = False,
) -> bool:
    """POST a BenchmarkResult to the API with exponential-backoff retry.

    Args:
        result: The completed benchmark result to upload.
        api_url: Full URL of the API endpoint
                 (e.g. ``https://api.your-instance.example.com/api/benchmarks``).
        api_key: Optional API key sent as the ``X-Calbredge-Key`` header.

    Returns:
        ``True`` if the server responded with HTTP 201 on any attempt.
        ``False`` if all :data:`_RETRY_ATTEMPTS` attempts failed.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Calbredge-Key"] = api_key

    payload = result.model_dump_json()

    if debug:
        print("\n--- DEBUG: JSON payload (first 1000 chars) ---")
        print(payload[:1000])
        print("--- END DEBUG ---\n")

    delay = _RETRY_BASE_DELAY
    failure_reason = "network"

    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            logger.info(
                "Uploading result %s to %s (attempt %d/%d)",
                result.run_id,
                api_url,
                attempt,
                _RETRY_ATTEMPTS,
            )
            response = requests.post(api_url, data=payload, headers=headers, timeout=30)

            if response.status_code == 201:
                logger.info("Upload succeeded (run_id=%s)", result.run_id)
                return True

            logger.warning(
                "Upload attempt %d/%d failed: HTTP %d — %s",
                attempt,
                _RETRY_ATTEMPTS,
                response.status_code,
                response.text[:200],
            )

            if response.status_code in _AUTH_STATUSES:
                failure_reason = "auth"
                break  # retrying won't help for auth failures

            failure_reason = "http"

        except requests.RequestException as exc:
            logger.warning(
                "Upload attempt %d/%d failed with network error: %s",
                attempt,
                _RETRY_ATTEMPTS,
                exc,
            )

        if attempt < _RETRY_ATTEMPTS:
            logger.info("Retrying in %.0fs…", delay)
            time.sleep(delay)
            delay *= 2

    logger.error(
        "All %d upload attempts failed for run_id=%s",
        _RETRY_ATTEMPTS,
        result.run_id,
    )

    if failure_reason == "auth":
        print("⚠ Upload failed: invalid or missing API key. Result saved locally instead.")
    else:
        print("⚠ Upload failed: could not reach the server. Result saved locally instead.")

    return False


def save_result_locally(
    result: BenchmarkResult,
    output_dir: Path | None = None,
) -> Path:
    """Save a BenchmarkResult as a JSON file in the local results cache.

    Args:
        result: The benchmark result to persist.
        output_dir: Directory to write the file to. Defaults to
                    ``~/.calbredge/results/``. Created if it does not exist.

    Returns:
        The :class:`~pathlib.Path` of the written JSON file.
    """
    if output_dir is None:
        output_dir = Path.home() / ".calbredge" / "results"

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{result.run_id}.json"
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Result saved locally: %s", out_path)
    return out_path


def upload_or_save(
    result: BenchmarkResult,
    api_url: str | None,
    api_key: str | None = None,
    debug: bool = False,
) -> dict:
    """Upload a BenchmarkResult, falling back to local save on failure.

    Behaviour:

    - If *api_url* is ``None``: skips upload entirely and saves locally.
    - If *api_url* is provided: attempts upload with retry. On success returns
      immediately. On failure saves locally as a fallback so the result is
      never lost.

    Args:
        result: The completed benchmark result.
        api_url: API endpoint URL, or ``None`` for local-only mode.
        api_key: Optional API key for the upload request.

    Returns:
        One of three outcome dicts:

        - ``{"uploaded": True, "run_id": str}`` — upload succeeded.
        - ``{"uploaded": False, "saved": True, "path": str}`` — upload failed,
          result saved locally.
        - ``{"saved": True, "path": str}`` — no URL provided, saved locally.
    """
    if api_url is None:
        path = save_result_locally(result)
        return {"saved": True, "path": str(path)}

    if upload_result(result, api_url, api_key, debug=debug):
        return {"uploaded": True, "run_id": result.run_id}

    path = save_result_locally(result)
    return {"uploaded": False, "saved": True, "path": str(path)}
