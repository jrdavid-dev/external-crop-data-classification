"""Bounded, checkpointed Earth Engine table retrieval with visible page progress."""
import hashlib
import json
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import requests


def save_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False, suffix=".part") as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, allow_nan=False)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transient(error):
    if isinstance(error, (requests.Timeout, requests.ConnectionError, TimeoutError, ConnectionError)):
        return True
    status = getattr(getattr(error, "response", None), "status_code", None)
    message = str(error).lower()
    return status in {429, 500, 502, 503, 504} or any(
        marker in message for marker in ("timed out", "timeout", "connection reset", "temporarily unavailable",
                                         "too many requests", "http 429", "http 500", "http 502", "http 503", "http 504"))


def download(ee, asset_id, cache_dir=None, page_size=5000, timeout_seconds=120, retries=1):
    if type(page_size) is not int or page_size <= 0 or timeout_seconds <= 0 or type(retries) is not int or retries < 0:
        raise ValueError("Use a positive page size/timeout and nonnegative retries")
    cache = Path(cache_dir) if cache_dir is not None else None
    manifest_path = cache / "manifest.json" if cache else None
    manifest = {"version": 1, "asset_id": asset_id, "page_size": page_size,
                "pages": [], "complete": False}
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("version") != 1 or manifest.get("asset_id") != asset_id
                or manifest.get("page_size") != page_size):
            raise ValueError("Table download checkpoint belongs to a different request")
    records, token, tokens = [], None, set()
    for page in manifest["pages"]:
        path = cache / page["file"]
        if not path.is_file() or digest(path) != page["sha256"]:
            raise ValueError(f"Corrupt download checkpoint: {path}. Preserve the asset and remove this task's download_pages cache to retry.")
        payload = json.loads(path.read_text())
        if page["request_token"] != token or len(payload) != page["rows"]:
            raise ValueError("Download checkpoint page order or count differs")
        records.extend(payload)
        if token is not None:
            tokens.add(token)
        token = page["next_token"]
    if records:
        print(f"  Resuming with {len(records):,} rows from {len(manifest['pages'])} verified cached pages.", flush=True)
    if manifest["complete"] and token is not None:
        raise ValueError("Complete download checkpoint has a continuation token")
    if manifest["complete"]:
        return pd.DataFrame.from_records(records)
    if manifest["pages"] and token is None:
        raise ValueError("Incomplete download checkpoint has no continuation token")
    # The pinned EE SDK exposes setters but no public getter for its retry count.
    # Disable its hidden retries here and perform a single visible bounded retry ourselves.
    previous_retries = ee.data._get_state().max_retries
    ee.data.setMaxRetries(0)
    try:
        # Keep a finite network timeout for later EE calls in this process as well.
        ee.data.setDeadline(timeout_seconds * 1000)
        expression = ee.FeatureCollection(asset_id)
        while True:
            number = len(manifest["pages"]) + 1
            parameters = {"expression": expression, "pageSize": page_size}
            if token is not None:
                parameters["pageToken"] = token
            print(f"  Fetching page {number} (up to {page_size:,} rows; {len(records):,} saved so far)…", flush=True)
            started = time.monotonic()
            for attempt in range(retries + 1):
                try:
                    response = ee.data.computeFeatures(parameters)
                    break
                except Exception as error:
                    if attempt == retries or not transient(error):
                        raise RuntimeError(
                            f"Table page {number} could not be retrieved. "
                            + (f"{len(records):,} rows remain checkpointed; " if cache else "No disk checkpoint was requested; ")
                            +
                            "the Earth Engine asset is preserved. Rerun collection to resume. "
                            f"Cause: {type(error).__name__}: {error}"
                        ) from error
                    print(f"  Page {number} request failed ({type(error).__name__}); retry {attempt+1}/{retries} in 2 seconds.", flush=True)
                    time.sleep(2)
            features = response.get("features", [])
            if not isinstance(features, list):
                raise ValueError("Earth Engine table response has invalid features")
            page_records = [feature.get("properties", {}) or {} for feature in features]
            next_token = response.get("nextPageToken") or response.get("next_page_token") or None
            if next_token is not None and (not isinstance(next_token, str) or next_token == token or next_token in tokens):
                raise ValueError("Earth Engine returned a repeated or invalid pagination token")
            record = {"request_token": token, "next_token": next_token, "rows": len(page_records)}
            if cache:
                name = f"page_{number:05d}.json"
                path = cache / name
                save_atomic(path, page_records)
                record.update(file=name, sha256=digest(path))
            manifest["pages"].append(record)
            manifest["complete"] = next_token is None
            if manifest_path:
                save_atomic(manifest_path, manifest)
            records.extend(page_records)
            print(f"  Page {number} verified: {len(page_records):,} rows; {len(records):,} total "
                  f"({time.monotonic()-started:.1f}s).", flush=True)
            if next_token is None:
                break
            if token is not None:
                tokens.add(token)
            token = next_token
    finally:
        ee.data.setMaxRetries(previous_retries)
    return pd.DataFrame.from_records(records)
