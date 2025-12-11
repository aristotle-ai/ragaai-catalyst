import os
import json
import requests
import logging
import time
from typing import Optional
from urllib3.exceptions import PoolError, MaxRetryError, NewConnectionError
from requests.exceptions import ConnectionError, Timeout, RequestException
from http.client import RemoteDisconnected

from ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst.session_manager import session_manager
from ragaai_catalyst.auth_manager import AuthManager

IGNORED_KEYS = {"log_source", "recorded_on"}
logger = logging.getLogger(__name__)

def create_dataset_schema_with_trace(
        project_name: str,
        dataset_name: str,
        base_url: Optional[str] = None,
        user_details: Optional[dict] = None,
        timeout: int = 120) -> requests.Response:
    schema_mapping = {}

    metadata = (
        user_details.get("trace_user_detail", {}).get("metadata", {})
        if user_details else {}
    )
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if key in IGNORED_KEYS:
                continue
            schema_mapping[key] = {"columnType": "metadata"}

    # Use AuthManager for headers
    headers = AuthManager.get_auth_header()
    headers.update({
        "Content-Type": "application/json",
        "X-Project-Name": project_name,
    })

    if schema_mapping:
        payload = json.dumps({
            "datasetName": dataset_name,
            "traceFolderUrl": None,
            "schemaMapping": schema_mapping
        })
    else:
        payload = json.dumps({
            "datasetName": dataset_name,
            "traceFolderUrl": None,
        })

    # Use provided base_url or fall back to environment variable or default
    if base_url is None:
        url_base = os.getenv("RAGAAI_CATALYST_BASE_URL", "https://catalyst.raga.ai/api")
    else:
        url_base = base_url
    logger.debug(f"Using base URL: {url_base}")
    endpoint = f"{url_base}/v1/llm/dataset/logs"

    def _make_request(retry_on_401=True):
        try:
            start_time = time.time()
            response = session_manager.make_request_with_retry(
                "POST", endpoint, headers=headers, data=payload, timeout=timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
            )

            if response.status_code in [200, 201]:
                logger.info(f"Dataset schema created successfully: {response.status_code}")
                return response
            elif response.status_code == 401 and retry_on_401:
                logger.warning("Received 401 error. Attempting to refresh token.")
                AuthManager.get_token(force_refresh=True)
                # Update header with new token
                headers.update(AuthManager.get_auth_header())
                return _make_request(retry_on_401=False)
            else:
                logger.error(f"Failed to create dataset schema: {response.status_code}")
                return None
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "creating dataset schema")
            return None
        except RequestException as e:
            logger.error(f"Failed to create dataset schema: {e}")
            return None

    return _make_request()