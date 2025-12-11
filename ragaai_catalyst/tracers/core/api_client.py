"""
api_client.py - Centralized API client for backend communication

This module provides a clean abstraction layer for all backend API calls,
handling authentication, retries, error handling, and response parsing.
"""

import os
import time
import json
from typing import Dict, Any, Optional, List
from urllib3.exceptions import PoolError, MaxRetryError, NewConnectionError
from requests.exceptions import ConnectionError, Timeout, RequestException
from http.client import RemoteDisconnected

from ragaai_catalyst.tracers.utils import get_logger

logger = get_logger(__name__)


class TraceAPIClient:
    """
    Centralized API client for trace-related backend operations.
    
    PUBLIC API (for external use):
    - get_presigned_url()
    - upload_to_presigned_url()
    - insert_traces()
    - get_dataset_code_hashes()
    - get_code_presigned_url()
    - upload_zip_to_presigned_url()
    - insert_code_metadata()
    - get_project_id()
    - validate_project()
    - list_all_projects()
    
    PRIVATE API (internal use only):
    - _get_headers()
    - _make_request()
    - _get_dataset_spans_from_file()
    
    Handles:
    - Authentication and token refresh
    - Request retries and error handling
    - Response parsing and validation
    - Presigned URL management
    - Project validation and fetching
    """
    
    def __init__(self, base_url: str, project_name: str, timeout: int = 120):
        self.base_url = base_url
        self.project_name = project_name
        self.timeout = timeout
        self._session_manager = None
        self._catalyst = None
    
    @property
    def session_manager(self):
        if self._session_manager is None:
            from ragaai_catalyst.session_manager import session_manager
            self._session_manager = session_manager
        return self._session_manager
    
    @property
    def catalyst(self):
        if self._catalyst is None:
            from ragaai_catalyst import RagaAICatalyst
            self._catalyst = RagaAICatalyst
        return self._catalyst
    
    def _get_headers(self, include_content_type: bool = True) -> Dict[str, str]:
        """Get standard headers with authentication."""
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            "X-Project-Name": self.project_name,
        }
        if include_content_type:
            headers["Content-Type"] = "application/json"
        return headers
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        retry_on_401: bool = True
    ) -> Optional[Any]:
        """
        Make an API request with automatic retry on 401.
        
        Returns:
            Response object on success, None on failure
        """
        if headers is None:
            headers = self._get_headers()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            start_time = time.time()
            response = self.session_manager.make_request_with_retry(
                method, url, headers=headers, data=data, timeout=self.timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [{method}] {url} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
            )
            
            if response.status_code in [200, 201]:
                return response
            elif response.status_code == 401 and retry_on_401:
                logger.warning("Received 401 error. Attempting to refresh token.")
                # Use AuthManager for token refresh
                from ragaai_catalyst.auth_manager import AuthManager
                AuthManager.get_token(force_refresh=True)
                headers["Authorization"] = f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}"
                return self._make_request(method, endpoint, data, headers, retry_on_401=False)
            else:
                error_msg = response.json().get('message', 'Unknown error') if response.text else 'No response'
                logger.error(f"API request failed: {error_msg}")
                return None
                
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            self.session_manager.handle_request_exceptions(e, f"{method} {endpoint}")
            return None
        except RequestException as e:
            logger.error(f"Request exception for {method} {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {method} {url}: {e}")
            return None
    
    def get_presigned_url(self, dataset_name: str, num_files: int = 1) -> Optional[str]:
        """
        Get presigned URL for file upload.
        
        Args:
            dataset_name: Name of the dataset
            num_files: Number of files to upload
            
        Returns:
            Presigned URL string or None on failure
        """
        payload = json.dumps({
            "datasetName": dataset_name,
            "numFiles": num_files,
        })
        
        response = self._make_request("POST", "/v1/llm/presigned-url", data=payload)
        
        if response:
            try:
                presigned_url = response.json()["data"]["presignedUrls"][0]
                return presigned_url
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to parse presigned URL from response: {e}")
                return None
        
        logger.warning("POST request failed for presigned URL, trying GET")
        return None
    
    def upload_to_presigned_url(self, presigned_url: str, file_path: str) -> bool:
        """
        Upload file to presigned URL (supports S3 and Azure Blob Storage).

        Args:
            presigned_url: Presigned URL for upload
            file_path: Path to file to upload

        Returns:
            True on success, False on failure
        """
        try:
            with open(file_path, "rb") as f:
                file_content = f.read()

            headers = {"Content-Type": "application/json"}

            if "blob.core.windows.net" in presigned_url:
                headers["x-ms-blob-type"] = "BlockBlob"
                logger.debug("Detected Azure Blob Storage, added x-ms-blob-type header")

            start_time = time.time()
            response = self.session_manager.make_presigned_request(
                "PUT", presigned_url, headers=headers, data=file_content, timeout=self.timeout
            )

            elapsed_ms = (time.time() - start_time) * 1000

            if response.status_code in [200, 201]:
                logger.info(f"Successfully uploaded file to presigned URL (Time: {elapsed_ms:.2f}ms)")
                return True

            error_detail = response.text[:200] if response.text else "No response body"
            logger.error(
                f"Failed to upload file: Status {response.status_code} | "
                f"File: {file_path} | Time: {elapsed_ms:.2f}ms | Response: {error_detail}"
            )
            return False

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return False
        except Exception as e:
            logger.error(f"Error uploading to presigned URL: {e} | File: {file_path}")
            return False

    def _get_dataset_spans_from_file(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract dataset spans from trace JSON file (private helper).

        Args:
            file_path: Path to trace JSON file

        Returns:
            List of dataset spans or None on failure
        """
        try:
            with open(file_path) as f:
                data = json.load(f)

            spans = data["data"][0]["spans"]
            dataset_spans = []

            for span in spans:
                try:
                    dataset_spans.append({
                        "spanId": span.get("context", {}).get("span_id", ""),
                        "spanName": span.get("name", ""),
                        "spanHash": span.get("hash_id", ""),
                        "spanType": span.get("attributes", {}).get("openinference.span.kind", ""),
                    })
                except Exception as e:
                    logger.warning(f"Error processing span: {e}")
                    continue

            return dataset_spans
        except Exception as e:
            logger.error(f"Error reading dataset spans from {file_path}: {e}")
            return None

    def insert_traces(
        self,
        dataset_name: str,
        presigned_url: str,
        dataset_spans: List[Dict[str, Any]],
    ) -> Optional[Any]:
        """
        Insert trace metadata after file upload.
        
        Args:
            dataset_name: Name of the dataset
            presigned_url: Presigned URL used for upload
            dataset_spans: List of dataset spans for agentic traces
        Returns:
            Response object on success, None on failure
        """
        payload = {
            "datasetName": dataset_name,
            "presignedUrl": presigned_url,
            "datasetSpans": dataset_spans,
        }
        
        response = self._make_request("POST", "/v1/llm/insert/trace", data=json.dumps(payload))
        return response
    
    def get_dataset_code_hashes(self, dataset_name: str) -> Optional[List[str]]:
        """
        Get list of existing code hashes for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of code hashes or None on failure
        """
        endpoint = f"/v2/llm/dataset/code?datasetName={dataset_name}"
        response = self._make_request("GET", endpoint, headers=self._get_headers(include_content_type=False))
        
        if response:
            try:
                return response.json()["data"]["codeHashes"]
            except KeyError as e:
                logger.error(f"Failed to parse code hashes from response: {e}")
                return None
        
        return None
    
    def get_code_presigned_url(self, dataset_name: str) -> Optional[str]:
        """
        Get presigned URL for code upload.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Presigned URL string or None on failure
        """
        payload = json.dumps({
            "datasetName": dataset_name,
            "numFiles": 1,
        })
        
        response = self._make_request("POST", "/v1/llm/presigned-url", data=payload)
        
        if response:
            try:
                presigned_url = response.json()["data"]["presignedUrls"][0]
                return presigned_url
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to parse code presigned URL from response: {e}")
                return None
        
        return None
    
    def upload_zip_to_presigned_url(self, presigned_url: str, zip_path: str) -> bool:
        """
        Upload zip file to presigned URL (supports S3 and Azure Blob Storage).

        Args:
            presigned_url: Presigned URL for upload
            zip_path: Path to zip file

        Returns:
            True on success, False on failure
        """
        try:
            with open(zip_path, "rb") as f:
                zip_content = f.read()

            # Use application/json for MinIO/S3 to match backend presigned URL signature
            headers = {"Content-Type": "application/json"}

            if "blob.core.windows.net" in presigned_url:
                headers["x-ms-blob-type"] = "BlockBlob"
                logger.debug("Detected Azure Blob Storage, added x-ms-blob-type header")

            start_time = time.time()
            response = self.session_manager.make_presigned_request(
                "PUT", presigned_url, headers=headers, data=zip_content, timeout=self.timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000

            if response.status_code in [200, 201, 204]:
                logger.info(f"Successfully uploaded zip file to presigned URL (Time: {elapsed_ms:.2f}ms)")
                return True

            error_detail = response.text[:200] if response.text else "No response body"
            logger.error(
                f"Failed to upload zip file: Status {response.status_code} | "
                f"File: {zip_path} | Time: {elapsed_ms:.2f}ms | Response: {error_detail}"
            )
            return False

        except FileNotFoundError as e:
            logger.error(f"Zip file not found: {zip_path} - {e}")
            return False
        except Exception as e:
            logger.error(f"Error uploading zip to presigned URL: {e}")
            return False
    
    def insert_code_metadata(
        self,
        dataset_name: str,
        hash_id: str,
        presigned_url: str
    ) -> Optional[Any]:
        """
        Insert code metadata after zip upload.
        
        Args:
            dataset_name: Name of the dataset
            hash_id: Hash ID of the code
            presigned_url: Presigned URL used for upload
            
        Returns:
            Response object on success, None on failure
        """
        payload = json.dumps({
            "datasetName": dataset_name,
            "codeHash": hash_id,
            "presignedUrl": presigned_url,
        })
        
        response = self._make_request("POST", "/v2/llm/dataset/code", data=payload)
        return response
    
    def get_project_id(self, project_name: Optional[str] = None) -> Optional[str]:
        """
        Get project ID for a given project name.
        
        Args:
            project_name: Name of the project (defaults to self.project_name)
            
        Returns:
            Project ID string or None if not found
        """
        name = project_name or self.project_name
        response = self._make_request("GET", "/v2/llm/projects?size=99999")
        
        if response:
            try:
                projects = response.json()["data"]["content"]
                for project in projects:
                    if project["name"] == name:
                        logger.debug(f"Found project '{name}' with ID: {project['id']}")
                        return project["id"]
                logger.error(f"Project '{name}' not found in project list")
                return None
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to parse projects from response: {e}")
                return None
        
        logger.error("Failed to fetch projects")
        return None
    
    def validate_project(self, project_name: Optional[str] = None) -> bool:
        """
        Validate that a project exists.
        
        Args:
            project_name: Name of the project (defaults to self.project_name)
            
        Returns:
            True if project exists, False otherwise
        """
        project_id = self.get_project_id(project_name)
        return project_id is not None
    
    def list_all_projects(self) -> List[Dict[str, Any]]:
        """
        Get list of all projects with their details.
        
        Returns:
            List of project dictionaries with 'id', 'name', and other fields
        """
        response = self._make_request("GET", "/v2/llm/projects?size=99999")
        
        if response:
            try:
                projects = response.json()["data"]["content"]
                logger.debug(f"Successfully fetched {len(projects)} projects")
                return projects
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to parse projects from response: {e}")
                return []
        
        logger.error("Failed to fetch projects")
        return []
