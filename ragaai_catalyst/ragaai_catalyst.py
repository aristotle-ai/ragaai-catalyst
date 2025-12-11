import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

from ragaai_catalyst.auth_manager import AuthManager

logger = logging.getLogger("RagaAICatalyst")

# Configure debug level if DEBUG environment variable is set
if os.getenv("DEBUG") == "1":
    logger.setLevel(logging.DEBUG)


class RagaAICatalyst:
    """RagaAI Catalyst SDK client for LLM evaluation and monitoring."""

    _DEFAULT_BASE_URL = "https://catalyst.raga.ai/api"
    _DEFAULT_TIMEOUT = 10
    
    # Backward compatibility
    BASE_URL = os.getenv("RAGAAI_CATALYST_BASE_URL") or _DEFAULT_BASE_URL
    TIMEOUT = _DEFAULT_TIMEOUT

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        api_keys: Optional[Dict[str, str]] = None,
        base_url: Optional[str] = None,
        token_expiry_time: float = 6.0,
    ):
        """
        Initialize RagaAI Catalyst client.

        Args:
            access_key: RagaAI Catalyst access key
            secret_key: RagaAI Catalyst secret key
            api_keys: Optional dict of third-party API keys
            base_url: Optional custom API base URL
            token_expiry_time: Token expiration time in hours (default: 6)

        Raises:
            ValueError: If access_key or secret_key is empty
            requests.RequestException: If initial token fetch fails
        """
        if not access_key or not secret_key:
            raise ValueError(
                "access_key and secret_key are required. "
                "Get your keys from Settings -> Authenticate in the RagaAI Catalyst dashboard."
            )

        self._set_access_key_secret_key(access_key, secret_key)

        self.base_url = (
            self._normalize_base_url(base_url) if base_url
            else os.getenv("RAGAAI_CATALYST_BASE_URL", self._DEFAULT_BASE_URL)
        )
        self.timeout = self._DEFAULT_TIMEOUT

        os.environ["RAGAAI_CATALYST_BASE_URL"] = self.base_url
        RagaAICatalyst.BASE_URL = self.base_url

        AuthManager.initialize(
            access_key=access_key,
            secret_key=secret_key,
            base_url=self.base_url,
            token_expiry_hours=token_expiry_time
        )

        try:
            AuthManager.get_token(force_refresh=True)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to authenticate with provided credentials: {e}")
            if base_url:
                logger.error("The provided base_url may not be accessible.")
            raise

        self.api_keys = api_keys or {}

        if self.api_keys:
            self._upload_keys()

    @staticmethod
    def _normalize_base_url(url: str) -> str:
        """Normalize and validate base URL format."""
        url = re.sub(r"(?<!:)//+", "/", url)
        url = url.rstrip("/")
        if not url.endswith("/api"):
            url = f"{url}/api"
        return url

    def _set_access_key_secret_key(self, access_key: str, secret_key: str) -> None:
        """Set credentials in environment variables for backward compatibility."""
        os.environ["RAGAAI_CATALYST_ACCESS_KEY"] = access_key
        os.environ["RAGAAI_CATALYST_SECRET_KEY"] = secret_key

    def _upload_keys(self) -> None:
        """
        Upload third-party API keys to RagaAI platform.

        Raises:
            requests.RequestException: If upload fails
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
        }
        secrets = [
            {"type": service, "key": service, "value": key}
            for service, key in self.api_keys.items()
        ]
        json_data = {"secrets": secrets}
        start_time = time.time()
        endpoint = f"{self.base_url}/v1/llm/secrets/upload"
        response = requests.post(
            endpoint,
            headers=headers,
            json=json_data,
            timeout=self.timeout,
        )
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
        )
        if response.status_code == 200:
            logger.info("API keys uploaded successfully")
        else:
            logger.error(f"Failed to upload API keys: Status {response.status_code}")

    def add_api_key(self, service: str, key: str) -> None:
        """Add or update an API key for a specific service."""
        self.api_keys[service] = key

    def get_api_key(self, service: str) -> Optional[str]:
        """Get the API key for a specific service."""
        return self.api_keys.get(service)

    def _make_authenticated_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        retry_on_401: bool = True
    ) -> Optional[requests.Response]:
        """
        Make an authenticated API request with automatic retry on 401.
        
        This centralizes request logic with:
        - Automatic authentication header injection
        - Token refresh on 401 errors
        - Connection retry logic via session_manager
        - Consistent logging
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (e.g., "/v2/llm/projects")
            json_data: Optional JSON payload for POST/PUT requests
            retry_on_401: Whether to retry once on 401 with token refresh
            
        Returns:
            Response object on success, None on failure
        """
        from ragaai_catalyst.session_manager import session_manager

        headers = self.get_auth_header()
        headers["Content-Type"] = "application/json"

        url = f"{self.base_url}{endpoint}"
        
        try:
            start_time = time.time()
            response = session_manager.make_request_with_retry(
                method,
                url,
                headers=headers,
                json=json_data,
                timeout=self.timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [{method}] {url} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
            )
            
            if response.status_code in [200, 201]:
                return response
            elif response.status_code == 401 and retry_on_401:
                logger.warning("Received 401 error. Attempting to refresh token.")
                AuthManager.get_token(force_refresh=True)
                headers = self.get_auth_header()
                headers["Content-Type"] = "application/json"
                return self._make_authenticated_request(method, endpoint, json_data, retry_on_401=False)
            else:
                error_msg = response.json().get('message', 'Unknown error') if response.text else 'No response'
                logger.error(f"API request failed with status {response.status_code}: {error_msg}")
                return response
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception for {method} {url}: {e}")
            session_manager.handle_request_exceptions(e, f"{method} {endpoint}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {method} {url}: {e}")
            return None

    @staticmethod
    def _get_credentials() -> tuple[str, str]:
        """
        Get access key and secret key from instance or environment.

        Deprecated: Use AuthManager._get_credentials() instead.
        Kept for backward compatibility.
        """
        return AuthManager._get_credentials()

    @staticmethod
    def _refresh_token_async() -> None:
        """
        Refresh token in background thread.

        Deprecated: Use AuthManager._refresh_token_async() instead.
        Kept for backward compatibility.
        """
        return AuthManager._refresh_token_async()

    @staticmethod
    def _schedule_token_refresh() -> None:
        """
        Schedule a token refresh to happen 20 seconds before expiration.

        Deprecated: Use AuthManager._schedule_token_refresh() instead.
        Kept for backward compatibility.
        """
        return AuthManager._schedule_token_refresh()

    @staticmethod
    def get_token(force_refresh: bool = False) -> Optional[str]:
        """
        Retrieves or refreshes a token using the provided credentials.

        Args:
            force_refresh (bool): If True, forces a token refresh regardless of expiration.

        Returns:
            - A string representing the token if successful.
            - None if credentials are not set or if there is an error.
            
        Note:
            This method delegates to AuthManager.get_token() for actual implementation.
        """
        return AuthManager.get_token(force_refresh=force_refresh)

    def ensure_valid_token(self) -> Optional[str]:
        """
        Ensures a valid token is available, with different handling for missing token vs expired token:
        - Missing token: Synchronous retrieval (fail fast)
        - Expired token: Synchronous refresh (since token is needed immediately)

        Returns:
            - A string representing the valid token if successful.
            - None if unable to obtain a valid token.
            
        Note:
            This method delegates to AuthManager.ensure_valid_token() for actual implementation.
        """
        return AuthManager.ensure_valid_token()

    def get_auth_header(self) -> Dict[str, str]:
        """
        Returns a dictionary containing the Authorization header with a valid token.
        This method should be used instead of directly accessing os.getenv("RAGAAI_CATALYST_TOKEN").

        Returns:
            - A dictionary with the Authorization header if successful.
            - An empty dictionary if no valid token could be obtained.
            
        Note:
            This method delegates to AuthManager.get_auth_header() for actual implementation.
        """
        return AuthManager.get_auth_header()

    def project_use_cases(self) -> List[str]:
        """
        Retrieve list of available project use cases.

        Returns:
            List of use case strings, or empty list on failure
        """
        response = self._make_authenticated_request("GET", "/v2/llm/usecase")
        
        if response and response.status_code in [200, 201]:
            try:
                return response.json()["data"]["usecase"]
            except (KeyError, TypeError) as e:
                logger.error(f"Failed to parse use cases from response: {e}")
                return []
        
        logger.error("Failed to retrieve project use cases")
        return []

    def create_project(
        self,
        project_name: str,
        usecase: str = "Q/A",
        type: str = "llm"
    ) -> str:
        """
        Create a new project.

        Args:
            project_name: Name of the project
            usecase: Use case for the project (default: "Q/A")
            type: Project type (default: "llm")

        Returns:
            Success or failure message

        Raises:
            ValueError: If project_name already exists or usecase is invalid
        """
        existing_projects = self.list_projects()
        if project_name in existing_projects:
            error_msg = f"Project name '{project_name}' already exists. Please choose a different name."
            logger.error(error_msg)
            return f"Failed to create project: {error_msg}"

        usecase_list = self.project_use_cases()
        if usecase not in usecase_list:
            error_msg = f"Invalid usecase '{usecase}'. Select a valid usecase from {usecase_list}"
            logger.error(error_msg)
            return f"Failed to create project: {error_msg}"

        json_data = {"name": project_name, "type": type, "usecase": usecase}
        response = self._make_authenticated_request("POST", "/v2/llm/project", json_data=json_data)
        
        if response and response.status_code in [200, 201]:
            try:
                project_data = response.json()["data"]
                success_msg = f"Project created successfully: {project_data['name']} (usecase: {usecase})"
                logger.info(success_msg)
                return success_msg
            except (KeyError, TypeError) as e:
                logger.error(f"Failed to parse project creation response: {e}")
                return "Project may have been created but response parsing failed"
        elif response:
            error_msg = response.json().get('message', 'Unknown error') if response.text else 'No response'
            return f"Failed to create project: {error_msg}"
        else:
            return "Failed to create project: Request failed"

    def list_projects(self, num_projects: int = 99999) -> List[str]:
        """
        Retrieve list of project names.

        Args:
            num_projects: Maximum number of projects to retrieve (default: 99999)

        Returns:
            List of project names, or empty list on failure
        """
        response = self._make_authenticated_request("GET", f"/v2/llm/projects?size={num_projects}")
        
        if response and response.status_code in [200, 201]:
            try:
                projects = response.json()["data"]["content"]
                project_names = [project["name"] for project in projects]
                logger.debug(f"Successfully retrieved {len(project_names)} projects")
                return project_names
            except (KeyError, TypeError) as e:
                logger.error(f"Failed to parse projects from response: {e}")
                return []
        elif response:
            error_msg = response.json().get('message', 'Unknown error') if response.text else 'No response'
            logger.error(f"Failed to list projects: {error_msg}")
            return []
        else:
            logger.error("Failed to list projects: Request failed")
            return []

    def list_metrics(self) -> List[str]:
        """
        Retrieve list of available LLM evaluation metric names.

        Returns:
            List of metric name strings, or empty list on failure
        """
        response = self._make_authenticated_request("GET", "/v1/llm/llm-metrics")
        
        if response and response.status_code in [200, 201]:
            try:
                metrics = response.json()["data"]["metrics"]
                return [metric["name"] for metric in metrics]
            except (KeyError, TypeError) as e:
                logger.error(f"Failed to parse metrics from response: {e}")
                return []
        
        logger.error("Failed to retrieve metrics")
        return []
