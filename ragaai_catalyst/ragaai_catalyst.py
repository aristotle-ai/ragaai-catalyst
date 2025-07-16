import logging
import os
import re
import threading
import time
from typing import Dict, Optional, Union

import requests
from urllib3.exceptions import PoolError, MaxRetryError, NewConnectionError
from requests.exceptions import ConnectionError, Timeout, RequestException, HTTPError
from http.client import RemoteDisconnected
from ragaai_catalyst.session_manager import session_manager

logger = logging.getLogger("RagaAICatalyst")
logging_level = (
    logger.setLevel(logging.DEBUG) if os.getenv("DEBUG") == "1" else logging.INFO
)


class RagaAICatalyst:
    BASE_URL = None
    TIMEOUT = 10  # Default timeout in seconds
    TOKEN_EXPIRY_TIME = 6  # Default token expiration time (6 hours in hours)

    def __init__(
        self,
        access_key,
        secret_key,
        api_keys: Optional[Dict[str, str]] = None,
        base_url: Optional[str] = None,
        token_expiry_time: Optional[float] = 6,
    ):
        """
        Initializes a new instance of the RagaAICatalyst class.

        Args:
            access_key (str): The access key for the RagaAICatalyst.
            secret_key (str): The secret key for the RagaAICatalyst.
            api_keys (Optional[Dict[str, str]]): A dictionary of API keys for different services. Defaults to None.
            base_url (Optional[str]): The base URL for the RagaAICatalyst API. Defaults to None.
            token_expiry_time (Optional[float]): The time in hours before the token expires. Defaults to 0.1 hours.

        Raises:
            ValueError: If the RAGAAI_CATALYST_ACCESS_KEY and RAGAAI_CATALYST_SECRET_KEY environment variables are not set.
            ConnectionError: If the provided base_url is not accessible.

        Returns:
            None
        """

        if not access_key or not secret_key:
            logger.error(
                "RAGAAI_CATALYST_ACCESS_KEY and RAGAAI_CATALYST_SECRET_KEY environment variables must be set"
            )

        RagaAICatalyst.access_key, RagaAICatalyst.secret_key = (
            self._set_access_key_secret_key(access_key, secret_key)
        )

        # Initialize token management
        RagaAICatalyst._token_expiry = None
        RagaAICatalyst._token_refresh_lock = threading.Lock()
        RagaAICatalyst._refresh_thread = None

        # Set token expiration time (convert hours to seconds)
        RagaAICatalyst.TOKEN_EXPIRY_TIME = token_expiry_time * 60 * 60

        RagaAICatalyst.BASE_URL = (
            os.getenv("RAGAAI_CATALYST_BASE_URL")
            if os.getenv("RAGAAI_CATALYST_BASE_URL")
            else "https://catalyst.raga.ai/api"
        )

        self.api_keys = api_keys or {}

        if base_url:
            RagaAICatalyst.BASE_URL = self._normalize_base_url(base_url)
            try:
                # set the os.environ["RAGAAI_CATALYST_BASE_URL"] before getting the token as it is used in the get_token method
                os.environ["RAGAAI_CATALYST_BASE_URL"] = RagaAICatalyst.BASE_URL
                RagaAICatalyst.get_token(force_refresh=True)
            except RequestException:
                logger.error("The provided base_url is not accessible. Please re-check the base_url.")
        else:
            # Get the token from the server
            RagaAICatalyst.get_token(force_refresh=True)

        # Set the API keys, if  available
        if self.api_keys:
            self._upload_keys()

    @staticmethod
    def _normalize_base_url(url):
        url = re.sub(
            r"(?<!:)//+", "/", url
        )  # Ignore the `://` part of URLs and remove extra // if any
        url = url.rstrip("/")  # To remove trailing slashes
        if not url.endswith("/api"):  # To ensure it ends with /api
            url = f"{url}/api"
        return url

    def _set_access_key_secret_key(self, access_key, secret_key):
        os.environ["RAGAAI_CATALYST_ACCESS_KEY"] = access_key
        os.environ["RAGAAI_CATALYST_SECRET_KEY"] = secret_key

        return access_key, secret_key

    def _upload_keys(self):
        """
        Uploads API keys to the server for the RagaAICatalyst.

        This function uploads the API keys stored in the `api_keys` attribute of the `RagaAICatalyst` object to the server. It sends a POST request to the server with the API keys in the request body. The request is authenticated using a bearer token obtained from the `RAGAAI_CATALYST_TOKEN` environment variable.

        Parameters:
            None

        Returns:
            None

        Raises:
            ValueError: If the `RAGAAI_CATALYST_ACCESS_KEY` or `RAGAAI_CATALYST_SECRET_KEY` environment variables are not set.

        Side Effects:
            - Sends a POST request to the server.
            - Prints "API keys uploaded successfully" if the request is successful.
            - Logs an error message if the request fails.

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

        try:
            start_time = time.time()
            endpoint = f"{RagaAICatalyst.BASE_URL}/v1/llm/secrets/upload"
            response = session_manager.make_request_with_retry(
                'POST',
                endpoint,
                headers=headers,
                json=json_data,
                timeout=RagaAICatalyst.TIMEOUT,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
            )
            if response.status_code == 200:
                print("API keys uploaded successfully")
            else:
                logger.error(f"Failed to upload API keys. Status: {response.status_code}")

        except (PoolError, MaxRetryError, NewConnectionError, RemoteDisconnected, ConnectionError, Timeout) as e:
            session_manager.handle_request_exceptions(e, "uploading API keys")
        except RequestException as e:
            session_manager.handle_request_exceptions(e, "uploading API keys")
        except Exception as e:
            logger.error(f"Unexpected error occurred while uploading API keys: {e}")

    def add_api_key(self, service: str, key: str):
        """Add or update an API key for a specific service."""
        self.api_keys[service] = key

    def get_api_key(self, service: str) -> Optional[str]:
        """Get the API key for a specific service."""
        return self.api_keys.get(service)

    # Token expiration time is now configurable via the token_expiry_time parameter
    # Default is 6 hours, but can be changed to 23 hours or any other value

    @staticmethod
    def _get_credentials() -> tuple[str, str]:
        """Get access key and secret key from instance or environment."""
        access_key = RagaAICatalyst.access_key or os.getenv(
            "RAGAAI_CATALYST_ACCESS_KEY"
        )
        secret_key = RagaAICatalyst.secret_key or os.getenv(
            "RAGAAI_CATALYST_SECRET_KEY"
        )
        return access_key, secret_key

    @staticmethod
    def _refresh_token_async():
        """Refresh token in background thread."""
        try:
            RagaAICatalyst.get_token(force_refresh=True)
        except Exception as e:
            logger.error(f"Background token refresh failed: {str(e)}")

    @staticmethod
    def _schedule_token_refresh():
        """Schedule a token refresh to happen 20 seconds before expiration."""
        if not RagaAICatalyst._token_expiry:
            return

        # Calculate when to refresh (20 seconds before expiration)
        current_time = time.time()
        refresh_buffer = min(
            20, RagaAICatalyst.TOKEN_EXPIRY_TIME * 0.05
        )  # 20 seconds or 5% of expiry time, whichever is smaller
        time_until_refresh = max(
            RagaAICatalyst._token_expiry - current_time - refresh_buffer, 1
        )  # At least 1 second

        def delayed_refresh():
            # Sleep until it's time to refresh
            time.sleep(time_until_refresh)
            logger.debug("Scheduled token refresh triggered")
            RagaAICatalyst._refresh_token_async()

        # Start a new thread for the delayed refresh
        if (
            not RagaAICatalyst._refresh_thread
            or not RagaAICatalyst._refresh_thread.is_alive()
        ):
            RagaAICatalyst._refresh_thread = threading.Thread(target=delayed_refresh)
            RagaAICatalyst._refresh_thread.daemon = True
            RagaAICatalyst._refresh_thread.start()
            logger.debug(f"Token refresh scheduled in {time_until_refresh:.1f} seconds")

    @staticmethod
    def get_token(force_refresh=True) -> Union[str, None]:
        """
        Retrieves or refreshes a token using the provided credentials.

        Args:
            force_refresh (bool): If True, forces a token refresh regardless of expiration.

        Returns:
            - A string representing the token if successful.
            - None if credentials are not set or if there is an error.
        """
        with RagaAICatalyst._token_refresh_lock:
            current_token = os.getenv("RAGAAI_CATALYST_TOKEN")
            current_time = time.time()

            # Check if we need to refresh the token
            if (
                not force_refresh
                and current_token
                and RagaAICatalyst._token_expiry
                and current_time < RagaAICatalyst._token_expiry
            ):
                return current_token

            access_key, secret_key = RagaAICatalyst._get_credentials()
            if not access_key or not secret_key:
                logger.error("Access key or secret key is not set")
                return None

            headers = {"Content-Type": "application/json"}
            json_data = {"accessKey": access_key, "secretKey": secret_key}

            start_time = time.time()
            endpoint = f"{RagaAICatalyst.BASE_URL}/token"
            response = session_manager.make_request_with_retry(
                'POST',
                endpoint,
                headers=headers,
                json=json_data,
                timeout=RagaAICatalyst.TIMEOUT,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
            )

            # Handle specific status codes before raising an error
            if response.status_code == 400:
                token_response = response.json()
                if token_response.get("message") == "Please enter valid credentials":
                    logger.error(
                        "Authentication failed. Invalid credentials provided. Please check your Access key and Secret key. \nTo view or create new keys, navigate to Settings -> Authenticate in the RagaAI Catalyst dashboard."
                    )
                    return None

            # Parse JSON response once
            token_response = response.json()

            # Validate response structure
            if not isinstance(token_response, dict):
                logger.error("Invalid response format - expected JSON object")
                return None

            if not token_response.get("success", False):
                logger.error(
                    "Token retrieval was not successful: %s",
                    token_response.get("message", "Unknown error"),
                )
                return None

            # Extract and validate token
            token = token_response.get("data", {}).get("token")
            if not token:
                logger.error("Token not found in response data")
                return None

            # Set environment and schedule refresh
            os.environ["RAGAAI_CATALYST_TOKEN"] = token
            RagaAICatalyst._token_expiry = (
                time.time() + RagaAICatalyst.TOKEN_EXPIRY_TIME
            )
            logger.debug(
                f"Token refreshed successfully. Next refresh in {RagaAICatalyst.TOKEN_EXPIRY_TIME / 3600:.1f} hours"
            )

            # Schedule token refresh 20 seconds before expiration
            RagaAICatalyst._schedule_token_refresh()

            return token

    def ensure_valid_token(self) -> Union[str, None]:
        """
        Ensures a valid token is available, with different handling for missing token vs expired token:
        - Missing token: Synchronous retrieval (fail fast)
        - Expired token: Synchronous refresh (since token is needed immediately)

        Returns:
            - A string representing the valid token if successful.
            - None if unable to obtain a valid token.
        """
        current_token = os.getenv("RAGAAI_CATALYST_TOKEN")
        current_time = time.time()

        # Case 1: No token - synchronous retrieval (fail fast)
        if not current_token:
            return self.get_token(force_refresh=True)

        # Case 2: Token expired - synchronous refresh (since we need a valid token now)
        if not self._token_expiry or current_time >= self._token_expiry:
            logger.info("Token expired, refreshing synchronously")
            return self.get_token(force_refresh=True)

        # Case 3: Token valid but approaching expiry (less than 10% of lifetime remaining)
        # Start background refresh but return current token
        token_remaining_time = self._token_expiry - current_time
        if token_remaining_time < (RagaAICatalyst.TOKEN_EXPIRY_TIME * 0.1):
            if not self._refresh_thread or not self._refresh_thread.is_alive():
                logger.info("Token approaching expiry, starting background refresh")
                self._refresh_thread = threading.Thread(
                    target=self._refresh_token_async
                )
                self._refresh_thread.daemon = True
                self._refresh_thread.start()

        # Return current token (which is valid)
        return current_token

    def get_auth_header(self) -> Dict[str, str]:
        """
        Returns a dictionary containing the Authorization header with a valid token.
        This method should be used instead of directly accessing os.getenv("RAGAAI_CATALYST_TOKEN").

        Returns:
            - A dictionary with the Authorization header if successful.
            - An empty dictionary if no valid token could be obtained.
        """
        token = self.ensure_valid_token()
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}

    def project_use_cases(self):
        try:
            headers = self.get_auth_header()
            start_time = time.time()
            endpoint = f"{RagaAICatalyst.BASE_URL}/v2/llm/usecase"
            response = session_manager.make_request_with_retry(
                'GET',
                endpoint,
                headers=headers,
                timeout=self.TIMEOUT
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [GET] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
            )

            # Check for successful status codes
            if response.status_code not in [200, 201]:
                logger.error(f"Failed to retrieve use cases - Status: {response.status_code}")
                return []

            # Parse JSON response once
            try:
                response_data = response.json()
            except ValueError as e:
                logger.error(f"Invalid JSON response from use cases endpoint: {e}")
                return []

            # Validate response structure and extract use cases
            if not isinstance(response_data, dict):
                logger.error("Invalid response format - expected JSON object")
                return []

            if not response_data.get("success", False):
                logger.error(f"Use cases retrieval was not successful: {response_data.get('message', 'Unknown error')}")
                return []

            usecase = response_data.get("data", {}).get("usecase", [])
            if not isinstance(usecase, list):
                logger.error("Invalid use cases format - expected list")
                return []

            return usecase

        except (PoolError, MaxRetryError, NewConnectionError, RemoteDisconnected, ConnectionError, Timeout) as e:
            session_manager.handle_request_exceptions(e, "retrieving project use cases")
            return []
        except RequestException as e:
            session_manager.handle_request_exceptions(e, "retrieving project use cases")
            return []
        except Exception as e:
            logger.error(f"Unexpected error occurred while retrieving project use cases: {e}")
            return []

    def create_project(self, project_name, usecase="Q/A", type="llm"):
        """
        Creates a project with the given project_name, type, and description.

        Parameters:
            project_name (str): The name of the project to be created.
            type (str, optional): The type of the project. Defaults to "llm".
            description (str, optional): Description of the project. Defaults to "".

        Returns:
            str: A message indicating the success or failure of the project creation.
        """
        # Check if the project already exists
        existing_projects = self.list_projects()
        if project_name in existing_projects:
            logger.error(
                f"Project name '{project_name}' already exists. Please choose a different name."
            )

        usecase_list = self.project_use_cases()
        if usecase not in usecase_list:
            logger.error(f"Select a valid usecase from {usecase_list}")

        json_data = {"name": project_name, "type": type, "usecase": usecase}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
        }
        try:
            start_time = time.time()
            endpoint = f"{RagaAICatalyst.BASE_URL}/v2/llm/project"
            response = session_manager.make_request_with_retry(
                'POST',
                endpoint,
                headers=headers,
                json=json_data,
                timeout=self.TIMEOUT,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
            )

            # Handle successful status codes first
            if response.status_code in [200, 201]:
                # Parse JSON response once
                try:
                    response_data = response.json()
                except ValueError as e:
                    logger.error(f"Invalid JSON response from create project endpoint: {e}")
                    return "Failed to create project: Invalid response format"

                # Validate response structure
                if not isinstance(response_data, dict):
                    logger.error("Invalid response format - expected JSON object")
                    return "Failed to create project: Invalid response format"

                if not response_data.get("success", False):
                    logger.error(f"Project creation was not successful: {response_data.get('message', 'Unknown error')}")
                    return f"Failed to create project: {response_data.get('message', 'Unknown error')}"

                project_name_response = response_data.get("data", {}).get("name")
                if not project_name_response:
                    logger.error("Project name not found in response")
                    return "Failed to create project: Project name not returned"

                success_message = f"Project Created Successfully with name {project_name_response} & usecase {usecase}"
                print(success_message)
                return success_message

            # Handle 401 status code (authentication error)
            elif response.status_code == 401:
                logger.warning("Received 401 error while creating project. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                headers["Authorization"] = f"Bearer {token}"

                start_time = time.time()
                response = session_manager.make_request_with_retry(
                    'POST',
                    endpoint,
                    headers=headers,
                    json=json_data,
                    timeout=self.TIMEOUT,
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"API Call: [POST] {endpoint} (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
                )

                if response.status_code in [200, 201]:
                    # Parse JSON response once
                    try:
                        response_data = response.json()
                    except ValueError as e:
                        logger.error(f"Invalid JSON response from create project endpoint after token refresh: {e}")
                        return "Failed to create project: Invalid response format"

                    # Validate response structure
                    if not isinstance(response_data, dict):
                        logger.error("Invalid response format after token refresh - expected JSON object")
                        return "Failed to create project after token refresh: Invalid response format"

                    if not response_data.get("success", False):
                        logger.error(f"Project creation was not successful after token refresh: {response_data.get('message', 'Unknown error')}")
                        return f"Failed to create project after token refresh: {response_data.get('message', 'Unknown error')}"

                    project_name_response = response_data.get("data", {}).get("name")
                    if not project_name_response:
                        logger.error("Project name not found in response after token refresh")
                        return "Failed to create project after token refresh: Project name not returned"

                    success_message = f"Project Created Successfully with name {project_name_response} & usecase {usecase}"
                    print(success_message)
                    return success_message
                else:
                    logger.error(f"Error while creating project after token refresh: Status {response.status_code}")
                    return f"Failed to create project after token refresh: Status {response.status_code}"

            # Handle all other status codes explicitly
            else:
                logger.error(f"Failed to create project - Status: {response.status_code}")
                try:
                    error_message = response.json().get('message', 'Unknown error')
                except (ValueError, AttributeError):
                    error_message = f'HTTP {response.status_code} error'
                return f"Failed to create project: {error_message}"

        except (PoolError, MaxRetryError, NewConnectionError, RemoteDisconnected, ConnectionError, Timeout) as e:
            session_manager.handle_request_exceptions(e, "creating project")
            return "Failed to create project: Connection error"
        except RequestException as e:
            session_manager.handle_request_exceptions(e, "creating project")
            return "Failed to create project: Request error"
        except Exception as general_err1:
            logger.error(
                "Unexpected error while creating project: %s", str(general_err1)
            )
            return "An unexpected error occurred while creating the project"

    def get_project_id(self, project_name):
        pass

    def list_projects(self, num_projects=99999):
        """
        Retrieves a list of projects with the specified number of projects.

        Parameters:
            num_projects (int, optional): Number of projects to retrieve. Defaults to 100.

        Returns:
            list: A list of project names retrieved successfully.
        """
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
        }
        try:
            start_time = time.time()
            endpoint = f"{RagaAICatalyst.BASE_URL}/v2/llm/projects?size={num_projects}"
            response = session_manager.make_request_with_retry(
                'GET',
                endpoint,
                headers=headers,
                timeout=self.TIMEOUT,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [GET] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
            )
            logger.debug("Projects list retrieved successfully")

            # Handle successful status codes first
            if response.status_code in [200, 201]:
                # Parse JSON response once
                try:
                    response_data = response.json()
                except ValueError as e:
                    logger.error(f"Invalid JSON response from list projects endpoint: {e}")
                    return "Failed to list projects: Invalid response format"

                # Validate response structure
                if not isinstance(response_data, dict):
                    logger.error("Invalid response format - expected JSON object")
                    return "Failed to list projects: Invalid response format"

                if not response_data.get("success", False):
                    logger.error(f"Projects listing was not successful: {response_data.get('message', 'Unknown error')}")
                    return f"Failed to list projects: {response_data.get('message', 'Unknown error')}"

                content = response_data.get("data", {}).get("content", [])
                if not isinstance(content, list):
                    logger.error("Invalid projects format - expected list")
                    return "Failed to list projects: Invalid data format"

                project_list = [
                    project.get("name") for project in content
                    if isinstance(project, dict) and project.get("name")
                ]

                return project_list

            # Handle 401 status code (authentication error)
            elif response.status_code == 401:
                logger.warning("Received 401 error while listing projects. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                headers["Authorization"] = f"Bearer {token}"

                start_time = time.time()
                response = session_manager.make_request_with_retry(
                    'GET',
                    f"{RagaAICatalyst.BASE_URL}/v2/llm/projects?size={num_projects}",
                    headers=headers,
                    timeout=self.TIMEOUT,
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"API Call: [GET] {endpoint} (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
                )

                if response.status_code in [200, 201]:
                    # Parse JSON response once
                    try:
                        response_data = response.json()
                    except ValueError as e:
                        logger.error(f"Invalid JSON response from list projects endpoint after token refresh: {e}")
                        return "Failed to list projects after token refresh: Invalid response format"

                    # Validate response structure
                    if not isinstance(response_data, dict):
                        logger.error("Invalid response format after token refresh - expected JSON object")
                        return "Failed to list projects after token refresh: Invalid response format"

                    if not response_data.get("success", False):
                        logger.error(f"Projects listing was not successful after token refresh: {response_data.get('message', 'Unknown error')}")
                        return f"Failed to list projects after token refresh: {response_data.get('message', 'Unknown error')}"

                    content = response_data.get("data", {}).get("content", [])
                    if not isinstance(content, list):
                        logger.error("Invalid projects format after token refresh - expected list")
                        return "Failed to list projects after token refresh: Invalid data format"

                    project_list = [
                        project.get("name") for project in content
                        if isinstance(project, dict) and project.get("name")
                    ]

                    return project_list
                else:
                    logger.error(f"Error while listing projects after token refresh: Status {response.status_code}")
                    return f"Failed to list projects after token refresh: Status {response.status_code}"

            # Handle all other status codes explicitly
            else:
                logger.error(f"Failed to list projects - Status: {response.status_code}")
                try:
                    error_message = response.json().get('message', 'Unknown error')
                except (ValueError, AttributeError):
                    error_message = f'HTTP {response.status_code} error'
                return f"Failed to list projects: {error_message}"

        except (PoolError, MaxRetryError, NewConnectionError, RemoteDisconnected, ConnectionError, Timeout) as e:
            session_manager.handle_request_exceptions(e, "listing projects")
            return "Failed to list projects: Connection error"
        except RequestException as e:
            session_manager.handle_request_exceptions(e, "listing projects")
            return "Failed to list projects: Request error"
        except Exception as general_err2:
            logger.error(
                "Unexpected error while listing projects: %s", str(general_err2)
            )
            return "An unexpected error occurred while listing projects"

    def list_metrics(self):
        return RagaAICatalyst.list_metrics()

    @staticmethod
    def list_metrics():
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
        }
        try:
            start_time = time.time()
            endpoint = f"{RagaAICatalyst.BASE_URL}/v1/llm/llm-metrics"
            response = session_manager.make_request_with_retry(
                'GET',
                endpoint,
                headers=headers,
                timeout=RagaAICatalyst.TIMEOUT,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"API Call: [GET] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
            )
            logger.debug("Metrics list retrieved successfully")

            # Handle successful status codes first
            if response.status_code in [200, 201]:
                # Parse JSON response once
                try:
                    response_data = response.json()
                except ValueError as e:
                    logger.error(f"Invalid JSON response from list metrics endpoint: {e}")
                    return []

                # Validate response structure
                if not isinstance(response_data, dict):
                    logger.error("Invalid response format - expected JSON object")
                    return []

                if not response_data.get("success", False):
                    logger.error(f"Metrics listing was not successful: {response_data.get('message', 'Unknown error')}")
                    return []

                metrics = response_data.get("data", {}).get("metrics", [])
                if not isinstance(metrics, list):
                    logger.error("Invalid metrics format - expected list")
                    return []

                sub_metrics = [
                    metric.get("name") for metric in metrics 
                    if isinstance(metric, dict) and metric.get("name")
                ]
                return sub_metrics

            # Handle 401 status code (authentication error)
            elif response.status_code == 401:
                logger.warning("Received 401 error while listing metrics. Attempting to refresh token.")
                token = RagaAICatalyst.get_token(force_refresh=True)
                headers["Authorization"] = f"Bearer {token}"

                start_time = time.time()
                response = session_manager.make_request_with_retry(
                    'GET',
                    f"{RagaAICatalyst.BASE_URL}/v1/llm/llm-metrics",
                    headers=headers,
                    timeout=RagaAICatalyst.TIMEOUT,
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"API Call: [GET] {endpoint} (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
                )

                if response.status_code in [200, 201]:
                    # Parse JSON response once
                    try:
                        response_data = response.json()
                    except ValueError as e:
                        logger.error(f"Invalid JSON response from list metrics endpoint after token refresh: {e}")
                        return []

                    # Validate response structure
                    if not isinstance(response_data, dict):
                        logger.error("Invalid response format after token refresh - expected JSON object")
                        return []

                    if not response_data.get("success", False):
                        logger.error(f"Metrics listing was not successful after token refresh: {response_data.get('message', 'Unknown error')}")
                        return []

                    metrics = response_data.get("data", {}).get("metrics", [])
                    if not isinstance(metrics, list):
                        logger.error("Invalid metrics format after token refresh - expected list")
                        return []

                    sub_metrics = [
                        metric.get("name") for metric in metrics
                        if isinstance(metric, dict) and metric.get("name")
                    ]
                    return sub_metrics
                else:
                    logger.error(f"Error while listing metrics after token refresh: Status {response.status_code}")
                    return []

            # Handle all other status codes explicitly
            else:
                logger.error(f"Failed to list metrics - Status: {response.status_code}")
                try:
                    error_message = response.json().get('message', 'Unknown error')
                except (ValueError, AttributeError):
                    error_message = f'HTTP {response.status_code} error'
                logger.error(f"Metrics listing error: {error_message}")
                return []

        except (PoolError, MaxRetryError, NewConnectionError, RemoteDisconnected, ConnectionError, Timeout) as e:
            session_manager.handle_request_exceptions(e, "listing metrics")
            return []
        except RequestException as e:
            session_manager.handle_request_exceptions(e, "listing metrics")
            return []
        except Exception as e:
            logger.error(f"Unexpected error occurred while listing metrics: {e}")
            return []
