import os
from ragaai_catalyst.session_manager import session_manager
import json
import re
import uuid
from typing import Optional, List, Dict, Any, Tuple
from .ragaai_catalyst import RagaAICatalyst
import logging
import time
from urllib3.exceptions import PoolError, MaxRetryError, NewConnectionError
from requests.exceptions import ConnectionError, Timeout, RequestException
from http.client import RemoteDisconnected
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages prompts for a specific project, providing CRUD operations and version management.
    
    This class handles authentication, token refresh, and API communication for prompt operations.
    """
    
    DEFAULT_TIMEOUT = 10
    DEFAULT_MAX_PROJECTS = 99999
    SUCCESS_STATUS_CODES = frozenset([200, 201])
    MAX_CONCURRENT_VERSION_FETCHES = 5
    
    _VARIABLE_PATTERN = re.compile(r'\{\{(.*?)\}\}')
    
    VALID_MODEL_PREFIXES = frozenset([
        "openai/", "azure/", "bedrock/", "gemini/", "anthropic/", "vertex_ai/"
    ])
    
    VALID_ROLES = frozenset(['system', 'user', 'assistant'])

    def __init__(self, project_name: str, timeout: Optional[int] = None, max_projects: Optional[int] = None):
        """
        Initialize the PromptManager with a project name.

        Args:
            project_name: The name of the project (non-empty string)
            timeout: Optional timeout for API requests in seconds
            max_projects: Optional maximum number of projects to fetch

        Raises:
            ValueError: If project_name is invalid or not found
            ConnectionError: If unable to connect to the API
            RuntimeError: If initialization fails for other reasons
        """
        if not project_name or not isinstance(project_name, str) or not project_name.strip():
            raise ValueError("Project name must be a non-empty string")
        
        self.project_name = project_name.strip()
        base_api_url = os.getenv("RAGAAI_CATALYST_BASE_URL", "https://catalyst.raga.ai/api")
        self.base_url = f"{base_api_url}/playground/prompt"
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.size = max_projects or self.DEFAULT_MAX_PROJECTS
        self.project_id: Optional[str] = None
        self.headers: Dict[str, str] = {}
        
        self._initialize_project()

    def _initialize_project(self) -> None:
        """
        Initialize project by fetching project list and finding project ID.
        
        Raises:
            ValueError: If project is not found
            ConnectionError: If API request fails
            RuntimeError: If response parsing fails
        """
        try:
            base_api_url = os.getenv("RAGAAI_CATALYST_BASE_URL", "https://catalyst.raga.ai/api")
            response = self._make_api_request(
                "GET",
                f"{base_api_url}/v2/llm/projects?size={self.size}",
                headers={"Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}'}
            )
            
            if response is None:
                raise ConnectionError("Failed to fetch project list from API")
            
            data = response.json()
            projects = data.get("data", {}).get("content", [])
            
            if not projects:
                raise RuntimeError("No projects found in the API response")
            
            project_list = [project["name"] for project in projects]
            
            if self.project_name not in project_list:
                raise ValueError(
                    f"Project '{self.project_name}' not found. "
                    f"Available projects: {', '.join(project_list[:10])}"
                    f"{'...' if len(project_list) > 10 else ''}"
                )
            
            matching_projects = [p["id"] for p in projects if p["name"] == self.project_name]
            
            if not matching_projects:
                raise RuntimeError(f"Project '{self.project_name}' found but has no ID")
            
            self.project_id = matching_projects[0]
            
            self.headers = {
                "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                "X-Project-Id": str(self.project_id)
            }
            
            logger.info(f"PromptManager initialized successfully for project '{self.project_name}' (ID: {self.project_id})")
            
        except (KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Error parsing project list response: {str(e)}") from e
        except Exception as e:
            if isinstance(e, (ValueError, ConnectionError, RuntimeError)):
                raise
            raise RuntimeError(f"Unexpected error during project initialization: {str(e)}") from e

    def _make_api_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Centralized API request handler with automatic 401 retry and token refresh.
        
        This method handles:
        - Making the initial request
        - Automatic token refresh on 401 errors
        - Comprehensive error handling
        - Request timing logging
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            url: Full API endpoint URL
            headers: Optional headers (uses self.headers if not provided)
            **kwargs: Additional arguments passed to session_manager.make_request_with_retry
            
        Returns:
            Response object if successful, None on failure
        """
        request_headers = headers or self.headers
        
        try:
            start_time = time.time()
            response = session_manager.make_request_with_retry(
                method,
                url,
                headers=request_headers,
                timeout=self.timeout,
                **kwargs
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [{method}] {url} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
            
            if response.status_code in self.SUCCESS_STATUS_CODES:
                return response
            
            if response.status_code == 401:
                logger.warning(f"Received 401 for [{method}] {url}, refreshing token...")
                token = RagaAICatalyst.get_token(force_refresh=True)
                
                new_headers = request_headers.copy()
                new_headers["Authorization"] = f"Bearer {token}"
                
                start_time = time.time()
                response = session_manager.make_request_with_retry(
                    method,
                    url,
                    headers=new_headers,
                    timeout=self.timeout,
                    **kwargs
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [{method}] {url} (retry) | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in self.SUCCESS_STATUS_CODES:
                    if headers is None:
                        self.headers["Authorization"] = f"Bearer {token}"
                    logger.info(f"Request successful after token refresh: [{method}] {url}")
                    return response
                else:
                    logger.error(f"Request failed after token refresh: [{method}] {url} | Status: {response.status_code}")
                    return None
            else:
                logger.error(f"HTTP {response.status_code} error for [{method}] {url}")
                return None
                
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, f"{method} {url}")
            return None
        except RequestException as e:
            logger.error(f"Request error for [{method}] {url}: {str(e)}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing response from [{method}] {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for [{method}] {url}: {str(e)}", exc_info=True)
            return None

    def list_prompts(self) -> List[str]:
        """
        List all prompts in the project.
        
        Returns:
            List of prompt names, empty list on failure
        """
        if not self.project_id:
            logger.error("PromptManager not properly initialized, cannot list prompts")
            return []
        
        response = self._make_api_request("GET", self.base_url)
        
        if response:
            try:
                data = response.json()
                return [prompt["name"] for prompt in data.get("data", [])]
            except (KeyError, json.JSONDecodeError, TypeError) as e:
                logger.error(f"Error parsing prompts list: {str(e)}")
                return []
        
        return []
    
    def get_prompt(self, prompt_name: str, version: Optional[str] = None) -> Optional['PromptObject']:
        """
        Fetch a prompt by name and optional version.
        
        Args:
            prompt_name: Name of the prompt (non-empty string)
            version: Optional version identifier
            
        Returns:
            PromptObject if found, None otherwise
            
        Raises:
            TypeError: If prompt_name is not a string
            ValueError: If prompt_name is empty or contains invalid characters
        """
        self._validate_prompt_name(prompt_name)
        
        try:
            prompt_list = self.list_prompts()
        except Exception as e:
            logger.error(f"Error fetching prompt list: {str(e)}")
            return None

        if prompt_name not in prompt_list:
            logger.error(f"Prompt '{prompt_name}' not found. Available prompts: {', '.join(prompt_list[:5])}")
            return None

        if version:
            try:
                prompt_versions = self.list_prompt_versions(prompt_name)
            except Exception as e:
                logger.error(f"Error fetching prompt versions: {str(e)}")
                return None

            if version not in prompt_versions:
                logger.error(f"Version '{version}' not found for prompt '{prompt_name}'")
                return None

        return Prompt.get_prompt(self.base_url, self.headers, prompt_name, version, self._make_api_request)

    def list_prompt_versions(self, prompt_name: str) -> Dict[str, List[Dict[str, str]]]:
        """
        List all versions of a prompt with concurrent fetching for performance.
        
        Args:
            prompt_name: Name of the prompt
            
        Returns:
            Dictionary mapping version names to their text fields
        """
        self._validate_prompt_name(prompt_name)
        
        try:
            prompt_list = self.list_prompts()
        except Exception as e:
            logger.error(f"Error fetching prompt list: {str(e)}")
            return {}

        if prompt_name not in prompt_list:
            logger.error(f"Prompt '{prompt_name}' not found")
            return {}
        
        return Prompt.list_prompt_versions(
            self.base_url,
            self.headers,
            prompt_name,
            self._make_api_request,
            self.MAX_CONCURRENT_VERSION_FETCHES
        )

    def _create_prompt(self, prompt_name: str, directory: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Internal method to create a new prompt.
        
        Args:
            prompt_name: Name for the new prompt
            directory: Optional directory/folder for organization
            
        Returns:
            API response data if successful, None otherwise
        """
        self._validate_prompt_name(prompt_name)

        try:
            existing_prompts = self.list_prompts()
            if prompt_name in existing_prompts:
                logger.info(f"Prompt '{prompt_name}' already exists, skipping creation")
                return None
        except Exception as e:
            logger.error(f"Error checking existing prompts for '{prompt_name}': {str(e)}")
            return None

        payload = {
            "name": prompt_name,
            "directory": directory
        }

        response = self._make_api_request("POST", self.base_url, json=payload)
        
        if response:
            try:
                return response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing create prompt response: {str(e)}")
                return None
        
        return None

    def delete_prompt(self, prompt_name: str) -> Dict[str, Any]:
        """
        Delete a prompt by name.
        
        Args:
            prompt_name: Name of the prompt to delete
            
        Returns:
            Dictionary with 'success', 'message', and 'prompt_name' keys
        """
        try:
            self._validate_prompt_name(prompt_name)
        except (TypeError, ValueError) as e:
            return {
                'success': False,
                'message': str(e),
                'prompt_name': prompt_name
            }

        try:
            existing_prompts = self.list_prompts()
            if prompt_name not in existing_prompts:
                error_msg = f"Prompt '{prompt_name}' not found"
                logger.error(error_msg)
                return {
                    'success': False,
                    'message': error_msg,
                    'prompt_name': prompt_name
                }
        except Exception as e:
            return {
                'success': False,
                'message': f"Error checking existing prompts: {str(e)}",
                'prompt_name': prompt_name
            }

        delete_url = f"{self.base_url}/{quote(prompt_name, safe='')}"
        response = self._make_api_request("DELETE", delete_url)
        
        if response:
            try:
                logger.info(f"Prompt '{prompt_name}' deleted successfully")
                return response.json()
            except json.JSONDecodeError as e:
                return {
                    'success': False,
                    'message': f"Error parsing response: {str(e)}",
                    'prompt_name': prompt_name
                }
        
        return {
            'success': False,
            'message': "Failed to delete prompt",
            'prompt_name': prompt_name
        }

    def set_version_as_default(self, version_id: int) -> Dict[str, Any]:
        """
        Set a specific version as the default version.
        
        Args:
            version_id: ID of the version to set as default
            
        Returns:
            Dictionary with operation results
        """
        if not version_id or not isinstance(version_id, int):
            error_msg = "Version ID must be a valid integer"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'version_id': version_id
            }

        default_url = f"{self.base_url}/version/{version_id}/default"
        response = self._make_api_request("PUT", default_url)
        
        if response:
            try:
                logger.info(f"Version '{version_id}' set as default successfully")
                return response.json()
            except json.JSONDecodeError as e:
                return {
                    'success': False,
                    'message': f"Error parsing response: {str(e)}",
                    'version_id': version_id
                }
        
        return {
            'success': False,
            'message': "Failed to set version as default",
            'version_id': version_id
        }

    def create_or_update_prompt(
        self,
        prompt_name: str,
        text_fields: List[Dict[str, str]],
        model: str,
        message: Optional[str] = None,
        directory: Optional[str] = None,
        is_default: bool = False,
        variable_specs: Optional[List[Dict[str, Any]]] = None,
        metrics_specs: Optional[List[Dict[str, Any]]] = None,
        model_parameters: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create or update a prompt with a new version.
        
        Args:
            prompt_name: Name of the prompt
            text_fields: List of message dictionaries with 'role' and 'content'
            model: Model identifier in format 'provider/model-name'
            message: Optional commit message
            directory: Optional directory for organization
            is_default: Whether to set this version as default
            variable_specs: Optional variable specifications
            metrics_specs: Optional metrics specifications
            model_parameters: Optional model parameters
            
        Returns:
            Dictionary with operation results
        """
        self._create_prompt(prompt_name=prompt_name, directory=directory)

        return self._save_prompt_version(
            prompt_name=prompt_name,
            text_fields=text_fields,
            message=message,
            is_default=is_default,
            model=model,
            variable_specs=variable_specs,
            metrics_specs=metrics_specs,
            model_parameters=model_parameters
        )

    @classmethod
    def _extract_variables_from_content(cls, content: str) -> List[str]:
        """
        Extract template variables from a content string.
        
        Args:
            content: String containing template variables in {{variable}} format
            
        Returns:
            List of variable names found in the content
        """
        matches = cls._VARIABLE_PATTERN.findall(content)
        return [match.strip() for match in matches if '"' not in match]

    def _extract_variables_from_text_fields(self, text_fields: List[Dict[str, str]]) -> List[str]:
        """
        Extract all unique variables from text fields.
        
        Args:
            text_fields: List of text field dictionaries
            
        Returns:
            Sorted list of unique variable names
        """
        variables = set()
        for field in text_fields:
            content = field.get('content', '')
            variables.update(self._extract_variables_from_content(content))
        return sorted(variables)

    def _get_supported_models(self, provider_name: str) -> List[str]:
        """
        Fetch list of supported models for a given provider.
        
        Args:
            provider_name: Name of the LLM provider
            
        Returns:
            List of model names, empty list on failure
        """
        base_api_url = os.getenv("RAGAAI_CATALYST_BASE_URL", "https://catalyst.raga.ai/api")
        models_url = f"{base_api_url}/v1/llm/models"
        response = self._make_api_request("POST", models_url, json={"providerName": provider_name})
        
        if response:
            try:
                data = response.json()
                if data.get("success") and "data" in data:
                    return [model["name"] for model in data["data"]]
            except (KeyError, json.JSONDecodeError, TypeError) as e:
                logger.error(f"Error parsing supported models response: {str(e)}")
        
        return []

    def _get_model_parameters(self, provider_name: str, model_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch model parameters for a specific model.
        
        Args:
            provider_name: Name of the provider
            model_name: Name of the model
            
        Returns:
            List of parameter dictionaries, None on failure
        """
        base_api_url = os.getenv("RAGAAI_CATALYST_BASE_URL", "https://catalyst.raga.ai/api")
        params_url = f"{base_api_url}/playground/providers/models/parameters/list"
        response = self._make_api_request(
            "POST",
            params_url,
            json={"providerName": provider_name, "modelName": model_name}
        )
        
        if response:
            try:
                data = response.json()
                if data.get("success") and "data" in data:
                    parameters = []
                    for param in data["data"]:
                        param_dict = {
                            "name": param["name"],
                            "value": param["value"],
                            "type": param["type"],
                            "minRange": param.get("minRange"),
                            "maxRange": param.get("maxRange")
                        }
                        parameters.append(param_dict)
                    return parameters
            except (KeyError, json.JSONDecodeError, TypeError) as e:
                logger.error(f"Error parsing model parameters response: {str(e)}")
        
        return None

    def _save_prompt_version(
        self,
        prompt_name: str,
        text_fields: List[Dict[str, str]],
        model: str,
        message: Optional[str] = None,
        is_default: bool = False,
        variable_specs: Optional[List[Dict[str, Any]]] = None,
        metrics_specs: Optional[List[Dict[str, Any]]] = None,
        model_parameters: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Internal method to save a new prompt version.
        
        Performs comprehensive validation and saves the prompt version.
        
        Args:
            prompt_name: Name of the prompt
            text_fields: List of message dictionaries
            model: Model identifier
            message: Optional commit message
            is_default: Whether to set as default
            variable_specs: Optional variable specifications
            metrics_specs: Optional metrics specifications
            model_parameters: Optional model parameters
            
        Returns:
            Dictionary with operation results
            
        Raises:
            ValueError: If any validation fails
        """
        self._validate_prompt_name(prompt_name)

        if not message or not message.strip():
            message = f"commit#{uuid.uuid4().hex[:8]}"
            logger.info(f"No message provided, auto-generated: {message}")

        self._validate_text_fields(text_fields)
        model = model.lower()
        provider_name, model_name = self._validate_and_parse_model(model)

        if metrics_specs is None:
            metrics_specs = []

        if variable_specs is None or variable_specs == []:
            extracted_variables = self._extract_variables_from_text_fields(text_fields)
            if extracted_variables:
                variable_specs = [
                    {
                        "name": var_name,
                        "type": "string",
                        "schema": "query"
                    }
                    for var_name in extracted_variables
                ]
                logger.info(f"Auto-extracted {len(extracted_variables)} variable(s): {', '.join(extracted_variables)}")
            else:
                variable_specs = []

        if model_parameters is None:
            fetched_params = self._get_model_parameters(provider_name, model_name)
            if not fetched_params:
                error_msg = (
                    f"Unable to fetch model parameters for '{model}'. "
                    f"Please verify the model name or provide model_parameters explicitly."
                )
                logger.error(error_msg)
                return {
                    'success': False,
                    'message': error_msg,
                    'prompt_name': prompt_name,
                    'version_id': None
                }
            model_parameters = fetched_params

        payload = {
            "isDefault": is_default,
            "message": message,
            "promptTemplate": {
                "textFields": text_fields,
                "variableSpecs": variable_specs,
                "modelSpecs": {
                    "parameters": model_parameters,
                    "model": model
                },
                "metricsSpecs": metrics_specs
            }
        }

        version_url = f"{self.base_url}/{quote(prompt_name, safe='')}/version"
        response = self._make_api_request("POST", version_url, json=payload)
        
        if response:
            try:
                response_data = response.json()
                success = response_data.get('success', True)
                message = response_data.get('message', 'Prompt version saved successfully')
                data = response_data.get('data', {})
                version_id = data.get('id')
                prompt_name_resp = data.get('name') or prompt_name

                result = {
                    'success': success,
                    'message': message,
                    'prompt_name': prompt_name_resp,
                    'version_id': version_id
                }

                logger.info(f"Prompt version saved successfully: {prompt_name} (version: {version_id})")
                return result
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Error parsing save response: {str(e)}")
                return {
                    'success': False,
                    'message': f"Error parsing response: {str(e)}",
                    'prompt_name': prompt_name,
                    'version_id': None
                }
        
        return {
            'success': False,
            'message': "Failed to save prompt version",
            'prompt_name': prompt_name,
            'version_id': None
        }

    @staticmethod
    def _validate_prompt_name(prompt_name: str) -> None:
        """
        Validate prompt name for security and correctness.
        
        Args:
            prompt_name: Name to validate
            
        Raises:
            TypeError: If not a string
            ValueError: If empty or contains invalid characters
        """
        if not isinstance(prompt_name, str):
            raise TypeError(f"prompt_name must be a string, not {type(prompt_name).__name__}")
        
        if not prompt_name.strip():
            raise ValueError("prompt_name cannot be empty")
        
        invalid_chars = ['/', '\\', '?', '&', '#', '%']
        if any(char in prompt_name for char in invalid_chars):
            raise ValueError(f"prompt_name contains invalid characters. Found: {[c for c in invalid_chars if c in prompt_name]}")

    def _validate_text_fields(self, text_fields: List[Dict[str, str]]) -> None:
        """
        Validate text fields structure and content.
        
        Args:
            text_fields: List of text field dictionaries to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(text_fields, list) or not text_fields:
            raise ValueError("text_fields must be a non-empty list")

        for idx, field in enumerate(text_fields):
            if not isinstance(field, dict) or 'role' not in field or 'content' not in field:
                raise ValueError(f"Each text_field must be a dict with 'role' and 'content' keys (field {idx})")

            role = field.get('role')
            if role not in self.VALID_ROLES:
                raise ValueError(
                    f"Invalid role '{role}' in text_field at index {idx}. "
                    f"Role must be one of: {', '.join(sorted(self.VALID_ROLES))}"
                )

            content = field.get('content')
            if not content or not isinstance(content, str) or content.strip() == "":
                raise ValueError(f"Content cannot be empty in text_field at index {idx}")

    def _validate_and_parse_model(self, model: str) -> Tuple[str, str]:
        """
        Validate model string and parse into provider and model name.
        
        Args:
            model: Model identifier in format 'provider/model-name'
            
        Returns:
            Tuple of (provider_name, model_name)
            
        Raises:
            ValueError: If model format is invalid or unsupported
        """
        if not model or not isinstance(model, str) or not model.strip():
            raise ValueError("Model must be a non-empty string")

        if "/" not in model:
            raise ValueError(
                f"Model must be in format 'provider/model-name' (e.g., 'openai/gpt-4o'). "
                f"Supported providers: {', '.join(sorted([p.rstrip('/') for p in self.VALID_MODEL_PREFIXES]))}"
            )

        if not any(model.startswith(prefix) for prefix in self.VALID_MODEL_PREFIXES):
            raise ValueError(
                f"Unsupported model provider in '{model}'. "
                f"Supported providers: {', '.join(sorted([p.rstrip('/') for p in self.VALID_MODEL_PREFIXES]))}"
            )

        provider_name = model.split('/')[0]
        model_name = model.split('/', 1)[1] if '/' in model else ""

        supported_models = self._get_supported_models(provider_name)
        if supported_models and model_name not in supported_models:
            raise ValueError(
                f"Model '{model_name}' is not supported by provider '{provider_name}'. "
                f"Supported models: {', '.join(supported_models[:10])}{'...' if len(supported_models) > 10 else ''}"
            )

        return provider_name, model_name


class Prompt:
    """
    Static helper class for prompt-related operations.
    
    All methods are static as this class maintains no state.
    """

    @staticmethod
    def list_prompts(url: str, headers: Dict[str, str], api_request_func) -> List[str]:
        """
        List all prompts using the provided API request function.
        
        Args:
            url: API endpoint URL
            headers: Request headers
            api_request_func: Function to make API requests
            
        Returns:
            List of prompt names
        """
        response = api_request_func("GET", url, headers=headers)
        
        if response:
            try:
                data = response.json()
                return [prompt["name"] for prompt in data.get("data", [])]
            except (KeyError, json.JSONDecodeError, TypeError) as e:
                logger.error(f"Error parsing prompts list: {str(e)}")
        
        return []

    @staticmethod
    def _get_response_by_version(
        base_url: str,
        headers: Dict[str, str],
        prompt_name: str,
        version: str,
        api_request_func
    ) -> Optional[Any]:
        """Fetch prompt response for a specific version."""
        url = f"{base_url}/version/{quote(prompt_name, safe='')}?version={quote(version, safe='')}"
        return api_request_func("GET", url, headers=headers)

    @staticmethod
    def _get_response(
        base_url: str,
        headers: Dict[str, str],
        prompt_name: str,
        api_request_func
    ) -> Optional[Any]:
        """Fetch prompt response for the latest version."""
        url = f"{base_url}/version/{quote(prompt_name, safe='')}"
        return api_request_func("GET", url, headers=headers)

    @staticmethod
    def _get_prompt_by_version(
        base_url: str,
        headers: Dict[str, str],
        prompt_name: str,
        version: str,
        api_request_func
    ) -> List[Dict[str, str]]:
        """Fetch prompt text fields for a specific version."""
        response = Prompt._get_response_by_version(
            base_url, headers, prompt_name, version, api_request_func
        )
        
        if response is None:
            return []
        
        try:
            data = response.json()
            return data["data"]["docs"][0]["textFields"]
        except (KeyError, json.JSONDecodeError, IndexError, TypeError) as e:
            logger.error(f"Error parsing prompt text for version {version}: {str(e)}")
            return []

    @staticmethod
    def get_prompt(
        base_url: str,
        headers: Dict[str, str],
        prompt_name: str,
        version: Optional[str],
        api_request_func
    ) -> Optional['PromptObject']:
        """
        Fetch a complete prompt object.
        
        Args:
            base_url: Base API URL
            headers: Request headers
            prompt_name: Name of the prompt
            version: Optional version identifier
            api_request_func: Function to make API requests
            
        Returns:
            PromptObject if successful, None otherwise
        """
        if version:
            response = Prompt._get_response_by_version(
                base_url, headers, prompt_name, version, api_request_func
            )
        else:
            response = Prompt._get_response(
                base_url, headers, prompt_name, api_request_func
            )

        if response is None:
            return None

        try:
            data = response.json()
            docs = data["data"]["docs"][0]
            prompt_text = docs["textFields"]
            prompt_parameters = docs["modelSpecs"]["parameters"]
            model = docs["modelSpecs"]["model"]
            return PromptObject(prompt_text, prompt_parameters, model)
        except (KeyError, json.JSONDecodeError, IndexError, TypeError) as e:
            logger.error(f"Error parsing prompt data: {str(e)}")
            return None

    @staticmethod
    def list_prompt_versions(
        base_url: str,
        headers: Dict[str, str],
        prompt_name: str,
        api_request_func,
        max_concurrent: int = 5
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        List all versions of a prompt with concurrent fetching for performance.
        
        This method uses ThreadPoolExecutor to fetch multiple versions concurrently,
        solving the N+1 query problem.
        
        Args:
            base_url: Base API URL
            headers: Request headers
            prompt_name: Name of the prompt
            api_request_func: Function to make API requests
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            Dictionary mapping version names to their text fields
        """
        url = f"{base_url}/{quote(prompt_name, safe='')}/version"
        response = api_request_func("GET", url, headers=headers)
        
        if not response:
            return {}
        
        try:
            data = response.json()
            version_names = [version["name"] for version in data.get("data", [])]
            
            if not version_names:
                return {}
            
            prompt_versions = {}
            
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                future_to_version = {
                    executor.submit(
                        Prompt._get_prompt_by_version,
                        base_url,
                        headers,
                        prompt_name,
                        version,
                        api_request_func
                    ): version
                    for version in version_names
                }
                
                for future in as_completed(future_to_version):
                    version = future_to_version[future]
                    try:
                        prompt_versions[version] = future.result()
                    except Exception as e:
                        logger.error(f"Error fetching version {version}: {str(e)}")
                        prompt_versions[version] = []
            
            return prompt_versions
            
        except (KeyError, json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error parsing prompt versions for '{prompt_name}': {str(e)}")
            return {}


class PromptObject:
    """
    Represents a prompt template with variables and model parameters.
    
    Provides methods to compile templates with variable substitution.
    """
    
    _VARIABLE_PATTERN = re.compile(r'\{\{(.*?)\}\}')

    def __init__(
        self,
        text: List[Dict[str, str]],
        parameters: List[Dict[str, Any]],
        model: str
    ):
        """
        Initialize a PromptObject.
        
        Args:
            text: List of message dictionaries with role and content
            parameters: List of model parameter dictionaries
            model: Model identifier
        """
        self.text = text
        self.parameters = parameters
        self.model = model

    def _extract_variable_from_content(self, content: str) -> List[str]:
        """
        Extract variables from a content string.
        
        Args:
            content: String containing template variables
            
        Returns:
            List of variable names
        """
        matches = self._VARIABLE_PATTERN.findall(content)
        return [match.strip() for match in matches if '"' not in match]

    def _add_variable_value_to_content(self, content: str, user_variables: Dict[str, str]) -> str:
        """
        Replace template variables in content with provided values.
        
        Args:
            content: Content string with variables
            user_variables: Dictionary of variable values
            
        Returns:
            Content with variables replaced
            
        Raises:
            ValueError: If any variable value is not a string
        """
        variables = self._extract_variable_from_content(content)
        
        for key, value in user_variables.items():
            if not isinstance(value, str):
                raise ValueError(f"Value for variable '{key}' must be a string, not {type(value).__name__}")
            if key in variables:
                content = content.replace(f"{{{{{key}}}}}", value)
        
        return content

    def compile(self, **kwargs) -> List[Dict[str, str]]:
        """
        Compile the prompt template with provided variable values.
        
        Args:
            **kwargs: Variable names and their string values
            
        Returns:
            New list of message dictionaries with variables replaced
            
        Raises:
            ValueError: If missing or extra variables provided, or if values aren't strings
        """
        required_variables = self.get_variables()
        provided_variables = set(kwargs.keys())

        missing_variables = [item for item in required_variables if item not in provided_variables]
        extra_variables = [item for item in provided_variables if item not in required_variables]

        if missing_variables:
            raise ValueError(f"Missing variable(s): {', '.join(missing_variables)}")
        if extra_variables:
            raise ValueError(f"Extra variable(s) provided: {', '.join(extra_variables)}")

        return [
            {
                "role": item["role"],
                "content": self._add_variable_value_to_content(item["content"], kwargs)
            }
            for item in self.text
        ]
    
    def get_variables(self) -> List[str]:
        """
        Get all variables used in the prompt template.
        
        Returns:
            List of unique variable names
        """
        try:
            variables = set()
            for item in self.text:
                content = item["content"]
                for var in self._extract_variable_from_content(content):
                    variables.add(var)
            return list(variables) if variables else []
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Error extracting variables: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in get_variables: {str(e)}")
            return []
    
    @staticmethod
    def _convert_value(value: Any, type_: str) -> Any:
        """
        Convert a value to the specified type.
        
        Args:
            value: Value to convert
            type_: Target type ('float', 'int', or other)
            
        Returns:
            Converted value
        """
        if type_ == "float":
            return float(value)
        elif type_ == "int":
            return int(value)
        return value

    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get model parameters as a dictionary.
        
        Returns:
            Dictionary of parameter names to values, plus 'model' key
        """
        parameters = {}
        for param in self.parameters:
            if "value" in param:
                parameters[param["name"]] = self._convert_value(param["value"], param["type"])
            else:
                parameters[param["name"]] = ""
        parameters["model"] = self.model
        return parameters    
    
    def get_prompt_content(self) -> List[Dict[str, str]]:
        """
        Get the raw prompt content (text fields).
        
        Returns:
            List of message dictionaries
        """
        return self.text
