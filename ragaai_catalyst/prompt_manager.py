import os
from .tracers.agentic_tracing.upload.session_manager import session_manager
import json
import re
import logging
import uuid
from typing import Optional, List, Dict, Any
from .ragaai_catalyst import RagaAICatalyst
import copy
from urllib3.exceptions import PoolError, MaxRetryError, NewConnectionError
from requests.exceptions import ConnectionError, Timeout, RequestException
from http.client import RemoteDisconnected

logger = logging.getLogger(__name__)

class PromptManager:
    NUM_PROJECTS = 100
    TIMEOUT = 10

    def __init__(self, project_name):
        """
        Initialize the PromptManager with a project name.

        Args:
            project_name (str): The name of the project.

        Raises:
            ValueError: If the project is not found.
        """
        self.project_name = project_name
        self.base_url = f"{RagaAICatalyst.BASE_URL}/playground/prompt"
        self.timeout = 10
        self.size = 99999 #Number of projects to fetch
        self.project_id = None
        self.headers = {}

        try:
            response = session_manager.make_request_with_retry(
                "GET",
                f"{RagaAICatalyst.BASE_URL}/v2/llm/projects?size={self.size}",
                headers={
                    "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            # logger.debug("Projects list retrieved successfully")

            data = response.json()
            project_list = [
                project["name"] for project in data["data"]["content"]
            ]
            matching_projects = [
                project["id"] for project in data["data"]["content"] if project["name"] == project_name
            ]

            if not matching_projects:
                logger.error(f"Project '{project_name}' not found. Please provide a valid project name.")
                return

            self.project_id = matching_projects[0]

        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "fetching project list")
            logger.error(f"Failed to fetch project list, PromptManager will have limited functionality")
            return
        except RequestException as e:
            logger.error(f"Error while fetching project list: {e}")
            logger.error(f"PromptManager will have limited functionality")
            return
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            logger.error(f"Error parsing project list: {str(e)}")
            return

        if not self.project_id:
            logger.error("Project not found. Please enter a valid project name")
            return

        self.headers = {
                "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                "X-Project-Id": str(self.project_id)
            }


    def list_prompts(self):
        if not self.project_id:
            logger.error("PromptManager not properly initialized, cannot list prompts")
            return []

        prompt = Prompt()
        try:
            prompt_list = prompt.list_prompts(self.base_url, self.headers, self.timeout)
            return prompt_list
        except Exception as e:
            logger.error(f"Error listing prompts: {str(e)}")
            return []
    
    def get_prompt(self, prompt_name, version=None):
        try:
            prompt_list = self.list_prompts()
        except Exception as e:
            logger.error(f"Error fetching prompt list: {str(e)}")
            return None

        if prompt_name not in prompt_list:
            logger.error("Prompt not found. Please enter a valid prompt name")
            return None

        try:
            prompt_versions = self.list_prompt_versions(prompt_name)
        except Exception as e:
            logger.error(f"Error fetching prompt versions: {str(e)}")
            return None

        if version and version not in prompt_versions.keys():
            logger.error("Version not found. Please enter a valid version name")
            return None

        prompt = Prompt()
        try:
            prompt_object = prompt.get_prompt(self.base_url, self.headers, self.timeout, prompt_name, version)
            return prompt_object
        except Exception as e:
            logger.error(f"Error fetching prompt: {str(e)}")
            return None

    def list_prompt_versions(self, prompt_name):
        try:
            prompt_list = self.list_prompts()
        except Exception as e:
            logger.error(f"Error fetching prompt list: {str(e)}")
            return {}

        if prompt_name not in prompt_list:
            logger.error("Prompt not found. Please enter a valid prompt name")
            return {}
        
        prompt = Prompt()
        try:
            prompt_versions = prompt.list_prompt_versions(self.base_url, self.headers, self.timeout, prompt_name)
            return prompt_versions
        except Exception as e:
            logger.error(f"Error fetching prompt versions: {str(e)}")
            return {}

    def _create_prompt(self, prompt_name: str, directory: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not prompt_name or not prompt_name.strip():
            logger.warning("Prompt name cannot be empty")
            return None

        try:
            existing_prompts = self.list_prompts()
            if prompt_name in existing_prompts:
                logger.info(f"Prompt '{prompt_name}' already exists, skipping creation")
                return
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, f"checking existing prompts for {prompt_name}")
            return
        except RequestException as e:
            logger.error(f"Error checking existing prompts: {str(e)}")
            return

        payload = {
            "name": prompt_name,
            "directory": directory
        }

        try:
            response = session_manager.make_request_with_retry(
                "POST",
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, f"creating prompt {prompt_name}")
            return
        except RequestException as e:
            logger.error(f"Error creating prompt: {str(e)}")
            return
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing response: {str(e)}")
            return

    def delete_prompt(self, prompt_name: str) -> Dict[str, Any]:
        if not prompt_name or not prompt_name.strip():
            error_msg = "Prompt name cannot be empty"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
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
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, f"checking existing prompts for {prompt_name}")
            return {
                'success': False,
                'message': f"Error checking existing prompts: {str(e)}",
                'prompt_name': prompt_name
            }
        except RequestException as e:
            logger.error(f"Error checking existing prompts: {str(e)}")
            return {
                'success': False,
                'message': f"Error checking existing prompts: {str(e)}",
                'prompt_name': prompt_name
            }

        try:
            delete_url = f"{self.base_url}/{prompt_name}"
            response = session_manager.make_request_with_retry(
                "DELETE",
                delete_url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            logger.info(f"Prompt '{prompt_name}' deleted successfully")
            return response.json()
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, f"deleting prompt {prompt_name}")
            return {
                'success': False,
                'message': f"Error deleting prompt: {str(e)}",
                'prompt_name': prompt_name
            }
        except RequestException as e:
            logger.error(f"Error deleting prompt: {str(e)}")
            return {
                'success': False,
                'message': f"Error deleting prompt: {str(e)}",
                'prompt_name': prompt_name
            }
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing response: {str(e)}")
            return {
                'success': False,
                'message': f"Error parsing response: {str(e)}",
                'prompt_name': prompt_name
            }

    def set_version_as_default(self, version_id: int) -> Dict[str, Any]:
        if not version_id or not isinstance(version_id, int):
            error_msg = "Version ID must be a valid integer"
            logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'version_id': version_id
            }

        try:
            default_url = f"{self.base_url}/version/{version_id}/default"
            response = session_manager.make_request_with_retry(
                "PUT",
                default_url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            logger.info(f"Version '{version_id}' set as default successfully")
            return response.json()
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, f"setting version {version_id} as default")
            return {
                'success': False,
                'message': f"Error setting version as default: {str(e)}",
                'version_id': version_id
            }
        except RequestException as e:
            logger.error(f"Error setting version as default: {str(e)}")
            return {
                'success': False,
                'message': f"Error setting version as default: {str(e)}",
                'version_id': version_id
            }
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing response: {str(e)}")
            return {
                'success': False,
                'message': f"Error parsing response: {str(e)}",
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

    def _extract_variables_from_text_fields(self, text_fields: List[Dict[str, str]]) -> List[str]:
        variables = set()
        pattern = r'\{\{(.*?)\}\}'

        for field in text_fields:
            content = field.get('content', '')
            matches = re.findall(pattern, content)
            for match in matches:
                var_name = match.strip()
                if '"' not in var_name:
                    variables.add(var_name)

        return sorted(list(variables))

    def _get_supported_models(self, provider_name):
        try:
            models_url = f"{RagaAICatalyst.BASE_URL}/v1/llm/models"
            response = session_manager.make_request_with_retry(
                "POST",
                models_url,
                headers=self.headers,
                json={"providerName": provider_name},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            if data.get("success") and "data" in data:
                return [model["name"] for model in data["data"]]
            return []
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, f"getting supported models for {provider_name}")
            return []
        except Exception:
            return []

    def _get_model_parameters(self, provider_name, model_name):
        try:
            params_url = f"{RagaAICatalyst.BASE_URL}/playground/providers/models/parameters/list"
            response = session_manager.make_request_with_retry(
                "POST",
                params_url,
                headers=self.headers,
                json={"providerName": provider_name, "modelName": model_name},
                timeout=self.timeout
            )
            response.raise_for_status()
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
            return None
        except Exception:
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
        if not prompt_name or not prompt_name.strip():
            raise ValueError("Prompt name cannot be empty")

        if not message or not message.strip():
            message = f"commit#{uuid.uuid4().hex[:8]}"
            logger.info(f"No message provided, auto-generated: {message}")

        if not isinstance(text_fields, list) or not text_fields:
            raise ValueError("text_fields must be a non-empty list")

        valid_roles = ['system', 'user', 'assistant']
        for idx, field in enumerate(text_fields):
            if not isinstance(field, dict) or 'role' not in field or 'content' not in field:
                raise ValueError("Each text_field must be a dict with 'role' and 'content' keys")

            role = field.get('role')
            if role not in valid_roles:
                raise ValueError(
                    f"Invalid role '{role}' in text_field at index {idx}. "
                    f"Role must be one of: {', '.join(valid_roles)}"
                )

            content = field.get('content')
            if not content or not isinstance(content, str) or content.strip() == "":
                raise ValueError(f"Content cannot be empty in text_field at index {idx}")

        if not model or not isinstance(model, str) or not model.strip():
            raise ValueError("Model must be a non-empty string")

        valid_model_prefixes = [
            "openai/", "azure/", "bedrock/", "gemini/", "anthropic/", "vertex_ai/"
        ]

        if "/" not in model:
            raise ValueError(
                f"Model must be in format 'provider/model-name' (e.g., 'openai/gpt-4o'). "
                f"Supported providers: {', '.join([p.rstrip('/') for p in valid_model_prefixes])}"
            )

        model = model.lower()
        if not any(model.startswith(prefix) for prefix in valid_model_prefixes):
            raise ValueError(
                f"Unsupported model provider in '{model}'. "
                f"Supported providers: {', '.join([p.rstrip('/') for p in valid_model_prefixes])}"
            )

        provider_name = model.split('/')[0]
        model_name = model.split('/', 1)[1] if '/' in model else ""

        supported_models = self._get_supported_models(provider_name)
        if supported_models and model_name not in supported_models:
            raise ValueError(
                f"Model '{model_name}' is not supported by provider '{provider_name}'. "
                f"Supported models: {', '.join(supported_models[:10])}{'...' if len(supported_models) > 10 else ''}"
            )

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

        try:
            version_url = f"{self.base_url}/{prompt_name}/version"
            response = session_manager.make_request_with_retry(
                "POST",
                version_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            response_data = response.json()

            # Extract relevant fields from API response
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

        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, f"saving prompt version for {prompt_name}")
            return {
                'success': False,
                'message': f"Error saving prompt version: {str(e)}",
                'prompt_name': prompt_name,
                'version_id': None
            }
        except RequestException as e:
            logger.error(f"Error saving prompt version: {str(e)}")
            return {
                'success': False,
                'message': f"Error saving prompt version: {str(e)}",
                'prompt_name': prompt_name,
                'version_id': None
            }
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing response: {str(e)}")
            return {
                'success': False,
                'message': f"Error parsing response: {str(e)}",
                'prompt_name': prompt_name,
                'version_id': None
            }


class Prompt:
    def __init__(self):
        pass

    def list_prompts(self, url, headers, timeout):
        try:
            response = session_manager.make_request_with_retry("GET", url, headers=headers, timeout=timeout)
            response.raise_for_status()
            prompt_list = [prompt["name"] for prompt in response.json()["data"]]                        
            return prompt_list
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "listing prompts")
            return []
        except RequestException as e:
            logger.error(f"Error while listing prompts: {e}")
            return []
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing prompt list: {str(e)}")
            return []

    def _get_response_by_version(self, base_url, headers, timeout, prompt_name, version):
        try:
            response = session_manager.make_request_with_retry("GET", f"{base_url}/version/{prompt_name}?version={version}",
                                    headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, f"fetching prompt version {version} for {prompt_name}")
            return None
        except RequestException as e:
            logger.error(f"Error while fetching prompt version {version} for {prompt_name}: {e}")
            return None
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            logger.error(f"Error parsing prompt version: {str(e)}")
            return None

    def _get_response(self, base_url, headers, timeout, prompt_name):
        try:
            response = session_manager.make_request_with_retry("GET", f"{base_url}/version/{prompt_name}",
                                headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, f"fetching latest prompt version for {prompt_name}")
            return None
        except RequestException as e:
            logger.error(f"Error while fetching latest prompt version for {prompt_name}: {e}")
            return None
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            logger.error(f"Error parsing prompt version: {str(e)}")
            return None

    def _get_prompt_by_version(self, base_url, headers, timeout, prompt_name, version):
        response = self._get_response_by_version(base_url, headers, timeout, prompt_name, version)
        if response is None:
            return ""
        try:
            prompt_text = response.json()["data"]["docs"][0]["textFields"]
            return prompt_text
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            logger.error(f"Error parsing prompt text: {str(e)}")
            return ""

    def get_prompt(self, base_url, headers, timeout, prompt_name, version=None):
        if version:
            response = self._get_response_by_version(base_url, headers, timeout, prompt_name, version)
        else:
            response = self._get_response(base_url, headers, timeout, prompt_name)

        if response is None:
            return None

        try:
            prompt_text = response.json()["data"]["docs"][0]["textFields"]
            prompt_parameters = response.json()["data"]["docs"][0]["modelSpecs"]["parameters"]
            model = response.json()["data"]["docs"][0]["modelSpecs"]["model"]
            return PromptObject(prompt_text, prompt_parameters, model)
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            logger.error(f"Error parsing prompt data: {str(e)}")
            return None


    def list_prompt_versions(self, base_url, headers, timeout, prompt_name):
        try:
            response = session_manager.make_request_with_retry("GET", f"{base_url}/{prompt_name}/version",
                                    headers=headers, timeout=timeout)
            response.raise_for_status()
            version_names = [version["name"] for version in response.json()["data"]]
            prompt_versions = {}
            for version in version_names:
                prompt_versions[version] = self._get_prompt_by_version(base_url, headers, timeout, prompt_name, version)
            return prompt_versions
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, f"listing prompt versions for {prompt_name}")
            return {}
        except RequestException as e:
            logger.error(f"Error while listing prompt versions for {prompt_name}: {e}")
            return {}
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing prompt versions: {str(e)}")
            return {}


class PromptObject:
    def __init__(self, text, parameters, model):
        self.text = text
        self.parameters = parameters
        self.model = model
    
    def _extract_variable_from_content(self, content):
        pattern = r'\{\{(.*?)\}\}'
        matches = re.findall(pattern, content)
        variables = [match.strip() for match in matches if '"' not in match]
        return variables

    def _add_variable_value_to_content(self, content, user_variables):
        variables = self._extract_variable_from_content(content)
        for key, value in user_variables.items():
            if not isinstance(value, str):
                raise ValueError(f"Value for variable '{key}' must be a string, not {type(value).__name__}")
            if key in variables:
                content = content.replace(f"{{{{{key}}}}}", value)
        return content

    def compile(self, **kwargs):
        required_variables = self.get_variables()
        provided_variables = set(kwargs.keys())

        missing_variables = [item for item in required_variables if item not in provided_variables]
        extra_variables = [item for item in provided_variables if item not in required_variables]

        if missing_variables:
            raise ValueError(f"Missing variable(s): {', '.join(missing_variables)}")
        if extra_variables:
            raise ValueError(f"Extra variable(s) provided: {', '.join(extra_variables)}")

        updated_text = copy.deepcopy(self.text)

        for item in updated_text:
            item["content"] = self._add_variable_value_to_content(item["content"], kwargs)

        return updated_text
    
    def get_variables(self):
        variables = set()
        for item in self.text:
            content = item["content"]
            for var in self._extract_variable_from_content(content):
                variables.add(var)
        if variables:
            return list(variables)
        else:
            return []
    
    def _convert_value(self, value, type_):
        if type_ == "float":
            return float(value)
        elif type_ == "int":
            return int(value)
        return value

    def get_model_parameters(self):
        parameters = {}
        for param in self.parameters:
            if "value" in param:
                parameters[param["name"]] = self._convert_value(param["value"], param["type"])
            else:
                parameters[param["name"]] = ""
        parameters["model"] = self.model
        return parameters    
    
    def get_prompt_content(self):
        return self.text
