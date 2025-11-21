import os
from sys import exception

import requests
import json
import re
import logging
import uuid
from typing import Optional, List, Dict, Any
from .ragaai_catalyst import RagaAICatalyst
import copy

logger = logging.getLogger(__name__)

class PromptManager:
    NUM_PROJECTS = 100
    TIMEOUT = 10

    def __init__(self, project_name):
        self.project_name = project_name
        self.base_url = f"{RagaAICatalyst.BASE_URL}/playground/prompt"
        self.timeout = 10
        self.size = 99999

        token = os.getenv("RAGAAI_CATALYST_TOKEN")
        if not token:
            raise EnvironmentError("RAGAAI_CATALYST_TOKEN is not set in environment variables")

        self.headers = {"Authorization": f"Bearer {token}"}

        # 1. Fetch Project List
        try:
            url = f"{RagaAICatalyst.BASE_URL}/v2/llm/projects?size={self.size}"
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise TimeoutError("Timed out while fetching project list")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch project list: {str(e)}")

        # 2. Parse JSON Response
        try:
            data = response.json()
            projects = data["data"]["content"]
            if not isinstance(projects, list):
                raise ValueError("Invalid project list format received from server")
        except (KeyError, json.JSONDecodeError):
            raise ValueError("Unexpected response structure while parsing project list")

        # 3. Validate project_name and extract project_id
        project_list = [p.get("name") for p in projects]
        if self.project_name not in project_list:
            raise ValueError(f"Project '{self.project_name}' not found. Please provide a valid project name.")

        matching_projects = [p["id"] for p in projects if p.get("name") == self.project_name]
        if not matching_projects:
            raise ValueError(f"Project ID for '{self.project_name}' not found in response")
        self.project_id = matching_projects[0]

        self.headers["X-Project-Id"] = str(self.project_id)

    def list_prompts(self):
        prompt = Prompt()
        try:
            prompt_list = prompt.list_prompts(self.base_url, self.headers, self.timeout)
            return prompt_list
        except requests.RequestException as e:
            raise requests.RequestException(f"Error listing prompts: {str(e)}")
    
    def get_prompt(self, prompt_name, version=None):
        try:
            prompt_list = self.list_prompts()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt list: {str(e)}")

        if prompt_name not in prompt_list:
            raise ValueError("Prompt not found. Please enter a valid prompt name")

        try:
            prompt_versions = self.list_prompt_versions(prompt_name)
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt versions: {str(e)}")

        if version and version not in prompt_versions.keys():
            raise ValueError("Version not found. Please enter a valid version name")

        prompt = Prompt()
        try:
            prompt_object = prompt.get_prompt(self.base_url, self.headers, self.timeout, prompt_name, version)
            return prompt_object
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt: {str(e)}")

    def list_prompt_versions(self, prompt_name):
        try:
            prompt_list = self.list_prompts()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt list: {str(e)}")

        if prompt_name not in prompt_list:
            raise ValueError("Prompt not found. Please enter a valid prompt name")

        prompt = Prompt()
        try:
            prompt_versions = prompt.list_prompt_versions(self.base_url, self.headers, self.timeout, prompt_name)
            return prompt_versions
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt versions: {str(e)}")

    def _create_prompt(self, prompt_name: str, directory: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not prompt_name or not prompt_name.strip():
            raise ValueError("Prompt name cannot be empty")

        try:
            existing_prompts = self.list_prompts()
            if prompt_name in existing_prompts:
                logger.info(f"Prompt '{prompt_name}' already exists, skipping creation")
                return
        except requests.RequestException as e:
            raise requests.RequestException(f"Error checking existing prompts: {str(e)}")

        payload = {
            "name": prompt_name,
            "directory": directory
        }

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error creating prompt: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing response: {str(e)}")

    def delete_prompt(self, prompt_name: str) -> Dict[str, Any]:
        if not prompt_name or not prompt_name.strip():
            raise ValueError("Prompt name cannot be empty")

        try:
            existing_prompts = self.list_prompts()
            if prompt_name not in existing_prompts:
                raise ValueError(f"Prompt '{prompt_name}' not found")
        except requests.RequestException as e:
            raise requests.RequestException(f"Error checking existing prompts: {str(e)}")

        try:
            delete_url = f"{self.base_url}/{prompt_name}"
            response = requests.delete(
                delete_url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            logger.info(f"Prompt '{prompt_name}' deleted successfully")
            return response.json()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error deleting prompt: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing response: {str(e)}")

    def set_version_as_default(self, version_id: int) -> Dict[str, Any]:
        if not version_id or not isinstance(version_id, int):
            raise ValueError("Version ID must be a valid integer")

        try:
            default_url = f"{self.base_url}/version/{version_id}/default"
            response = requests.put(
                default_url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            logger.info(f"Version '{version_id}' set as default successfully")
            return response.json()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error setting version as default: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing response: {str(e)}")

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
            response = requests.post(
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
        except Exception:
            return []

    def _get_model_parameters(self, provider_name, model_name):
        try:
            params_url = f"{RagaAICatalyst.BASE_URL}/playground/providers/models/parameters/list"
            response = requests.post(
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
                raise ValueError(
                    f"Unable to fetch model parameters for '{model}'. "
                    f"Please verify the model name or provide model_parameters explicitly."
                )
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
            response = requests.post(
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

        except requests.RequestException as e:
            raise requests.RequestException(f"Error saving prompt version: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing response: {str(e)}")


class Prompt:
    def __init__(self):
        pass

    def list_prompts(self, url, headers, timeout):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            prompt_list = [prompt["name"] for prompt in response.json()["data"]]                        
            return prompt_list
        except requests.RequestException as e:
            raise requests.RequestException(f"Error listing prompts: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing prompt list: {str(e)}")

    def _get_response_by_version(self, base_url, headers, timeout, prompt_name, version):
        try:
            response = requests.get(f"{base_url}/version/{prompt_name}?version={version}",
                                    headers=headers, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt version: {str(e)}")
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            raise ValueError(f"Error parsing prompt version: {str(e)}")
        return response

    def _get_response(self, base_url, headers, timeout, prompt_name):
        try:
            response = requests.get(f"{base_url}/version/{prompt_name}",
                                headers=headers, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise requests.RequestException(f"Error fetching prompt version: {str(e)}")
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            raise ValueError(f"Error parsing prompt version: {str(e)}")
        return response

    def _get_prompt_by_version(self, base_url, headers, timeout, prompt_name, version):
        response = self._get_response_by_version(base_url, headers, timeout, prompt_name, version)
        prompt_text = response.json()["data"]["docs"][0]["textFields"]
        return prompt_text

    def get_prompt(self, base_url, headers, timeout, prompt_name, version=None):
        if version:
            response = self._get_response_by_version(base_url, headers, timeout, prompt_name, version)
            prompt_text = response.json()["data"]["docs"][0]["textFields"]
            prompt_parameters = response.json()["data"]["docs"][0]["modelSpecs"]["parameters"]
            model = response.json()["data"]["docs"][0]["modelSpecs"]["model"]
        else:
            response = self._get_response(base_url, headers, timeout, prompt_name)
            prompt_text = response.json()["data"]["docs"][0]["textFields"]
            prompt_parameters = response.json()["data"]["docs"][0]["modelSpecs"]["parameters"]
            model = response.json()["data"]["docs"][0]["modelSpecs"]["model"]
        return PromptObject(prompt_text, prompt_parameters, model)


    def list_prompt_versions(self, base_url, headers, timeout, prompt_name):
        try:
            response = requests.get(f"{base_url}/{prompt_name}/version",
                                    headers=headers, timeout=timeout)
            response.raise_for_status()
            version_names = [version["name"] for version in response.json()["data"]]
            prompt_versions = {}
            for version in version_names:
                prompt_versions[version] = self._get_prompt_by_version(base_url, headers, timeout, prompt_name, version)
            return prompt_versions
        except requests.RequestException as e:
            raise requests.RequestException(f"Error listing prompt versions: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing prompt versions: {str(e)}")


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
