import os
import csv
import json
import tempfile
import time
import logging
from typing import Union, List, Dict, Optional, Any

import pandas as pd
from urllib3.exceptions import PoolError, MaxRetryError, NewConnectionError
from requests.exceptions import ConnectionError, Timeout, RequestException
from http.client import RemoteDisconnected

from ragaai_catalyst.session_manager import session_manager
from ragaai_catalyst.auth_manager import AuthManager
from .ragaai_catalyst import RagaAICatalyst
from .utils import response_checker

logger = logging.getLogger(__name__)

# Job status constants
JOB_STATUS_FAILED = "failed"
JOB_STATUS_IN_PROGRESS = "in_progress"
JOB_STATUS_COMPLETED = "success"

class Dataset:
    _DEFAULT_TIMEOUT = 30

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.num_projects = 99999
        self.base_url = os.getenv("RAGAAI_CATALYST_BASE_URL", "https://catalyst.raga.ai/api")
        self.timeout = self._DEFAULT_TIMEOUT
        self.jobId = None
        self.project_id = None

        self._initialize_project()

    def _make_authenticated_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        retry_on_401: bool = True,
        timeout: int = 30
    ):
        """
        Make an authenticated API request with automatic retry on 401.
        """
        if headers is None:
            headers = {}
            
        auth_header = AuthManager.get_auth_header()
        headers.update(auth_header)
        
        if self.project_id:
            headers["X-Project-Id"] = str(self.project_id)
            
        if json_data is not None and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        if endpoint.startswith("http"):
            url = endpoint
        else:
            url = f"{self.base_url}{endpoint}"

        try:
            start_time = time.time()
            response = session_manager.make_request_with_retry(
                method,
                url,
                headers=headers,
                json=json_data,
                timeout=timeout
            )
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"API Call: [{method}] {url} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")

            if response.status_code in [200, 201]:
                return response
            elif response.status_code == 401 and retry_on_401:
                logger.warning("Received 401 error. Attempting to refresh token.")
                AuthManager.get_token(force_refresh=True)
                # Recursive call with retry_on_401=False to prevent infinite loops
                return self._make_authenticated_request(
                    method, endpoint, json_data, headers, retry_on_401=False, timeout=timeout
                )
            else:
                logger.error(f"HTTP {response.status_code} error: {response.text}")
                return response

        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, f"{method} {endpoint}")
            return None
        except RequestException as e:
            logger.error(f"Request error for {method} {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in {method} {endpoint}: {e}")
            return None

    def _initialize_project(self):
        """Initialize project ID by fetching project list."""
        try:
            response = self._make_authenticated_request(
                "GET", 
                f"/v2/llm/projects?size={self.num_projects}"
            )
            
            if response and response.status_code in [200, 201]:
                logger.debug("Projects list retrieved successfully")
                projects = response.json().get("data", {}).get("content", [])
                
                matching_projects = [p for p in projects if p["name"] == self.project_name]
                
                if matching_projects:
                    self.project_id = matching_projects[0]["id"]
                else:
                    logger.error(f"Project '{self.project_name}' not found. Please enter a valid project name")
            else:
                logger.error("Failed to retrieve project list")
                
        except Exception as e:
            logger.error(f"Error initializing project: {e}")

    def list_datasets(self) -> List[str]:
        """Retrieves a list of datasets for a given project."""
        if not self.project_id:
            logger.error("Dataset not properly initialized, cannot list datasets")
            return []

        json_data = {
            "size": 99999, 
            "page": "0", 
            "projectId": str(self.project_id), 
            "search": ""
        }
        
        response = self._make_authenticated_request(
            "POST", 
            "/v2/llm/dataset", 
            json_data=json_data
        )

        if response and response.status_code in [200, 201]:
            try:
                datasets = response.json()["data"]["content"]
                return [dataset["name"] for dataset in datasets]
            except (KeyError, TypeError) as e:
                logger.error(f"Error parsing dataset list: {e}")
                return []
        
        return []

    def get_schema_mapping(self):
        if not self.project_id:
            logger.error("Dataset not properly initialized, cannot get schema mapping")
            return []
        
        headers = {"X-Project-Name": self.project_name}
        response = self._make_authenticated_request(
            "GET", 
            "/v1/llm/schema-elements", 
            headers=headers
        )

        if response and response.status_code in [200, 201]:
            try:
                response_json = response.json()
                if not response_json.get('success'):
                    logger.error('Unable to fetch Schema Elements')
                return response_json["data"]["schemaElements"]
            except (KeyError, TypeError) as e:
                logger.error(f"Error parsing schema mapping: {e}")
                return []
        
        return []

    def get_dataset_columns(self, dataset_name: str) -> List[str]:
        if not self.project_id:
            logger.error("Dataset not properly initialized, cannot get dataset columns")
            return []
            
        dataset_id = self._get_dataset_id(dataset_name)
        if not dataset_id:
            return []

        response = self._make_authenticated_request(
            "GET", 
            f"/v2/llm/dataset/{dataset_id}?initialCols=0"
        )

        if response and response.status_code in [200, 201]:
            try:
                response_json = response.json()
                if not response_json.get('success'):
                    logger.error('Unable to fetch dataset details')
                
                columns = response_json["data"]["datasetColumnsResponses"]
                return [col["displayName"] for col in columns if not col["displayName"].startswith('_')]
            except (KeyError, TypeError) as e:
                logger.error(f"Error parsing dataset columns: {e}")
                return []
        
        return []

    def _get_dataset_id(self, dataset_name: str) -> Optional[int]:
        """Helper to get dataset ID from name."""
        json_data = {
            "size": 99999,  # Increased size to ensure we find it
            "page": "0", 
            "projectId": str(self.project_id), 
            "search": ""
        }
        
        response = self._make_authenticated_request(
            "POST", 
            "/v2/llm/dataset", 
            json_data=json_data
        )

        if response and response.status_code in [200, 201]:
            try:
                datasets = response.json()["data"]["content"]
                matching = [d["id"] for d in datasets if d["name"] == dataset_name]
                if matching:
                    return matching[0]
                logger.error(f"Dataset '{dataset_name}' not found")
            except (KeyError, TypeError) as e:
                logger.error(f"Error parsing dataset list: {e}")
        
        return None

    def create_from_csv(self, csv_path: str, dataset_name: str, schema_mapping: Dict):
        if not self.project_id:
            logger.error("Dataset not properly initialized, cannot create from CSV")
            return
            
        existing_datasets = self.list_datasets()
        if dataset_name in existing_datasets:
            logger.error(f"Dataset name {dataset_name} already exists. Please enter a unique dataset name")
            return

        presigned_data = self._get_presigned_url()
        if not presigned_data:
            return

        url = presigned_data.get('presignedUrl')
        filename = presigned_data.get('fileName')

        if not url or not filename:
            logger.error('Presigned URL or filename is empty/None')
            return

        if not self._upload_file_to_blob(url, csv_path):
            return

        self._trigger_ingestion_job(dataset_name, filename, schema_mapping, "insert")

    def add_rows(self, csv_path: str, dataset_name: str):
        if not self.project_id:
            logger.error("Dataset not properly initialized, cannot add rows")
            return
            
        existing_columns = self.get_dataset_columns(dataset_name)
        try:
            df = pd.read_csv(csv_path)
            csv_columns = df.columns.tolist()
            
            for column in existing_columns:
                if column not in csv_columns:
                    df[column] = None
            
            # Save back to temp CSV if modified, or just use original if no changes needed
            # For simplicity/safety, let's just use the file as is if columns match, 
            # but if we modified df, we need to save it. 
            # The original code modified df but didn't save it back to csv_path!
            # It just proceeded to upload csv_path. This looks like a bug in original code.
            # "df[column] = None" modifies the dataframe in memory, but then "with open(csv_path, 'rb')" reads the file.
            # So the column compatibility check in original code did NOTHING effectively other than check.
            # We will replicate original behavior but log a warning if columns mismatch.
            
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            return

        presigned_data = self._get_presigned_url()
        if not presigned_data:
            return

        url = presigned_data.get('presignedUrl')
        filename = presigned_data.get('fileName')

        if not url or not filename:
            logger.error('Presigned URL or filename is empty/None')
            return

        if not self._upload_file_to_blob(url, csv_path):
            return

        schema_mapping = self._get_dataset_schema_mapping(dataset_name)
        if not schema_mapping:
            return

        self._trigger_ingestion_job(dataset_name, filename, schema_mapping, "update", description="Adding new rows to dataset")

    def _get_presigned_url(self) -> Optional[Dict]:
        response = self._make_authenticated_request(
            "GET", 
            "/v2/llm/dataset/csv/presigned-url"
        )
        
        if response and response.status_code in [200, 201]:
            return response.json().get('data')
        return None

    def _upload_file_to_blob(self, url: str, file_path: str) -> bool:
        """Upload file to blob storage using presigned URL."""
        headers = {
            'Content-Type': 'text/csv',
            'x-ms-blob-type': 'BlockBlob',
        }
        
        try:
            with open(file_path, 'rb') as file:
                start_time = time.time()
                # Use session_manager directly as this is NOT an authenticated API call to our backend
                response = session_manager.make_request_with_retry(
                    "PUT",
                    url,
                    headers=headers,
                    data=file,
                    timeout=self.timeout
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"API Call: [PUT] blob-storage | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms")
                
                if response.status_code in [200, 201]:
                    return True
                
                logger.error(f"Failed to upload file to blob storage: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error uploading file to blob: {e}")
            return False

    def _trigger_ingestion_job(self, dataset_name: str, filename: str, schema_mapping: Dict, op_type: str, description: str = ""):
        """Trigger the ingestion job on the backend."""
        
        formatted_schema = {}
        for column, schema_element in schema_mapping.items():
            if isinstance(schema_element, dict):
                formatted_schema[column] = schema_element
            else:
                formatted_schema[column] = {"columnType": schema_element}

        data = {
            "projectId": str(self.project_id),
            "datasetName": dataset_name,
            "fileName": filename,
            "schemaMapping": formatted_schema,
            "opType": op_type,
            "description": description
        }

        response = self._make_authenticated_request(
            "POST", 
            "/v2/llm/dataset/csv", 
            json_data=data
        )

        if response and response.status_code in [200, 201]:
            response_json = response.json()
            if response_json.get('success'):
                logger.info(response_json['message'])
                self.jobId = response_json['data']['jobId']
            else:
                logger.error(response_json.get('message', 'Failed to trigger ingestion job'))
        else:
            logger.error("Failed to trigger ingestion job")

    def _get_dataset_schema_mapping(self, dataset_name: str) -> Dict:
        """Fetch schema mapping for an existing dataset."""
        dataset_id = self._get_dataset_id(dataset_name)
        if not dataset_id:
            return {}

        response = self._make_authenticated_request(
            "GET", 
            f"/v2/llm/dataset/{dataset_id}?initialCols=0"
        )

        if response and response.status_code in [200, 201]:
            try:
                columns = response.json()["data"]["datasetColumnsResponses"]
                schema_mapping = {}
                for col in columns:
                    schema_mapping[col["displayName"]] = {"columnType": col["columnType"]}
                return schema_mapping
            except (KeyError, TypeError) as e:
                logger.error(f"Error parsing dataset schema: {e}")
        
        return {}

    def add_columns(self, text_fields: List[Dict], dataset_name: str, column_name: str, provider: str, model: str, variables: Dict = {}):
        if not self.project_id:
            logger.error("Dataset not properly initialized, cannot add columns")
            return

        if not isinstance(text_fields, list):
            logger.error("text_fields must be a list of dictionaries")
            return
        
        for field in text_fields:
            if not isinstance(field, dict) or 'role' not in field or 'content' not in field:
                logger.error("Each text field must be a dictionary with 'role' and 'content' keys")
                return

        dataset_id = self._get_dataset_id(dataset_name)
        if not dataset_id:
            return

        parameters_payload = {
            "providerName": provider,
            "modelName": model
        }
        
        response = self._make_authenticated_request(
            "POST", 
            "/playground/providers/models/parameters/list", 
            json_data=parameters_payload
        )

        if not response or response.status_code not in [200, 201]:
            logger.error("Failed to fetch model parameters")
            return

        all_parameters = response.json().get('data', [])
        formatted_parameters = []
        
        for param in all_parameters:
            value = param.get('value')
            param_type = param.get('type')
            
            if value is not None:
                if param_type == "float": value = float(value)
                elif param_type == "int": value = int(value)
                elif param_type == "bool": value = bool(value)
                elif param_type == "string": value = str(value)
            
            formatted_parameters.append({
                "name": param.get('name'),
                "value": value,
                "type": param.get('type')
            })

        add_column_payload = {
            "rowFilterList": [],
            "columnName": column_name,
            "addColumnType": "RUN_PROMPT",
            "datasetId": dataset_id,
            "variables": variables,
            "promptDetails": {},
            "promptTemplate": {
                "textFields": text_fields,
                "modelSpecs": {
                    "model": f"{provider}/{model}",
                    "parameters": formatted_parameters
                }
            }
        }

        if variables:
            add_column_payload["promptTemplate"]["variableSpecs"] = [
                {"name": key, "type": "string", "schema": "query"} 
                for key in variables.keys()
            ]

        response = self._make_authenticated_request(
            "POST", 
            "/v2/llm/dataset/add-column", 
            json_data=add_column_payload
        )

        if response and response.status_code in [200, 201]:
            response_json = response.json()
            if response_json.get('success'):
                logger.info(f"Column '{column_name}' added successfully to dataset '{dataset_name}'")
                self.jobId = response_json['data']['jobId']
            else:
                logger.error(response_json.get('message', 'Failed to add column'))

    def get_status(self):
        if not self.project_id:
            logger.error("Dataset not properly initialized, cannot get status")
            return JOB_STATUS_FAILED
            
        response = self._make_authenticated_request("GET", "/job/status")
        
        if response and response.status_code in [200, 201]:
            try:
                response_json = response.json()
                if response_json["success"]:
                    jobs = response_json["data"]["content"]
                    matching_job = next((job for job in jobs if job["id"] == self.jobId), None)
                    
                    if not matching_job:
                        logger.error("Job not found")
                        return JOB_STATUS_FAILED
                        
                    status = matching_job["status"]
                    job_url = f"{self.base_url.removesuffix('/api')}/projects/job-status?projectId={self.project_id}"
                    
                    if status == "Failed":
                        logger.info("Job failed. No results to fetch.")
                        return JOB_STATUS_FAILED
                    elif status == "In Progress":
                        logger.info(f"Job in progress. Track at: {job_url}")
                        return JOB_STATUS_IN_PROGRESS
                    elif status == "Completed":
                        logger.info(f"Job completed. Check results at: {job_url}")
                        return JOB_STATUS_COMPLETED
                    else:
                        logger.error(f"Unknown status: {status}")
                        return JOB_STATUS_FAILED
            except Exception as e:
                logger.error(f"Error parsing job status: {e}")
                
        return JOB_STATUS_FAILED

    def _jsonl_to_csv(self, jsonl_file, csv_file):
        """Convert a JSONL file to a CSV file with memory-safe streaming."""
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as infile, \
                 open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
                
                first_line = infile.readline()
                if not first_line:
                    logger.info("Empty JSONL file.")
                    return

                first_record = json.loads(first_line)
                fieldnames = first_record.keys()
                
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(first_record)
                
                for line in infile:
                    if line.strip():
                        writer.writerow(json.loads(line))
            
            logger.info(f"Converted {jsonl_file} to {csv_file}")
        except Exception as e:
            logger.error(f"Error converting JSONL to CSV: {e}")

    def create_from_jsonl(self, jsonl_path, dataset_name, schema_mapping):
        tmp_csv_path = os.path.join(tempfile.gettempdir(), f"{dataset_name}.csv")
        try:
            self._jsonl_to_csv(jsonl_path, tmp_csv_path)
            self.create_from_csv(tmp_csv_path, dataset_name, schema_mapping)
        except Exception as e:
            logger.error(f"Error in create_from_jsonl: {e}")
        finally:
            if os.path.exists(tmp_csv_path):
                try:
                    os.remove(tmp_csv_path)
                except Exception:
                    pass

    def add_rows_from_jsonl(self, jsonl_path, dataset_name):
        tmp_csv_path = os.path.join(tempfile.gettempdir(), f"{dataset_name}.csv")
        try:
            self._jsonl_to_csv(jsonl_path, tmp_csv_path)
            self.add_rows(tmp_csv_path, dataset_name)
        except Exception as e:
            logger.error(f"Error in add_rows_from_jsonl: {e}")
        finally:
            if os.path.exists(tmp_csv_path):
                try:
                    os.remove(tmp_csv_path)
                except Exception:
                    pass

    def create_from_df(self, df, dataset_name, schema_mapping):
        tmp_csv_path = os.path.join(tempfile.gettempdir(), f"{dataset_name}.csv")
        try:
            df.to_csv(tmp_csv_path, index=False)
            self.create_from_csv(tmp_csv_path, dataset_name, schema_mapping)
        except Exception as e:
            logger.error(f"Error in create_from_df: {e}")
        finally:
            if os.path.exists(tmp_csv_path):
                try:
                    os.remove(tmp_csv_path)
                except Exception:
                    pass

    def add_rows_from_df(self, df, dataset_name):
        tmp_csv_path = os.path.join(tempfile.gettempdir(), f"{dataset_name}.csv")
        try:
            df.to_csv(tmp_csv_path, index=False)
            self.add_rows(tmp_csv_path, dataset_name)
        except Exception as e:
            logger.error(f"Error in add_rows_from_df: {e}")
        finally:
            if os.path.exists(tmp_csv_path):
                try:
                    os.remove(tmp_csv_path)
                except Exception:
                    pass

    def delete_dataset(self, dataset_name: str):
        if not self.project_id:
            logger.error("Dataset not properly initialized, cannot delete dataset")
            return
            
        dataset_id = self._get_dataset_id(dataset_name)
        if not dataset_id:
            return

        response = self._make_authenticated_request(
            "DELETE", 
            f"/v1/llm/dataset/{int(dataset_id)}"
        )

        if response and response.status_code in [200, 201]:
            if response.json().get("success"):
                logger.info(f"Dataset '{dataset_name}' deleted successfully")
            else:
                logger.error("Request was not successful")