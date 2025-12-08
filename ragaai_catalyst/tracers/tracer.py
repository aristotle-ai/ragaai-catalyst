"""
Clean, standalone Tracer class for OpenTelemetry-based tracing.
Independent of AgenticTracing - focused solely on instrumentation and export.
"""

import os
import datetime
import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import json

from ragaai_catalyst import RagaAICatalyst
from ragaai_catalyst.session_manager import session_manager
from ragaai_catalyst.tracers.constants import TracerConstants, TracerType
from ragaai_catalyst.tracers.instrumentor_registry import get_instrumentors_for_type
from ragaai_catalyst.tracers.agentic_tracing.utils.file_name_tracker import TrackName

from urllib3.exceptions import PoolError, MaxRetryError, NewConnectionError
from requests.exceptions import ConnectionError, Timeout
from http.client import RemoteDisconnected

logger = logging.getLogger(__name__)


class Tracer:
    """
    Standalone tracer for OpenTelemetry-based instrumentation.
    
    Handles:
    - Project validation
    - Instrumentor setup (LangChain, OpenAI, etc.)
    - Dynamic trace export
    - Feedback management
    - Context and ground truth injection
    """
    
    def __init__(
        self,
        project_name: str,
        dataset_name: str,
        tracer_type: Optional[str] = None,
        trace_name: Optional[str] = None,
        pipeline: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: int = TracerConstants.DEFAULT_TIMEOUT,
        max_upload_workers: int = TracerConstants.DEFAULT_MAX_UPLOAD_WORKERS,
        external_id: Optional[str] = None
    ):
        """
        Initialize Tracer.
        
        Args:
            project_name: Name of the project
            dataset_name: Name of the dataset
            tracer_type: Type of tracer ('langchain', 'openai', 'agentic', etc.)
            trace_name: Optional custom trace name
            pipeline: Pipeline configuration
            metadata: Additional metadata
            description: Tracer description
            timeout: Upload timeout in seconds
            update_ll_cost: Whether to update model costs
            interval_time: Upload interval
            max_upload_workers: Maximum number of upload workers
            external_id: External identifier for traces
        """
        self.project_name = project_name
        self.dataset_name = dataset_name
        self.tracer_type = tracer_type
        self.pipeline = pipeline
        self.timeout = timeout
        self.max_upload_workers = max_upload_workers
        self.external_id = external_id
        
        self.trace_name = trace_name or f"trace_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.metadata = self._prepare_metadata(metadata, tracer_type)
        self.base_url = f"{RagaAICatalyst.BASE_URL}"
        self.start_time = datetime.datetime.now().astimezone().isoformat()
        
        self.user_context = ""
        self.user_gt = ""
        self.model_custom_cost = {}
        self.file_tracker = TrackName()
        self.post_processor: Optional[Callable] = None
        
        self._tracer = None
        self._tracer_provider = None
        self.dynamic_exporter = None
        self.user_details = self._build_user_details()
        
        if TracerType.requires_instrumentation(tracer_type):
            self._setup_instrumentation()
    
    def _prepare_metadata(self, metadata: Optional[Dict], tracer_type: Optional[str]) -> Dict[str, Any]:
        """Prepare and enrich metadata."""
        if metadata is None:
            metadata = {}
        metadata.setdefault("log_source", f"{tracer_type}_tracer" if tracer_type else "tracer")
        metadata.setdefault("recorded_on", str(datetime.datetime.now()))
        return metadata
    
    def _build_user_details(self) -> Dict[str, Any]:
        """Build user details structure for export."""
        return {
            "project_name": self.project_name,
            "dataset_name": self.dataset_name,
            "trace_user_detail": {
                "trace_id": "",
                "session_id": None,
                "trace_type": self.tracer_type,
                "traces": [],
                "metadata": self.metadata,
                "pipeline": {
                    "llm_model": (self.pipeline or {}).get("llm_model", ""),
                    "vector_store": (self.pipeline or {}).get("vector_store", ""),
                    "embed_model": (self.pipeline or {}).get("embed_model", "")
                }
            }
        }
    
    def _setup_instrumentation(self):
        """Setup OpenTelemetry instrumentation."""
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from ragaai_catalyst.tracers.exporters.dynamic_trace_exporter import DynamicTraceExporter
        from openinference.instrumentation import TracerProvider, TraceConfig
        
        instrumentors = get_instrumentors_for_type(self.tracer_type)
        
        if not instrumentors and self.tracer_type != "custom":
            logger.warning(f"No instrumentors available for type: {self.tracer_type}")
            return
        
        self.file_tracker.trace_main_file()
        list_of_unique_files = self.file_tracker.get_unique_files()
        
        self.dynamic_exporter = DynamicTraceExporter(
            project_name=self.project_name,
            dataset_name=self.dataset_name,
            base_url=self.base_url,
            tracer_type=self.tracer_type,
            files_to_zip=list_of_unique_files,
            user_details=self.user_details,
            custom_model_cost=self.model_custom_cost,
            timeout=self.timeout,
            post_processor=self.post_processor,
            max_upload_workers=self.max_upload_workers,
            user_context=self.user_context,
            user_gt=self.user_gt,
            external_id=self.external_id
        )
        
        self._tracer_provider = TracerProvider(config=TraceConfig())
        self._tracer_provider.add_span_processor(SimpleSpanProcessor(self.dynamic_exporter))
        
        for instrumentor_class, args in instrumentors:
            try:
                instrumentor = instrumentor_class()
                instrumentor.instrument(tracer_provider=self._tracer_provider, *args)
                logger.info(f"Instrumented {instrumentor_class.__name__}")
                
            except Exception as e:
                logger.error(f"Failed to instrument {instrumentor_class.__name__}: {e}")
        
        self._tracer = self._tracer_provider.get_tracer(__name__)
    
    def register_masking_function(self, masking_func: Callable):
        """
        Register a masking function for sensitive data.
        
        Args:
            masking_func: Function that takes a value and returns masked value
        """
        if not callable(masking_func):
            logger.error("masking_func must be callable")
            return
        
        def recursive_mask_values(obj, parent_key=None):
            try:
                if isinstance(obj, dict):
                    return {k: recursive_mask_values(v, k) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_mask_values(item, parent_key) for item in obj]
                elif isinstance(obj, str):
                    excluded_keys = {
                        'start_time', 'end_time', 'name', 'id',
                        'hash_id', 'parent_id', 'source_hash_id',
                        'cost', 'type', 'feedback', 'error', 'ctx','telemetry.sdk.version',
                        'telemetry.sdk.language','service.name', 'llm.model_name',
                        'llm.invocation_parameters', 'metadata', 'openinference.span.kind',
                        'llm.token_count.prompt', 'llm.token_count.completion', 'llm.token_count.total',
                        "input_cost", "output_cost", "total_cost", "status_code", "output.mime_type",
                        "span_id", "trace_id"
                    }
                    if parent_key and parent_key.lower() not in excluded_keys:
                        return masking_func(obj)
                    return obj
                else:
                    return obj
            except Exception as e:
                logger.error(f"Error masking value: {e}")
                return obj
        
        def file_post_processor(original_trace_json_path: os.PathLike) -> os.PathLike:
            original_path = Path(original_trace_json_path)
            
            with open(original_path, 'r') as f:
                data = json.load(f)
            
            if 'data' in data:
                data['data'] = recursive_mask_values(data['data'])
            
            new_filename = f"processed_{original_path.name}"
            dir_name = os.path.dirname(original_trace_json_path)
            final_trace_json_path = Path(dir_name) / new_filename
            
            with open(final_trace_json_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            return final_trace_json_path
        
        self.register_post_processor(file_post_processor)
    
    def register_post_processor(self, post_processor_func: Callable):
        """Register a post-processing function for traces."""
        if not callable(post_processor_func):
            logger.error("post_processor_func must be callable")
            return
        
        self.post_processor = post_processor_func
        
        if self.dynamic_exporter:
            self.dynamic_exporter._exporter.post_processor = post_processor_func
            self.dynamic_exporter._post_processor = post_processor_func
        
        logger.info(f"Registered post-processor: {post_processor_func}")
    
    def set_external_id(self, external_id: str):
        """Update external ID."""
        self.external_id = external_id
        if self.dynamic_exporter:
            self.dynamic_exporter.external_id = external_id
    
    def set_dataset_name(self, dataset_name: str):
        """Update dataset name."""
        self.dataset_name = dataset_name
        if self.dynamic_exporter:
            self.dynamic_exporter.dataset_name = dataset_name
    
    def set_project_name(self, project_name: str):
        """Update project name."""
        self.project_name = project_name
        if self.dynamic_exporter:
            self.dynamic_exporter.project_name = project_name
    
    def add_context(self, context: str):
        """Add context to trace (for langchain/llamaindex)."""
        if self.tracer_type not in ["langchain", "llamaindex"]:
            logger.warning("add_context only supported for langchain/llamaindex")
            return
        
        if isinstance(context, str):
            self.user_context = context
            if self.dynamic_exporter:
                self.dynamic_exporter.user_context = context
        else:
            logger.warning("context must be a string")
    
    def add_gt(self, gt: str):
        """Add ground truth to trace (for langchain/llamaindex)."""
        if self.tracer_type not in ["langchain", "llamaindex"]:
            logger.warning("add_gt only supported for langchain/llamaindex")
            return
        
        if isinstance(gt, str):
            self.user_gt = gt
            if self.dynamic_exporter:
                self.dynamic_exporter.user_gt = gt
        else:
            logger.warning("gt must be a string")

    def add_metadata(self, metadata):
        """
        Add metadata information to the trace. If metadata is a dictionary, it will be merged with existing metadata.
        Non-dictionary metadata or keys not present in the existing metadata will be logged as warnings.

        Args:
            metadata: Additional metadata information to be added to the trace. Should be a dictionary.
        """
        # Convert string metadata to string if needed
        user_details = self.user_details
        user_metadata = user_details["trace_user_detail"]["metadata"]
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if key in user_metadata:
                    user_metadata[key] = value
                else:
                    logger.warning(f"Key '{key}' not found in metadata")
            self.dynamic_exporter.user_details = user_details
            self.metadata = user_metadata
        else:
            logger.warning("metadata must be a dictionary")
    
    def update_file_list(self):
        """Update file list in dynamic exporter."""
        if not self.dynamic_exporter:
            logger.error("Dynamic exporter not initialized")
            return
        
        list_of_unique_files = self.file_tracker.get_unique_files()
        self.dynamic_exporter.files_to_zip = list_of_unique_files
        logger.debug(f"Updated file list: {len(list_of_unique_files)} files")
    
    def set_feedback(self, external_id: str, feedback: Any):
        """Set feedback for a trace by external ID."""
        try:
            if not external_id:
                logger.error("external_id is required")
                return None
            
            if not feedback:
                logger.error("feedback is required")
                return None
            
            # Get project ID for the feedback request
            from ragaai_catalyst.tracers.agentic_tracing.core.api_client import TraceAPIClient
            api_client = TraceAPIClient(self.base_url, self.project_name, self.timeout)
            project_id = api_client.get_project_id(self.project_name)

            if not project_id:
                logger.error("Failed to get project ID for feedback request")
                return None

            url = f"{self.base_url}/v1/llm/feedback"
            headers = {
                'Accept': 'application/json, text/plain, */*',
                'Authorization': f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                'X-Project-Id': str(project_id),
                'Content-Type': 'application/json'
            }
            payload = json.dumps({
                "externalId": str(external_id),
                "feedbackColumnName": TracerConstants.FEEDBACK_COLUMN_NAME,
                "feedback": feedback,
                "datasetName": self.dataset_name
            })
            
            response = session_manager.make_request_with_retry(
                "POST", url, headers=headers, data=payload, timeout=self.timeout
            )
            print("response=", response)
            return response.json() if response else None
            
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "setting feedback")
            return None
        except Exception as e:
            logger.error(f"Error setting feedback: {e}")
            return None
    
    @property
    def tracer(self):
        """Get the OpenInference tracer instance."""
        return self._tracer
    
    def shutdown(self):
        """Shutdown tracer and cleanup resources."""
        if self._tracer_provider:
            try:
                self._tracer_provider.shutdown()
                logger.info("Tracer provider shut down successfully")
            except Exception as e:
                logger.error(f"Error during tracer shutdown: {e}")
