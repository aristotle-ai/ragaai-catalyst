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
from ragaai_catalyst.tracers.utils.file_name_tracker import TrackName
from ragaai_catalyst.tracers.core.api_client import TraceAPIClient

from urllib3.exceptions import PoolError, MaxRetryError, NewConnectionError
from requests.exceptions import ConnectionError, Timeout
from http.client import RemoteDisconnected

logger = logging.getLogger(__name__)


class TracerTypeSupport:
    """Supported tracer types and their capabilities."""
    LANGCHAIN = "langchain"
    LLAMAINDEX = "llamaindex"
    OPENAI = "openai"
    AGENTIC = "agentic"
    CUSTOM = "custom"
    
    CONTEXT_SUPPORTED = frozenset({LANGCHAIN, LLAMAINDEX})


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
    
    MASKING_EXCLUDED_KEYS = frozenset({
        'start_time', 'end_time', 'name', 'id',
        'hash_id', 'parent_id', 'source_hash_id',
        'cost', 'type', 'feedback', 'error', 'ctx',
        'telemetry.sdk.version', 'telemetry.sdk.language',
        'service.name', 'llm.model_name', 'llm.invocation_parameters',
        'metadata', 'openinference.span.kind',
        'llm.token_count.prompt', 'llm.token_count.completion', 'llm.token_count.total',
        'input_cost', 'output_cost', 'total_cost',
        'status_code', 'output.mime_type', 'span_id', 'trace_id'
    })
    
    _PROCESSED_FILE_PREFIX = "processed_"
    _JSON_INDENT = 4
    
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
    ):
        # Core configuration
        self.project_name = project_name
        self.dataset_name = dataset_name
        self.tracer_type = tracer_type
        self.timeout = timeout
        self.max_upload_workers = max_upload_workers

        # Optional configuration
        self.pipeline = pipeline or {}
        self.metadata = self._prepare_metadata(metadata, tracer_type)
        self.post_processor: Optional[Callable] = None

        # Runtime state
        self.trace_name = trace_name or f"trace_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.base_url = RagaAICatalyst.BASE_URL
        self.start_time = datetime.datetime.now().astimezone().isoformat()
        self.model_custom_cost = {}
        self.file_tracker = TrackName()

        # OpenTelemetry components
        self._tracer = None
        self._tracer_provider = None
        self.exporter = None
        self.user_details = self._build_user_details()

        # Initialize instrumentation if needed
        if TracerType.requires_instrumentation(tracer_type):
            self._setup_instrumentation()
    
    def _prepare_metadata(self, metadata: Optional[Dict], tracer_type: Optional[str]) -> Dict[str, Any]:
        result = metadata.copy() if metadata else {}
        result.setdefault("log_source", f"{tracer_type}_tracer" if tracer_type else "tracer")
        result.setdefault("recorded_on", str(datetime.datetime.now()))
        return result
    
    def _build_user_details(self) -> Dict[str, Any]:
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
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from ragaai_catalyst.tracers.exporters.ragaai_trace_exporter import RAGATraceExporter
        from ragaai_catalyst.tracers.processors.custom_span_processor import CustomSpanProcessor
        from openinference.instrumentation import TracerProvider, TraceConfig
        
        instrumentors = get_instrumentors_for_type(self.tracer_type)
        
        if not instrumentors and self.tracer_type != "custom":
            logger.warning(f"No instrumentors available for type: {self.tracer_type}")
            return
        
        self.file_tracker.trace_main_file()
        list_of_unique_files = self.file_tracker.get_unique_files()
        
        self.exporter = RAGATraceExporter(
            project_name=self.project_name,
            dataset_name=self.dataset_name,
            base_url=self.base_url,
            tracer_type=self.tracer_type,
            files_to_zip=list_of_unique_files,
            user_details=self.user_details,
            timeout=self.timeout,
            post_processor=self.post_processor,
            max_upload_workers=self.max_upload_workers
        )
        
        self._tracer_provider = TracerProvider(config=TraceConfig())
        self._tracer_provider.add_span_processor(CustomSpanProcessor())
        self._tracer_provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        
        for instrumentor_class, args in instrumentors:
            try:
                instrumentor = instrumentor_class()
                instrumentor.instrument(tracer_provider=self._tracer_provider, *args)
                logger.info(f"Instrumented {instrumentor_class.__name__}")
                
            except Exception as e:
                logger.error(f"Failed to instrument {instrumentor_class.__name__}: {e}")
        
        self._tracer = self._tracer_provider.get_tracer(__name__)
    
    def register_masking_function(self, masking_func: Callable) -> None:
        """
        Register a masking function for sensitive data in traces.
        
        The masking function will be applied to all string values in trace data,
        except for keys listed in MASKING_EXCLUDED_KEYS.
        
        Args:
            masking_func: Function that takes a string and returns the masked version
            
        Raises:
            TypeError: If masking_func is not callable
            
        Example:
            >>> def mask_email(text):
            ...     return re.sub(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', '***@***.***', text)
            >>> tracer.register_masking_function(mask_email)
        """
        if not callable(masking_func):
            raise TypeError(f"masking_func must be callable, got {type(masking_func)}")
        
        def recursive_mask_values(obj, parent_key=None):
            try:
                if isinstance(obj, dict):
                    return {k: recursive_mask_values(v, k) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_mask_values(item, parent_key) for item in obj]
                elif isinstance(obj, str):
                    if parent_key and parent_key.lower() not in self.MASKING_EXCLUDED_KEYS:
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
            
            new_filename = f"{self._PROCESSED_FILE_PREFIX}{original_path.name}"
            dir_name = os.path.dirname(original_trace_json_path)
            final_trace_json_path = Path(dir_name) / new_filename
            
            with open(final_trace_json_path, 'w') as f:
                json.dump(data, f, indent=self._JSON_INDENT)
            
            return final_trace_json_path
        
        self.register_post_processor(file_post_processor)
    
    def register_post_processor(self, post_processor_func: Callable) -> None:
        """
        Register a post-processing function for trace files.
        
        Args:
            post_processor_func: Callable that takes a file path and returns processed file path
            
        Raises:
            TypeError: If post_processor_func is not callable
        """
        if not callable(post_processor_func):
            raise TypeError(f"post_processor_func must be callable, got {type(post_processor_func)}")
        
        self.post_processor = post_processor_func
        if self.exporter:
            self.exporter.post_processor = post_processor_func
        
        logger.info(f"Registered post-processor: {post_processor_func.__name__}")
    
    def set_external_id(self, external_id: str) -> None:
        """
        Set external ID for subsequent traces.
        Uses OpenTelemetry context variables to attach to spans.
        
        Args:
            external_id: External identifier to associate with traces
        """
        from ragaai_catalyst.tracers.processors.custom_span_processor import set_trace_config
        set_trace_config(external_id=external_id)
        logger.debug(f"Set external_id: {external_id}")

    def set_dataset_name(self, dataset_name: str) -> None:
        """
        Update dataset name for subsequent traces.
        
        Args:
            dataset_name: New dataset name
        """
        self.dataset_name = dataset_name
        if self.exporter:
            self.exporter.dataset_name = dataset_name
        from ragaai_catalyst.tracers.processors.custom_span_processor import set_trace_config
        set_trace_config(dataset_name=dataset_name)
        logger.debug(f"Set dataset_name: {dataset_name}")

    def set_project_name(self, project_name: str) -> None:
        """
        Update project name for subsequent traces.
        
        Args:
            project_name: New project name
        """
        self.project_name = project_name
        if self.exporter:
            self.exporter.project_name = project_name
        from ragaai_catalyst.tracers.processors.custom_span_processor import set_trace_config
        set_trace_config(project_name=project_name)
        logger.debug(f"Set project_name: {project_name}")
    
    def add_context(self, context: str) -> None:
        """
        Add context to trace. Only supported for LangChain and LlamaIndex tracers.
        Uses OpenTelemetry context variables to attach to spans.
        
        Args:
            context: Context string to associate with traces
            
        Raises:
            ValueError: If tracer_type doesn't support context or context is not a string
        """
        if self.tracer_type not in TracerTypeSupport.CONTEXT_SUPPORTED:
            raise ValueError(
                f"add_context only supported for {', '.join(TracerTypeSupport.CONTEXT_SUPPORTED)}. "
                f"Current tracer_type: {self.tracer_type}"
            )
        
        if not isinstance(context, str):
            raise TypeError(f"context must be a string, got {type(context)}")
        
        from ragaai_catalyst.tracers.processors.custom_span_processor import set_trace_config
        set_trace_config(user_context=context)
        logger.debug(f"Added context: {context[:50]}..." if len(context) > 50 else f"Added context: {context}")
    
    def add_gt(self, gt: str) -> None:
        """
        Add ground truth to trace. Only supported for LangChain and LlamaIndex tracers.
        Uses OpenTelemetry context variables to attach to spans.
        
        Args:
            gt: Ground truth string to associate with traces
            
        Raises:
            ValueError: If tracer_type doesn't support ground truth or gt is not a string
        """
        if self.tracer_type not in TracerTypeSupport.CONTEXT_SUPPORTED:
            raise ValueError(
                f"add_gt only supported for {', '.join(TracerTypeSupport.CONTEXT_SUPPORTED)}. "
                f"Current tracer_type: {self.tracer_type}"
            )
        
        if not isinstance(gt, str):
            raise TypeError(f"gt must be a string, got {type(gt)}")
        
        from ragaai_catalyst.tracers.processors.custom_span_processor import set_trace_config
        set_trace_config(user_gt=gt)
        logger.debug(f"Added ground truth: {gt[:50]}..." if len(gt) > 50 else f"Added ground truth: {gt}")

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Add or update metadata for traces.
        
        Args:
            metadata: Dictionary of metadata key-value pairs to merge
            
        Raises:
            TypeError: If metadata is not a dictionary
        """
        if not isinstance(metadata, dict):
            raise TypeError(f"metadata must be a dictionary, got {type(metadata)}")
        
        user_details = self.user_details
        user_metadata = user_details["trace_user_detail"]["metadata"]
        user_metadata.update(metadata)
        self.metadata = user_metadata
        if self.exporter:
            self.exporter.user_details = self.user_details
        logger.debug(f"Updated metadata with {len(metadata)} keys")
    
    def update_file_list(self) -> None:
        """
        Update the list of files to be included in trace uploads.
        
        Raises:
            RuntimeError: If exporter is not initialized (tracer_type doesn't require instrumentation)
        """
        if not self.exporter:
            raise RuntimeError(
                "Exporter not initialized. Ensure tracer_type requires instrumentation "
                "(e.g., 'langchain', 'openai', 'agentic')"
            )
        
        list_of_unique_files = self.file_tracker.get_unique_files()
        self.exporter.files_to_zip = list_of_unique_files
        logger.debug(f"Updated file list: {len(list_of_unique_files)} files")
    
    def set_feedback(self, external_id: str, feedback: Any) -> Optional[Dict[str, Any]]:
        """
        Submit feedback for a specific trace.
        
        Args:
            external_id: External ID of the trace to attach feedback to
            feedback: Feedback data (can be any JSON-serializable value)
            
        Returns:
            Response data from the API if successful, None otherwise
            
        Raises:
            ValueError: If external_id or feedback is empty
        """
        if not external_id:
            raise ValueError("external_id is required")
        if feedback is None or feedback == "":
            raise ValueError("feedback is required")
        
        try:
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
            
            if response:
                logger.info(f"Feedback submitted successfully for external_id: {external_id}")
                return response.json()
            return None
            
        except (PoolError, MaxRetryError, NewConnectionError, ConnectionError, Timeout, RemoteDisconnected) as e:
            session_manager.handle_request_exceptions(e, "setting feedback")
            return None
        except Exception as e:
            logger.error(f"Error setting feedback: {e}")
            return None
    
    @property
    def tracer(self):
        """Get the OpenTelemetry tracer instance for manual span creation."""
        return self._tracer
    
    def shutdown(self) -> None:
        """
        Shutdown the tracer and cleanup resources.
        
        This flushes any pending spans and releases OpenTelemetry resources.
        Should be called when done tracing, or use the context manager pattern.
        """
        if self._tracer_provider:
            try:
                self._tracer_provider.shutdown()
                logger.info("Tracer provider shut down successfully")
            except Exception as e:
                logger.error(f"Error during tracer shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically shutdown tracer."""
        self.shutdown()
        return False
