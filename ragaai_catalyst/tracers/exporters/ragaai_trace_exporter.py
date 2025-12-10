"""
RagaAI Trace Exporter - High-performance OpenTelemetry span exporter.
"""
import json
import logging
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from ragaai_catalyst.tracers.uploader.uploader import AbstractTraceUploader
from ragaai_catalyst.tracers.models import TraceId, ProcessingContext, UploadRequest
from ragaai_catalyst.tracers.exporters.trace_pipeline import TraceExportPipeline

logger = logging.getLogger("RagaAICatalyst")
logger.setLevel(logging.DEBUG if os.getenv("DEBUG") == "1" else logging.INFO)


class TracerJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, bytes):
            try:
                return obj.decode("utf-8")
            except UnicodeDecodeError:
                return str(obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return {
                k: v
                for k, v in obj.__dict__.items()
                if v is not None and not k.startswith("_")
            }
        try:
            return str(obj)
        except Exception:
            return None


class RAGATraceExporter(SpanExporter):
    """OpenTelemetry span exporter for RagaAI platform."""

    OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
    DEFAULT_SPAN_KIND = "UNKNOWN"
    EXCLUDED_METADATA_KEYS = {"log_source", "recorded_on"}

    def __init__(
            self,
            project_name: str,
            dataset_name: str,
            base_url: str,
            tracer_type: str,
            files_to_zip: Optional[List[str]] = None,
            user_details: Optional[Dict] = None,
            timeout: int = 120,
            post_processor: Optional[Callable] = None,
            max_upload_workers: int = 30,
            uploader: Optional[AbstractTraceUploader] = None
    ):
        self.project_name = project_name
        self.dataset_name = dataset_name
        self.base_url = base_url
        self.tracer_type = tracer_type
        self.timeout = timeout
        self.files_to_zip = files_to_zip or []
        self.user_details = user_details
        self.post_processor = post_processor
        self.max_upload_workers = max_upload_workers
        self._lock = threading.Lock()

        if uploader is None:
            from ragaai_catalyst.tracers.uploader.trace_uploader import TraceUploader
            uploader = TraceUploader.get_instance()
        self.uploader = uploader

        custom_dir = os.getenv("RAGAAI_TRACE_DIR")
        if custom_dir:
            try:
                os.makedirs(custom_dir, exist_ok=True)
                self.tmp_dir = Path(custom_dir)
            except Exception:
                self.tmp_dir = Path(tempfile.gettempdir())
        else:
            self.tmp_dir = Path(tempfile.gettempdir())

        print("tmp directory", self.tmp_dir)
        self._pipeline = TraceExportPipeline(output_dir=self.tmp_dir)

    def export(self, spans: Any) -> SpanExportResult:
        for span in spans:
            try:
                span_json = json.loads(span.to_json())
                trace_id = TraceId(span_json.get("context", {}).get("trace_id"))

                if not trace_id:
                    logger.error("Trace ID is None")
                    continue

                if span_json.get("attributes", {}).get(self.OPENINFERENCE_SPAN_KIND) is None:
                    span_json["attributes"][self.OPENINFERENCE_SPAN_KIND] = self.DEFAULT_SPAN_KIND

                self._pipeline.add_span(trace_id, span_json)

                if span_json.get("parent_id") is None:
                    try:
                        self._process_complete_trace(trace_id, span_json)
                    except Exception as e:
                        logger.exception(f"Error processing complete trace {trace_id}: {e}")

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.error(f"Invalid span data, skipping: {e}")
                continue
            except Exception as e:
                logger.exception(f"Unexpected error processing span: {e}")
                continue

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        base_context = self._create_processing_context(TraceId("shutdown"))
        remaining_traces = self._pipeline.process_all_remaining(base_context)

        for prepared_trace in remaining_traces:
            try:
                trace_context = self._create_processing_context(
                    prepared_trace.trace_data.trace_id
                )
                self._upload_prepared_trace(prepared_trace, trace_context)
            except Exception as e:
                logger.exception(f"Error uploading trace {prepared_trace.trace_data.trace_id}: {e}")

    def _process_complete_trace(self, trace_id: TraceId, root_span: Dict[str, Any]) -> None:
        context = self._create_processing_context(trace_id, root_span)
        prepared_trace = self._pipeline.process_trace(context)

        if not prepared_trace:
            logger.error(f"Failed to prepare trace {trace_id}")
            return

        self._upload_prepared_trace(prepared_trace, context)

    def _create_processing_context(self, trace_id: TraceId, root_span: Optional[Dict[str, Any]] = None) -> ProcessingContext:
        project_name = self.project_name
        dataset_name = self.dataset_name
        user_context = None
        user_gt = None
        external_id = None

        if root_span:
            attrs = root_span.get('attributes', {})
            project_name = attrs.get('ragaai.project_name', project_name)
            dataset_name = attrs.get('ragaai.dataset_name', dataset_name)
            user_context = attrs.get('ragaai.user_context', user_context)
            user_gt = attrs.get('ragaai.user_gt', user_gt)
            external_id = attrs.get('ragaai.external_id', external_id)

        return ProcessingContext(
            trace_id=trace_id,
            project_name=project_name,
            dataset_name=dataset_name,
            base_url=self.base_url,
            tracer_type=self.tracer_type,
            timeout=self.timeout,
            files_to_zip=self.files_to_zip.copy(),
            user_details=self.user_details,
            user_context=user_context,
            user_gt=user_gt,
            external_id=external_id
        )

    def _upload_prepared_trace(self, prepared_trace, context: ProcessingContext) -> None:
        """
        Submit prepared trace for upload using type-safe UploadRequest.

        Args:
            prepared_trace: PreparedTrace with files ready for upload
            context: ProcessingContext with metadata
        """
        filepath = prepared_trace.trace_file.path

        if self.post_processor:
            try:
                processed_filepath = self.post_processor(filepath)
                logger.info(f"Post-processor transformed {filepath} -> {processed_filepath}")
                filepath = processed_filepath
            except Exception as e:
                logger.exception(f"Post-processor failed for {prepared_trace.trace_data.trace_id}: {e}")

        # Create upload request with processed filepath
        upload_request = UploadRequest(
            trace_file=filepath,
            code_archive=prepared_trace.code_archive,
            project_name=context.project_name,
            dataset_name=context.dataset_name,
            base_url=context.base_url,
            tracer_type=context.tracer_type,
            user_details=context.user_details or {},
            timeout=context.timeout
        )

        upload_task_id = self.uploader.submit(upload_request)
        logger.info(f"Submitted upload task with ID: {upload_task_id}")
