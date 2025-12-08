"""
Trace export pipeline orchestrator.

This module contains the main pipeline that coordinates all trace export operations,
following the Pipeline and Chain of Responsibility patterns.
"""
import logging
from pathlib import Path
from typing import List, Optional

from ragaai_catalyst.tracers.models import (
    TraceId, PreparedTrace, ProcessingContext
)
from ragaai_catalyst.tracers.exporters.pipeline_components import (
    SpanCollector,
    TraceFormatter,
    TraceProcessor,
    WorkflowEnricher,
    SystemMetadataEnricher,
    TimestampEnricher,
    ProjectInfoEnricher,
    UserMetadataEnricher,
    SourceCodeArchiver,
    TraceSerializer
)

logger = logging.getLogger("RagaAICatalyst")


class TraceProcessingPipeline:
    """
    Extensible pipeline for processing traces.

    This pipeline applies a sequence of processors to enrich trace data.
    Processors can be added, removed, or reordered easily.
    """

    def __init__(self, processors: List[TraceProcessor]):
        self._processors = processors

    def process(self, trace_data, context: ProcessingContext):
        """
        Apply all processors to trace data.

        Args:
            trace_data: Trace data to process
            context: Processing context

        Returns:
            Processed trace data

        Raises:
            Exception: If a critical processor fails
        """
        result = trace_data

        for processor in self._processors:
            try:
                result = processor.process(result, context)
            except Exception as e:
                logger.exception(
                    f"Processor {processor.__class__.__name__} failed for {context.trace_id} with exception {e}"
                )
                if processor.is_critical:
                    raise
        return result


class TraceExportPipeline:
    """
    Main pipeline orchestrator for trace export.

    Coordinates all components of the trace export process:
    1. Span collection and buffering
    2. Trace formatting
    3. Metadata enrichment
    4. Source code archiving
    5. File serialization
    6. Preparation for upload

    This class follows the Façade pattern, providing a simple interface
    to a complex subsystem.
    """

    def __init__(
        self,
        output_dir: Path,
        collector: Optional[SpanCollector] = None,
        formatter: Optional[TraceFormatter] = None,
        enrichment_pipeline: Optional[TraceProcessingPipeline] = None,
        archiver: Optional[SourceCodeArchiver] = None,
        serializer: Optional[TraceSerializer] = None
    ):
        """
        Initialize the pipeline with components.

        Args:
            output_dir: Directory for output files
            collector: Span collector (created if None)
            formatter: Trace formatter (created if None)
            enrichment_pipeline: Processing pipeline (created if None)
            archiver: Source code archiver (created if None)
            serializer: Trace serializer (created if None)
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._collector = collector or SpanCollector()
        self._formatter = formatter or TraceFormatter()
        self._enrichment_pipeline = enrichment_pipeline or self._create_default_pipeline()
        self._archiver = archiver or SourceCodeArchiver(self._output_dir)
        self._serializer = serializer or TraceSerializer(self._output_dir)

    @staticmethod
    def _create_default_pipeline() -> TraceProcessingPipeline:
        """Create default enrichment pipeline."""
        return TraceProcessingPipeline([
            WorkflowEnricher(),
            SystemMetadataEnricher(),
            TimestampEnricher(),
            ProjectInfoEnricher(),
            UserMetadataEnricher(),
        ])

    def add_span(self, trace_id: TraceId, span: dict) -> None:
        """Add a span to the collection buffer."""
        self._collector.add_span(trace_id, span)

    def is_trace_complete(self, trace_id: TraceId) -> bool:
        """Check if a trace is complete and ready for processing."""
        return self._collector.is_trace_complete(trace_id)

    def process_trace(self, context: ProcessingContext) -> Optional[PreparedTrace]:
        """
        Process a complete trace through the entire pipeline.

        Args:
            context: Processing context with all configuration

        Returns:
            PreparedTrace ready for upload, or None if processing fails
        """
        trace_id = context.trace_id

        # 1. Get spans from collector
        spans = self._collector.get_trace(trace_id)
        if not spans:
            logger.error(f"No spans found for trace {trace_id}")
            return None

        # 2. Format to RagaAI format
        trace_data = self._formatter.format(spans, context)
        if not trace_data:
            logger.error(f"Failed to format trace {trace_id}")
            return None

        # 3. Create source code archive
        code_archive = self._archiver.create_archive(
            context.files_to_zip or [],
            trace_id
        )
        if not code_archive:
            logger.error(f"Failed to create code archive for {trace_id}")
            return None

        # 4. Update context with code hash
        context.code_hash = code_archive.hash_id
        context.code_zip_path = code_archive.path

        try:
            enriched_trace = self._enrichment_pipeline.process(trace_data, context)
        except Exception as e:
            logger.exception(f"Critical error in enrichment pipeline for {trace_id}: {e}")
            return None

        # 6. Serialize to file
        trace_file = self._serializer.save(enriched_trace)
        if not trace_file:
            logger.error(f"Failed to save trace file for {trace_id}")
            return None

        # 7. Create prepared trace
        prepared = PreparedTrace(
            trace_file=trace_file,
            code_archive=code_archive,
            trace_data=enriched_trace
        )

        # 8. Clear from buffer
        self._collector.clear_trace(trace_id)

        return prepared

    def process_all_remaining(self, base_context: ProcessingContext) -> List[PreparedTrace]:
        """
        Process all remaining buffered traces (called during shutdown).

        Args:
            base_context: Base context to use for all traces

        Returns:
            List of prepared traces
        """
        prepared_traces = []

        for trace_id, spans in self._collector.get_all_traces().items():
            context = ProcessingContext(
                trace_id=trace_id,
                project_name=base_context.project_name,
                dataset_name=base_context.dataset_name,
                base_url=base_context.base_url,
                tracer_type=base_context.tracer_type,
                timeout=base_context.timeout,
                files_to_zip=base_context.files_to_zip,
                user_details=base_context.user_details,
                user_context=base_context.user_context,
                user_gt=base_context.user_gt,
                external_id=base_context.external_id
            )

            prepared = self.process_trace(context)
            if prepared:
                prepared_traces.append(prepared)

        self._collector.clear_all()
        return prepared_traces

    def get_collector(self) -> SpanCollector:
        """Get the span collector (for direct access if needed)."""
        return self._collector
