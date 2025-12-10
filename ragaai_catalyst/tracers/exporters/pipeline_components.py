"""
Pipeline components for trace export system.

This module contains the individual components that make up the trace export pipeline,
following the Single Responsibility Principle.
"""
import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from ragaai_catalyst.tracers.models import (
    TraceId, TraceFile, CodeArchive, TraceData, TraceMetadata,
    ProcessingContext, HashId
)
from ragaai_catalyst.tracers.json_converter.trace_json_converter import DefaultTraceConverter
from ragaai_catalyst.tracers.utils.system_monitor import SystemMonitor
from ragaai_catalyst.tracers.utils.trace_utils import format_interactions
from ragaai_catalyst.tracers.utils.zip_list_of_unique_files import zip_list_of_unique_files

logger = logging.getLogger("RagaAICatalyst")


class SpanCollector:
    """
    High-performance span collector using lock striping.

    Uses 256 locks sharded by trace_id hash for scalable concurrency.
    """

    NUM_SHARDS = 256

    def __init__(self):
        self._shards = [
            {
                'lock': threading.RLock(),
                'data': {}
            }
            for _ in range(self.NUM_SHARDS)
        ]

    def _get_shard(self, trace_id: TraceId) -> Dict:
        """Get shard for trace_id using hash-based sharding."""
        shard_idx = hash(trace_id) & (self.NUM_SHARDS - 1)
        return self._shards[shard_idx]

    def add_span(self, trace_id: TraceId, span: Dict[str, Any]) -> None:
        """Add a span to its trace buffer."""
        shard = self._get_shard(trace_id)
        with shard['lock']:
            if trace_id not in shard['data']:
                shard['data'][trace_id] = []
            shard['data'][trace_id].append(span)

    def is_trace_complete(self, trace_id: TraceId) -> bool:
        """Check if trace has a root span (parent_id is None)."""
        shard = self._get_shard(trace_id)
        with shard['lock']:
            spans = shard['data'].get(trace_id)
            if not spans:
                return False
            return any(span.get("parent_id") is None for span in spans)

    def get_trace(self, trace_id: TraceId) -> Optional[List[Dict]]:
        """Get all spans for a trace."""
        shard = self._get_shard(trace_id)
        with shard['lock']:
            spans = shard['data'].get(trace_id)
            return list(spans) if spans else None

    def clear_trace(self, trace_id: TraceId) -> None:
        """Remove a trace from the buffer."""
        shard = self._get_shard(trace_id)
        with shard['lock']:
            shard['data'].pop(trace_id, None)

    def get_all_traces(self) -> Dict[TraceId, List[Dict]]:
        """Get all buffered traces (used during shutdown)."""
        all_traces = {}
        for shard in self._shards:
            with shard['lock']:
                for trace_id, spans in shard['data'].items():
                    all_traces[trace_id] = list(spans)
        return all_traces

    def clear_all(self) -> None:
        """Clear all buffered traces."""
        for shard in self._shards:
            with shard['lock']:
                shard['data'].clear()


class TraceFormatter:
    """
    Converts OpenTelemetry spans to RagaAI trace format.

    Responsibilities:
    - Extract external_id from spans
    - Convert span format using TraceConverter
    - Create TraceData domain object
    """

    def __init__(self, trace_converter: Optional[DefaultTraceConverter] = None):
        self._converter = trace_converter or DefaultTraceConverter()

    def format(
        self,
        spans: List[Dict],
        context: ProcessingContext
    ) -> Optional[TraceData]:
        """
        Format spans into RagaAI trace format.

        Args:
            spans: List of OpenTelemetry spans
            context: Processing context with configuration

        Returns:
            TraceData object or None if formatting fails
        """
        converted = self._converter.convert(
            spans,
            context.user_context,
            context.user_gt,
            context.external_id
        )

        if not converted:
            return None

        metadata = TraceMetadata(
            project_name=context.project_name,
            dataset_name=context.dataset_name,
            tracer_type=context.tracer_type
        )

        return TraceData(
            trace_id=context.trace_id,
            data=converted,
            metadata=metadata
        )


class TraceProcessor(ABC):
    """Base class for trace processors in the enrichment pipeline."""

    @abstractmethod
    def process(self, trace_data: TraceData, context: ProcessingContext) -> TraceData:
        """
        Process trace data.

        Args:
            trace_data: Trace data to process
            context: Processing context

        Returns:
            Processed trace data
        """
        pass

    @property
    def is_critical(self) -> bool:
        """Whether this processor is critical (failure should stop pipeline)."""
        return False


class WorkflowEnricher(TraceProcessor):
    """Adds workflow interactions to trace."""

    def process(self, trace_data: TraceData, context: ProcessingContext) -> TraceData:
        try:
            interactions = format_interactions(trace_data.data)
            trace_data.data["workflow"] = interactions.get('workflow', {})
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Could not format workflow interactions for {context.trace_id}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error formatting interactions for {context.trace_id}: {e}")

        return trace_data


class SystemMetadataEnricher(TraceProcessor):
    """Adds system metadata and source code hash to trace."""

    def __init__(self, system_monitor: Optional[SystemMonitor] = None):
        self._system_monitor = system_monitor

    @property
    def is_critical(self) -> bool:
        return True

    def process(self, trace_data: TraceData, context: ProcessingContext) -> TraceData:
        if not self._system_monitor:
            self._system_monitor = SystemMonitor(context.dataset_name)

        try:
            system_info = asdict(self._system_monitor.get_system_info())
            resources = asdict(self._system_monitor.get_resources())

            if context.code_hash:
                system_info["source_code"] = context.code_hash

            trace_data.data.setdefault("metadata", {}).update({
                "system_info": system_info,
                "resources": resources
            })
        except (AttributeError, TypeError) as e:
            logger.exception(f"Failed to collect system metadata for {context.trace_id}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error adding system metadata for {context.trace_id}: {e}")
            raise

        return trace_data


class TimestampEnricher(TraceProcessor):
    """Adds start/end timestamps to trace data array."""

    def process(self, trace_data: TraceData, context: ProcessingContext) -> TraceData:
        try:
            data_array = trace_data.data.get("data", [])
            if data_array and len(data_array) > 0:
                data_array[0]["start_time"] = trace_data.data.get("start_time")
                data_array[0]["end_time"] = trace_data.data.get("end_time")
            else:
                logger.warning(f"Data array empty for {context.trace_id}")
        except (KeyError, TypeError, IndexError) as e:
            logger.warning(f"Could not add timestamps for {context.trace_id}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error adding timestamps for {context.trace_id}: {e}")

        return trace_data


class ProjectInfoEnricher(TraceProcessor):
    """Adds project and tracer information to trace."""

    def process(self, trace_data: TraceData, context: ProcessingContext) -> TraceData:
        try:
            trace_data.data["project_name"] = context.project_name
            trace_data.data["tracer_type"] = context.tracer_type
        except (TypeError, AttributeError) as e:
            logger.warning(f"Could not add project info for {context.trace_id}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error adding project info for {context.trace_id}: {e}")

        return trace_data


class UserMetadataEnricher(TraceProcessor):
    """Adds user-provided metadata to trace."""

    EXCLUDED_KEYS = {"log_source", "recorded_on"}

    def process(self, trace_data: TraceData, context: ProcessingContext) -> TraceData:
        try:
            if not context.user_details:
                return trace_data

            metadata = context.user_details.get("trace_user_detail", {}).get("metadata", {})

            if isinstance(metadata, dict):
                trace_metadata = trace_data.data.setdefault("metadata", {})
                for key, value in metadata.items():
                    if key not in self.EXCLUDED_KEYS:
                        trace_metadata[key] = value
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Could not add user metadata for {context.trace_id}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error adding user metadata for {context.trace_id}: {e}")

        return trace_data


class SourceCodeArchiver:
    """Creates source code archives with hash."""

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir

    def create_archive(
        self,
        files_to_zip: List[str],
        trace_id: TraceId
    ) -> Optional[CodeArchive]:
        """
        Create a zip archive of source files.

        Args:
            files_to_zip: List of file paths to include
            trace_id: Trace identifier

        Returns:
            CodeArchive object or None if creation fails

        Raises:
            IOError: If archive cannot be created
            OSError: If filesystem error occurs
        """
        try:
            hash_id, zip_path = zip_list_of_unique_files(
                files_to_zip,
                output_dir=str(self._output_dir)
            )

            return CodeArchive(
                path=Path(zip_path),
                hash_id=HashId(hash_id),
                file_count=len(files_to_zip)
            )
        except (IOError, OSError, PermissionError) as e:
            logger.exception(f"Failed to create code archive for {trace_id}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error creating code archive for {trace_id}: {e}")
            raise


class TraceSerializer:
    """
    Serializes trace data to JSON files.

    Responsibilities:
    - Convert TraceData to JSON
    - Write to filesystem
    - Create TraceFile domain object
    """

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, trace_data: TraceData) -> Optional[TraceFile]:
        """
        Save trace data to JSON file.

        Args:
            trace_data: Trace data to serialize

        Returns:
            TraceFile object or None if save fails

        Raises:
            IOError: If file cannot be written
            OSError: If filesystem error occurs
            TypeError: If data cannot be serialized
        """
        try:
            from ragaai_catalyst.tracers.exporters.ragaai_trace_exporter import TracerJSONEncoder

            file_path = self._output_dir / f"{trace_data.trace_id}.json"

            with open(file_path, "w") as f:
                json.dump(trace_data.data, f, cls=TracerJSONEncoder, indent=2)

            trace_file = TraceFile(
                path=file_path,
                trace_id=trace_data.trace_id,
                size_bytes=file_path.stat().st_size
            )

            return trace_file

        except (IOError, OSError, PermissionError) as e:
            logger.exception(f"Failed to write trace file for {trace_data.trace_id}: {e}")
            raise
        except (TypeError, ValueError) as e:
            logger.exception(f"Failed to serialize trace data for {trace_data.trace_id}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error saving trace {trace_data.trace_id}: {e}")
            raise
