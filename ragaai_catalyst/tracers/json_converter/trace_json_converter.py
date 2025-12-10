import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytz

logger = logging.getLogger(__name__)


class SpanAttributes:
    """Constants for span attribute keys."""
    KIND = "openinference.span.kind"
    LLM_KIND = "LLM"
    UNKNOWN_KIND = "UNKNOWN"
    PROMPT_TOKENS = "llm.token_count.prompt"
    COMPLETION_TOKENS = "llm.token_count.completion"
    MODEL_NAME = "llm.model_name"
    INPUT_VALUE = "input.value"
    OUTPUT_VALUE = "output.value"
    METADATA = "metadata"
    AIQ_METADATA = "aiq.metadata"
    LLM_COST = "llm.cost"


class StatusCode:
    """Constants for span status codes."""
    OK = "OK"
    ERROR = "ERROR"


class SpanKind:
    """Constants for span kinds."""
    INTERNAL = "SpanKind.INTERNAL"


DEFAULT_TIMEZONE = os.getenv("RAGAAI_TIMEZONE", "UTC")


class TraceConverter(ABC):
    """
    Abstract base class for converting trace data into a specific JSON format.
    """

    @abstractmethod
    def convert(
        self,
        input_trace: List[Dict[str, Any]],
        user_context: Optional[str] = None,
        user_gt: Optional[str] = None,
        external_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Convert the input trace data.

        Args:
            input_trace: List of span dictionaries.
            user_context: Optional user context string.
            user_gt: Optional user ground truth string.
            external_id: Optional external ID for the trace.

        Returns:
            A dictionary representing the converted trace, or None if conversion fails.
        """
        pass


class DefaultTraceConverter(TraceConverter):
    """
    Default implementation of TraceConverter.

    Converts OpenTelemetry trace spans into RagaAI's internal trace format.
    """

    def convert(
        self,
        input_trace: List[Dict[str, Any]],
        user_context: Optional[str] = None,
        user_gt: Optional[str] = None,
        external_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not input_trace:
            logger.error("Cannot convert empty trace")
            return None

        if not isinstance(input_trace, list):
            logger.error(f"input_trace must be a list, got {type(input_trace)}")
            return None

        try:
            root_span = next((span for span in input_trace if span.get('parent_id') is None), None)
            if root_span:
                attrs = root_span.get('attributes', {})
                external_id = attrs.get('ragaai.external_id', external_id)
                user_context = attrs.get('ragaai.user_context', user_context)
                user_gt = attrs.get('ragaai.user_gt', user_gt)

            final_trace = self._create_base_trace(input_trace, external_id)
            spans = self._get_spans(input_trace)

            trace_id = final_trace["id"]
            parent_id = spans[0].get("parent_id") if spans else None
            spans = self._add_custom_spans(
                spans, trace_id, parent_id, user_context, user_gt
            )

            final_trace["data"][0]["spans"] = spans
            return final_trace

        except KeyError as e:
            logger.exception(f"Missing required field in trace data for trace_id={trace_id}: {e}")
            raise
        except (TypeError, AttributeError, ValueError) as e:
            logger.exception(f"Invalid trace data structure for trace_id={trace_id}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error converting trace {trace_id}: {e}")
            raise

    @staticmethod
    def _convert_time_format(
            original_time_str: str, target_timezone_str: Optional[str] = None
    ) -> str:
        """
        Convert time string to target timezone format.

        Args:
            original_time_str: UTC time string in format "%Y-%m-%dT%H:%M:%S.%fZ"
            target_timezone_str: Target timezone name (default: DEFAULT_TIMEZONE)

        Returns:
            Formatted time string in target timezone

        Raises:
            ValueError: If timezone string is invalid
            ValueError: If time string format is invalid
        """
        if target_timezone_str is None:
            target_timezone_str = DEFAULT_TIMEZONE

        try:
            target_timezone = pytz.timezone(target_timezone_str)
        except pytz.exceptions.UnknownTimeZoneError as e:
            logger.error(f"Invalid timezone '{target_timezone_str}': {e}")
            raise ValueError(f"Invalid timezone: {target_timezone_str}") from e

        try:
            utc_time = datetime.strptime(original_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError as e:
            logger.error(f"Invalid time format '{original_time_str}': {e}")
            raise ValueError(f"Invalid time format: {original_time_str}") from e

        utc_time = utc_time.replace(tzinfo=pytz.UTC)
        target_time = utc_time.astimezone(target_timezone)

        formatted_time = target_time.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        # Add colon in timezone offset (e.g., +0530 -> +05:30)
        formatted_time = formatted_time[:-2] + ":" + formatted_time[-2:]

        return formatted_time

    @staticmethod
    def _get_uuid(name: str) -> str:
        """Generate deterministic UUID from name."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))

    def _process_span_naming(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add occurrence count and hash_id to spans with duplicate names.

        Mutates the input spans list.

        Args:
            spans: List of span dictionaries

        Returns:
            The same spans list (mutated)
        """
        name_counts = defaultdict(int)

        for span in spans:
            span_name = span["name"]
            span["name_occurrences"] = name_counts[span_name]
            name_counts[span_name] += 1
            span["name"] = f"{span_name}.{span['name_occurrences']}"
            span["hash_id"] = self._get_uuid(span["name"])

        return spans

    def _get_spans(self, input_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Shallow copy input trace and process span naming.

        Creates shallow copies of span dictionaries, which is sufficient since
        we only modify top-level fields (name, name_occurrences, hash_id).
        This is 10-100x faster than deep copy for large traces.

        Args:
            input_trace: Original trace spans

        Returns:
            Shallow copied and processed spans
        """
        data = [span.copy() for span in input_trace]
        return self._process_span_naming(data)

    def _create_base_trace(
        self, input_trace: List[Dict[str, Any]], external_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Create base trace structure from input spans.

        Args:
            input_trace: List of span dictionaries (must not be empty)
            external_id: Optional external identifier

        Returns:
            Base trace dictionary structure

        Raises:
            ValueError: If input_trace is empty
            KeyError: If required fields are missing
        """
        if not input_trace:
            raise ValueError("input_trace cannot be empty")

        try:
            trace_id = input_trace[0]["context"]["trace_id"]
        except (KeyError, IndexError) as e:
            logger.error(f"Invalid trace structure: {e}")
            raise KeyError("Missing required field: context.trace_id") from e

        try:
            start_times = [item["start_time"] for item in input_trace]
            end_times = [item["end_time"] for item in input_trace]
        except KeyError as e:
            logger.error(f"Missing time field in trace: {e}")
            raise KeyError("Missing required time fields") from e

        return {
            "id": trace_id,
            "trace_name": "",
            "project_name": "",
            "start_time": self._convert_time_format(min(start_times)),
            "end_time": self._convert_time_format(max(end_times)),
            "external_id": external_id,
            "metadata": {},
            "replays": {"source": None},
            "data": [{}],
            "network_calls": [],
            "interactions": [],
        }

    def _create_custom_span(
        self, text: str, span_type: str, trace_id: str, parent_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Create a custom span (e.g., for context or ground truth).

        Args:
            text: Content for the span
            span_type: Type of custom span (e.g., "Context", "GroundTruth")
            trace_id: ID of the parent trace
            parent_id: Optional parent span ID

        Returns:
            Custom span dictionary
        """
        status = {"status_code": StatusCode.OK}

        try:
            time_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            current_time = self._convert_time_format(time_str)
        except (ValueError, Exception) as e:
            # Fallback to ISO format if conversion fails
            logger.warning(f"Error converting time for custom {span_type} span: {e}")
            current_time = datetime.now().isoformat()
            status = {"status_code": StatusCode.ERROR, "description": str(e)}

        return {
            "name": f"Custom{span_type}Span",
            "context": {
                "trace_id": trace_id,
                "span_id": f"0x{uuid.uuid4().hex[:16]}",
                "trace_state": "[]",
            },
            "kind": SpanKind.INTERNAL,
            "parent_id": parent_id,
            "start_time": current_time,
            "end_time": current_time,
            "status": status,
            "attributes": {
                "input.value": text,
                SpanAttributes.KIND: SpanAttributes.UNKNOWN_KIND,
            },
            "events": [],
            "name_occurrences": 0,
            "hash_id": self._get_uuid(f"Custom{span_type}Span"),
        }

    def _add_custom_spans(
        self,
        spans: List[Dict[str, Any]],
        trace_id: str,
        parent_id: Optional[str],
        user_context: Optional[str],
        user_gt: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Add custom spans for user context and ground truth if provided.

        Args:
            spans: Existing spans list
            trace_id: Trace identifier
            parent_id: Parent span ID for custom spans
            user_context: Optional user context text
            user_gt: Optional ground truth text

        Returns:
            Spans list with custom spans appended (mutates input)
        """
        if user_context:
            try:
                context_span = self._create_custom_span(
                    user_context, "Context", trace_id, parent_id
                )
                spans.append(context_span)
            except (TypeError, ValueError) as e:
                logger.warning(f"Could not add context span for {trace_id}: {e}")
            except Exception as e:
                logger.exception(f"Unexpected error adding context span for {trace_id}: {e}")

        if user_gt:
            try:
                gt_span = self._create_custom_span(
                    user_gt, "GroundTruth", trace_id, parent_id
                )
                spans.append(gt_span)
            except (TypeError, ValueError) as e:
                logger.warning(f"Could not add ground truth span for {trace_id}: {e}")
            except Exception as e:
                logger.exception(f"Unexpected error adding ground truth span for {trace_id}: {e}")

        return spans


# Module-level singleton for efficient reuse
_DEFAULT_CONVERTER = DefaultTraceConverter()


def convert_json_format(
    input_trace: List[Dict[str, Any]],
    user_context: Optional[str],
    user_gt: Optional[str],
    external_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Convert trace to JSON format using the default converter.

    This is a convenience function for backward compatibility.
    Reuses a module-level singleton converter instance for efficiency.

    Args:
        input_trace: List of span dictionaries
        user_context: Optional user context string
        user_gt: Optional user ground truth string
        external_id: Optional external ID for the trace

    Returns:
        Converted trace dictionary, or None if conversion fails
    """
    return _DEFAULT_CONVERTER.convert(input_trace, user_context, user_gt, external_id)


def custom_spans(
    text: str, span_type: str, trace_id: str, parent_id: Optional[str]
) -> Dict[str, Any]:
    """
    Create a custom span using the default converter.

    This is a convenience function for backward compatibility.
    Reuses a module-level singleton converter instance for efficiency.

    Note: This function accesses a private method for backward compatibility.
    Consider using DefaultTraceConverter directly in new code.

    Args:
        text: Content for the span
        span_type: Type of custom span
        trace_id: Trace identifier
        parent_id: Optional parent span ID

    Returns:
        Custom span dictionary
    """
    return _DEFAULT_CONVERTER._create_custom_span(text, span_type, trace_id, parent_id)


def convert_time_format(
    original_time_str: str, target_timezone_str: Optional[str] = None
) -> str:
    """
    Convert time string to target timezone format.

    This is a convenience function for backward compatibility.
    Reuses a module-level singleton converter instance for efficiency.

    Args:
        original_time_str: UTC time string
        target_timezone_str: Target timezone name

    Returns:
        Formatted time string in target timezone

    Raises:
        ValueError: If timezone or time format is invalid
    """
    return _DEFAULT_CONVERTER._convert_time_format(original_time_str, target_timezone_str)
