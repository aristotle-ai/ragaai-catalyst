import json
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytz

logger = logging.getLogger(__name__)


class SpanAttributes:
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
    OK = "OK"
    ERROR = "ERROR"


class SpanKind:
    INTERNAL = "SpanKind.INTERNAL"


DEFAULT_TIMEZONE = os.getenv("RAGAAI_TIMEZONE", "UTC")


def convert_time_format(
    original_time_str: str, target_timezone_str: Optional[str] = None
) -> str:
    if target_timezone_str is None:
        target_timezone_str = DEFAULT_TIMEZONE

    utc_time = datetime.strptime(original_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    utc_time = utc_time.replace(tzinfo=pytz.UTC)
    
    target_timezone = pytz.timezone(target_timezone_str)
    target_time = utc_time.astimezone(target_timezone)
    
    formatted_time = target_time.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    formatted_time = formatted_time[:-2] + ":" + formatted_time[-2:]
    
    return formatted_time


def get_uuid(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))


def _process_span_naming(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    name_counts = defaultdict(int)

    for span in spans:
        span_name = span["name"]
        span["name_occurrences"] = name_counts[span_name]
        name_counts[span_name] += 1
        span["name"] = f"{span_name}.{span['name_occurrences']}"
        span["hash_id"] = get_uuid(span["name"])

    return spans


def get_spans(input_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    data = input_trace.copy()
    return _process_span_naming(data)


def _find_model_name(spans: List[Dict[str, Any]]) -> str:
    for span in reversed(spans):
        model_name = span.get("attributes", {}).get(SpanAttributes.MODEL_NAME)
        if model_name:
            return model_name

    for span in reversed(spans):
        attrs = span.get("attributes", {})
        if attrs.get(SpanAttributes.KIND) == SpanAttributes.LLM_KIND:
            try:
                metadata_str = attrs.get(SpanAttributes.METADATA) or attrs.get(
                    SpanAttributes.AIQ_METADATA
                )
                if metadata_str:
                    metadata = json.loads(metadata_str)
                    model_name = metadata.get("ls_model_name", "")
                    if model_name:
                        return model_name
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse metadata: {e}")
                continue

    return ""


def _create_base_trace(
    input_trace: List[Dict[str, Any]], external_id: Optional[str]
) -> Dict[str, Any]:
    trace_id = input_trace[0]["context"]["trace_id"]
    
    start_times = [item["start_time"] for item in input_trace]
    end_times = [item["end_time"] for item in input_trace]

    return {
        "id": trace_id,
        "trace_name": "",
        "project_name": "",
        "start_time": convert_time_format(min(start_times)),
        "end_time": convert_time_format(max(end_times)),
        "external_id": external_id,
        "metadata": {},
        "replays": {"source": None},
        "data": [{}],
        "network_calls": [],
        "interactions": [],
    }


def _create_custom_span(
    text: str, span_type: str, trace_id: str, parent_id: Optional[str]
) -> Dict[str, Any]:
    status = {"status_code": StatusCode.OK}
    
    try:
        current_time = convert_time_format(
            datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        )
    except Exception as e:
        logger.warning(f"Error creating custom {span_type} span: {e}")
        current_time = convert_time_format(
            datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        )
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
        "hash_id": get_uuid(f"Custom{span_type}Span"),
    }


def _add_custom_spans(
    spans: List[Dict[str, Any]],
    trace_id: str,
    parent_id: Optional[str],
    user_context: Optional[str],
    user_gt: Optional[str],
) -> List[Dict[str, Any]]:
    if user_context:
        try:
            context_span = _create_custom_span(user_context, "Context", trace_id, parent_id)
            spans.append(context_span)
        except Exception as e:
            logger.error(f"Error adding context: {e}")

    if user_gt:
        try:
            gt_span = _create_custom_span(user_gt, "GroundTruth", trace_id, parent_id)
            spans.append(gt_span)
        except Exception as e:
            logger.error(f"Error adding ground truth: {e}")

    return spans


def convert_json_format(
    input_trace: List[Dict[str, Any]],
    user_context: Optional[str],
    user_gt: Optional[str],
    external_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    try:
        final_trace = _create_base_trace(input_trace, external_id)
        
        spans = get_spans(input_trace)
        
        trace_id = final_trace["id"]
        parent_id = spans[0].get("parent_id") if spans else None
        spans = _add_custom_spans(spans, trace_id, parent_id, user_context, user_gt)
        
        final_trace["data"][0]["spans"] = spans
        return final_trace

    except Exception as e:
        logger.error(f"Error in convert_json_format: {e}")
        return None


def custom_spans(
    text: str, span_type: str, trace_id: str, parent_id: Optional[str]
) -> Dict[str, Any]:
    return _create_custom_span(text, span_type, trace_id, parent_id)
