"""CustomSpanProcessor - Attaches per-trace configuration to root spans."""
import json
import logging
from contextvars import ContextVar
from typing import Optional, Dict, Any, Callable
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.context import Context

logger = logging.getLogger(__name__)

_trace_config: ContextVar[Dict[str, Any]] = ContextVar('trace_config', default={})


class CustomSpanProcessor(SpanProcessor):
    """Attaches per-trace configuration to root spans via span attributes."""

    ATTR_EXTERNAL_ID = "ragaai.external_id"
    ATTR_USER_CONTEXT = "ragaai.user_context"
    ATTR_USER_GT = "ragaai.user_gt"
    ATTR_PROJECT_NAME = "ragaai.project_name"
    ATTR_DATASET_NAME = "ragaai.dataset_name"
    ATTR_CUSTOM_METADATA = "ragaai.custom_metadata"

    def on_start(self, span: "Span", parent_context: Optional[Context] = None) -> None:
        try:
            if span.parent is None:
                config = _trace_config.get({})
                if not config:
                    return

                if external_id := config.get('external_id'):
                    span.set_attribute(self.ATTR_EXTERNAL_ID, str(external_id))

                if user_context := config.get('user_context'):
                    span.set_attribute(self.ATTR_USER_CONTEXT, str(user_context))

                if user_gt := config.get('user_gt'):
                    span.set_attribute(self.ATTR_USER_GT, str(user_gt))

                if project_name := config.get('project_name'):
                    span.set_attribute(self.ATTR_PROJECT_NAME, str(project_name))

                if dataset_name := config.get('dataset_name'):
                    span.set_attribute(self.ATTR_DATASET_NAME, str(dataset_name))

                if custom_metadata := config.get('custom_metadata'):
                    try:
                        span.set_attribute(self.ATTR_CUSTOM_METADATA, json.dumps(custom_metadata))
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Failed to serialize custom metadata: {e}")

        except Exception as e:
            logger.exception(f"Error in CustomSpanProcessor.on_start: {e}")

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class trace_context:
    """Context manager to set per-trace configuration."""

    def __init__(
        self,
        external_id: Optional[str] = None,
        user_context: Optional[str] = None,
        user_gt: Optional[str] = None,
        project_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        post_processor: Optional[Callable] = None,
        **custom_metadata
    ):
        self.config = {
            'external_id': external_id,
            'user_context': user_context,
            'user_gt': user_gt,
            'project_name': project_name,
            'dataset_name': dataset_name,
            'post_processor': post_processor,
            'custom_metadata': custom_metadata if custom_metadata else None
        }
        self.config = {k: v for k, v in self.config.items() if v is not None}
        self.token = None

    def __enter__(self):
        self.token = _trace_config.set(self.config)
        return self

    def __exit__(self, *args):
        if self.token:
            _trace_config.reset(self.token)


def set_trace_config(
    external_id: Optional[str] = None,
    user_context: Optional[str] = None,
    user_gt: Optional[str] = None,
    project_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    post_processor: Optional[Callable] = None,
    **custom_metadata
) -> None:
    current = _trace_config.get({}).copy()

    if external_id is not None:
        current['external_id'] = external_id
    if user_context is not None:
        current['user_context'] = user_context
    if user_gt is not None:
        current['user_gt'] = user_gt
    if project_name is not None:
        current['project_name'] = project_name
    if dataset_name is not None:
        current['dataset_name'] = dataset_name
    if post_processor is not None:
        current['post_processor'] = post_processor

    if custom_metadata:
        if 'custom_metadata' not in current:
            current['custom_metadata'] = {}
        current['custom_metadata'].update(custom_metadata)

    _trace_config.set(current)


def get_trace_config() -> Dict[str, Any]:
    return _trace_config.get({}).copy()


def clear_trace_config() -> None:
    _trace_config.set({})


set_trace_metadata = set_trace_config
get_trace_metadata = get_trace_config
clear_trace_metadata = clear_trace_config
