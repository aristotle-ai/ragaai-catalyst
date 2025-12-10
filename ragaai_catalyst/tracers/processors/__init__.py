"""Span processors for RagaAI Catalyst tracing."""

from ragaai_catalyst.tracers.processors.custom_span_processor import (
    CustomSpanProcessor,
    trace_context,
    set_trace_config,
    get_trace_config,
    clear_trace_config,
    # Legacy aliases for backward compatibility
    set_trace_metadata,
    get_trace_metadata,
    clear_trace_metadata
)

__all__ = [
    'CustomSpanProcessor',
    'trace_context',
    'set_trace_config',
    'get_trace_config',
    'clear_trace_config',
    # Legacy exports
    'set_trace_metadata',
    'get_trace_metadata',
    'clear_trace_metadata'
]
