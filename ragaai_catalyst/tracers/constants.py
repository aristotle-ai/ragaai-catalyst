"""Constants for the Tracer module."""

from enum import Enum


class TracerConstants:
    """Constant values used throughout tracer."""
    MAX_PROJECTS = 99999
    DEFAULT_TIMEOUT = 120
    DEFAULT_INTERVAL_TIME = 2
    DEFAULT_MAX_UPLOAD_WORKERS = 30
    FEEDBACK_COLUMN_NAME = "_response-feedBack"


class TracerType(str, Enum):
    """Supported tracer types."""
    AGENTIC = "agentic"
    LANGCHAIN = "langchain"
    LLAMAINDEX = "llamaindex"
    GOOGLE_ADK = "google-adk"
    OPENAI = "openai"
    CUSTOM = "custom"
    
    AGENTIC_LANGCHAIN = "agentic/langchain"
    AGENTIC_LANGGRAPH = "agentic/langgraph"
    AGENTIC_LLAMAINDEX = "agentic/llamaindex"
    AGENTIC_CREWAI = "agentic/crewai"
    AGENTIC_HAYSTACK = "agentic/haystack"
    AGENTIC_AUTOGEN = "agentic/autogen"
    AGENTIC_SMOLAGENTS = "agentic/smolagents"
    AGENTIC_OPENAI_AGENTS = "agentic/openai_agents"
    
    @classmethod
    def is_agentic(cls, tracer_type: str) -> bool:
        """Check if tracer type is agentic."""
        return tracer_type == cls.AGENTIC or tracer_type.startswith("agentic/")
    
    @classmethod
    def requires_instrumentation(cls, tracer_type: str) -> bool:
        """Check if tracer type requires instrumentation."""
        return (
            cls.is_agentic(tracer_type) or
            tracer_type in [cls.LANGCHAIN, cls.LLAMAINDEX, cls.OPENAI, 
                           cls.GOOGLE_ADK, cls.CUSTOM]
        )

