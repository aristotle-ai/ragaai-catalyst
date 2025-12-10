"""Simple instrumentor registry for OpenInference instrumentors."""

import logging
from typing import List, Tuple, Type, Optional

logger = logging.getLogger(__name__)


INSTRUMENTOR_REGISTRY = {
    "langchain": "openinference.instrumentation.langchain.LangChainInstrumentor",
    "llamaindex": "openinference.instrumentation.llama_index.LlamaIndexInstrumentor",
    "openai": "openinference.instrumentation.openai.OpenAIInstrumentor",
    "anthropic": "openinference.instrumentation.anthropic.AnthropicInstrumentor",
    "vertexai": "openinference.instrumentation.vertexai.VertexAIInstrumentor",
    "bedrock": "openinference.instrumentation.bedrock.BedrockInstrumentor",
    "crewai": "openinference.instrumentation.crewai.CrewAIInstrumentor",
    "haystack": "openinference.instrumentation.haystack.HaystackInstrumentor",
    "autogen": "openinference.instrumentation.autogen.AutogenInstrumentor",
    "groq": "openinference.instrumentation.groq.GroqInstrumentor",
    "litellm": "openinference.instrumentation.litellm.LiteLLMInstrumentor",
    "mistralai": "openinference.instrumentation.mistralai.MistralAIInstrumentor",
    "smolagents": "openinference.instrumentation.smolagents.SmolagentsInstrumentor",
    "openai_agents": "openinference.instrumentation.openai_agents.OpenAIAgentsInstrumentor",
    "google_adk": "openinference.instrumentation.google_adk.GoogleADKInstrumentor",
}


TRACER_TYPE_MAPPING = {
    "langchain": ["langchain"],
    "agentic/langchain": ["langchain"],
    "agentic/langgraph": ["langchain"],
    "llamaindex": ["llamaindex"],
    "agentic/llamaindex": ["llamaindex"],
    "openai": ["openai"],
    "agentic/openai": ["openai"],
    "agentic/crewai": ["crewai", "langchain", "openai", "anthropic", "vertexai",
                       "groq", "litellm", "mistralai", "bedrock"],
    "agentic/haystack": ["haystack"],
    "agentic/autogen": ["autogen"],
    "agentic/smolagents": ["smolagents"],
    "agentic/openai_agents": ["openai_agents"],
    "google-adk": ["google_adk"],
    "custom": [],
}


def load_instrumentor(name: str) -> Optional[Type]:
    """
    Load a single instrumentor class by name.
    
    Args:
        name: Framework name (e.g., 'langchain', 'openai')
        
    Returns:
        Instrumentor class or None if not available
    """
    if name not in INSTRUMENTOR_REGISTRY:
        logger.debug(f"Unknown instrumentor: {name}")
        return None
    
    module_path = INSTRUMENTOR_REGISTRY[name]
    module_name, class_name = module_path.rsplit(".", 1)
    
    try:
        module = __import__(module_name, fromlist=[class_name])
        instrumentor_class = getattr(module, class_name)
        return instrumentor_class
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        logger.debug(f"{name} instrumentor not available: {e}")
        return None


def get_instrumentors_for_type(tracer_type: str) -> List[Tuple[Type, List]]:
    """
    Get list of instrumentor classes for given tracer type.
    
    Args:
        tracer_type: Type of tracer (e.g., 'langchain', 'agentic/crewai')
        
    Returns:
        List of tuples (InstrumentorClass, args)
    """
    if tracer_type == "agentic":
        framework_names = list(INSTRUMENTOR_REGISTRY.keys())
        logger.info("Auto-detecting all available instrumentors for 'agentic' type")
    else:
        framework_names = TRACER_TYPE_MAPPING.get(tracer_type, [])
    
    if not framework_names and tracer_type != "custom":
        logger.warning(f"Unknown tracer type: {tracer_type}")
        return []
    
    instrumentors = []
    for name in framework_names:
        instrumentor_class = load_instrumentor(name)
        if instrumentor_class:
            instrumentors.append((instrumentor_class, []))
            logger.info(f"Loaded {name} instrumentor")
    
    return instrumentors
