"""Core agent implementation for DocQuest intelligent document analysis."""

import logging
import time
from typing import List, Optional, Dict, Any
from .registry import PluginRegistry
from .plugin import Plugin
from .orchestrator import OrchestratorAgent

logger = logging.getLogger(__name__)

# Specialized loggers for verbose output - maintained for backward compatibility
classification_logger = logging.getLogger('agent.classification')
execution_logger = logging.getLogger('agent.execution')
synthesis_logger = logging.getLogger('agent.synthesis')
timing_logger = logging.getLogger('timing')


class Agent:
    """Core agent for intelligent document analysis.
    
    The agent provides a simple interface that delegates to the OrchestratorAgent
    for intelligent agentic workflow coordination.
    """
    
    def __init__(self, registry: Optional[PluginRegistry] = None):
        """Initialize the agent with a plugin registry.
        
        Args:
            registry: Plugin registry instance. If None, creates a new one.
        """
        self.registry = registry or PluginRegistry()
        self.orchestrator = OrchestratorAgent(self.registry)
        self._last_query = ""
        self._last_execution_time = 0.0
    
    def process_query(self, question: str) -> str:
        """Process natural language query and return response.
        
        Args:
            question: Natural language question to process
            
        Returns:
            Formatted response string
        """
        start_time = time.time()
        self._last_query = question
        
        try:
            # Delegate all processing to the orchestrator
            response = self.orchestrator.process_query(question)
            
            self._last_execution_time = time.time() - start_time
            timing_logger.info(f"Total execution time: {self._last_execution_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query '{question}': {e}")
            return f"❌ Error processing query: {e}"
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities for introspection.
        
        Returns:
            List of capabilities provided by all registered plugins
        """
        capabilities = []
        for plugin_info in self.registry._plugin_info.values():
            capabilities.extend(plugin_info.capabilities)
        return list(set(capabilities))  # Remove duplicates
    
    def explain_reasoning(self) -> Optional[str]:
        """Return explanation of last query processing steps.
        
        Returns:
            Human-readable explanation of reasoning process, or None if no query processed
        """
        if not self._last_query:
            return None
        
        # Get reasoning trace from orchestrator
        reasoning_trace = self.orchestrator.get_last_reasoning_trace()
        execution_plan = self.orchestrator.get_last_execution_plan()
        
        explanation = [
            f"Query: {self._last_query}",
            f"Execution time: {self._last_execution_time:.2f}s",
            "",
            "Agentic Execution Plan:"
        ]
        
        if execution_plan:
            explanation.append(f"  Intent: {execution_plan.intent.value}")
            explanation.append(f"  Strategy: {execution_plan.description}")
            explanation.append(f"  Steps: {len(execution_plan.steps)}")
            explanation.append("")
            explanation.append("Execution Steps:")
            for i, step in enumerate(execution_plan.steps, 1):
                explanation.append(f"  {i}. {step.description} ({step.plugin_name})")
        
        explanation.append("")
        explanation.append("Reasoning trace:")
        
        if reasoning_trace:
            for i, step in enumerate(reasoning_trace, 1):
                explanation.append(f"  {i}. {step}")
        else:
            explanation.append("  No reasoning steps recorded")
        
        return "\n".join(explanation)