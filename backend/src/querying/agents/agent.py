"""Core agent implementation for DocQuest intelligent document analysis."""

import logging
import time
from typing import List, Optional
from .registry import PluginRegistry
from .agentic.orchestrator_agent import OrchestratorAgent

logger = logging.getLogger(__name__)


class Agent:
    """Core agent for intelligent document analysis.
    
    This agent serves as a facade over the agentic OrchestratorAgent,
    providing a simple interface while leveraging sophisticated multi-step
    reasoning capabilities internally.
    """
    
    def __init__(self, registry: Optional[PluginRegistry] = None):
        """Initialize the agent with agentic orchestration capabilities.
        
        Args:
            registry: Plugin registry instance. If None, creates a new one.
        """
        self.registry = registry or PluginRegistry()
        self.orchestrator = OrchestratorAgent(self.registry)
        self._last_query = ""
        self._last_execution_time = 0.0
    
    def process_query(self, question: str) -> str:
        """Process natural language query using agentic orchestration.
        
        Args:
            question: Natural language question to process
            
        Returns:
            Formatted response string from agentic processing
        """
        start_time = time.time()
        self._last_query = question
        
        try:
            # Delegate to agentic orchestrator for intelligent processing
            response = self.orchestrator.process_query(question)
            
            self._last_execution_time = time.time() - start_time
            return response
            
        except Exception as e:
            logger.error(f"Error processing query '{question}': {e}")
            return f"❌ Error processing query: {e}"
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities.
        
        Returns:
            List of capabilities provided by the agentic system
        """
        return [
            "Multi-step reasoning and planning",
            "Intent analysis and classification", 
            "Document discovery and search",
            "Content analysis and extraction",
            "Knowledge graph relationships",
            "Cross-document synthesis",
            "Adaptive execution planning"
        ]
    
    def explain_reasoning(self) -> Optional[str]:
        """Return explanation of last query processing steps.
        
        Returns:
            Human-readable explanation of agentic reasoning process
        """
        if not self._last_query:
            return None
        
        # Get reasoning from orchestrator if available
        if hasattr(self.orchestrator, '_reasoning_trace') and self.orchestrator._reasoning_trace:
            explanation = [
                f"Query: {self._last_query}",
                f"Execution time: {self._last_execution_time:.2f}s",
                "",
                "Agentic reasoning trace:"
            ]
            
            for i, step in enumerate(self.orchestrator._reasoning_trace, 1):
                explanation.append(f"  {i}. {step}")
            
            return "\n".join(explanation)
        
        return f"Query: {self._last_query}\nExecution time: {self._last_execution_time:.2f}s\nAgentic processing completed"
