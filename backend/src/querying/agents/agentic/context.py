"""Agent context for cross-step communication and state management."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Shared context for cross-step communication in agentic workflows.
    
    This context object maintains state across multiple execution steps,
    allowing agents to share information and build upon previous results.
    """
    session_id: str
    query: str = ""
    intent: str = ""
    discovered_documents: List[Dict[str, Any]] = field(default_factory=list)
    extracted_content: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, Any] = field(default_factory=dict)
    entities: Set[str] = field(default_factory=set)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    execution_history: List[str] = field(default_factory=list)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def add_discovered_document(self, doc_info: Dict[str, Any]) -> None:
        """Add a discovered document to the context."""
        if doc_info not in self.discovered_documents:
            self.discovered_documents.append(doc_info)
            logger.debug(f"Added document to context: {doc_info.get('path', 'unknown')}")
    
    def get_discovered_paths(self) -> List[str]:
        """Get list of discovered document paths."""
        return [doc.get('path', '') for doc in self.discovered_documents if doc.get('path')]
    
    def add_extracted_content(self, source: str, content: Any) -> None:
        """Add extracted content from a specific source."""
        self.extracted_content[source] = content
        logger.debug(f"Added extracted content for source: {source}")
    
    def get_extracted_content(self, source: Optional[str] = None) -> Any:
        """Get extracted content for a specific source or all content."""
        if source:
            return self.extracted_content.get(source)
        return self.extracted_content
    
    def add_relationship(self, rel_type: str, relationship_data: Any) -> None:
        """Add relationship information."""
        self.relationships[rel_type] = relationship_data
        logger.debug(f"Added relationship: {rel_type}")
    
    def add_entity(self, entity: str) -> None:
        """Add an entity to the context."""
        self.entities.add(entity)
    
    def get_entities(self) -> List[str]:
        """Get all entities as a list."""
        return list(self.entities)
    
    def set_confidence(self, step_id: str, confidence: float) -> None:
        """Set confidence score for a step."""
        self.confidence_scores[step_id] = confidence
    
    def get_confidence(self, step_id: str) -> float:
        """Get confidence score for a step."""
        return self.confidence_scores.get(step_id, 1.0)
    
    def add_to_history(self, action: str) -> None:
        """Add an action to the execution history."""
        timestamp = time.time()
        self.execution_history.append(f"[{timestamp:.2f}] {action}")
        logger.debug(f"Added to history: {action}")
    
    def set_shared_data(self, key: str, value: Any) -> None:
        """Set shared data that can be accessed by any agent."""
        self.shared_data[key] = value
    
    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """Get shared data."""
        return self.shared_data.get(key, default)
    
    def has_documents(self) -> bool:
        """Check if any documents have been discovered."""
        return len(self.discovered_documents) > 0
    
    def has_content(self) -> bool:
        """Check if any content has been extracted."""
        return len(self.extracted_content) > 0
    
    def has_relationships(self) -> bool:
        """Check if any relationships have been identified."""
        return len(self.relationships) > 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current context state."""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "intent": self.intent,
            "documents_count": len(self.discovered_documents),
            "content_sources": list(self.extracted_content.keys()),
            "relationship_types": list(self.relationships.keys()),
            "entities_count": len(self.entities),
            "execution_steps": len(self.execution_history),
            "age_seconds": time.time() - self.created_at
        }
    
    def clear_documents(self) -> None:
        """Clear discovered documents (useful for retries)."""
        self.discovered_documents.clear()
        logger.debug("Cleared discovered documents from context")
    
    def clear_content(self) -> None:
        """Clear extracted content (useful for retries)."""
        self.extracted_content.clear()
        logger.debug("Cleared extracted content from context")
    
    def reset(self) -> None:
        """Reset the context to initial state (except session_id, query, intent)."""
        self.discovered_documents.clear()
        self.extracted_content.clear()
        self.relationships.clear()
        self.entities.clear()
        self.confidence_scores.clear()
        self.execution_history.clear()
        self.shared_data.clear()
        self.created_at = time.time()
        logger.debug("Reset context to initial state")