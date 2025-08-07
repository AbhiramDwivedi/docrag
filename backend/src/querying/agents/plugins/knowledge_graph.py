"""Knowledge graph plugin for DocQuest agent framework.

This plugin provides knowledge graph-enhanced querying capabilities,
allowing the agent to find related entities and explore document relationships
through the knowledge graph.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from querying.agents.plugin import Plugin, PluginInfo
from ingestion.storage.knowledge_graph import KnowledgeGraph
from shared.config import settings

logger = logging.getLogger(__name__)


class KnowledgeGraphPlugin(Plugin):
    """Plugin for knowledge graph-enhanced querying.
    
    This plugin leverages the knowledge graph to find related entities,
    explore relationships, and provide enhanced context for queries.
    """
    
    def __init__(self):
        """Initialize the knowledge graph plugin."""
        self._knowledge_graph = None
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute knowledge graph operations.
        
        Args:
            params: Dictionary containing:
                - operation: The operation to perform ('find_entities', 'get_relationships', 'explore_entity')
                - entity_type: Type of entities to search for (optional)
                - entity_id: Specific entity ID to explore (optional)
                - question: Natural language question for entity extraction (optional)
                - max_depth: Maximum depth for relationship traversal (default: 2)
                
        Returns:
            Dictionary containing:
                - results: The operation results
                - entities: List of entities found
                - relationships: List of relationships found
                - metadata: Additional metadata about the operation
        """
        operation = params.get("operation", "find_entities")
        
        try:
            # Initialize knowledge graph if needed
            if self._knowledge_graph is None:
                # Check for demo KG first, then default location
                kg_paths = [
                    Path("data/demo_knowledge_graph.db"),
                    Path("data/knowledge_graph.db")
                ]
                
                kg_path = None
                for path in kg_paths:
                    if path.exists():
                        kg_path = path
                        break
                
                if kg_path is None:
                    return {
                        "results": "Knowledge graph database not found. Please run document ingestion first.",
                        "entities": [],
                        "relationships": [],
                        "metadata": {"error": "kg_not_found"}
                    }
                    
                self._knowledge_graph = KnowledgeGraph(str(kg_path))
            
            if operation == "find_entities":
                return self._find_entities(params)
            elif operation == "get_relationships":
                return self._get_relationships(params)
            elif operation == "explore_entity":
                return self._explore_entity(params)
            elif operation == "get_statistics":
                return self._get_statistics()
            elif operation == "extract_entities_from_question":
                return self._extract_entities_from_question(params)
            else:
                return {
                    "results": f"Unknown operation: {operation}",
                    "entities": [],
                    "relationships": [],
                    "metadata": {"error": "unknown_operation"}
                }
                
        except Exception as e:
            logger.error(f"Error in knowledge graph operation: {e}")
            return {
                "results": f"Error in knowledge graph operation: {e}",
                "entities": [],
                "relationships": [],
                "metadata": {"error": str(e)}
            }
    
    def _find_entities(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Find entities by type or other criteria."""
        entity_type = params.get("entity_type", "person")
        entities = self._knowledge_graph.find_entities_by_type(entity_type)
        
        entity_list = []
        for entity in entities:
            entity_list.append({
                "id": entity.id,
                "type": entity.type,
                "name": entity.name,
                "properties": entity.properties,
                "confidence": entity.confidence
            })
        
        return {
            "results": f"Found {len(entities)} entities of type '{entity_type}'",
            "entities": entity_list,
            "relationships": [],
            "metadata": {
                "operation": "find_entities",
                "entity_type": entity_type,
                "count": len(entities)
            }
        }
    
    def _get_relationships(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get relationships in the knowledge graph."""
        # This is a simplified approach - in a real implementation,
        # you would query the relationships table directly
        stats = self._knowledge_graph.get_statistics()
        
        return {
            "results": f"Knowledge graph contains {stats.get('total_relationships', 0)} relationships",
            "entities": [],
            "relationships": list(stats.get('relationship_types', {}).keys()),
            "metadata": {
                "operation": "get_relationships",
                "total_relationships": stats.get('total_relationships', 0),
                "relationship_types": stats.get('relationship_types', {})
            }
        }
    
    def _explore_entity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Explore relationships for a specific entity."""
        entity_id = params.get("entity_id")
        max_depth = params.get("max_depth", 2)
        
        if not entity_id:
            return {
                "results": "No entity_id provided for exploration",
                "entities": [],
                "relationships": [],
                "metadata": {"error": "missing_entity_id"}
            }
        
        # Get the entity
        entity = self._knowledge_graph.get_entity(entity_id)
        if not entity:
            return {
                "results": f"Entity '{entity_id}' not found",
                "entities": [],
                "relationships": [],
                "metadata": {"error": "entity_not_found"}
            }
        
        # Find related entities
        related = self._knowledge_graph.find_related_entities(entity_id, max_depth=max_depth)
        
        entity_list = [{
            "id": entity.id,
            "type": entity.type,
            "name": entity.name,
            "properties": entity.properties,
            "confidence": entity.confidence
        }]
        
        relationship_list = []
        for rel_entity, rel_type, distance in related:
            entity_list.append({
                "id": rel_entity.id,
                "type": rel_entity.type,
                "name": rel_entity.name,
                "properties": rel_entity.properties,
                "confidence": rel_entity.confidence
            })
            relationship_list.append({
                "from": entity_id,
                "to": rel_entity.id,
                "type": rel_type,
                "distance": distance
            })
        
        return {
            "results": f"Found {len(related)} entities related to '{entity.name}'",
            "entities": entity_list,
            "relationships": relationship_list,
            "metadata": {
                "operation": "explore_entity",
                "entity_id": entity_id,
                "max_depth": max_depth,
                "related_count": len(related)
            }
        }
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        stats = self._knowledge_graph.get_statistics()
        
        return {
            "results": "Knowledge graph statistics retrieved",
            "entities": [],
            "relationships": [],
            "metadata": {
                "operation": "get_statistics",
                "statistics": stats
            }
        }
    
    def _extract_entities_from_question(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract potential entities from a natural language question."""
        question = params.get("question", "")
        
        if not question.strip():
            return {
                "results": "No question provided for entity extraction",
                "entities": [],
                "relationships": [],
                "metadata": {"error": "empty_question"}
            }
        
        # Simple entity extraction from question
        # In a real implementation, this would use NLP to identify entities
        potential_entities = []
        
        # Look for existing entities that match words in the question
        words = question.lower().split()
        
        # Check all entity types
        for entity_type in ['person', 'organization', 'topic', 'location', 'document']:
            entities = self._knowledge_graph.find_entities_by_type(entity_type)
            for entity in entities:
                # Simple name matching
                entity_words = entity.name.lower().split()
                if any(word in words for word in entity_words):
                    potential_entities.append({
                        "id": entity.id,
                        "type": entity.type,
                        "name": entity.name,
                        "properties": entity.properties,
                        "confidence": entity.confidence,
                        "relevance": "mentioned_in_question"
                    })
        
        return {
            "results": f"Found {len(potential_entities)} potential entities in question",
            "entities": potential_entities,
            "relationships": [],
            "metadata": {
                "operation": "extract_entities_from_question",
                "question": question,
                "entity_count": len(potential_entities)
            }
        }
    
    def get_info(self) -> PluginInfo:
        """Return plugin metadata and capabilities."""
        return PluginInfo(
            name="knowledge_graph",
            description="Knowledge graph querying and entity exploration capabilities",
            version="1.0.0",
            capabilities=[
                "knowledge_graph",
                "entity_search", 
                "relationship_exploration",
                "graph_statistics",
                "entity_extraction"
            ],
            parameters={
                "operation": "str - Operation to perform: 'find_entities', 'get_relationships', 'explore_entity', 'get_statistics', 'extract_entities_from_question'",
                "entity_type": "str - Type of entities to find (optional)",
                "entity_id": "str - Entity ID to explore (optional)",
                "question": "str - Question for entity extraction (optional)",
                "max_depth": "int - Maximum depth for relationship traversal (default: 2)"
            }
        )
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters before execution."""
        operation = params.get("operation", "find_entities")
        
        valid_operations = [
            "find_entities", 
            "get_relationships", 
            "explore_entity", 
            "get_statistics",
            "extract_entities_from_question"
        ]
        
        if operation not in valid_operations:
            return False
        
        # Validate specific operation requirements
        if operation == "explore_entity" and "entity_id" not in params:
            return False
        
        if operation == "extract_entities_from_question" and "question" not in params:
            return False
        
        max_depth = params.get("max_depth", 2)
        if not isinstance(max_depth, int) or max_depth < 1:
            return False
        
        return True
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self._knowledge_graph = None
        logger.info("Knowledge graph plugin cleaned up")