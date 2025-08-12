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

import sys
from pathlib import Path

# Add src to path for absolute imports
src_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_path))

from ..plugin import Plugin, PluginInfo
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
                # Get knowledge graph path from configuration
                kg_path = settings.resolve_storage_path(settings.knowledge_graph_path)
                
                # Check for demo KG first, then configured location
                kg_paths = [
                    Path("data/demo_knowledge_graph.db"),  # Demo KG for testing
                    kg_path  # Configured location
                ]
                
                kg_path_to_use = None
                for path in kg_paths:
                    if path.exists():
                        kg_path_to_use = path
                        break
                
                # If neither exists, use the configured location (will be created)
                if kg_path_to_use is None:
                    kg_path_to_use = kg_path
                
                self._knowledge_graph = KnowledgeGraph(str(kg_path_to_use))
            
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
            elif operation == "hybrid_search":
                return self._hybrid_search(params)
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
    
    def _hybrid_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hybrid search combining knowledge graph and vector search.
        
        Args:
            params: Dictionary containing:
                - question: The search query
                - vector_results: Optional vector search results to enhance
                - max_entities: Maximum number of entities to include (default: 10)
                
        Returns:
            Dictionary with enhanced search results
        """
        question = params.get("question", "")
        vector_results = params.get("vector_results", [])
        max_entities = params.get("max_entities", 10)
        
        if not question.strip():
            return {
                "results": "No question provided for hybrid search",
                "entities": [],
                "relationships": [],
                "metadata": {"error": "empty_question"}
            }
        
        try:
            # Step 1: Extract entities from the question
            question_entities_result = self._extract_entities_from_question({"question": question})
            question_entities = question_entities_result.get("entities", [])
            
            # Step 2: For each entity found, get related entities
            all_related_entities = []
            relationships_found = []
            
            for entity_data in question_entities[:max_entities]:
                entity_id = entity_data["id"]
                related = self._knowledge_graph.find_related_entities(entity_id, max_depth=2)
                
                for rel_entity, rel_type, distance in related:
                    all_related_entities.append({
                        "id": rel_entity.id,
                        "type": rel_entity.type,
                        "name": rel_entity.name,
                        "properties": rel_entity.properties,
                        "confidence": rel_entity.confidence,
                        "relation_to_query": rel_type,
                        "distance": distance
                    })
                    
                    relationships_found.append({
                        "from": entity_id,
                        "to": rel_entity.id,
                        "type": rel_type,
                        "distance": distance
                    })
            
            # Step 3: Use graph analytics to find important entities
            centrality = self._knowledge_graph.get_entity_centrality('betweenness')
            
            # Enhance entities with centrality scores
            for entity in all_related_entities:
                entity["centrality_score"] = centrality.get(entity["id"], 0.0)
            
            # Sort by relevance (combination of distance and centrality)
            all_related_entities.sort(key=lambda x: (1.0 - x["distance"]) + x["centrality_score"], reverse=True)
            
            # Step 4: Build enhanced results
            enhanced_results = {
                "graph_entities": question_entities + all_related_entities[:max_entities],
                "graph_relationships": relationships_found,
                "vector_results": vector_results,
                "centrality_insights": self._get_top_central_entities(centrality, 5)
            }
            
            # Step 5: Create narrative summary
            summary_parts = []
            if question_entities:
                entity_names = [e["name"] for e in question_entities]
                summary_parts.append(f"Found {len(question_entities)} entities in your question: {', '.join(entity_names)}")
            
            if all_related_entities:
                summary_parts.append(f"Discovered {len(all_related_entities)} related entities through the knowledge graph")
            
            if centrality:
                top_entity = max(centrality.items(), key=lambda x: x[1])
                top_entity_obj = self._knowledge_graph.get_entity(top_entity[0])
                if top_entity_obj:
                    summary_parts.append(f"Most central entity: {top_entity_obj.name}")
            
            summary = ". ".join(summary_parts) if summary_parts else "No significant knowledge graph insights found."
            
            return {
                "results": summary,
                "entities": enhanced_results["graph_entities"],
                "relationships": enhanced_results["graph_relationships"],
                "metadata": {
                    "operation": "hybrid_search",
                    "question": question,
                    "entity_count": len(enhanced_results["graph_entities"]),
                    "relationship_count": len(enhanced_results["graph_relationships"]),
                    "vector_result_count": len(vector_results),
                    "centrality_insights": enhanced_results["centrality_insights"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return {
                "results": f"Error in hybrid search: {e}",
                "entities": [],
                "relationships": [],
                "metadata": {"error": str(e)}
            }
    
    def _get_top_central_entities(self, centrality: Dict[str, float], top_n: int = 5) -> List[Dict[str, Any]]:
        """Get the top N most central entities."""
        top_entities = []
        sorted_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        for entity_id, score in sorted_entities:
            entity = self._knowledge_graph.get_entity(entity_id)
            if entity:
                top_entities.append({
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type,
                    "centrality_score": score
                })
        
        return top_entities
    
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
                "operation": "str - Operation to perform: 'find_entities', 'get_relationships', 'explore_entity', 'get_statistics', 'extract_entities_from_question', 'hybrid_search'",
                "entity_type": "str - Type of entities to find (optional)",
                "entity_id": "str - Entity ID to explore (optional)",
                "question": "str - Question for entity extraction or hybrid search (optional)",
                "max_depth": "int - Maximum depth for relationship traversal (default: 2)",
                "vector_results": "list - Vector search results to enhance with graph data (optional)",
                "max_entities": "int - Maximum number of entities for hybrid search (default: 10)"
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
            "extract_entities_from_question",
            "hybrid_search"
        ]
        
        if operation not in valid_operations:
            return False
        
        # Validate specific operation requirements
        if operation == "explore_entity" and "entity_id" not in params:
            return False
        
        if operation == "extract_entities_from_question" and "question" not in params:
            return False
        
        if operation == "hybrid_search" and "question" not in params:
            return False
        
        max_depth = params.get("max_depth", 2)
        if not isinstance(max_depth, int) or max_depth < 1:
            return False
        
        return True
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self._knowledge_graph = None
        logger.info("Knowledge graph plugin cleaned up")