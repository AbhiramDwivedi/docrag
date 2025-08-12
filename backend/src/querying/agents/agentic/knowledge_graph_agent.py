"""Knowledge Graph agent for relationship and entity analysis operations."""

from typing import Dict, Any, List
import logging

from .base_agent import BaseAgent
from .execution_plan import ExecutionStep, StepResult, StepType
from .context import AgentContext
from ..registry import PluginRegistry

logger = logging.getLogger(__name__)


class KnowledgeGraphAgent(BaseAgent):
    """Specialized agent for relationship and entity analysis operations.
    
    This agent focuses on identifying entities, relationships, and 
    knowledge patterns within the document collection.
    """
    
    def __init__(self, plugin_registry: PluginRegistry):
        """Initialize the knowledge graph agent.
        
        Args:
            plugin_registry: Registry of available plugins
        """
        super().__init__("knowledge_graph")
        self.registry = plugin_registry
    
    def can_handle(self, step: ExecutionStep) -> bool:
        """Check if this agent can handle the given step type.
        
        Args:
            step: ExecutionStep to evaluate
            
        Returns:
            True if this agent can handle the step
        """
        return step.step_type in [
            StepType.FIND_RELATIONSHIPS
        ]
    
    def get_capabilities(self) -> List[str]:
        """Get capabilities provided by this agent."""
        return [
            "entity_extraction",
            "relationship_analysis", 
            "knowledge_mapping",
            "entity_linking",
            "network_analysis"
        ]
    
    def execute_step(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Execute a knowledge graph step.
        
        Args:
            step: ExecutionStep to execute
            context: AgentContext for communication
            
        Returns:
            StepResult with knowledge graph analysis results
        """
        if step.step_type == StepType.FIND_RELATIONSHIPS:
            return self._find_relationships(step, context)
        else:
            return self._create_failure_result(
                step, f"Unsupported step type: {step.step_type.value}"
            )
    
    def _find_relationships(self, step: ExecutionStep, context: AgentContext) -> StepResult:
        """Find relationships and entities using knowledge graph capabilities.
        
        Args:
            step: ExecutionStep with relationship parameters
            context: AgentContext for storing results
            
        Returns:
            StepResult with relationship analysis
        """
        self.agent_logger.info("Executing relationship analysis")
        
        try:
            # Get knowledge graph plugin
            kg_plugin = self.registry.get_plugin("knowledge_graph")
            if not kg_plugin:
                return self._create_failure_result(
                    step, "Knowledge graph plugin not available"
                )
            
            # Prepare parameters for knowledge graph analysis
            kg_params = self._prepare_kg_params(step, context)
            
            # Execute knowledge graph search
            if not kg_plugin.validate_params(kg_params):
                return self._create_failure_result(
                    step, "Invalid parameters for knowledge graph analysis"
                )
            
            result = kg_plugin.execute(kg_params)
            
            # Process and store results
            relationship_data = self._process_kg_results(result, step, context)
            
            # Update context with relationship information
            rel_type = step.parameters.get("relationship_type", "general")
            context.add_relationship(rel_type, relationship_data)
            
            # Extract and store entities
            entities = self._extract_entities_from_results(relationship_data)
            for entity in entities:
                context.add_entity(entity)
            
            self.agent_logger.info(f"Found {len(entities)} entities and relationship data")
            
            return self._create_success_result(
                step,
                {
                    "relationships": relationship_data,
                    "entities": entities,
                    "kg_params": kg_params,
                    "raw_response": result.get("response", "")
                },
                confidence=0.8 if entities else 0.4
            )
            
        except Exception as e:
            self.agent_logger.error(f"Relationship analysis failed: {e}")
            return self._create_failure_result(step, f"Knowledge graph error: {e}")
    
    def _prepare_kg_params(self, step: ExecutionStep, context: AgentContext) -> Dict[str, Any]:
        """Prepare parameters for knowledge graph plugin.
        
        Args:
            step: ExecutionStep with user parameters
            context: AgentContext with shared information
            
        Returns:
            Dictionary of parameters for knowledge graph plugin
        """
        base_params = step.parameters.copy()
        query = base_params.get("query", context.query)
        
        # Determine operation type based on query and step parameters
        operation = self._determine_kg_operation(query, base_params)
        
        params = {
            "operation": operation,
            "question": query,
            "max_depth": base_params.get("max_depth", 2),
            "max_entities": base_params.get("max_entities", 10)
        }
        
        # Add entity type if specified
        entity_type = self._extract_entity_type(query)
        if entity_type:
            params["entity_type"] = entity_type
        
        # Use context information to enhance search
        if context.has_documents():
            params["document_context"] = [doc.get("name", "") for doc in context.discovered_documents]
        
        params.update(base_params)
        
        self.agent_logger.debug(f"Prepared KG params: {params}")
        return params
    
    def _determine_kg_operation(self, query: str, parameters: Dict[str, Any]) -> str:
        """Determine the appropriate knowledge graph operation.
        
        Args:
            query: User query
            parameters: Step parameters
            
        Returns:
            Operation name for knowledge graph plugin
        """
        query_lower = query.lower()
        
        # Check for explicit operation in parameters
        if "operation" in parameters:
            return parameters["operation"]
        
        # Determine based on query patterns
        if any(word in query_lower for word in ["who works", "who is", "employee", "staff"]):
            return "find_entities"
        elif any(word in query_lower for word in ["what company", "organization", "corp"]):
            return "find_entities"
        elif any(word in query_lower for word in ["related to", "connected", "relationship"]):
            return "hybrid_search"
        elif any(word in query_lower for word in ["statistics", "stats", "overview"]):
            return "get_statistics"
        else:
            return "hybrid_search"  # Default to hybrid search
    
    def _extract_entity_type(self, query: str) -> str:
        """Extract entity type from query.
        
        Args:
            query: User query
            
        Returns:
            Entity type string or empty string
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["who", "person", "employee", "staff"]):
            return "person"
        elif any(word in query_lower for word in ["company", "organization", "corp"]):
            return "organization"
        elif any(word in query_lower for word in ["where", "location", "city", "office"]):
            return "location"
        elif any(word in query_lower for word in ["project", "initiative", "program"]):
            return "topic"
        else:
            return ""
    
    def _process_kg_results(self, result: Dict[str, Any], 
                           step: ExecutionStep, context: AgentContext) -> Dict[str, Any]:
        """Process knowledge graph plugin results.
        
        Args:
            result: Raw result from knowledge graph plugin
            step: ExecutionStep that was executed
            context: AgentContext for additional processing
            
        Returns:
            Processed relationship data
        """
        response = result.get("response", "")
        
        # Parse the response for structured information
        relationship_data = {
            "raw_response": response,
            "entities": [],
            "relationships": [],
            "analysis_type": step.parameters.get("relationship_type", "general"),
            "confidence": 0.7
        }
        
        # Extract entities and relationships from response
        if response and not response.startswith("❌"):
            # Parse entities
            entities = self._parse_entities_from_response(response)
            relationship_data["entities"] = entities
            
            # Parse relationships
            relationships = self._parse_relationships_from_response(response)
            relationship_data["relationships"] = relationships
            
            # Update confidence based on findings
            if entities or relationships:
                relationship_data["confidence"] = 0.8
            else:
                relationship_data["confidence"] = 0.4
        else:
            relationship_data["confidence"] = 0.2
        
        return relationship_data
    
    def _parse_entities_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse entities from knowledge graph response.
        
        Args:
            response: Response text from knowledge graph plugin
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        lines = response.split('\n')
        
        current_entity = None
        for line in lines:
            line = line.strip()
            
            # Look for entity markers
            if line.startswith('Entity:') or line.startswith('Person:') or line.startswith('Organization:'):
                if current_entity:
                    entities.append(current_entity)
                
                entity_name = line.split(':', 1)[1].strip()
                entity_type = line.split(':')[0].lower()
                
                current_entity = {
                    "name": entity_name,
                    "type": entity_type,
                    "properties": {}
                }
            elif current_entity and ':' in line:
                # Add properties to current entity
                key, value = line.split(':', 1)
                current_entity["properties"][key.strip().lower()] = value.strip()
        
        # Add the last entity
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def _parse_relationships_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse relationships from knowledge graph response.
        
        Args:
            response: Response text from knowledge graph plugin
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for relationship patterns
            if any(indicator in line.lower() for indicator in [
                "works with", "manages", "reports to", "connected to", "related to"
            ]):
                relationships.append({
                    "description": line,
                    "type": "extracted_relationship",
                    "confidence": 0.7
                })
        
        return relationships
    
    def _extract_entities_from_results(self, relationship_data: Dict[str, Any]) -> List[str]:
        """Extract entity names from relationship analysis results.
        
        Args:
            relationship_data: Processed relationship data
            
        Returns:
            List of entity names
        """
        entities = []
        
        # Extract from structured entities
        for entity in relationship_data.get("entities", []):
            if isinstance(entity, dict) and "name" in entity:
                entities.append(entity["name"])
            elif isinstance(entity, str):
                entities.append(entity)
        
        # Extract from relationships
        for rel in relationship_data.get("relationships", []):
            if isinstance(rel, dict) and "description" in rel:
                # Simple entity extraction from relationship descriptions
                desc = rel["description"]
                # This is a simple approach - could be enhanced with NLP
                words = desc.split()
                entities.extend([word for word in words if word[0].isupper() and len(word) > 2])
        
        return list(set(entities))  # Remove duplicates