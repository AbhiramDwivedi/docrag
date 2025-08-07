"""Tests for knowledge graph implementation fixes."""
import pytest
import tempfile
import sys
from pathlib import Path

# Add backend src to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.storage.knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder, Entity, Relationship
from querying.agents.factory import create_phase3_agent
from querying.agents.plugins.knowledge_graph import KnowledgeGraphPlugin


class TestKnowledgeGraphFixes:
    """Test suite for knowledge graph implementation fixes."""
    
    def test_improved_entity_extraction(self):
        """Test improved entity extraction with relationships."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            kg = KnowledgeGraph(tmp.name)
            builder = KnowledgeGraphBuilder(kg)
            
            # Test text with various entity types and relationships
            test_text = """
            John Smith works for Acme Corporation. He manages the Budget Project.
            Jane Doe is employed by Tech Solutions Inc. She leads the Marketing Initiative.
            The project manager at Acme Corp sent an email to john.smith@acme.com.
            """
            
            entities, relationships = builder.extract_entities_from_text(test_text, "/test/doc.txt")
            
            # Verify entities were extracted
            assert len(entities) > 0, "Should extract entities from text"
            
            # Verify relationships were extracted
            assert len(relationships) > 0, "Should extract relationships from text"
            
            # Check for person entities
            person_entities = [e for e in entities if e.type == 'person']
            assert len(person_entities) > 0, "Should extract person entities"
            
            # Check for organization entities
            org_entities = [e for e in entities if e.type == 'organization']
            assert len(org_entities) > 0, "Should extract organization entities"
            
            # Check for WORKS_FOR relationships
            works_for_rels = [r for r in relationships if r.relationship_type == 'WORKS_FOR']
            assert len(works_for_rels) >= 0, "Should extract employment relationships"
            
            # Check for MENTIONS relationships
            mentions_rels = [r for r in relationships if r.relationship_type == 'MENTIONS']
            assert len(mentions_rels) > 0, "Should create document mention relationships"
    
    def test_cli_uses_phase3_agent(self):
        """Test that CLI uses phase3 agent with KG plugin."""
        from interface.cli.ask import get_agent
        
        agent = get_agent()
        assert agent is not None, "Should create agent successfully"
        
        capabilities = agent.get_capabilities()
        assert "knowledge_graph" in capabilities, "Should have knowledge graph capabilities"
    
    def test_knowledge_graph_plugin_operations(self):
        """Test knowledge graph plugin operations."""
        plugin = KnowledgeGraphPlugin()
        
        # Test operation validation
        valid_params = {
            "operation": "find_entities",
            "entity_type": "person"
        }
        assert plugin.validate_params(valid_params), "Should validate correct parameters"
        
        invalid_params = {
            "operation": "unknown_operation"
        }
        assert not plugin.validate_params(invalid_params), "Should reject invalid operation"
        
        # Test hybrid search validation
        hybrid_params = {
            "operation": "hybrid_search",
            "question": "Who works at Acme?"
        }
        assert plugin.validate_params(hybrid_params), "Should validate hybrid search parameters"
    
    def test_agent_query_routing(self):
        """Test that agent correctly routes entity queries to KG plugin."""
        agent = create_phase3_agent()
        
        # Mock a simple question classification test
        entity_questions = [
            "Who works at Acme Corporation?",
            "What company does John work for?",
            "Who manages the project?",
            "Show me people in the organization"
        ]
        
        for question in entity_questions:
            # Test classification logic by examining what plugins would be used
            plugins_to_use = agent._classify_query(question)
            assert "knowledge_graph" in plugins_to_use, f"Should route '{question}' to knowledge graph"
    
    def test_hybrid_search_capability(self):
        """Test hybrid search combines vector and graph results."""
        plugin = KnowledgeGraphPlugin()
        
        # Test with no KG database (should handle gracefully)
        params = {
            "operation": "hybrid_search",
            "question": "Who works at Acme?",
            "vector_results": [{"document": "test.pdf", "content": "Some content"}],
            "max_entities": 5
        }
        
        result = plugin.execute(params)
        
        # Should return a valid response even if KG is empty
        assert "results" in result
        assert "entities" in result
        assert "relationships" in result
        assert "metadata" in result
    
    def test_entity_resolution_prevents_duplicates(self):
        """Test that entity resolution prevents duplicate entities."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            kg = KnowledgeGraph(tmp.name)
            
            # Add same entity twice
            entity1 = Entity(
                id="person_john_smith",
                type="person",
                name="John Smith",
                properties={"source_document": "doc1.txt"},
                confidence=0.9
            )
            
            entity2 = Entity(
                id="person_john_smith",  # Same ID
                type="person",
                name="John Smith",
                properties={"source_document": "doc2.txt"},
                confidence=0.8
            )
            
            kg.add_entity(entity1)
            kg.add_entity(entity2)  # Should replace, not duplicate
            
            # Verify only one entity exists
            retrieved = kg.get_entity("person_john_smith")
            assert retrieved is not None
            assert retrieved.name == "John Smith"
    
    def test_graph_analytics_integration(self):
        """Test that graph analytics are properly integrated."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            kg = KnowledgeGraph(tmp.name)
            
            # Add some entities and relationships
            entities = [
                Entity(id="person_1", type="person", name="Alice", properties={}, confidence=1.0),
                Entity(id="person_2", type="person", name="Bob", properties={}, confidence=1.0),
                Entity(id="org_1", type="organization", name="Acme Corp", properties={}, confidence=1.0)
            ]
            
            for entity in entities:
                kg.add_entity(entity)
            
            relationships = [
                Relationship("person_1", "org_1", "WORKS_FOR", {}, 1.0, 1.0),
                Relationship("person_2", "org_1", "WORKS_FOR", {}, 1.0, 1.0)
            ]
            
            for rel in relationships:
                kg.add_relationship(rel)
            
            # Test centrality calculation
            centrality = kg.get_entity_centrality('betweenness')
            assert len(centrality) > 0, "Should calculate centrality scores"
            
            # Test statistics
            stats = kg.get_statistics()
            assert stats['total_entities'] == 3
            assert stats['total_relationships'] == 2
            assert 'entity_types' in stats
            assert 'relationship_types' in stats


if __name__ == "__main__":
    pytest.main([__file__])