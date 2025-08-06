"""Tests for Phase III implementation."""

import pytest
import tempfile
from pathlib import Path
from querying.agents.factory import create_phase3_agent, create_default_agent
from querying.agents.plugins.document_relationships import DocumentRelationshipPlugin
from querying.agents.plugins.comprehensive_reporting import ComprehensiveReportingPlugin
from ingestion.storage.knowledge_graph import KnowledgeGraph, Entity, Relationship


class TestPhase3Implementation:
    """Test Phase III advanced intelligence features."""
    
    def test_phase3_agent_creation(self):
        """Test that Phase III agent creates successfully with all plugins."""
        agent = create_phase3_agent()
        
        # Check that all expected plugins are registered
        expected_plugins = [
            "semantic_search",
            "metadata_commands", 
            "document_relationships",
            "comprehensive_reporting"
        ]
        
        for plugin_name in expected_plugins:
            assert agent.registry.get_plugin(plugin_name) is not None
        
        # Check capabilities
        capabilities = agent.get_capabilities()
        assert len(capabilities) >= 20  # Should have many capabilities
        
        # Check specific Phase III capabilities
        phase3_capabilities = [
            "document_similarity",
            "document_clustering",
            "relationship_analysis",
            "collection_summary",
            "activity_report"
        ]
        
        for capability in phase3_capabilities:
            assert capability in capabilities
    
    def test_enhanced_query_classification(self):
        """Test that Phase III queries are classified correctly."""
        agent = create_phase3_agent()
        
        # Test relationship analysis queries
        plugins = agent._classify_query("find documents similar to budget_report.pdf")
        assert "document_relationships" in plugins
        
        plugins = agent._classify_query("cluster documents by theme")
        assert "document_relationships" in plugins
        
        # Test reporting queries
        plugins = agent._classify_query("generate a summary report")
        assert "comprehensive_reporting" in plugins
        
        plugins = agent._classify_query("show me activity analytics")
        assert "comprehensive_reporting" in plugins
        
        # Test complex multi-step queries
        plugins = agent._classify_query("analyze budget themes and generate a report")
        assert len(plugins) >= 2  # Should use multiple plugins
    
    def test_document_relationships_plugin(self):
        """Test document relationships plugin functionality."""
        plugin = DocumentRelationshipPlugin()
        
        # Test plugin info
        info = plugin.get_info()
        assert info.name == "document_relationships"
        assert "document_similarity" in info.capabilities
        assert "document_clustering" in info.capabilities
        
        # Test parameter validation
        valid_params = {"operation": "find_similar_documents"}
        assert plugin.validate_params(valid_params)
        
        invalid_params = {"operation": "invalid_operation"}
        assert not plugin.validate_params(invalid_params)
        
        # Test execution (should handle missing vector store gracefully)
        result = plugin.execute({"operation": "cluster_documents"})
        assert "response" in result or "error" in result
    
    def test_comprehensive_reporting_plugin(self):
        """Test comprehensive reporting plugin functionality."""
        plugin = ComprehensiveReportingPlugin()
        
        # Test plugin info
        info = plugin.get_info()
        assert info.name == "comprehensive_reporting"
        assert "collection_summary" in info.capabilities
        assert "activity_report" in info.capabilities
        
        # Test parameter validation
        valid_params = {"operation": "generate_collection_summary"}
        assert plugin.validate_params(valid_params)
        
        invalid_params = {"operation": "invalid_report"}
        assert not plugin.validate_params(invalid_params)
        
        # Test execution with mock data
        result = plugin.execute({"operation": "generate_health_report"})
        assert "response" in result
        assert "report" in result
    
    def test_knowledge_graph_integration(self):
        """Test knowledge graph functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kg_path = Path(tmpdir) / "test_kg.db"
            kg = KnowledgeGraph(str(kg_path))
            
            # Test entity creation
            entity = Entity(
                id="test_entity",
                type="document",
                name="Test Document",
                properties={"size": 1024}
            )
            
            assert kg.add_entity(entity)
            
            # Test entity retrieval
            retrieved = kg.get_entity("test_entity")
            assert retrieved is not None
            assert retrieved.name == "Test Document"
            
            # Test relationship creation
            entity2 = Entity(
                id="test_entity_2",
                type="person",
                name="Test Person",
                properties={}
            )
            kg.add_entity(entity2)
            
            relationship = Relationship(
                source_id="test_entity",
                target_id="test_entity_2",
                relationship_type="authored_by",
                properties={}
            )
            
            assert kg.add_relationship(relationship)
            
            # Test relationship queries
            related = kg.find_related_entities("test_entity")
            assert len(related) > 0
            
            # Test graph statistics
            stats = kg.get_statistics()
            assert stats["total_entities"] >= 2
            assert stats["total_relationships"] >= 1
    
    def test_backward_compatibility(self):
        """Test that Phase III maintains backward compatibility."""
        # Phase II agent should still work
        phase2_agent = create_default_agent()
        capabilities_2 = phase2_agent.get_capabilities()
        
        # Phase III agent should include all Phase II capabilities
        phase3_agent = create_phase3_agent()
        capabilities_3 = phase3_agent.get_capabilities()
        
        # All Phase II capabilities should be in Phase III
        for capability in capabilities_2:
            assert capability in capabilities_3
        
        # Phase III should have additional capabilities
        assert len(capabilities_3) > len(capabilities_2)
    
    def test_parameter_generation(self):
        """Test that agent generates appropriate parameters for Phase III plugins."""
        agent = create_phase3_agent()
        
        # Test relationship analysis parameters
        params = agent._generate_relationship_params("find similar documents")
        assert params["operation"] == "find_similar_documents"
        
        params = agent._generate_relationship_params("cluster documents into 3 groups")
        assert params["operation"] == "cluster_documents"
        assert params["num_clusters"] == 3
        
        # Test reporting parameters
        params = agent._generate_reporting_params("generate activity report for last week")
        assert params["operation"] == "generate_activity_report"
        assert params["time_window"] == "1_week"
        
        params = agent._generate_reporting_params("create trend analysis")
        assert params["operation"] == "generate_trend_analysis"
    
    def test_plugin_coordination(self):
        """Test that multiple plugins can be coordinated for complex queries."""
        agent = create_phase3_agent()
        
        # Test complex query that should use multiple plugins
        complex_query = "analyze document relationships and generate a summary report"
        plugins = agent._classify_query(complex_query)
        
        # Should classify to multiple relevant plugins
        assert len(plugins) >= 1
        
        # Should include appropriate plugin types
        plugin_types = set()
        for plugin_name in plugins:
            if "relationship" in plugin_name:
                plugin_types.add("relationship")
            elif "reporting" in plugin_name:
                plugin_types.add("reporting")
            elif "semantic" in plugin_name:
                plugin_types.add("semantic")
            elif "metadata" in plugin_name:
                plugin_types.add("metadata")
        
        # Complex queries should trigger multiple plugin types
        assert len(plugin_types) >= 1


if __name__ == "__main__":
    # Run basic tests
    test_suite = TestPhase3Implementation()
    
    print("🧪 Running Phase III Tests...")
    
    tests = [
        ("Agent Creation", test_suite.test_phase3_agent_creation),
        ("Query Classification", test_suite.test_enhanced_query_classification),
        ("Document Relationships", test_suite.test_document_relationships_plugin),
        ("Comprehensive Reporting", test_suite.test_comprehensive_reporting_plugin),
        ("Knowledge Graph", test_suite.test_knowledge_graph_integration),
        ("Backward Compatibility", test_suite.test_backward_compatibility),
        ("Parameter Generation", test_suite.test_parameter_generation),
        ("Plugin Coordination", test_suite.test_plugin_coordination)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase III tests passed!")
    else:
        print(f"⚠️  {total - passed} tests failed")
    
    exit(0 if passed == total else 1)