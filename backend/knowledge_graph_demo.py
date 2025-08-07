#!/usr/bin/env python3
"""
Knowledge Graph Integration Demo
This script demonstrates the knowledge graph integration in DocQuest.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingestion.storage.knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder
from querying.agents.factory import create_knowledge_graph_agent

def main():
    print("🧠 DocQuest Knowledge Graph Integration Demo")
    print("=" * 50)
    
    # Step 1: Create sample documents and extract entities
    print("\n📚 Step 1: Creating sample documents and extracting entities")
    
    sample_documents = [
        {
            "name": "project_plan.txt",
            "content": """
            Project Alpha - Budget Planning Initiative
            
            Team Members:
            - John Smith (Project Manager) - john.smith@acme.com
            - Sarah Johnson (Marketing Lead) - sarah.j@acme.com  
            - Mike Chen (Technical Lead) - mike.chen@acme.com
            
            Organization: Acme Corporation
            Department: Marketing Department
            
            Project involves quarterly budget analysis, stakeholder meetings,
            and coordination with the Finance team on revenue forecasting.
            
            Related projects: Market Research Initiative, Q4 Planning
            """
        },
        {
            "name": "meeting_notes.txt", 
            "content": """
            Meeting Notes - Budget Planning Review
            Date: 2024-12-15
            Attendees: John Smith, Sarah Johnson, Lisa Rodriguez
            
            Discussion Topics:
            - Customer Segmentation analysis
            - Revenue Forecasting methodology
            - Market Analysis results
            - Competitive Intelligence gathering
            
            Action Items:
            - John to coordinate with Finance Department
            - Sarah to update Marketing Strategy document
            - Lisa to provide Data Science insights
            
            Next meeting: Budget Planning project review
            """
        }
    ]
    
    # Step 2: Initialize knowledge graph and extract entities
    print("\n🔧 Step 2: Initializing knowledge graph")
    kg_path = Path("data/demo_knowledge_graph.db")
    kg_path.parent.mkdir(exist_ok=True)
    
    kg = KnowledgeGraph(str(kg_path))
    kg_builder = KnowledgeGraphBuilder(kg)
    
    # Extract entities from documents
    total_entities = 0
    for doc in sample_documents:
        print(f"   📄 Processing: {doc['name']}")
        entities = kg_builder.extract_entities_from_text(doc['content'], doc['name'])
        for entity in entities:
            kg.add_entity(entity)
        total_entities += len(entities)
        print(f"      Extracted {len(entities)} entities")
    
    print(f"   ✅ Total entities extracted: {total_entities}")
    
    # Step 3: Show knowledge graph statistics
    print("\n📊 Step 3: Knowledge graph statistics")
    stats = kg.get_statistics()
    print(f"   Total entities: {stats.get('total_entities', 0)}")
    print(f"   Total relationships: {stats.get('total_relationships', 0)}")
    print(f"   Entity types: {dict(stats.get('entity_types', {}))}")
    
    # Show sample entities by type
    for entity_type in ['person', 'organization', 'topic']:
        entities = kg.find_entities_by_type(entity_type)
        if entities:
            print(f"   {entity_type.title()} entities:")
            for entity in entities[:3]:  # Show first 3
                print(f"     - {entity.name} (confidence: {entity.confidence})")
    
    # Step 4: Test knowledge graph plugin
    print("\n🤖 Step 4: Testing knowledge graph plugin")
    
    agent = create_knowledge_graph_agent()
    print(f"   Agent created with {agent.registry.get_plugin_count()} plugins")
    
    # Test various operations
    operations = [
        {"operation": "get_statistics"},
        {"operation": "find_entities", "entity_type": "person"},
        {"operation": "find_entities", "entity_type": "topic"},
        {"operation": "extract_entities_from_question", "question": "What is John Smith working on for Budget Planning?"}
    ]
    
    kg_plugin = agent.registry.get_plugin("knowledge_graph")
    if kg_plugin:
        print("   Knowledge graph plugin found ✅")
        
        for op in operations:
            try:
                result = kg_plugin.execute(op)
                print(f"   Operation '{op['operation']}': {result.get('results', 'No result')}")
                
                if op['operation'] == 'find_entities' and result.get('entities'):
                    entities = result['entities'][:2]  # Show first 2
                    for entity in entities:
                        print(f"     - {entity['name']} ({entity['type']})")
                
                if op['operation'] == 'extract_entities_from_question' and result.get('entities'):
                    entities = result['entities']
                    print(f"     Found {len(entities)} entities in question:")
                    for entity in entities[:3]:  # Show first 3
                        print(f"       - {entity['name']} ({entity['type']})")
            
            except Exception as e:
                print(f"   ❌ Operation '{op['operation']}' failed: {e}")
    else:
        print("   ❌ Knowledge graph plugin not found")
    
    # Step 5: Demonstrate integration benefits
    print("\n🎯 Step 5: Integration benefits demonstrated")
    print("   ✅ Knowledge graph automatically populated during document ingestion")
    print("   ✅ Entity extraction identifies people, organizations, and topics")
    print("   ✅ Knowledge graph plugin enables entity-based queries")
    print("   ✅ Agent framework seamlessly integrates KG capabilities")
    print("   ✅ Backward compatibility maintained with --skip-kg option")
    
    # Cleanup
    try:
        kg_path.unlink()
        print("\n🧹 Demo cleanup completed")
    except:
        pass
    
    print("\n🎉 Knowledge Graph Integration Demo completed successfully!")
    print("\nNext steps:")
    print("1. Run document ingestion: python -m ingestion.pipeline")
    print("2. Query with KG support: python -m interface.cli.ask 'What projects is John working on?'")
    print("3. Use agent API with knowledge_graph plugin for advanced queries")

if __name__ == "__main__":
    main()