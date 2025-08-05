#!/usr/bin/env python3
"""Demo script showcasing Phase III advanced intelligence capabilities.

This script demonstrates the sophisticated reasoning and analysis capabilities
enabled by Phase III enhancements, including document relationship analysis,
knowledge graph capabilities, and comprehensive reporting.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.querying.agents.factory import create_phase3_agent
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

console = Console()


def demo_phase3_agent_creation():
    """Demonstrate Phase III agent creation with all capabilities."""
    rprint("\n🤖 [bold]Phase III Agent Creation Demo[/bold]")
    
    try:
        agent = create_phase3_agent()
        capabilities = agent.get_capabilities()
        
        rprint(f"\n✅ [bold green]Agent created successfully![/bold green]")
        rprint(f"🔧 Plugin count: {len(agent.registry._plugins) if hasattr(agent.registry, '_plugins') else 'Unknown'}")
        rprint(f"⚡ Available capabilities: {len(capabilities)}")
        
        # Display capabilities in a table
        table = Table(title="Phase III Agent Capabilities")
        table.add_column("Capability", style="cyan")
        table.add_column("Source Plugin", style="magenta")
        
        capability_mapping = {
            "document_query": "semantic_search",
            "vector_search": "semantic_search", 
            "semantic_search": "semantic_search",
            "content_analysis": "semantic_search",
            "find_files": "metadata_commands",
            "get_latest_files": "metadata_commands",
            "get_file_stats": "metadata_commands",
            "get_file_count": "metadata_commands",
            "get_file_types": "metadata_commands",
            "find_files_by_content": "metadata_commands",
            "document_similarity": "document_relationships",
            "document_clustering": "document_relationships",
            "cross_reference_detection": "document_relationships",
            "thematic_grouping": "document_relationships",
            "content_evolution_tracking": "document_relationships",
            "citation_analysis": "document_relationships",
            "relationship_analysis": "document_relationships",
            "collection_summary": "comprehensive_reporting",
            "thematic_analysis_report": "comprehensive_reporting",
            "activity_report": "comprehensive_reporting",
            "cross_document_insights": "comprehensive_reporting",
            "custom_reports": "comprehensive_reporting",
            "trend_analysis": "comprehensive_reporting",
            "usage_analytics": "comprehensive_reporting",
            "document_health_report": "comprehensive_reporting"
        }
        
        for capability in sorted(capabilities):
            plugin_source = capability_mapping.get(capability, "unknown")
            table.add_row(capability, plugin_source)
        
        console.print(table)
        return agent
        
    except Exception as e:
        rprint(f"\n❌ [bold red]Agent creation failed:[/bold red] {e}")
        return None


def demo_advanced_query_classification(agent):
    """Demonstrate Phase III advanced query classification."""
    rprint("\n🔍 [bold]Advanced Query Classification Demo[/bold]")
    
    phase3_test_queries = [
        # Document relationship queries
        ("Document Similarity", "find documents similar to budget_report.pdf"),
        ("Document Clustering", "cluster documents by theme into 5 groups"),
        ("Cross-references", "find cross-references between documents"),
        ("Thematic Analysis", "analyze themes across all documents"),
        ("Content Evolution", "track how budget documents have evolved"),
        ("Citation Analysis", "find citations in research papers"),
        
        # Reporting queries
        ("Collection Summary", "generate a summary report of the document collection"),
        ("Activity Report", "show me activity report for last week"),
        ("Insights Report", "provide insights about document relationships"),
        ("Trend Analysis", "analyze trends in document usage"),
        ("Usage Analytics", "show usage analytics and performance metrics"),
        ("Health Report", "generate a health report for the collection"),
        
        # Complex multi-step queries
        ("Multi-step Complex", "analyze budget themes and generate a trend report"),
        ("Comprehensive Analysis", "find similar financial documents and summarize relationships"),
        ("Advanced Synthesis", "what patterns exist in project documents over time")
    ]
    
    for category, query in phase3_test_queries:
        plugins = agent._classify_query(query)
        reasoning = agent._reasoning_trace
        
        rprint(f"\n📋 {category}:")
        rprint(f"   Query: [italic]\"{query}\"[/italic]")
        rprint(f"   Plugins: [bold]{plugins}[/bold]")
        if reasoning:
            rprint(f"   Reasoning: {reasoning[-1] if reasoning else 'No reasoning recorded'}")
        
        # Clear reasoning trace for next query
        agent._reasoning_trace = []


def demo_knowledge_graph_integration():
    """Demonstrate knowledge graph integration."""
    rprint("\n🕸️ [bold]Knowledge Graph Integration Demo[/bold]")
    
    try:
        from backend.ingestion.storage.knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder, Entity, Relationship
        
        # Create a demo knowledge graph
        kg = KnowledgeGraph("data/demo_knowledge_graph.db")
        
        # Add sample entities
        entities = [
            Entity(
                id="budget_project_2024",
                type="project",
                name="Budget Project 2024",
                properties={"status": "active", "priority": "high"},
                confidence=1.0
            ),
            Entity(
                id="finance_team",
                type="organization",
                name="Finance Team",
                properties={"department": "finance", "size": 12},
                confidence=1.0
            ),
            Entity(
                id="quarterly_review",
                type="concept",
                name="Quarterly Review",
                properties={"frequency": "quarterly", "importance": "high"},
                confidence=0.9
            )
        ]
        
        for entity in entities:
            kg.add_entity(entity)
        
        # Add sample relationships
        relationships = [
            Relationship(
                source_id="finance_team",
                target_id="budget_project_2024",
                relationship_type="manages",
                properties={"role": "primary_owner"},
                weight=1.0,
                confidence=0.95
            ),
            Relationship(
                source_id="budget_project_2024",
                target_id="quarterly_review",
                relationship_type="involves",
                properties={"frequency": "quarterly"},
                weight=0.8,
                confidence=0.85
            )
        ]
        
        for relationship in relationships:
            kg.add_relationship(relationship)
        
        # Demonstrate graph statistics
        stats = kg.get_statistics()
        
        rprint(f"\n📊 Knowledge Graph Statistics:")
        rprint(f"   • Total entities: {stats.get('total_entities', 0)}")
        rprint(f"   • Total relationships: {stats.get('total_relationships', 0)}")
        rprint(f"   • Entity types: {list(stats.get('entity_types', {}).keys())}")
        rprint(f"   • Relationship types: {list(stats.get('relationship_types', {}).keys())}")
        rprint(f"   • Graph density: {stats.get('density', 0):.3f}")
        
        # Demonstrate entity relationships
        related_entities = kg.find_related_entities("budget_project_2024")
        rprint(f"\n🔗 Related entities to 'budget_project_2024':")
        for entity, rel_type, distance in related_entities:
            rprint(f"   • {entity.name} ({rel_type}, distance: {distance:.2f})")
        
        rprint("\n✅ [bold green]Knowledge graph integration working![/bold green]")
        
    except Exception as e:
        rprint(f"\n❌ [bold red]Knowledge graph demo failed:[/bold red] {e}")


def demo_comprehensive_reporting():
    """Demonstrate comprehensive reporting capabilities."""
    rprint("\n📊 [bold]Comprehensive Reporting Demo[/bold]")
    
    try:
        from backend.querying.agents.plugins.comprehensive_reporting import ComprehensiveReportingPlugin
        
        plugin = ComprehensiveReportingPlugin()
        
        # Test different report types
        report_types = [
            ("Collection Summary", {"operation": "generate_collection_summary"}),
            ("Activity Report", {"operation": "generate_activity_report", "time_window": "1_week"}),
            ("Health Report", {"operation": "generate_health_report"}),
            ("Usage Analytics", {"operation": "generate_usage_analytics"})
        ]
        
        for report_name, params in report_types:
            rprint(f"\n📋 {report_name}:")
            
            if plugin.validate_params(params):
                result = plugin.execute(params)
                
                if "error" not in result:
                    rprint(f"   ✅ [green]Generated successfully[/green]")
                    
                    # Show brief summary of the report
                    report_data = result.get("report", {})
                    if report_data:
                        rprint(f"   📄 Title: {report_data.get('title', 'Unknown')}")
                        rprint(f"   📅 Generated: {report_data.get('generated_at', 'Unknown')}")
                        rprint(f"   📊 Sections: {len(report_data.get('sections', []))}")
                        
                        # Show summary if available
                        summary = report_data.get('summary', '')
                        if summary:
                            rprint(f"   📝 Summary: {summary[:100]}{'...' if len(summary) > 100 else ''}")
                else:
                    rprint(f"   ❌ [red]Failed: {result.get('error', 'Unknown error')}[/red]")
            else:
                rprint(f"   ❌ [red]Invalid parameters[/red]")
        
        rprint("\n✅ [bold green]Comprehensive reporting demo complete![/bold green]")
        
    except Exception as e:
        rprint(f"\n❌ [bold red]Reporting demo failed:[/bold red] {e}")


def demo_document_relationships():
    """Demonstrate document relationship analysis."""
    rprint("\n🔗 [bold]Document Relationship Analysis Demo[/bold]")
    
    try:
        from backend.querying.agents.plugins.document_relationships import DocumentRelationshipPlugin
        
        plugin = DocumentRelationshipPlugin()
        
        # Test different relationship analysis operations
        operations = [
            ("Find Similar Documents", {
                "operation": "find_similar_documents",
                "query": "budget analysis",
                "threshold": 0.7
            }),
            ("Cluster Documents", {
                "operation": "cluster_documents",
                "num_clusters": 3
            }),
            ("Analyze Themes", {
                "operation": "analyze_themes"
            }),
            ("Comprehensive Analysis", {
                "operation": "analyze_relationships",
                "types": ["similarity", "themes"]
            })
        ]
        
        for operation_name, params in operations:
            rprint(f"\n🔍 {operation_name}:")
            
            if plugin.validate_params(params):
                result = plugin.execute(params)
                
                if "error" not in result:
                    rprint(f"   ✅ [green]Analysis completed[/green]")
                    
                    # Show brief results
                    response = result.get("response", "")
                    if response:
                        # Show first line of response
                        first_line = response.split('\n')[0]
                        rprint(f"   📋 Result: {first_line}")
                    
                    # Show metadata
                    metadata = result.get("metadata", {})
                    if metadata:
                        for key, value in metadata.items():
                            if key != "operation":
                                rprint(f"   📊 {key}: {value}")
                else:
                    rprint(f"   ❌ [red]Failed: {result.get('error', 'Unknown error')}[/red]")
            else:
                rprint(f"   ❌ [red]Invalid parameters[/red]")
        
        rprint("\n✅ [bold green]Document relationship analysis demo complete![/bold green]")
        
    except Exception as e:
        rprint(f"\n❌ [bold red]Relationship analysis demo failed:[/bold red] {e}")


def demo_phase3_example_queries(agent):
    """Demonstrate Phase III example queries from the implementation plan."""
    rprint("\n🎯 [bold]Phase III Example Queries Demo[/bold]")
    
    example_queries = [
        "What's the latest on the server migration project?",
        "Compare the Q1 and Q2 budget reports",
        "Find similar documents to the compliance policy",
        "Generate a comprehensive report on document trends",
        "Show me relationships between financial documents",
        "Analyze themes in recent project files"
    ]
    
    for i, query in enumerate(example_queries, 1):
        rprint(f"\n{i}. [bold]Query:[/bold] [italic]\"{query}\"[/italic]")
        
        try:
            # Show classification
            plugins = agent._classify_query(query)
            rprint(f"   🎯 Classified plugins: {plugins}")
            
            # Show reasoning
            if agent._reasoning_trace:
                rprint(f"   💭 Reasoning: {agent._reasoning_trace[-1]}")
            
            # Clear trace for next query
            agent._reasoning_trace = []
            
            rprint(f"   ✅ [green]Successfully classified[/green]")
            
        except Exception as e:
            rprint(f"   ❌ [red]Classification failed: {e}[/red]")


def main():
    """Main demo function."""
    console.print(Panel.fit(
        Text("DocQuest Phase III Demo\nAdvanced Intelligence & Comprehensive Analysis", 
             style="bold magenta", justify="center"),
        style="blue"
    ))
    
    rprint("\n🚀 [bold green]Phase III Implementation Complete![/bold green]")
    rprint("This demo showcases the advanced intelligence capabilities of DocQuest Phase III:")
    
    try:
        # 1. Agent creation
        agent = demo_phase3_agent_creation()
        
        if agent:
            # 2. Query classification
            demo_advanced_query_classification(agent)
            
            # 3. Knowledge graph integration
            demo_knowledge_graph_integration()
            
            # 4. Comprehensive reporting
            demo_comprehensive_reporting()
            
            # 5. Document relationships
            demo_document_relationships()
            
            # 6. Example queries
            demo_phase3_example_queries(agent)
        
        rprint("\n✅ [bold green]Phase III Demo Complete![/bold green]")
        rprint("\n🎉 [bold]DocQuest now features:[/bold]")
        rprint("   • 🧠 Advanced intelligence with sophisticated reasoning")
        rprint("   • 🔗 Document relationship analysis and knowledge graphs")
        rprint("   • 📊 Comprehensive reporting and analytics")
        rprint("   • 🎯 Multi-step query planning and execution")
        rprint("   • ⚡ Performance optimization and caching")
        rprint("   • 🌐 Rich plugin architecture for extensibility")
        
        rprint("\n🔮 [bold]Ready for advanced document analysis![/bold]")
        
    except Exception as e:
        rprint(f"\n❌ [bold red]Demo error:[/bold red] {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())