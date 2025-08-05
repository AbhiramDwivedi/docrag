#!/usr/bin/env python3
"""Demo script showcasing Phase 2 enhanced metadata capabilities.

This script demonstrates the advanced querying capabilities enabled by
Phase 2 enhancements, including email analysis, file filtering, and
multi-step query planning.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.querying.agents.factory import create_default_agent
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def demo_query_classification():
    """Demonstrate enhanced query classification."""
    rprint("\n🔍 [bold]Enhanced Query Classification Demo[/bold]")
    
    agent = create_default_agent()
    
    test_queries = [
        ("Basic metadata", "how many files do we have?"),
        ("Email filtering", "emails from john@example.com last week"),
        ("File size filtering", "show me PDF files larger than 1MB"),
        ("Content search", "what is the compliance policy"),
        ("Multi-step query", "latest email about budget and related files"),
        ("Time-based query", "files modified yesterday"),
        ("Email subject search", "emails about meeting"),
        ("File type query", "list all Excel files"),
    ]
    
    for category, query in test_queries:
        plugins = agent._classify_query(query)
        reasoning = agent._reasoning_trace
        
        rprint(f"\n📋 {category}:")
        rprint(f"   Query: [italic]\"{query}\"[/italic]")
        rprint(f"   Plugins: [bold]{plugins}[/bold]")
        if reasoning:
            rprint(f"   Reasoning: {reasoning[-1] if reasoning else 'No reasoning recorded'}")
        
        # Clear reasoning trace for next query
        agent._reasoning_trace = []


def demo_metadata_plugin_capabilities():
    """Demonstrate enhanced metadata plugin capabilities."""
    rprint("\n🧩 [bold]Enhanced Metadata Plugin Capabilities[/bold]")
    
    from backend.querying.agents.plugins.metadata_commands import MetadataCommandsPlugin
    
    plugin = MetadataCommandsPlugin()
    info = plugin.get_info()
    
    rprint(f"\n📦 Plugin: {info.name} v{info.version}")
    rprint(f"📋 Description: {info.description}")
    rprint("\n🎯 Capabilities:")
    
    for capability in info.capabilities:
        rprint(f"   • {capability}")
    
    rprint("\n🔧 Example Query Classifications:")
    
    test_classifications = [
        "emails from alice@company.com",
        "emails about project status last month", 
        "PDF files larger than 5MB",
        "files modified this week",
        "Excel files created yesterday"
    ]
    
    for query in test_classifications:
        query_type, params = plugin._classify_enhanced_metadata_query(query)
        rprint(f"\n   Query: [italic]\"{query}\"[/italic]")
        rprint(f"   Type: [bold]{query_type}[/bold]")
        if params:
            rprint(f"   Parameters: {params}")


def demo_vector_store():
    """Demonstrate vector store schema."""
    rprint("\n💾 [bold]Vector Store Schema[/bold]")
    
    from backend.ingestion.storage.enhanced_vector_store import EnhancedVectorStore
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        store = EnhancedVectorStore(tmpdir / "demo.index", tmpdir / "demo.db", dim=384)
        
        # Show the schema capabilities
        rprint("\n📊 Database Schema Features:")
        rprint("   • Files table: comprehensive file metadata")
        rprint("   • Email metadata table: sender, recipients, dates, subjects")
        rprint("   • Indexed queries: fast filtering by type, date, size, sender")
        rprint("   • JSON support: flexible additional metadata storage")
        
        # Demonstrate empty statistics
        stats = store.get_file_statistics()
        rprint(f"\n📈 Collection Statistics (Empty Database):")
        rprint(f"   • Total files: {stats['total_files']}")
        rprint(f"   • Email files: {stats['total_emails']}")
        rprint(f"   • Total chunks: {stats['total_chunks']}")


def demo_migration_capabilities():
    """Demonstrate migration from Phase 1 to Phase 2."""
    rprint("\n🔄 [bold]Migration Capabilities[/bold]")
    
    rprint("Phase 1 → Phase 2 Migration Features:")
    rprint("   • Backward compatibility: existing data preserved")
    rprint("   • Safe upgrade: automatic backup creation")
    rprint("   • Metadata extraction: existing files get enhanced metadata")
    rprint("   • Zero downtime: migration can run alongside existing system")
    
    rprint("\n🛠️  Migration Tools:")
    rprint("   • [bold]ingest/migrate_to_enhanced.py[/bold] - Safe schema upgrade")
    rprint("   • [bold]ingest/enhanced_ingest.py[/bold] - Enhanced ingestion pipeline")
    rprint("   • Automatic detection of schema version in plugins")


def demo_example_use_cases():
    """Show example use cases enabled by Phase 2."""
    rprint("\n💡 [bold]Example Use Cases Enabled by Phase 2[/bold]")
    
    use_cases = [
        {
            "scenario": "Executive Dashboard",
            "query": "Show me all email activity from the CEO last month",
            "plugins": ["metadata"],
            "description": "Filter emails by sender and date range"
        },
        {
            "scenario": "Compliance Audit", 
            "query": "Find all financial documents larger than 10MB from Q4",
            "plugins": ["metadata"],
            "description": "Filter by file type, size, and date range"
        },
        {
            "scenario": "Project Research",
            "query": "Latest emails about project Alpha and related documents",
            "plugins": ["metadata", "semantic_search"],
            "description": "Multi-step query combining email search and content analysis"
        },
        {
            "scenario": "Document Cleanup",
            "query": "Show me duplicate PDF files larger than 1MB",
            "plugins": ["metadata"],
            "description": "File management with size and type filtering"
        },
        {
            "scenario": "Team Communication Analysis",
            "query": "Emails from team leads about deadlines this week",
            "plugins": ["metadata"],
            "description": "Complex filtering by sender patterns and content keywords"
        }
    ]
    
    for i, use_case in enumerate(use_cases, 1):
        rprint(f"\n{i}. [bold]{use_case['scenario']}[/bold]")
        rprint(f"   Query: [italic]\"{use_case['query']}\"[/italic]")
        rprint(f"   Plugins: {use_case['plugins']}")
        rprint(f"   Description: {use_case['description']}")


def main():
    """Main demo function."""
    console.print(Panel.fit(
        Text("DocQuest Phase 2 Demo\nEnhanced Metadata + Database Schema", 
             style="bold magenta", justify="center"),
        style="blue"
    ))
    
    rprint("\n🎉 [bold green]Phase 2 Implementation Complete![/bold green]")
    rprint("This demo showcases the enhanced capabilities of DocQuest Phase 2:")
    
    try:
        demo_query_classification()
        demo_metadata_plugin_capabilities()
        demo_vector_store()
        demo_migration_capabilities()
        demo_example_use_cases()
        
        rprint("\n✅ [bold green]Phase 2 Demo Complete![/bold green]")
        rprint("\n🚀 [bold]Ready for Phase 3:[/bold]")
        rprint("   • Document relationship analysis")
        rprint("   • Advanced LLM-based query planning")
        rprint("   • Comprehensive reporting capabilities")
        
    except Exception as e:
        rprint(f"\n❌ [bold red]Demo error:[/bold red] {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())