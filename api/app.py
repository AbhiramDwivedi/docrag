from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path

# Add parent directory to path so we can import from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import Agent, PluginRegistry
from agent.plugins import SemanticSearchPlugin, MetadataPlugin

app = FastAPI()

# Initialize agent with plugins at module level for reuse
registry = PluginRegistry()
registry.register(SemanticSearchPlugin())
registry.register(MetadataPlugin())
agent = Agent(registry)

class QueryIn(BaseModel):
    question: str

@app.post('/query')
def query_endpoint(q: QueryIn):
    """
    Process user queries through the DocQuest agent framework.
    
    Maintains backward compatibility with the original API interface
    while using the new plugin-based agent system.
    """
    return {'answer': agent.process_query(q.question)}

@app.get('/capabilities')
def capabilities_endpoint():
    """Get agent capabilities for introspection."""
    return {'capabilities': agent.get_capabilities()}

@app.get('/reasoning')
def reasoning_endpoint():
    """Get explanation of last query processing."""
    reasoning = agent.explain_reasoning()
    return {'reasoning': reasoning}
