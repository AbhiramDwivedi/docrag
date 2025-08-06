from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path

# Add backend root to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.src.interface.cli.ask import answer

app = FastAPI()

class QueryIn(BaseModel):
    question: str

@app.post('/query')
def query_endpoint(q: QueryIn):
    return {'answer': answer(q.question)}
