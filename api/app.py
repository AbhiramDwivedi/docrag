from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path

# Add parent directory to path so we can import from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.ask import answer

app = FastAPI()

class QueryIn(BaseModel):
    question: str

@app.post('/query')
def query_endpoint(q: QueryIn):
    return {'answer': answer(q.question)}
