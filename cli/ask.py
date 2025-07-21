import sys
from pathlib import Path

# Add parent directory to path so we can import from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.embed import embed_texts
from ingest.vector_store import VectorStore
from config.config import settings
from openai import OpenAI


def answer(question: str):
    # Check if OpenAI API key is configured
    if not settings.openai_api_key or settings.openai_api_key == "your-openai-api-key-here":
        return ("❌ OpenAI API key not configured.\n"
                "Please set your API key in config/config.yaml:\n"
                "1. Get your API key from: https://platform.openai.com/api-keys\n"
                "2. Edit config/config.yaml and replace 'your-openai-api-key-here' with your actual key\n"
                "3. Alternatively, set the OPENAI_API_KEY environment variable")
    
    store = VectorStore(settings.vector_path, settings.db_path, dim=384)
    q_vec = embed_texts([question], settings.embed_model)[0]
    top = store.query(q_vec, k=8)
    
    if not top:
        return "No relevant information found."
    
    context = "\n\n".join([row.get('text','') for row in top])
    prompt = f"Answer the question using only the context below.\nContext:\n\"\"\"\n{context}\n\"\"\"\nQ: {question}\nA:"
    
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=512,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Error calling OpenAI API: {e}\nPlease check your API key in config/config.yaml"


if __name__ == '__main__':
    print(answer(' '.join(sys.argv[1:])))
