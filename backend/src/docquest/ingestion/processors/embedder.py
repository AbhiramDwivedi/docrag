from sentence_transformers import SentenceTransformer
from functools import lru_cache

@lru_cache
def get_model(name):
    return SentenceTransformer(name)

def embed_texts(texts, model_name):
    model = get_model(model_name)
    return model.encode(texts, batch_size=32, show_progress_bar=False)
