"""Sentence-aware text splitter with overlap."""
from typing import List
import hashlib
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Fallback to old punkt if punkt_tab doesn't work
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def chunk_text(text: str, file_id: str, unit_id: str, max_len: int, overlap: int):
    from nltk.tokenize import sent_tokenize
    sents = sent_tokenize(text)
    chunks, buf = [], []
    for sent in sents:
        if sum(len(s) for s in buf) + len(sent) > max_len:
            _add_chunk(chunks, buf, file_id, unit_id)
            buf = buf[-overlap:]
        buf.append(sent)
    _add_chunk(chunks, buf, file_id, unit_id)
    return chunks

def _add_chunk(dest: List, sentences: List[str], file_id: str, unit_id: str):
    text = " ".join(sentences)
    if not text.strip():
        return
    text_hash = hashlib.sha1(text.encode()).hexdigest()[:12]
    chunk_id = f"{file_id}_{unit_id}_{text_hash}"
    dest.append({"id": chunk_id, "text": text})
