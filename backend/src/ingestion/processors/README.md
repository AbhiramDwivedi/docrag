# Text Processors

This directory contains text processing components that transform extracted document content into searchable vector embeddings. These processors bridge the gap between raw text extraction and vector storage.

## Components

### [chunker.py](chunker.py)
Intelligent text chunking with NLTK-based sentence boundary detection:

**Purpose**: Splits long documents into overlapping chunks optimized for semantic search and context preservation.

**Key Features**:
- **Sentence-aware chunking**: Uses NLTK to respect sentence boundaries
- **Configurable overlap**: Maintains context across chunk boundaries
- **Size optimization**: Balances chunk size with embedding model limitations
- **Automatic NLTK data**: Downloads required punkt tokenizer data

**Configuration**:
```yaml
chunk_size: 800        # Target characters per chunk
chunk_overlap: 150     # Overlap between adjacent chunks
```

### [embedder.py](embedder.py)
Vector embedding generation using sentence transformers:

**Purpose**: Converts text chunks into dense vector representations for semantic similarity search.

**Key Features**:
- **Sentence Transformers**: Uses `all-MiniLM-L6-v2` model by default (384 dimensions)
- **Batch processing**: Efficient processing of multiple chunks
- **GPU acceleration**: Automatic CUDA detection when available
- **Model caching**: Local model storage for offline usage

**Configuration**:
```yaml
embed_model: "sentence-transformers/all-MiniLM-L6-v2"
batch_size: 32         # Embedding batch size
```

## Data Flow

```mermaid
graph LR
    A[Extracted Text] --> B[Chunker]
    B --> C[Text Chunks]
    C --> D[Embedder]  
    D --> E[Vector Embeddings]
    E --> F[Vector Store]
    
    style B fill:#e1f5fe
    style D fill:#e8f5e8
```

## Usage Patterns

### Chunking Text
```python
from backend.src.ingestion.processors.chunker import chunk_text

# Process extracted document text
chunks = chunk_text(
    text="Long document content...",
    chunk_size=800,
    chunk_overlap=150
)
# Returns: List[str] of text chunks
```

### Generating Embeddings
```python
from backend.src.ingestion.processors.embedder import embed_texts

# Convert chunks to vectors
embeddings = embed_texts(chunks)
# Returns: numpy.ndarray of shape (n_chunks, 384)
```

### Pipeline Integration
```python
# Automatic processing in pipeline.py
text_chunks = chunk_text(extracted_text, settings.chunk_size, settings.chunk_overlap)
embeddings = embed_texts(text_chunks)
```

## Technical Details

### Chunking Strategy
- **Primary method**: Sentence boundary detection with NLTK
- **Fallback method**: Character-based splitting for non-sentence text
- **Overlap preservation**: Maintains context across chunk boundaries
- **Size constraints**: Respects embedding model token limits

### Embedding Model
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensions**: 384 (optimized for speed/quality balance)
- **Context window**: ~512 tokens
- **Performance**: ~1000 chunks/second on CPU

## Configuration Options

### Chunking Parameters
```yaml
chunking:
  chunk_size: 800           # Target characters per chunk
  chunk_overlap: 150        # Character overlap between chunks
  min_chunk_size: 100       # Minimum viable chunk size
  sentence_boundary: true   # Use NLTK sentence detection
```

### Embedding Parameters
```yaml
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32           # Chunks processed per batch
  normalize: true          # L2 normalize embeddings
  device: "auto"           # "cpu", "cuda", or "auto"
```

## Extension Points

### Custom Chunking Strategies
```python
def custom_chunker(text: str, **kwargs) -> List[str]:
    """Implement domain-specific chunking logic"""
    # Custom implementation
    return chunks
```

### Alternative Embedding Models
```python
def custom_embedder(texts: List[str]) -> np.ndarray:
    """Use different embedding model or provider"""
    # Custom implementation
    return embeddings
```

### Preprocessing Enhancements
- **Text cleaning**: Remove artifacts and formatting
- **Language detection**: Process multilingual documents
- **Domain adaptation**: Optimize for specific document types
- **Quality filtering**: Remove low-quality chunks

## Performance Considerations

### Memory Usage
- **Chunking**: Minimal memory overhead
- **Embedding**: Scales with batch size and model size
- **Peak usage**: During batch embedding generation

### Processing Speed
- **Chunking**: ~10MB text/second
- **Embedding**: Varies by hardware (GPU >> CPU)
- **Bottleneck**: Usually embedding generation

### Optimization Tips
- Increase batch_size for GPU processing
- Use CPU for small document collections
- Consider larger embedding models for quality over speed
- Implement async processing for large corpora

## Quality Metrics

### Chunking Quality
- Sentence boundary preservation
- Semantic coherence within chunks
- Appropriate overlap for context retention

### Embedding Quality  
- Semantic similarity preservation
- Dimensionality vs. quality trade-offs
- Retrieval performance in downstream tasks

## Links

- **Ingestion Pipeline**: [../README.md](../README.md)
- **Configuration**: [../../shared/README.md](../../shared/README.md)
- **Vector Storage**: [../storage/README.md](../storage/README.md)
- **Architecture Overview**: [../../../docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md)