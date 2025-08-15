# Phase 3 Embedding Model Upgrade - User Guide

This guide explains how to use the new embedding model capabilities introduced in Phase 3.

## Overview

Phase 3 adds support for advanced embedding models that provide better retrieval quality:

- **E5 models** (intfloat/e5-base-v2): High-quality with query/passage prefixes
- **BGE models** (BAAI/bge-small-en-v1.5): Balanced speed/quality 
- **GTE models** (thenlper/gte-base): General purpose

## Quick Start

### 1. Choose Your Model

Edit your `config.yaml` to use one of the new models:

```yaml
# Option 1: E5 Base v2 (recommended for quality)
embed_model: "intfloat/e5-base-v2"
embed_model_version: "2.0.0"

# Option 2: BGE Small v1.5 (balanced)  
embed_model: "BAAI/bge-small-en-v1.5"
embed_model_version: "2.0.0"

# Option 3: GTE Base (general purpose)
embed_model: "thenlper/gte-base" 
embed_model_version: "2.0.0"
```

### 2. Migrate Existing Embeddings

Run the migration script to upgrade your existing documents:

```bash
# Start migration to E5 model
python scripts/migrate_embeddings.py --model intfloat/e5-base-v2 --version 2.0.0

# Resume if interrupted
python scripts/migrate_embeddings.py --resume

# Dry run to see what would be migrated
python scripts/migrate_embeddings.py --model intfloat/e5-base-v2 --dry-run
```

### 3. Start Using Improved Retrieval

The system will automatically use the new model with proper formatting. No code changes needed!

## Model Comparison

| Model | Dimensions | Prefixes | Strengths | Best For |
|-------|------------|----------|-----------|----------|
| all-MiniLM-L6-v2 (current) | 384 | None | Fast, lightweight | General use |
| intfloat/e5-base-v2 | 768 | query:/passage: | High quality | Best retrieval |
| BAAI/bge-small-en-v1.5 | 384 | None | Balanced | Speed + quality |
| thenlper/gte-base | 768 | None | General purpose | Versatile |

## Model-Specific Features

### E5 Models

E5 models require specific prefixes for optimal performance:

- **Query text**: Automatically prefixed with "query: "
- **Passage text**: Automatically prefixed with "passage: "

Example:
```python
# Input: "What is machine learning?"
# E5 query: "query: What is machine learning?"
# E5 passage: "passage: What is machine learning?"
```

### BGE Models

BGE models work well without prefixes and use CLS token pooling:

- No text formatting required
- Optimized for retrieval tasks
- Good balance of speed and quality

### GTE Models  

GTE models are general-purpose and use mean pooling:

- No text formatting required
- Good for various text understanding tasks
- Reliable baseline performance

## Migration Process

The migration script provides several safety features:

### Backup Creation
- Automatic backup of database and vector index
- Backup metadata saved with timestamp
- Easy rollback if needed

### Progress Tracking
- Migration progress saved to file
- Resume capability if interrupted
- Batch processing for memory efficiency

### Safety Features
- No data loss during migration
- Preserves all existing document metadata
- Validates new embeddings before committing

## Performance Expectations

Phase 3 targets these improvements:

- **≥20% recall@10 improvement** on evaluation datasets
- **Better proper noun handling** with dense-only retrieval
- **Improved semantic understanding** with larger models
- **Task-specific optimization** for retrieval scenarios

## Troubleshooting

### Migration Issues

**Problem**: Migration script fails to start
```bash
# Check dependencies
pip install -e backend/

# Verify database exists
ls data/docmeta.db
```

**Problem**: Out of memory during migration
```bash
# Reduce batch size
python scripts/migrate_embeddings.py --model intfloat/e5-base-v2 --batch-size 50
```

**Problem**: Interrupted migration
```bash
# Simply resume - progress is automatically saved
python scripts/migrate_embeddings.py --resume
```

### Model Loading Issues

**Problem**: Model download fails
- Ensure internet connection
- Check disk space (models are 1-3GB)
- Verify model name is correct

**Problem**: CUDA out of memory
- Models will automatically fall back to CPU
- Consider using smaller model (bge-small-en-v1.5)

## Advanced Usage

### Custom Model Configuration

You can add support for other models by extending the configuration in `embedder.py`:

```python
# Add to get_model_config() function
"your-model-name": {
    "query_prefix": "Q: ",
    "passage_prefix": "P: ", 
    "requires_prefixes": True,
    "pooling": "mean"
}
```

### Batch Processing

For large datasets, use larger batch sizes:

```bash
python scripts/migrate_embeddings.py \
    --model intfloat/e5-base-v2 \
    --batch-size 500 \
    --version 2.0.0
```

### Version Tracking

The `embed_model_version` field tracks compatibility:

- `1.0.0`: Original all-MiniLM-L6-v2 model
- `2.0.0`: Phase 3 upgraded models (E5, BGE, GTE)
- Future versions for additional model upgrades

## Best Practices

1. **Test with small datasets first** before migrating large corpora
2. **Monitor memory usage** during migration
3. **Keep backups** until you're satisfied with results
4. **Update version numbers** when changing models
5. **Validate retrieval quality** after migration

## Next Steps

After completing Phase 3:

1. Test improved retrieval on your use cases
2. Measure recall improvements with your queries
3. Fine-tune configuration based on results
4. Consider Phase 4 features for advanced query understanding

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review migration logs in console output
3. Check backup files in `data/migration_backup/`
4. Validate configuration with dry-run mode