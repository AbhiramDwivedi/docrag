"""Entity-aware indexing utilities for Phase 4 advanced features.

This module provides optional NER (Named Entity Recognition) functionality
for extracting and indexing entities during document ingestion to enable
entity-document mapping and boosting.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Entity extractor using spaCy for NER during ingestion."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize entity extractor.
        
        Args:
            model_name: spaCy model name to use for NER
        """
        self.model_name = model_name
        self._nlp = None
        self._available = False
        
        # Check if spaCy is available
        try:
            import spacy
            # Basic security check - verify the module is from expected source
            module_path = spacy.__file__
            if module_path and 'site-packages' not in module_path:
                logger.warning(f"spaCy loaded from unexpected location: {module_path}")
            
            self._spacy = spacy
            self._available = True
            logger.info("spaCy library validated and available for entity extraction")
        except ImportError:
            logger.warning(
                "spaCy not available. Entity-aware indexing will be disabled. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )
        except Exception as e:
            logger.error(f"Error loading spaCy: {e}")
            self._available = False
    
    def _load_model(self):
        """Lazy load spaCy model."""
        if not self._available or self._nlp is not None:
            return self._nlp
        
        try:
            self._nlp = self._spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logger.error(
                f"Failed to load spaCy model '{self.model_name}'. "
                f"Install with: python -m spacy download {self.model_name}"
            )
            self._available = False
            return None
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            self._available = False
            return None
        
        return self._nlp
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of entity dictionaries with text, label, start, end positions
        """
        if not self._available:
            return []
        
        nlp = self._load_model()
        if nlp is None:
            return []
        
        try:
            doc = nlp(text)
            entities = []
            
            for ent in doc.ents:
                # Filter for relevant entity types
                if ent.label_ in {
                    'PERSON', 'ORG', 'GPE',  # Person, Organization, Geopolitical entity
                    'PRODUCT', 'EVENT', 'LAW',  # Product, Event, Law
                    'NORP', 'FAC', 'LOC'  # Nationality/Religious group, Facility, Location
                }:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 1.0  # spaCy doesn't provide confidence scores
                    })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if entity extraction is available."""
        return self._available


def create_entity_document_mapping(entities: List[Dict[str, Any]], 
                                 document_id: str, 
                                 chunk_id: str) -> List[Dict[str, Any]]:
    """Create entity-to-document mappings for indexing.
    
    Args:
        entities: List of extracted entities
        document_id: Document identifier
        chunk_id: Chunk identifier
        
    Returns:
        List of entity mapping records for database storage
    """
    mappings = []
    
    for entity in entities:
        mapping = {
            'entity_text': entity['text'].lower(),  # Normalize for matching
            'entity_label': entity['label'],
            'document_id': document_id,
            'chunk_id': chunk_id,
            'start_pos': entity['start'],
            'end_pos': entity['end'],
            'confidence': entity.get('confidence', 1.0)
        }
        mappings.append(mapping)
    
    return mappings


def enhance_chunk_with_entities(chunk_data: Dict[str, Any], 
                               extractor: EntityExtractor) -> Dict[str, Any]:
    """Enhance chunk data with extracted entities.
    
    Args:
        chunk_data: Chunk data dictionary
        extractor: Entity extractor instance
        
    Returns:
        Enhanced chunk data with entities
    """
    enhanced_chunk = chunk_data.copy()
    
    if not extractor.is_available():
        enhanced_chunk['entities'] = []
        return enhanced_chunk
    
    text = chunk_data.get('content', '')
    if not text:
        enhanced_chunk['entities'] = []
        return enhanced_chunk
    
    # Extract entities
    entities = extractor.extract_entities(text)
    enhanced_chunk['entities'] = entities
    
    # Add entity count for quick filtering
    enhanced_chunk['entity_count'] = len(entities)
    
    # Add entity types for faceted search
    entity_types = list(set(ent['label'] for ent in entities))
    enhanced_chunk['entity_types'] = entity_types
    
    return enhanced_chunk


def get_entity_boost_score(query_entities: List[str], 
                         chunk_entities: List[Dict[str, Any]], 
                         boost_factor: float = 0.2) -> float:
    """Calculate entity boost score for a chunk.
    
    Args:
        query_entities: Entities detected in the query
        chunk_entities: Entities extracted from the chunk
        boost_factor: Base boost factor per matching entity
        
    Returns:
        Entity boost score to add to similarity
    """
    if not query_entities or not chunk_entities:
        return 0.0
    
    # Normalize query entities for matching
    normalized_query_entities = [ent.lower() for ent in query_entities]
    
    # Count matching entities
    chunk_entity_texts = [ent['text'].lower() for ent in chunk_entities]
    matches = 0
    
    for query_entity in normalized_query_entities:
        if query_entity in chunk_entity_texts:
            matches += 1
    
    # Calculate boost score
    boost_score = matches * boost_factor
    
    # Cap the boost to avoid overwhelming similarity scores
    max_boost = 0.5
    return min(boost_score, max_boost)


# SQL schema for entity indexing (for reference)
ENTITY_MAPPING_SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_mappings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_text TEXT NOT NULL,
    entity_label TEXT NOT NULL,
    document_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    start_pos INTEGER,
    end_pos INTEGER,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_entity_text (entity_text),
    INDEX idx_document_id (document_id),
    INDEX idx_chunk_id (chunk_id),
    INDEX idx_entity_label (entity_label)
);
"""


# Example usage in ingestion pipeline
def example_entity_aware_ingestion():
    """Example of how to integrate entity extraction in ingestion pipeline."""
    print("Entity-aware Ingestion Example")
    print("-" * 40)
    
    # Initialize entity extractor
    extractor = EntityExtractor()
    
    if not extractor.is_available():
        print("spaCy not available - entity extraction disabled")
        return
    
    # Example document chunk
    sample_chunk = {
        'content': "Tesla Motors, founded by Elon Musk, is revolutionizing electric vehicles. "
                  "The company's Model S has impressive performance compared to traditional cars.",
        'document_id': 'tesla_doc_001',
        'chunk_id': 'tesla_chunk_001'
    }
    
    # Extract entities and enhance chunk
    enhanced_chunk = enhance_chunk_with_entities(sample_chunk, extractor)
    
    print(f"Original chunk: {sample_chunk['content'][:50]}...")
    print(f"Entities found: {len(enhanced_chunk['entities'])}")
    
    for entity in enhanced_chunk['entities']:
        print(f"  - {entity['text']} ({entity['label']})")
    
    # Create entity mappings for database
    mappings = create_entity_document_mapping(
        enhanced_chunk['entities'],
        sample_chunk['document_id'],
        sample_chunk['chunk_id']
    )
    
    print(f"Entity mappings created: {len(mappings)}")
    
    # Example entity boost calculation
    query_entities = ["Tesla", "Elon Musk"]
    boost_score = get_entity_boost_score(
        query_entities, 
        enhanced_chunk['entities'],
        boost_factor=0.2
    )
    
    print(f"Entity boost score for query {query_entities}: {boost_score}")


if __name__ == "__main__":
    example_entity_aware_ingestion()