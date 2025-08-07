"""Knowledge Graph Implementation for DocQuest Phase III.

This module provides a knowledge graph representation of document relationships,
entities, and semantic connections to enable advanced querying and analysis.
"""

import logging
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    type: str  # 'person', 'organization', 'topic', 'document', 'concept'
    name: str
    properties: Dict[str, Any]
    confidence: float = 1.0
    created_at: Optional[datetime] = None


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source_id: str
    target_id: str
    relationship_type: str  # 'mentions', 'authored_by', 'relates_to', 'cites', 'contains'
    properties: Dict[str, Any]
    weight: float = 1.0
    confidence: float = 1.0
    created_at: Optional[datetime] = None


@dataclass
class DocumentNode:
    """Represents a document node in the knowledge graph."""
    document_path: str
    title: str
    content_summary: str
    entities: List[str]  # Entity IDs mentioned in the document
    themes: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    modified_at: datetime


class KnowledgeGraph:
    """Knowledge graph for document relationships and entity extraction."""
    
    def __init__(self, db_path: str):
        """Initialize the knowledge graph.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize NetworkX graph for analysis
        self.graph = nx.MultiDiGraph()
        
        # Initialize database
        self._init_database()
        self._load_graph_from_db()
    
    def _init_database(self):
        """Initialize the knowledge graph database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Entities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    properties TEXT,  -- JSON
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    properties TEXT,  -- JSON
                    weight REAL DEFAULT 1.0,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES entities (id),
                    FOREIGN KEY (target_id) REFERENCES entities (id)
                )
            """)
            
            # Document nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_nodes (
                    document_path TEXT PRIMARY KEY,
                    title TEXT,
                    content_summary TEXT,
                    entities TEXT,  -- JSON list of entity IDs
                    themes TEXT,    -- JSON list of themes
                    metadata TEXT,  -- JSON
                    created_at TIMESTAMP,
                    modified_at TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities (name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships (source_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships (target_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships (relationship_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_nodes_modified ON document_nodes (modified_at)")
            
            conn.commit()
    
    def _load_graph_from_db(self):
        """Load the graph structure from database into NetworkX."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Load entities as nodes
            cursor.execute("SELECT id, type, name, properties, confidence FROM entities")
            for entity_id, entity_type, name, properties_json, confidence in cursor.fetchall():
                properties = json.loads(properties_json) if properties_json else {}
                self.graph.add_node(
                    entity_id,
                    type=entity_type,
                    name=name,
                    properties=properties,
                    confidence=confidence
                )
            
            # Load relationships as edges
            cursor.execute("""
                SELECT source_id, target_id, relationship_type, properties, weight, confidence
                FROM relationships
            """)
            for source_id, target_id, rel_type, properties_json, weight, confidence in cursor.fetchall():
                properties = json.loads(properties_json) if properties_json else {}
                self.graph.add_edge(
                    source_id,
                    target_id,
                    relationship_type=rel_type,
                    properties=properties,
                    weight=weight,
                    confidence=confidence
                )
    
    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the knowledge graph.
        
        Args:
            entity: Entity to add
            
        Returns:
            True if successfully added, False if already exists
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO entities 
                    (id, type, name, properties, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    entity.id,
                    entity.type,
                    entity.name,
                    json.dumps(entity.properties),
                    entity.confidence,
                    entity.created_at or datetime.now()
                ))
                conn.commit()
            
            # Add to NetworkX graph
            self.graph.add_node(
                entity.id,
                type=entity.type,
                name=entity.name,
                properties=entity.properties,
                confidence=entity.confidence
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding entity {entity.id}: {e}")
            return False
    
    def add_relationship(self, relationship: Relationship) -> bool:
        """Add a relationship to the knowledge graph.
        
        Args:
            relationship: Relationship to add
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO relationships 
                    (source_id, target_id, relationship_type, properties, weight, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    relationship.source_id,
                    relationship.target_id,
                    relationship.relationship_type,
                    json.dumps(relationship.properties),
                    relationship.weight,
                    relationship.confidence,
                    relationship.created_at or datetime.now()
                ))
                conn.commit()
            
            # Add to NetworkX graph
            self.graph.add_edge(
                relationship.source_id,
                relationship.target_id,
                relationship_type=relationship.relationship_type,
                properties=relationship.properties,
                weight=relationship.weight,
                confidence=relationship.confidence
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding relationship {relationship.source_id}->{relationship.target_id}: {e}")
            return False
    
    def add_document_node(self, document_node: DocumentNode) -> bool:
        """Add a document node to the knowledge graph.
        
        Args:
            document_node: Document node to add
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO document_nodes 
                    (document_path, title, content_summary, entities, themes, metadata, created_at, modified_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    document_node.document_path,
                    document_node.title,
                    document_node.content_summary,
                    json.dumps(document_node.entities),
                    json.dumps(document_node.themes),
                    json.dumps(document_node.metadata),
                    document_node.created_at,
                    document_node.modified_at
                ))
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding document node {document_node.document_path}: {e}")
            return False
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID.
        
        Args:
            entity_id: Entity ID to retrieve
            
        Returns:
            Entity if found, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, type, name, properties, confidence, created_at
                    FROM entities WHERE id = ?
                """, (entity_id,))
                
                row = cursor.fetchone()
                if row:
                    entity_id, entity_type, name, properties_json, confidence, created_at = row
                    properties = json.loads(properties_json) if properties_json else {}
                    return Entity(
                        id=entity_id,
                        type=entity_type,
                        name=name,
                        properties=properties,
                        confidence=confidence,
                        created_at=datetime.fromisoformat(created_at) if created_at else None
                    )
                    
        except Exception as e:
            logger.error(f"Error getting entity {entity_id}: {e}")
        
        return None
    
    def find_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Find all entities of a specific type.
        
        Args:
            entity_type: Type of entities to find
            
        Returns:
            List of entities of the specified type
        """
        entities = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, type, name, properties, confidence, created_at
                    FROM entities WHERE type = ?
                """, (entity_type,))
                
                for row in cursor.fetchall():
                    entity_id, entity_type, name, properties_json, confidence, created_at = row
                    properties = json.loads(properties_json) if properties_json else {}
                    entities.append(Entity(
                        id=entity_id,
                        type=entity_type,
                        name=name,
                        properties=properties,
                        confidence=confidence,
                        created_at=datetime.fromisoformat(created_at) if created_at else None
                    ))
                    
        except Exception as e:
            logger.error(f"Error finding entities by type {entity_type}: {e}")
        
        return entities
    
    def find_related_entities(self, entity_id: str, relationship_types: Optional[List[str]] = None, max_depth: int = 2) -> List[Tuple[Entity, str, float]]:
        """Find entities related to a given entity.
        
        Args:
            entity_id: Source entity ID
            relationship_types: Optional list of relationship types to filter by
            max_depth: Maximum depth for traversal
            
        Returns:
            List of tuples (entity, relationship_path, distance)
        """
        related_entities = []
        try:
            # Use NetworkX for graph traversal
            if entity_id not in self.graph:
                return related_entities
            
            # Find all nodes within max_depth
            if max_depth == 1:
                # Direct neighbors only
                for neighbor in self.graph.neighbors(entity_id):
                    edge_data = self.graph.get_edge_data(entity_id, neighbor)
                    for edge_key, edge_attrs in edge_data.items():
                        if relationship_types is None or edge_attrs.get('relationship_type') in relationship_types:
                            entity = self.get_entity(neighbor)
                            if entity:
                                related_entities.append((entity, edge_attrs.get('relationship_type', 'unknown'), 1.0))
            else:
                # Multi-hop traversal
                paths = nx.single_source_shortest_path(self.graph, entity_id, cutoff=max_depth)
                for target_id, path in paths.items():
                    if target_id != entity_id and len(path) <= max_depth + 1:
                        entity = self.get_entity(target_id)
                        if entity:
                            distance = len(path) - 1
                            # Get relationship type from first edge in path
                            first_edge = self.graph.get_edge_data(path[0], path[1])
                            rel_type = 'connected'
                            if first_edge:
                                rel_type = list(first_edge.values())[0].get('relationship_type', 'connected')
                            
                            if relationship_types is None or rel_type in relationship_types:
                                related_entities.append((entity, rel_type, 1.0 / distance))
                                
        except Exception as e:
            logger.error(f"Error finding related entities for {entity_id}: {e}")
        
        return related_entities
    
    def find_shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Find shortest path between two entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            
        Returns:
            List of entity IDs in the shortest path, or None if no path exists
        """
        try:
            if source_id in self.graph and target_id in self.graph:
                return nx.shortest_path(self.graph, source_id, target_id)
        except nx.NetworkXNoPath:
            pass
        except Exception as e:
            logger.error(f"Error finding shortest path from {source_id} to {target_id}: {e}")
        
        return None
    
    def get_entity_centrality(self, centrality_type: str = 'betweenness') -> Dict[str, float]:
        """Calculate centrality measures for entities.
        
        Args:
            centrality_type: Type of centrality ('betweenness', 'closeness', 'degree', 'pagerank')
            
        Returns:
            Dictionary mapping entity IDs to centrality scores
        """
        try:
            if centrality_type == 'betweenness':
                return nx.betweenness_centrality(self.graph)
            elif centrality_type == 'closeness':
                return nx.closeness_centrality(self.graph)
            elif centrality_type == 'degree':
                return dict(self.graph.degree())
            elif centrality_type == 'pagerank':
                return nx.pagerank(self.graph)
            else:
                logger.warning(f"Unknown centrality type: {centrality_type}")
                return {}
                
        except Exception as e:
            logger.error(f"Error calculating {centrality_type} centrality: {e}")
            return {}
    
    def detect_communities(self) -> List[List[str]]:
        """Detect communities in the knowledge graph.
        
        Returns:
            List of communities, where each community is a list of entity IDs
        """
        try:
            # Convert to undirected graph for community detection
            undirected_graph = self.graph.to_undirected()
            
            # Use Louvain algorithm for community detection
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.louvain_communities(undirected_graph)
            
            return [list(community) for community in communities]
            
        except Exception as e:
            logger.error(f"Error detecting communities: {e}")
            return []
    
    def query_subgraph(self, entity_ids: List[str], include_connections: bool = True) -> 'KnowledgeGraph':
        """Extract a subgraph containing specified entities.
        
        Args:
            entity_ids: List of entity IDs to include
            include_connections: Whether to include connecting entities
            
        Returns:
            New KnowledgeGraph containing the subgraph
        """
        try:
            if include_connections:
                # Find all entities connected to any of the specified entities
                all_entities = set(entity_ids)
                for entity_id in entity_ids:
                    if entity_id in self.graph:
                        all_entities.update(self.graph.neighbors(entity_id))
                subgraph_nodes = list(all_entities)
            else:
                subgraph_nodes = entity_ids
            
            # Create subgraph
            subgraph = self.graph.subgraph(subgraph_nodes)
            
            # Create new KnowledgeGraph with subgraph data
            temp_db_path = str(self.db_path.parent / f"subgraph_{datetime.now().timestamp()}.db")
            sub_kg = KnowledgeGraph(temp_db_path)
            
            # Add nodes (entities)
            for node_id in subgraph.nodes():
                node_data = subgraph.nodes[node_id]
                entity = Entity(
                    id=node_id,
                    type=node_data.get('type', 'unknown'),
                    name=node_data.get('name', node_id),
                    properties=node_data.get('properties', {}),
                    confidence=node_data.get('confidence', 1.0)
                )
                sub_kg.add_entity(entity)
            
            # Add edges (relationships)
            for source, target, edge_data in subgraph.edges(data=True):
                relationship = Relationship(
                    source_id=source,
                    target_id=target,
                    relationship_type=edge_data.get('relationship_type', 'related'),
                    properties=edge_data.get('properties', {}),
                    weight=edge_data.get('weight', 1.0),
                    confidence=edge_data.get('confidence', 1.0)
                )
                sub_kg.add_relationship(relationship)
            
            return sub_kg
            
        except Exception as e:
            logger.error(f"Error creating subgraph: {e}")
            return self
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        try:
            stats = {
                'total_entities': self.graph.number_of_nodes(),
                'total_relationships': self.graph.number_of_edges(),
                'entity_types': {},
                'relationship_types': {}
            }
            
            # Add network analysis stats that work with multigraphs
            if self.graph.number_of_nodes() > 0:
                undirected = self.graph.to_undirected()
                # Convert to simple graph for some calculations
                simple_graph = nx.Graph(undirected)
                
                stats.update({
                    'is_connected': nx.is_connected(simple_graph) if simple_graph.number_of_nodes() > 0 else True,
                    'number_of_components': nx.number_connected_components(simple_graph),
                    'density': nx.density(simple_graph)
                })
                
                # Only calculate clustering if we have nodes
                if simple_graph.number_of_nodes() > 0:
                    try:
                        stats['average_clustering'] = nx.average_clustering(simple_graph)
                    except:
                        stats['average_clustering'] = 0.0
                else:
                    stats['average_clustering'] = 0.0
            else:
                stats.update({
                    'is_connected': True,
                    'number_of_components': 0,
                    'density': 0.0,
                    'average_clustering': 0.0
                })
            
            # Count entity types
            for node_id in self.graph.nodes():
                node_data = self.graph.nodes[node_id]
                entity_type = node_data.get('type', 'unknown')
                stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
            
            # Count relationship types
            for source, target, edge_data in self.graph.edges(data=True):
                rel_type = edge_data.get('relationship_type', 'unknown')
                stats['relationship_types'][rel_type] = stats['relationship_types'].get(rel_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def export_to_json(self, filepath: str) -> bool:
        """Export the knowledge graph to JSON format.
        
        Args:
            filepath: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            export_data = {
                'entities': [],
                'relationships': [],
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'total_entities': self.graph.number_of_nodes(),
                    'total_relationships': self.graph.number_of_edges()
                }
            }
            
            # Export entities
            for node_id in self.graph.nodes():
                node_data = self.graph.nodes[node_id]
                export_data['entities'].append({
                    'id': node_id,
                    'type': node_data.get('type'),
                    'name': node_data.get('name'),
                    'properties': node_data.get('properties', {}),
                    'confidence': node_data.get('confidence', 1.0)
                })
            
            # Export relationships
            for source, target, edge_data in self.graph.edges(data=True):
                export_data['relationships'].append({
                    'source_id': source,
                    'target_id': target,
                    'relationship_type': edge_data.get('relationship_type'),
                    'properties': edge_data.get('properties', {}),
                    'weight': edge_data.get('weight', 1.0),
                    'confidence': edge_data.get('confidence', 1.0)
                })
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return False
    
    def import_from_json(self, filepath: str) -> bool:
        """Import knowledge graph from JSON format.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Import entities
            for entity_data in import_data.get('entities', []):
                entity = Entity(
                    id=entity_data['id'],
                    type=entity_data['type'],
                    name=entity_data['name'],
                    properties=entity_data.get('properties', {}),
                    confidence=entity_data.get('confidence', 1.0)
                )
                self.add_entity(entity)
            
            # Import relationships
            for rel_data in import_data.get('relationships', []):
                relationship = Relationship(
                    source_id=rel_data['source_id'],
                    target_id=rel_data['target_id'],
                    relationship_type=rel_data['relationship_type'],
                    properties=rel_data.get('properties', {}),
                    weight=rel_data.get('weight', 1.0),
                    confidence=rel_data.get('confidence', 1.0)
                )
                self.add_relationship(relationship)
            
            return True
            
        except Exception as e:
            logger.error(f"Error importing from JSON: {e}")
            return False


class KnowledgeGraphBuilder:
    """Builder for constructing knowledge graphs from document collections."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """Initialize the builder.
        
        Args:
            knowledge_graph: Knowledge graph to build into
        """
        self.kg = knowledge_graph
    
    def extract_entities_from_text(self, text: str, document_path: str) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from text using improved heuristics.
        
        Args:
            text: Text to extract entities from
            document_path: Path of the source document
            
        Returns:
            Tuple of (entities, relationships) extracted from text
        """
        entities = []
        relationships = []
        
        import re
        
        # Extract email addresses as person entities
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        for email in emails:
            entity_id = f"person_{email.replace('@', '_at_').replace('.', '_')}"
            entities.append(Entity(
                id=entity_id,
                type='person',
                name=email,
                properties={'email': email, 'source_document': document_path},
                confidence=0.9
            ))
        
        # Extract person names using improved patterns
        person_patterns = [
            r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Titles
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b(?:\s+(?:said|wrote|reported|presented|worked|manages|leads))',  # Action context
            r'(?:authored by|written by|created by|presented by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Authorship
        ]
        
        for pattern in person_patterns:
            matches = re.findall(pattern, text)
            for name in matches:
                if len(name.strip()) > 3:  # Filter out short matches
                    entity_id = f"person_{name.replace(' ', '_').lower()}"
                    entities.append(Entity(
                        id=entity_id,
                        type='person',
                        name=name,
                        properties={'source_document': document_path, 'extraction_method': 'pattern'},
                        confidence=0.8
                    ))
        
        # Extract organization names with improved patterns
        org_patterns = [
            r'\b([A-Z][a-zA-Z\s&]+(?:Inc|LLC|Corp|Company|Corporation|Organization|Foundation|University|Institute)\.?)\b',
            r'\b([A-Z][a-zA-Z\s&]+(?:Ltd|Limited|Group|Association|Agency|Department)\.?)\b',
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, text)
            for org_name in matches:
                if len(org_name.strip()) > 3:
                    entity_id = f"org_{org_name.replace(' ', '_').replace('&', 'and').lower()}"
                    entities.append(Entity(
                        id=entity_id,
                        type='organization',
                        name=org_name.strip(),
                        properties={'source_document': document_path, 'extraction_method': 'pattern'},
                        confidence=0.8
                    ))
        
        # Extract locations 
        location_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})\b',  # City, State
            r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,?\s*(?:USA|United States|Canada|UK)\b',  # Cities with countries
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                location = match if isinstance(match, str) else ' '.join(filter(None, match))
                if len(location.strip()) > 2:
                    entity_id = f"location_{location.replace(' ', '_').replace(',', '').lower()}"
                    entities.append(Entity(
                        id=entity_id,
                        type='location',
                        name=location.strip(),
                        properties={'source_document': document_path, 'extraction_method': 'pattern'},
                        confidence=0.7
                    ))
        
        # Extract projects and topics with context
        topic_patterns = [
            r'\b(?:Project|Initiative|Program)\s+([A-Z][a-zA-Z\s]+)',
            r'\b([A-Z][a-zA-Z\s]+)\s+(?:project|initiative|program)\b',
            r'\bregarding\s+([a-zA-Z\s]+)',
            r'\babout\s+([a-zA-Z\s]+)',
        ]
        
        for pattern in topic_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for topic in matches:
                topic = topic.strip()
                if len(topic) > 3 and len(topic) < 50:  # Reasonable topic length
                    entity_id = f"topic_{topic.replace(' ', '_').lower()}"
                    entities.append(Entity(
                        id=entity_id,
                        type='topic',
                        name=topic.title(),
                        properties={'source_document': document_path, 'extraction_method': 'pattern'},
                        confidence=0.6
                    ))
        
        # Extract relationships
        relationships.extend(self._extract_relationships(entities, text, document_path))
        
        # Add document entity
        doc_entity_id = f"document_{Path(document_path).stem}"
        entities.append(Entity(
            id=doc_entity_id,
            type='document',
            name=Path(document_path).name,
            properties={'path': document_path, 'type': Path(document_path).suffix},
            confidence=1.0
        ))
        
        # Create MENTIONS relationships between document and all other entities
        for entity in entities[:-1]:  # Exclude the document entity itself
            relationships.append(Relationship(
                source_id=doc_entity_id,
                target_id=entity.id,
                relationship_type='MENTIONS',
                properties={'source_document': document_path},
                weight=1.0,
                confidence=0.8
            ))
        
        return entities, relationships
    
    def _extract_relationships(self, entities: List[Entity], text: str, document_path: str) -> List[Relationship]:
        """Extract relationships between entities from text patterns.
        
        Args:
            entities: List of entities found in the text
            text: Original text
            document_path: Path of the source document
            
        Returns:
            List of relationships found
        """
        relationships = []
        import re  # Import re module here
        
        # Create entity lookup by name for easier matching
        entity_by_name = {entity.name.lower(): entity for entity in entities}
        
        # Employment/affiliation relationships
        employment_patterns = [
            r'([A-Z][a-zA-Z\s]+)\s+(?:works? (?:at|for)|employed by|hired by)\s+([A-Z][a-zA-Z\s&]+)',
            r'([A-Z][a-zA-Z\s&]+)\s+(?:employee|staff|member)\s+([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-zA-Z\s]+)\s+(?:at|from)\s+([A-Z][a-zA-Z\s&]+(?:Inc|Corp|Company|Organization))',
        ]
        
        for pattern in employment_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for person_name, org_name in matches:
                person_entity = entity_by_name.get(person_name.lower().strip())
                org_entity = entity_by_name.get(org_name.lower().strip())
                
                if person_entity and org_entity:
                    relationships.append(Relationship(
                        source_id=person_entity.id,
                        target_id=org_entity.id,
                        relationship_type='WORKS_FOR',
                        properties={'source_document': document_path, 'extraction_method': 'pattern'},
                        weight=1.0,
                        confidence=0.7
                    ))
        
        # Management relationships
        management_patterns = [
            r'([A-Z][a-zA-Z\s]+)\s+(?:manages|leads|heads|supervises)\s+([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-zA-Z\s]+)\s+(?:manager|director|head)\s+of\s+([A-Z][a-zA-Z\s]+)',
        ]
        
        for pattern in management_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for manager_name, managed_entity in matches:
                manager_entity = entity_by_name.get(manager_name.lower().strip())
                managed = entity_by_name.get(managed_entity.lower().strip())
                
                if manager_entity and managed:
                    relationships.append(Relationship(
                        source_id=manager_entity.id,
                        target_id=managed.id,
                        relationship_type='MANAGES',
                        properties={'source_document': document_path, 'extraction_method': 'pattern'},
                        weight=1.0,
                        confidence=0.7
                    ))
        
        # Co-occurrence relationships (entities mentioned together)
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity1.type != entity2.type:  # Only cross-type relationships
                    # Simple proximity check: if entities appear in same sentence
                    sentences = text.split('.')
                    for sentence in sentences:
                        if entity1.name.lower() in sentence.lower() and entity2.name.lower() in sentence.lower():
                            relationships.append(Relationship(
                                source_id=entity1.id,
                                target_id=entity2.id,
                                relationship_type='CO_OCCURS',
                                properties={'source_document': document_path, 'context': sentence.strip()},
                                weight=0.5,
                                confidence=0.5
                            ))
                            break  # Only add one co-occurrence per pair
        
        return relationships
    
    def build_from_document_collection(self, vector_store) -> bool:
        """Build knowledge graph from a document collection.
        
        Args:
            vector_store: Vector store containing the documents
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # This is a simplified implementation
            # In a real system, you would iterate through all documents
            # and extract entities and relationships
            
            logger.info("Building knowledge graph from document collection...")
            
            # Add sample entities and relationships for demonstration
            sample_entities = [
                Entity(
                    id="doc_analysis_system",
                    type="concept", 
                    name="Document Analysis System",
                    properties={"description": "AI-powered document analysis"},
                    confidence=1.0
                ),
                Entity(
                    id="user_queries",
                    type="concept",
                    name="User Queries", 
                    properties={"description": "Natural language questions from users"},
                    confidence=1.0
                )
            ]
            
            for entity in sample_entities:
                self.kg.add_entity(entity)
            
            # Add sample relationship
            relationship = Relationship(
                source_id="user_queries",
                target_id="doc_analysis_system",
                relationship_type="processed_by",
                properties={"description": "Queries are processed by the system"},
                weight=1.0,
                confidence=1.0
            )
            self.kg.add_relationship(relationship)
            
            logger.info("Knowledge graph building completed")
            return True
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return False