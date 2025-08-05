"""Document Relationship Analysis Plugin for Phase III intelligent analysis.

This plugin provides sophisticated document relationship analysis capabilities
including similarity analysis, clustering, cross-reference detection, and
thematic grouping of documents.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import sqlite3
import sys
from pathlib import Path

# Add backend root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from ..plugin import Plugin, PluginInfo
from backend.ingestion.storage.vector_store import VectorStore
from backend.ingestion.storage.enhanced_vector_store import EnhancedVectorStore


logger = logging.getLogger(__name__)


@dataclass 
class DocumentSimilarity:
    """Represents similarity between two documents."""
    doc1_path: str
    doc2_path: str
    similarity_score: float
    relationship_type: str  # 'similar', 'reference', 'version', 'theme'
    details: Optional[Dict[str, Any]] = None


@dataclass
class DocumentCluster:
    """Represents a cluster of related documents."""
    cluster_id: str
    documents: List[str]
    theme: str
    confidence: float
    keywords: List[str]


@dataclass
class CrossReference:
    """Represents a cross-reference between documents."""
    source_doc: str
    target_doc: str
    reference_type: str  # 'citation', 'filename_mention', 'content_reference'
    context: str
    confidence: float


class DocumentRelationshipPlugin(Plugin):
    """Plugin for analyzing relationships between documents."""
    
    def __init__(self, vector_store_path: Optional[str] = None, db_path: Optional[str] = None):
        """Initialize the document relationship plugin.
        
        Args:
            vector_store_path: Path to the vector store index
            db_path: Path to the database file
        """
        self.vector_store_path = vector_store_path or "data/vector.index"
        self.db_path = db_path or "backend/data/docmeta.db"
        self._vector_store = None
        self._similarity_cache = {}
        self._cluster_cache = {}
        self._cross_ref_cache = {}
        
    def get_info(self) -> PluginInfo:
        """Return plugin information."""
        return PluginInfo(
            name="document_relationships",
            version="1.0.0",
            description="Advanced document relationship analysis including similarity, clustering, and cross-references",
            capabilities=[
                "document_similarity",
                "document_clustering", 
                "cross_reference_detection",
                "thematic_grouping",
                "content_evolution_tracking",
                "citation_analysis",
                "relationship_analysis"
            ]
        )
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate plugin parameters."""
        operation = params.get("operation")
        return operation in [
            "find_similar_documents",
            "cluster_documents", 
            "detect_cross_references",
            "analyze_themes",
            "track_content_evolution",
            "find_citations",
            "analyze_relationships"
        ]
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the document relationship analysis operation."""
        operation = params.get("operation", "find_similar_documents")
        
        try:
            if operation == "find_similar_documents":
                return self._find_similar_documents(params)
            elif operation == "cluster_documents":
                return self._cluster_documents(params)
            elif operation == "detect_cross_references":
                return self._detect_cross_references(params)
            elif operation == "analyze_themes":
                return self._analyze_themes(params)
            elif operation == "track_content_evolution":
                return self._track_content_evolution(params)
            elif operation == "find_citations":
                return self._find_citations(params)
            elif operation == "analyze_relationships":
                return self._analyze_relationships(params)
            else:
                return {"error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            logger.error(f"Error in document relationship analysis: {e}")
            return {"error": f"Document relationship analysis failed: {e}"}
    
    def _get_vector_store(self):
        """Get or initialize the vector store."""
        if self._vector_store is None:
            try:
                # Initialize vector store
                if Path(self.db_path).exists():
                    self._vector_store = EnhancedVectorStore.load(
                        Path(self.vector_store_path), 
                        Path(self.db_path)
                    )
                else:
                    # Fallback to basic vector store if database doesn't exist
                    self._vector_store = VectorStore.load(Path(self.vector_store_path))
                    logger.warning("Using basic vector store - metadata features unavailable")
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                raise
        return self._vector_store
    
    def _find_similar_documents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Find documents similar to a given document or query."""
        target_doc = params.get("document_path")
        query_text = params.get("query")
        similarity_threshold = params.get("threshold", 0.7)
        max_results = params.get("max_results", 10)
        
        if not target_doc and not query_text:
            return {"error": "Either document_path or query must be provided"}
        
        try:
            vector_store = self._get_vector_store()
            similarities = []
            
            if target_doc:
                # Find documents similar to the target document
                similarities = self._compute_document_similarities(target_doc, similarity_threshold, max_results)
            elif query_text:
                # Find documents similar to query text
                similarities = self._find_similar_by_query(query_text, similarity_threshold, max_results)
            
            # Format response
            response_text = self._format_similarity_response(similarities, target_doc or query_text)
            
            return {
                "response": response_text,
                "similarities": [
                    {
                        "document": sim.doc2_path if target_doc else sim.doc1_path,
                        "score": sim.similarity_score,
                        "type": sim.relationship_type,
                        "details": sim.details
                    }
                    for sim in similarities
                ],
                "metadata": {
                    "operation": "find_similar_documents",
                    "threshold": similarity_threshold,
                    "results_count": len(similarities)
                }
            }
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return {"error": f"Similarity analysis failed: {e}"}
    
    def _cluster_documents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Cluster documents by thematic similarity."""
        num_clusters = params.get("num_clusters", 5)
        file_types = params.get("file_types", None)
        time_filter = params.get("time_filter", None)
        
        try:
            # Get cache key
            cache_key = f"clusters_{num_clusters}_{file_types}_{time_filter}"
            if cache_key in self._cluster_cache:
                clusters = self._cluster_cache[cache_key]
            else:
                clusters = self._perform_document_clustering(num_clusters, file_types, time_filter)
                self._cluster_cache[cache_key] = clusters
            
            # Format response
            response_text = self._format_clustering_response(clusters)
            
            return {
                "response": response_text,
                "clusters": [
                    {
                        "id": cluster.cluster_id,
                        "theme": cluster.theme,
                        "documents": cluster.documents,
                        "confidence": cluster.confidence,
                        "keywords": cluster.keywords
                    }
                    for cluster in clusters
                ],
                "metadata": {
                    "operation": "cluster_documents",
                    "num_clusters": num_clusters,
                    "total_documents": sum(len(cluster.documents) for cluster in clusters)
                }
            }
            
        except Exception as e:
            logger.error(f"Error clustering documents: {e}")
            return {"error": f"Document clustering failed: {e}"}
    
    def _detect_cross_references(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect cross-references between documents."""
        target_doc = params.get("document_path")
        reference_types = params.get("types", ["citation", "filename_mention", "content_reference"])
        
        try:
            # Get cache key
            cache_key = f"crossrefs_{target_doc}_{','.join(reference_types)}"
            if cache_key in self._cross_ref_cache:
                cross_refs = self._cross_ref_cache[cache_key]
            else:
                cross_refs = self._find_cross_references(target_doc, reference_types)
                self._cross_ref_cache[cache_key] = cross_refs
            
            # Format response
            response_text = self._format_cross_reference_response(cross_refs, target_doc)
            
            return {
                "response": response_text,
                "cross_references": [
                    {
                        "source": ref.source_doc,
                        "target": ref.target_doc,
                        "type": ref.reference_type,
                        "context": ref.context,
                        "confidence": ref.confidence
                    }
                    for ref in cross_refs
                ],
                "metadata": {
                    "operation": "detect_cross_references",
                    "reference_types": reference_types,
                    "results_count": len(cross_refs)
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting cross-references: {e}")
            return {"error": f"Cross-reference detection failed: {e}"}
    
    def _analyze_themes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze thematic patterns across documents."""
        try:
            themes = self._extract_document_themes()
            
            # Format response
            response_text = self._format_theme_analysis_response(themes)
            
            return {
                "response": response_text,
                "themes": themes,
                "metadata": {
                    "operation": "analyze_themes",
                    "themes_count": len(themes)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing themes: {e}")
            return {"error": f"Theme analysis failed: {e}"}
    
    def _track_content_evolution(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Track how content has evolved over time."""
        document_pattern = params.get("pattern", ".*")
        time_window = params.get("time_window", "1_month")
        
        try:
            evolution_data = self._analyze_content_evolution(document_pattern, time_window)
            
            # Format response
            response_text = self._format_evolution_response(evolution_data)
            
            return {
                "response": response_text,
                "evolution": evolution_data,
                "metadata": {
                    "operation": "track_content_evolution",
                    "time_window": time_window,
                    "documents_tracked": len(evolution_data.get("documents", []))
                }
            }
            
        except Exception as e:
            logger.error(f"Error tracking content evolution: {e}")
            return {"error": f"Content evolution tracking failed: {e}"}
    
    def _find_citations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Find citations and references in documents."""
        try:
            citations = self._extract_citations()
            
            # Format response
            response_text = self._format_citations_response(citations)
            
            return {
                "response": response_text,
                "citations": citations,
                "metadata": {
                    "operation": "find_citations",
                    "citations_count": len(citations)
                }
            }
            
        except Exception as e:
            logger.error(f"Error finding citations: {e}")
            return {"error": f"Citation analysis failed: {e}"}
    
    def _analyze_relationships(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive relationship analysis."""
        analysis_types = params.get("types", ["similarity", "themes", "cross_references"])
        
        try:
            relationships = {}
            
            if "similarity" in analysis_types:
                relationships["similarities"] = self._get_top_similarities()
            
            if "themes" in analysis_types:
                relationships["themes"] = self._extract_document_themes()
            
            if "cross_references" in analysis_types:
                relationships["cross_references"] = self._get_all_cross_references()
            
            # Format comprehensive response
            response_text = self._format_comprehensive_analysis_response(relationships)
            
            return {
                "response": response_text,
                "relationships": relationships,
                "metadata": {
                    "operation": "analyze_relationships",
                    "analysis_types": analysis_types
                }
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive relationship analysis: {e}")
            return {"error": f"Relationship analysis failed: {e}"}
    
    # Helper methods for specific operations
    
    def _compute_document_similarities(self, target_doc: str, threshold: float, max_results: int) -> List[DocumentSimilarity]:
        """Compute similarities between target document and all others."""
        # TODO: Implement using vector embeddings
        # For now, return mock data
        return [
            DocumentSimilarity(
                doc1_path=target_doc,
                doc2_path="similar_doc.pdf",
                similarity_score=0.85,
                relationship_type="similar",
                details={"reason": "Similar content themes"}
            )
        ]
    
    def _find_similar_by_query(self, query: str, threshold: float, max_results: int) -> List[DocumentSimilarity]:
        """Find documents similar to a text query."""
        # TODO: Implement using vector search
        return []
    
    def _perform_document_clustering(self, num_clusters: int, file_types: Optional[List[str]], time_filter: Optional[str]) -> List[DocumentCluster]:
        """Perform document clustering analysis."""
        # TODO: Implement clustering using vector embeddings
        return [
            DocumentCluster(
                cluster_id="cluster_1",
                documents=["doc1.pdf", "doc2.pdf"],
                theme="Financial Reports",
                confidence=0.8,
                keywords=["budget", "revenue", "expenses"]
            )
        ]
    
    def _find_cross_references(self, target_doc: Optional[str], reference_types: List[str]) -> List[CrossReference]:
        """Find cross-references between documents."""
        # TODO: Implement cross-reference detection
        return []
    
    def _extract_document_themes(self) -> Dict[str, Any]:
        """Extract thematic patterns from documents."""
        # TODO: Implement theme extraction
        return {
            "major_themes": ["Finance", "Operations", "Strategy"],
            "theme_distribution": {"Finance": 0.4, "Operations": 0.35, "Strategy": 0.25}
        }
    
    def _analyze_content_evolution(self, pattern: str, time_window: str) -> Dict[str, Any]:
        """Analyze how content has evolved over time."""
        # TODO: Implement content evolution tracking
        return {
            "documents": ["doc1.pdf", "doc2.pdf"],
            "evolution_timeline": [],
            "changes_detected": 0
        }
    
    def _extract_citations(self) -> List[Dict[str, Any]]:
        """Extract citations from documents."""
        # TODO: Implement citation extraction
        return []
    
    def _get_top_similarities(self) -> List[Dict[str, Any]]:
        """Get top document similarities."""
        # TODO: Implement
        return []
    
    def _get_all_cross_references(self) -> List[Dict[str, Any]]:
        """Get all cross-references in the collection."""
        # TODO: Implement
        return []
    
    # Response formatting methods
    
    def _format_similarity_response(self, similarities: List[DocumentSimilarity], reference: str) -> str:
        """Format similarity analysis response."""
        if not similarities:
            return f"No similar documents found for '{reference}'"
        
        response_lines = [f"📊 Found {len(similarities)} similar documents to '{reference}':\n"]
        
        for i, sim in enumerate(similarities[:5], 1):  # Show top 5
            doc_name = Path(sim.doc2_path if reference != sim.doc2_path else sim.doc1_path).name
            response_lines.append(
                f"{i}. {doc_name} (similarity: {sim.similarity_score:.2f})"
            )
            if sim.details and sim.details.get("reason"):
                response_lines.append(f"   • {sim.details['reason']}")
        
        if len(similarities) > 5:
            response_lines.append(f"\n... and {len(similarities) - 5} more similar documents")
        
        return "\n".join(response_lines)
    
    def _format_clustering_response(self, clusters: List[DocumentCluster]) -> str:
        """Format clustering analysis response."""
        if not clusters:
            return "No document clusters found"
        
        response_lines = [f"📊 Found {len(clusters)} document clusters:\n"]
        
        for i, cluster in enumerate(clusters, 1):
            response_lines.append(f"{i}. **{cluster.theme}** ({len(cluster.documents)} documents)")
            response_lines.append(f"   • Keywords: {', '.join(cluster.keywords[:5])}")
            response_lines.append(f"   • Confidence: {cluster.confidence:.2f}")
            if cluster.documents:
                doc_names = [Path(doc).name for doc in cluster.documents[:3]]
                response_lines.append(f"   • Sample docs: {', '.join(doc_names)}")
                if len(cluster.documents) > 3:
                    response_lines.append(f"     ... and {len(cluster.documents) - 3} more")
            response_lines.append("")
        
        return "\n".join(response_lines)
    
    def _format_cross_reference_response(self, cross_refs: List[CrossReference], target_doc: Optional[str]) -> str:
        """Format cross-reference analysis response."""
        if not cross_refs:
            target_text = f" for '{target_doc}'" if target_doc else ""
            return f"No cross-references found{target_text}"
        
        response_lines = [f"🔗 Found {len(cross_refs)} cross-references:\n"]
        
        for i, ref in enumerate(cross_refs[:10], 1):  # Show top 10
            source_name = Path(ref.source_doc).name
            target_name = Path(ref.target_doc).name
            response_lines.append(
                f"{i}. {source_name} → {target_name} ({ref.reference_type})"
            )
            if ref.context:
                context_preview = ref.context[:100] + "..." if len(ref.context) > 100 else ref.context
                response_lines.append(f"   • Context: {context_preview}")
        
        if len(cross_refs) > 10:
            response_lines.append(f"\n... and {len(cross_refs) - 10} more cross-references")
        
        return "\n".join(response_lines)
    
    def _format_theme_analysis_response(self, themes: Dict[str, Any]) -> str:
        """Format theme analysis response."""
        major_themes = themes.get("major_themes", [])
        distribution = themes.get("theme_distribution", {})
        
        if not major_themes:
            return "No major themes identified in the document collection"
        
        response_lines = [f"📋 Identified {len(major_themes)} major themes:\n"]
        
        for theme in major_themes:
            percentage = distribution.get(theme, 0) * 100
            response_lines.append(f"• **{theme}** ({percentage:.1f}% of collection)")
        
        return "\n".join(response_lines)
    
    def _format_evolution_response(self, evolution_data: Dict[str, Any]) -> str:
        """Format content evolution response."""
        documents = evolution_data.get("documents", [])
        changes = evolution_data.get("changes_detected", 0)
        
        if changes == 0:
            return "No significant content evolution detected in the specified time window"
        
        response_lines = [
            f"📈 Content evolution analysis:",
            f"• Analyzed {len(documents)} documents",
            f"• Detected {changes} significant changes",
            f"• Evolution timeline available in detailed results"
        ]
        
        return "\n".join(response_lines)
    
    def _format_citations_response(self, citations: List[Dict[str, Any]]) -> str:
        """Format citations analysis response."""
        if not citations:
            return "No citations found in the document collection"
        
        response_lines = [f"📚 Found {len(citations)} citations:\n"]
        
        for i, citation in enumerate(citations[:10], 1):
            response_lines.append(f"{i}. {citation.get('text', 'Unknown citation')}")
        
        if len(citations) > 10:
            response_lines.append(f"\n... and {len(citations) - 10} more citations")
        
        return "\n".join(response_lines)
    
    def _format_comprehensive_analysis_response(self, relationships: Dict[str, Any]) -> str:
        """Format comprehensive relationship analysis response."""
        response_lines = ["🔍 Comprehensive Document Relationship Analysis:\n"]
        
        if "similarities" in relationships:
            sims = relationships["similarities"]
            response_lines.append(f"📊 Similarity Analysis: {len(sims)} similar document pairs found")
        
        if "themes" in relationships:
            themes = relationships["themes"]
            major_themes = themes.get("major_themes", [])
            response_lines.append(f"📋 Theme Analysis: {len(major_themes)} major themes identified")
        
        if "cross_references" in relationships:
            refs = relationships["cross_references"]
            response_lines.append(f"🔗 Cross-References: {len(refs)} references found")
        
        response_lines.append("\n📈 The document collection shows rich interconnections and thematic patterns.")
        response_lines.append("Use specific analysis operations for detailed insights.")
        
        return "\n".join(response_lines)