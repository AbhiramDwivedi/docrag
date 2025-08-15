"""LLM-based intent analysis for intelligent query processing."""

import logging
import re
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass

# Import OpenAI for LLM-based intent classification
try:
    from openai import OpenAI
    from shared.config import settings
except ImportError:
    # Fallback to src imports
    try:
        from src.shared.config import settings
        from openai import OpenAI
    except ImportError:
        OpenAI = None
        settings = None

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of query intents."""
    DOCUMENT_DISCOVERY = "document_discovery"   # Find specific documents
    CONTENT_ANALYSIS = "content_analysis"       # Analyze document content
    COMPARISON = "comparison"                    # Compare documents
    INVESTIGATION = "investigation"              # Multi-step investigation
    METADATA_QUERY = "metadata_query"           # Get file information and statistics
    RELATIONSHIP_ANALYSIS = "relationship_analysis"  # Analyze relationships between entities


class QueryComplexity(Enum):
    """Complexity levels for queries."""
    SIMPLE = "simple"
    MODERATE = "moderate" 
    COMPLEX = "complex"


@dataclass
class IntentAnalysisResult:
    """Result of intent analysis."""
    primary_intent: QueryIntent
    secondary_intents: List[QueryIntent]
    confidence: float
    complexity: QueryComplexity
    reasoning: str
    key_entities: List[str]
    action_verbs: List[str]
    scope_indicators: List[str]


class IntentAnalyzer:
    """Analyzes user queries to determine intent using LLM."""
    
    def __init__(self):
        """Initialize the intent analyzer."""
        self.intent_logger = logging.getLogger('agent.classification')
        self._openai_client = None
        
    def analyze_intent(self, query: str) -> IntentAnalysisResult:
        """Analyze query intent using LLM-based classification.
        
        Args:
            query: Natural language query to analyze
            
        Returns:
            IntentAnalysisResult with classification details
        """
        self.intent_logger.debug(f"Analyzing intent for query: {query}")
        
        try:
            # Use LLM for intelligent intent classification
            intent_result = self._classify_with_llm(query)
            return intent_result
            
        except Exception as e:
            # Fallback to simple rule-based classification if LLM fails
            self.intent_logger.warning(f"LLM intent classification failed: {e}, using fallback")
            return self._fallback_classification(query)
    
    def _classify_with_llm(self, query: str) -> IntentAnalysisResult:
        """Use LLM to classify query intent intelligently."""
        if not OpenAI or not settings or not settings.openai_api_key:
            raise Exception("OpenAI not configured")
        
        # Initialize OpenAI client if needed
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        
        # Create prompt for intent classification
        prompt = f"""Analyze this document query and classify its intent:

Query: "{query}"

Available intent types:
1. DOCUMENT_DISCOVERY - Finding specific documents by name, type, or content
2. CONTENT_ANALYSIS - Analyzing content within documents (summaries, key points, etc.)
3. COMPARISON - Comparing content across multiple documents  
4. INVESTIGATION - Complex multi-step research requiring analysis
5. METADATA_QUERY - Getting file information, statistics, counts, lists, recent files
6. RELATIONSHIP_ANALYSIS - Finding relationships between entities or documents

Guidelines:
- METADATA_QUERY: queries about file counts, lists, types, recent files, statistics, "how many", "show me files", "list documents"
- DOCUMENT_DISCOVERY: finding specific documents or content
- CONTENT_ANALYSIS: analyzing what's inside documents
- COMPARISON: comparing documents or content
- INVESTIGATION: complex research questions
- RELATIONSHIP_ANALYSIS: entity relationships

Examples:
- "how many files" -> METADATA_QUERY
- "list all documents" -> METADATA_QUERY  
- "show me recent files" -> METADATA_QUERY
- "what file types do we have" -> METADATA_QUERY
- "find budget document" -> DOCUMENT_DISCOVERY
- "summarize the report" -> CONTENT_ANALYSIS
- "what does AWS stand for" -> CONTENT_ANALYSIS
- "what is GCP" -> CONTENT_ANALYSIS
- "define machine learning" -> CONTENT_ANALYSIS
- "explain the concept of" -> CONTENT_ANALYSIS

Respond with JSON:
{{
    "primary_intent": "METADATA_QUERY|DOCUMENT_DISCOVERY|CONTENT_ANALYSIS|COMPARISON|INVESTIGATION|RELATIONSHIP_ANALYSIS",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "complexity": "simple|moderate|complex"
}}"""

        # Get LLM classification
        resp = self._openai_client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=200,
            temperature=0.1  # Low temperature for consistent classification
        )
        
        # Parse LLM response
        response_text = resp.choices[0].message.content.strip()
        
        # Extract JSON from response
        import json
        try:
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]
            
            result = json.loads(response_text)
            
            # Map string to enum
            intent_map = {
                "DOCUMENT_DISCOVERY": QueryIntent.DOCUMENT_DISCOVERY,
                "CONTENT_ANALYSIS": QueryIntent.CONTENT_ANALYSIS,
                "COMPARISON": QueryIntent.COMPARISON,
                "INVESTIGATION": QueryIntent.INVESTIGATION,
                "METADATA_QUERY": QueryIntent.METADATA_QUERY,
                "RELATIONSHIP_ANALYSIS": QueryIntent.RELATIONSHIP_ANALYSIS
            }
            
            complexity_map = {
                "simple": QueryComplexity.SIMPLE,
                "moderate": QueryComplexity.MODERATE,
                "complex": QueryComplexity.COMPLEX
            }
            
            primary_intent = intent_map.get(result.get("primary_intent"), QueryIntent.DOCUMENT_DISCOVERY)
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "LLM classification")
            complexity = complexity_map.get(result.get("complexity", "simple"), QueryComplexity.SIMPLE)
            
            # Extract additional features
            key_entities = self._extract_entities(query)
            action_verbs = self._extract_action_verbs(query.lower())
            scope_indicators = self._extract_scope_indicators(query.lower())
            
            return IntentAnalysisResult(
                primary_intent=primary_intent,
                secondary_intents=[],
                confidence=confidence,
                complexity=complexity,
                reasoning=reasoning,
                key_entities=key_entities,
                action_verbs=action_verbs,
                scope_indicators=scope_indicators
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.intent_logger.warning(f"Failed to parse LLM response: {e}")
            raise Exception(f"LLM response parsing failed: {e}")
    
    def _fallback_classification(self, query: str) -> IntentAnalysisResult:
        """Fallback rule-based classification when LLM is unavailable."""
        query_lower = query.lower()
        
        # Simple rule-based fallback prioritizing metadata queries
        if any(term in query_lower for term in ["how many", "count", "list", "recent", "show me", "file types", "statistics", "files", "documents"]):
            intent = QueryIntent.METADATA_QUERY
            confidence = 0.8
        elif any(term in query_lower for term in ["find", "locate", "get", "where is"]):
            intent = QueryIntent.DOCUMENT_DISCOVERY  
            confidence = 0.6
        elif any(term in query_lower for term in ["what does", "what is", "define", "explain", "stands for", "mean", "meaning", "definition", "analyze", "summarize", "content"]):
            intent = QueryIntent.CONTENT_ANALYSIS
            confidence = 0.7
        elif any(term in query_lower for term in ["compare", "difference", "similar"]):
            intent = QueryIntent.COMPARISON
            confidence = 0.6
        elif any(term in query_lower for term in ["relationship", "connected", "related"]):
            intent = QueryIntent.RELATIONSHIP_ANALYSIS
            confidence = 0.6
        else:
            intent = QueryIntent.DOCUMENT_DISCOVERY
            confidence = 0.4
        
        return IntentAnalysisResult(
            primary_intent=intent,
            secondary_intents=[],
            confidence=confidence,
            complexity=QueryComplexity.SIMPLE,
            reasoning="Fallback rule-based classification",
            key_entities=self._extract_entities(query),
            action_verbs=self._extract_action_verbs(query_lower),
            scope_indicators=self._extract_scope_indicators(query_lower)
        )
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from the query."""
        # Simple entity extraction - can be enhanced with NER later
        entities = []
        
        # Look for capitalized words (potential proper nouns)
        words = re.findall(r'\b[A-Z][a-zA-Z]*\b', query)
        entities.extend(words)
        
        # Look for email patterns
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', query)
        entities.extend(emails)
        
        # Look for file extensions
        extensions = re.findall(r'\.[a-zA-Z]{2,4}\b', query)
        entities.extend(extensions)
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_action_verbs(self, query_lower: str) -> List[str]:
        """Extract action verbs that indicate user intent."""
        action_verbs = [
            "find", "get", "show", "list", "analyze", "compare", "explain", "describe",
            "summarize", "track", "investigate", "identify", "locate", "extract",
            "count", "search", "explore", "examine", "review"
        ]
        
        found_verbs = []
        for verb in action_verbs:
            if re.search(r'\b' + re.escape(verb) + r'\b', query_lower):
                found_verbs.append(verb)
        
        return found_verbs
    
    def _extract_scope_indicators(self, query_lower: str) -> List[str]:
        """Extract indicators of query scope (single doc, multiple docs, etc.)."""
        scope_indicators = []
        
        single_doc_patterns = ["the document", "this file", "the file", "specific document"]
        multi_doc_patterns = ["all documents", "across files", "multiple documents", "various files"]
        temporal_patterns = ["recent", "latest", "newest", "oldest", "last week", "yesterday"]
        
        for pattern in single_doc_patterns:
            if pattern in query_lower:
                scope_indicators.append("single_document")
                break
        
        for pattern in multi_doc_patterns:
            if pattern in query_lower:
                scope_indicators.append("multiple_documents")
                break
        
        for pattern in temporal_patterns:
            if pattern in query_lower:
                scope_indicators.append("temporal")
                break
        
        return scope_indicators
