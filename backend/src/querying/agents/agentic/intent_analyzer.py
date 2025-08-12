"""Intent analysis module for understanding user goals beyond keywords."""

import re
import logging
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of user intents that can be identified."""
    DOCUMENT_DISCOVERY = "document_discovery"  # Find specific documents
    CONTENT_ANALYSIS = "content_analysis"      # Analyze content within documents
    COMPARISON = "comparison"                  # Compare across multiple documents
    INVESTIGATION = "investigation"            # Research and gather evidence
    METADATA_QUERY = "metadata_query"         # Get file information and statistics
    RELATIONSHIP_ANALYSIS = "relationship_analysis"  # Find connections and patterns


class QueryComplexity(Enum):
    """Complexity levels for query processing."""
    SIMPLE = "simple"          # Single-step queries
    MODERATE = "moderate"      # 2-3 step queries
    COMPLEX = "complex"        # Multi-step analytical queries


@dataclass
class IntentAnalysisResult:
    """Result of intent analysis including confidence and reasoning."""
    primary_intent: QueryIntent
    secondary_intents: List[QueryIntent]
    complexity: QueryComplexity
    confidence: float
    reasoning: str
    key_entities: List[str]
    action_verbs: List[str]
    scope_indicators: List[str]


class IntentAnalyzer:
    """Analyzes user queries to understand intent beyond keyword matching."""
    
    def __init__(self):
        """Initialize the intent analyzer with pattern definitions."""
        self.patterns = self._initialize_patterns()
        self.complexity_indicators = self._initialize_complexity_indicators()
    
    def analyze_intent(self, query: str) -> IntentAnalysisResult:
        """Analyze a query to determine user intent and complexity.
        
        Args:
            query: Natural language query to analyze
            
        Returns:
            IntentAnalysisResult with primary intent, complexity, and reasoning
        """
        query_lower = query.lower()
        
        # Extract linguistic features
        entities = self._extract_entities(query)
        action_verbs = self._extract_action_verbs(query_lower)
        scope_indicators = self._extract_scope_indicators(query_lower)
        
        # Analyze intent patterns
        intent_scores = self._score_intents(query_lower)
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        
        # Identify secondary intents
        secondary_intents = [
            intent for intent, score in intent_scores.items()
            if score > 0.3 and intent != primary_intent
        ]
        
        # Determine complexity
        complexity = self._determine_complexity(query_lower, action_verbs, scope_indicators)
        
        # Calculate confidence based on pattern strength
        confidence = self._calculate_confidence(intent_scores, primary_intent)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(query, primary_intent, complexity, intent_scores)
        
        return IntentAnalysisResult(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            complexity=complexity,
            confidence=confidence,
            reasoning=reasoning,
            key_entities=entities,
            action_verbs=action_verbs,
            scope_indicators=scope_indicators
        )
    
    def _initialize_patterns(self) -> Dict[QueryIntent, List[Dict[str, any]]]:
        """Initialize intent recognition patterns."""
        return {
            QueryIntent.DOCUMENT_DISCOVERY: [
                {
                    "patterns": [
                        r"\b(find|get|show|locate|where is)\s+(the\s+)?(document|file)",
                        r"\b(document|file)\s+(called|named|titled|about)",
                        r"\bfull\s+path\s+(for|of|to)",
                        r"\bfind\s+.*\s+(document|file|doc)",
                        r"\bget\s+.*\s+(document|file)",
                        r"\b(find|get|show|locate)\s+(the\s+)?[a-zA-Z]+\s+(report|file|document)",
                        r"\b(find|get|show)\s+.*\s+(report|presentation|spreadsheet|email)",
                        r"\bwhere is\s+.*",
                        r"\bshow me\s+.*",
                        r"\bfind\s+(?:the\s+)?[a-zA-Z\s]+"
                    ],
                    "keywords": ["find", "locate", "get", "show", "where", "path", "document", "file", "report"],
                    "weight": 1.0
                }
            ],
            
            QueryIntent.CONTENT_ANALYSIS: [
                {
                    "patterns": [
                        r"\bwhat\s+(does|do|is)\s+.*\s+(say|contain|mention)",
                        r"\b(analyze|explain|describe|summarize)",
                        r"\b(content|information|details)\s+(about|of|in)",
                        r"\b(according\s+to|as\s+mentioned\s+in|from\s+the)",
                        r"\bthe\s+document\s+says"
                    ],
                    "keywords": ["analyze", "explain", "describe", "content", "information", "says", "contains"],
                    "weight": 1.0
                }
            ],
            
            QueryIntent.COMPARISON: [
                {
                    "patterns": [
                        r"\b(compare|contrast|difference|similar)",
                        r"\bacross\s+(documents|files|all)",
                        r"\b(between|versus|vs\.?)\s+",
                        r"\ball\s+.*\s+(say|mention|contain)",
                        r"\bmultiple\s+(documents|sources)"
                    ],
                    "keywords": ["compare", "contrast", "across", "between", "multiple", "all", "different"],
                    "weight": 1.0
                }
            ],
            
            QueryIntent.INVESTIGATION: [
                {
                    "patterns": [
                        r"\bwhich\s+(documents|files)\s+(mention|contain|reference)",
                        r"\btrack\s+(evolution|changes|development)",
                        r"\b(investigate|research|explore)",
                        r"\bevidence\s+(of|for|about)",
                        r"\bidentify\s+(patterns|trends|stakeholders)"
                    ],
                    "keywords": ["which", "track", "investigate", "evidence", "identify", "patterns", "trends"],
                    "weight": 1.0
                }
            ],
            
            QueryIntent.METADATA_QUERY: [
                {
                    "patterns": [
                        r"\b(how\s+many|count|number\s+of|total)",
                        r"\b(list|show)\s+(all\s+)?(files|documents)",
                        r"\b(recent|latest|newest|oldest)",
                        r"\b(size|type|modified|created)",
                        r"\b(email|pdf|docx|xlsx|file\s+types)"
                    ],
                    "keywords": ["how many", "count", "list", "recent", "size", "type", "files"],
                    "weight": 1.0
                }
            ],
            
            QueryIntent.RELATIONSHIP_ANALYSIS: [
                {
                    "patterns": [
                        r"\b(relationship|connection|linked|related)",
                        r"\b(who\s+(works|is|manages)|people|organization)",
                        r"\b(network|graph|cluster|group)",
                        r"\b(similar|related)\s+(documents|content)",
                        r"\bcross-reference"
                    ],
                    "keywords": ["relationship", "connection", "who", "people", "network", "similar", "related"],
                    "weight": 1.0
                }
            ]
        }
    
    def _initialize_complexity_indicators(self) -> Dict[str, List[str]]:
        """Initialize complexity assessment indicators."""
        return {
            "simple": [
                "what is", "where is", "show me", "list", "count", "find the"
            ],
            "moderate": [
                "analyze", "compare", "explain", "describe", "summarize", "which documents"
            ],
            "complex": [
                "comprehensive", "detailed analysis", "across all", "relationships between",
                "evolution of", "track changes", "investigate", "identify patterns"
            ]
        }
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entities (proper nouns, quoted terms, etc.)."""
        entities = []
        
        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted_terms)
        
        # Extract potential proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        entities.extend(proper_nouns)
        
        # Extract file extensions and technical terms
        technical_terms = re.findall(r'\b(?:pdf|docx|xlsx|pptx|msg|txt|email|document|file)\b', query.lower())
        entities.extend(technical_terms)
        
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
                scope_indicators.append("temporal_filter")
                break
        
        return scope_indicators
    
    def _score_intents(self, query_lower: str) -> Dict[QueryIntent, float]:
        """Score each intent based on pattern matching."""
        scores = {}
        
        for intent, pattern_groups in self.patterns.items():
            total_score = 0.0
            
            for group in pattern_groups:
                group_score = 0.0
                
                # Score based on regex patterns
                for pattern in group["patterns"]:
                    if re.search(pattern, query_lower):
                        group_score += 0.4
                
                # Score based on keywords
                for keyword in group["keywords"]:
                    if keyword in query_lower:
                        group_score += 0.1
                
                # Apply weight
                total_score += group_score * group["weight"]
            
            scores[intent] = min(total_score, 1.0)  # Cap at 1.0
        
        return scores
    
    def _determine_complexity(self, query_lower: str, action_verbs: List[str], 
                            scope_indicators: List[str]) -> QueryComplexity:
        """Determine query complexity based on multiple factors."""
        complexity_score = 0
        
        # Check for complex indicators
        for indicator in self.complexity_indicators["complex"]:
            if indicator in query_lower:
                complexity_score += 3
        
        # Check for moderate indicators
        for indicator in self.complexity_indicators["moderate"]:
            if indicator in query_lower:
                complexity_score += 2
        
        # Multiple action verbs suggest complexity
        if len(action_verbs) > 2:
            complexity_score += 2
        
        # Multiple scope indicators suggest complexity
        if len(scope_indicators) > 1:
            complexity_score += 1
        
        # Query length as a factor
        word_count = len(query_lower.split())
        if word_count > 15:
            complexity_score += 2
        elif word_count > 8:
            complexity_score += 1
        
        # Determine complexity level
        if complexity_score >= 5:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 2:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _calculate_confidence(self, intent_scores: Dict[QueryIntent, float], 
                            primary_intent: QueryIntent) -> float:
        """Calculate confidence in the intent analysis."""
        primary_score = intent_scores[primary_intent]
        
        # If primary score is very high, confidence is high
        if primary_score > 0.8:
            return 0.9
        elif primary_score > 0.5:
            return 0.7
        elif primary_score > 0.3:
            return 0.5
        else:
            return 0.3
    
    def _generate_reasoning(self, query: str, primary_intent: QueryIntent, 
                          complexity: QueryComplexity, 
                          intent_scores: Dict[QueryIntent, float]) -> str:
        """Generate human-readable reasoning for the intent analysis."""
        reasoning_parts = []
        
        # Primary intent reasoning
        reasoning_parts.append(f"Primary intent: {primary_intent.value} (score: {intent_scores[primary_intent]:.2f})")
        
        # Complexity reasoning
        reasoning_parts.append(f"Complexity: {complexity.value}")
        
        # Additional context
        if intent_scores[primary_intent] > 0.7:
            reasoning_parts.append("High confidence based on strong pattern matches")
        elif intent_scores[primary_intent] > 0.4:
            reasoning_parts.append("Moderate confidence based on keyword patterns")
        else:
            reasoning_parts.append("Low confidence - may need clarification")
        
        return ". ".join(reasoning_parts)