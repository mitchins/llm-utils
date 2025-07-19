"""
Embedding support for ruNER co-referencing capabilities.

This module provides embedding extraction for entities to enable
future co-referencing functionality.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union
import numpy as np
from dataclasses import dataclass

from .core import Entity


class EmbeddingExtractor(ABC):
    """Abstract base for embedding extraction strategies."""
    
    @abstractmethod
    def extract_embeddings(self, entities: List[Entity], text: str) -> List[Entity]:
        """Extract embeddings for entities and return updated entities."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced."""
        pass


class DummyEmbeddingExtractor(EmbeddingExtractor):
    """Dummy embedder for testing and development."""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
    
    def extract_embeddings(self, entities: List[Entity], text: str) -> List[Entity]:
        """Generate random embeddings for testing."""
        updated_entities = []
        
        for entity in entities:
            # Create a simple hash-based embedding for consistency
            entity_hash = hash(entity.text.lower()) % 1000000
            np.random.seed(entity_hash)  # Deterministic for same text
            
            embedding = np.random.normal(0, 1, self.dimension).tolist()
            
            # Create new entity with embedding
            updated_entity = Entity(
                text=entity.text,
                label=entity.label,
                start=entity.start,
                end=entity.end,
                confidence=entity.confidence,
                normalized_text=entity.normalized_text,
                embedding=embedding
            )
            updated_entities.append(updated_entity)
        
        return updated_entities
    
    def get_embedding_dimension(self) -> int:
        return self.dimension


class TransformerEmbeddingExtractor(EmbeddingExtractor):
    """
    Transformer-based embedding extractor using sentence-transformers.
    
    This provides real embeddings suitable for co-referencing tasks.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model.
        
        Args:
            model_name: HuggingFace model name for sentence-transformers
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for TransformerEmbeddingExtractor. "
                "Install with: pip install sentence-transformers"
            )
    
    def extract_embeddings(self, entities: List[Entity], text: str) -> List[Entity]:
        """Extract embeddings using the transformer model."""
        if not entities:
            return entities
        
        # Extract entity texts with context
        entity_contexts = []
        for entity in entities:
            # Get context around entity (±50 characters)
            start_ctx = max(0, entity.start - 50)
            end_ctx = min(len(text), entity.end + 50)
            context = text[start_ctx:end_ctx]
            
            # Highlight the entity in context for better embedding
            entity_in_context = (
                context[:entity.start-start_ctx] + 
                f"[{entity.text}]" + 
                context[entity.end-start_ctx:]
            )
            entity_contexts.append(entity_in_context)
        
        # Generate embeddings
        embeddings = self.model.encode(entity_contexts)
        
        # Create updated entities with embeddings
        updated_entities = []
        for entity, embedding in zip(entities, embeddings):
            updated_entity = Entity(
                text=entity.text,
                label=entity.label,
                start=entity.start,
                end=entity.end,
                confidence=entity.confidence,
                normalized_text=entity.normalized_text,
                embedding=embedding.tolist()
            )
            updated_entities.append(updated_entity)
        
        return updated_entities
    
    def get_embedding_dimension(self) -> int:
        return self._dimension


@dataclass
class CorefCluster:
    """Represents a cluster of co-referent entities."""
    entities: List[Entity]
    canonical_text: str
    confidence: float = 1.0
    
    def add_entity(self, entity: Entity):
        """Add an entity to this cluster."""
        self.entities.append(entity)
    
    def get_mentions(self) -> List[str]:
        """Get all mention texts in this cluster."""
        return [entity.text for entity in self.entities]


class CoreferenceResolver:
    """
    Simple coreference resolution using embedding similarity.
    
    This is a basic implementation suitable for simple co-referencing tasks.
    For production use, consider more sophisticated approaches.
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize coreference resolver.
        
        Args:
            similarity_threshold: Minimum cosine similarity for coreference
        """
        self.similarity_threshold = similarity_threshold
    
    def resolve_coreferences(self, entities: List[Entity]) -> List[CorefCluster]:
        """
        Resolve coreferences among entities using embedding similarity.
        
        Args:
            entities: List of entities with embeddings
            
        Returns:
            List of coreference clusters
        """
        if not entities or not entities[0].embedding:
            return [CorefCluster([entity], entity.text) for entity in entities]
        
        clusters = []
        remaining_entities = entities.copy()
        
        while remaining_entities:
            seed_entity = remaining_entities.pop(0)
            cluster = CorefCluster([seed_entity], seed_entity.text)
            
            # Find similar entities
            to_remove = []
            for i, entity in enumerate(remaining_entities):
                similarity = self._cosine_similarity(
                    seed_entity.embedding, 
                    entity.embedding
                )
                
                if similarity >= self.similarity_threshold:
                    cluster.add_entity(entity)
                    to_remove.append(i)
            
            # Remove clustered entities
            for i in reversed(to_remove):
                remaining_entities.pop(i)
            
            clusters.append(cluster)
        
        return clusters
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norms == 0:
            return 0.0
        
        return dot_product / norms


class EmbeddingRuNER:
    """
    Extended ruNER with embedding and coreference capabilities.
    
    This wraps the basic ruNER with embedding extraction and 
    simple coreference resolution.
    """
    
    def __init__(self, 
                 base_runer,
                 embedding_extractor: EmbeddingExtractor = None,
                 coreference_resolver: CoreferenceResolver = None):
        """
        Initialize embedding-enabled ruNER.
        
        Args:
            base_runer: Base RuNER instance
            embedding_extractor: Embedding extraction strategy
            coreference_resolver: Coreference resolution strategy
        """
        self.base_runer = base_runer
        self.embedding_extractor = embedding_extractor or DummyEmbeddingExtractor()
        self.coreference_resolver = coreference_resolver or CoreferenceResolver()
    
    def process_with_coref(self, text: str, entity_map: Dict[str, str]) -> tuple:
        """
        Process text with entity recognition, embeddings, and coreference resolution.
        
        Returns:
            Tuple of (entities_with_embeddings, bio_tagged, coref_clusters)
        """
        # Basic entity recognition
        entities, bio_tagged = self.base_runer.process(text, entity_map)
        
        # Extract embeddings
        entities_with_embeddings = self.embedding_extractor.extract_embeddings(entities, text)
        
        # Resolve coreferences
        coref_clusters = self.coreference_resolver.resolve_coreferences(entities_with_embeddings)
        
        return entities_with_embeddings, bio_tagged, coref_clusters


# Convenience function
def create_embedding_runer(base_runer, 
                          use_transformers: bool = False,
                          transformer_model: str = "all-MiniLM-L6-v2") -> EmbeddingRuNER:
    """
    Create an embedding-enabled ruNER.
    
    Args:
        base_runer: Base RuNER instance
        use_transformers: Whether to use real transformer embeddings
        transformer_model: Model name if using transformers
        
    Returns:
        EmbeddingRuNER instance
    """
    if use_transformers:
        try:
            embedding_extractor = TransformerEmbeddingExtractor(transformer_model)
        except ImportError:
            print("Warning: sentence-transformers not available, using dummy embeddings")
            embedding_extractor = DummyEmbeddingExtractor()
    else:
        embedding_extractor = DummyEmbeddingExtractor()
    
    return EmbeddingRuNER(base_runer, embedding_extractor)