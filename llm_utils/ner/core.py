"""
Really Uncomplicated NER (ruNER) - Core Components

A simple, intuitive Named Entity Recognition system that:
- Takes entity mappings like {"Bilbo Baggins": "Person"}
- Automatically generates BIO tags
- Handles basic possessive lemmatization
- Provides embeddings for co-referencing
- Follows SOLID principles for maintainability
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass
from enum import Enum
import re


@dataclass
class Entity:
    """Represents a detected entity with all relevant information."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    normalized_text: Optional[str] = None
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        if self.normalized_text is None:
            self.normalized_text = self.text


class BIOTag(Enum):
    """BIO tagging scheme enumeration."""
    O = "O"  # Outside
    B = "B"  # Beginning
    I = "I"  # Inside
    
    def with_label(self, label: str) -> str:
        """Create BIO tag with entity label."""
        if self == BIOTag.O:
            return "O"
        return f"{self.value}-{label.upper()}"


@dataclass
class TokenizedSpan:
    """Represents a tokenized span with BIO tags."""
    tokens: List[str]
    tags: List[str]
    original_text: str
    start_positions: List[int]
    end_positions: List[int]


class TextNormalizer(ABC):
    """Abstract base for text normalization strategies."""
    
    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize text for entity matching."""
        pass


class BasicTextNormalizer(TextNormalizer):
    """Basic text normalizer handling possessives and common variations."""
    
    def __init__(self):
        # Common possessive patterns - order matters!
        self.possessive_patterns = [
            (r"(\w+)s'\s+", r"\1s "),  # "Bilbos' house" -> "Bilbos house" (plural possessive first)
            (r"(\w+)s'$", r"\1s"),     # "Bilbos'" -> "Bilbos" (end of string plural)
            (r"(\w+)'s\s+", r"\1 "),  # "Bilbo's hat" -> "Bilbo hat" 
            (r"(\w+)'s$", r"\1"),     # "Bilbo's" -> "Bilbo"
        ]
    
    def normalize(self, text: str) -> str:
        """Apply basic normalization including possessive handling."""
        normalized = text.lower().strip()
        
        # Handle possessives
        for pattern, replacement in self.possessive_patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Clean up extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized


class EntityMatcher(ABC):
    """Abstract base for entity matching strategies."""
    
    @abstractmethod
    def find_matches(self, text: str, entity_map: Dict[str, str]) -> List[Entity]:
        """Find entity matches in text."""
        pass


class ExactEntityMatcher(EntityMatcher):
    """Exact string matching with normalization support."""
    
    def __init__(self, normalizer: TextNormalizer = None):
        self.normalizer = normalizer or BasicTextNormalizer()
    
    def find_matches(self, text: str, entity_map: Dict[str, str]) -> List[Entity]:
        """Find exact matches with normalization support."""
        entities = []
        normalized_text = self.normalizer.normalize(text)
        
        # Create mapping of normalized entity names to original names and labels
        normalized_map = {}
        for entity_name, label in entity_map.items():
            normalized_name = self.normalizer.normalize(entity_name)
            normalized_map[normalized_name] = (entity_name, label)
        
        # Find matches in normalized text
        for normalized_name, (original_name, label) in normalized_map.items():
            start_idx = 0
            while True:
                idx = normalized_text.find(normalized_name, start_idx)
                if idx == -1:
                    break
                
                # Find corresponding position in original text
                orig_start, orig_end = self._map_to_original_position(
                    text, normalized_text, idx, idx + len(normalized_name)
                )
                
                if orig_start is not None and orig_end is not None:
                    entity_text = text[orig_start:orig_end]
                    entities.append(Entity(
                        text=entity_text,
                        label=label,
                        start=orig_start,
                        end=orig_end,
                        normalized_text=normalized_name
                    ))
                
                start_idx = idx + 1
        
        # Remove overlapping entities (keep longest)
        return self._remove_overlaps(entities)
    
    def _map_to_original_position(self, original: str, normalized: str, 
                                 norm_start: int, norm_end: int) -> Tuple[Optional[int], Optional[int]]:
        """Map normalized positions back to original text positions."""
        # Simple approximation - in practice, you'd want more sophisticated mapping
        # For basic use cases, this provides reasonable results
        
        # Find approximate positions by scanning character by character
        orig_pos = 0
        norm_pos = 0
        start_found = None
        
        while orig_pos < len(original) and norm_pos < len(normalized):
            if norm_pos == norm_start:
                start_found = orig_pos
            if norm_pos == norm_end:
                return start_found, orig_pos
            
            orig_char = original[orig_pos].lower()
            norm_char = normalized[norm_pos]
            
            if orig_char == norm_char:
                orig_pos += 1
                norm_pos += 1
            elif orig_char.isspace() and norm_char == ' ':
                # Skip multiple whitespace in original
                while orig_pos < len(original) and original[orig_pos].isspace():
                    orig_pos += 1
                norm_pos += 1
            elif orig_char in "'":
                # Skip apostrophes that were removed
                orig_pos += 1
            else:
                orig_pos += 1
        
        if norm_pos == norm_end and start_found is not None:
            return start_found, orig_pos
        
        return None, None
    
    def _remove_overlaps(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping the longest ones."""
        if not entities:
            return entities
        
        # Sort by start position, then by length (descending)
        sorted_entities = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
        
        result = []
        for entity in sorted_entities:
            # Check if this entity overlaps with any already selected
            overlaps = any(
                not (entity.end <= existing.start or entity.start >= existing.end)
                for existing in result
            )
            
            if not overlaps:
                result.append(entity)
        
        return sorted(result, key=lambda e: e.start)


class BIOTagger:
    """Converts entity spans to BIO tagged sequences."""
    
    def __init__(self, tokenizer_fn=None):
        """Initialize with optional custom tokenizer."""
        self.tokenizer_fn = tokenizer_fn or self._simple_tokenize
    
    def tag_text(self, text: str, entities: List[Entity]) -> TokenizedSpan:
        """Convert text and entities to BIO-tagged token sequence."""
        tokens, positions = self.tokenizer_fn(text)
        tags = ["O"] * len(tokens)
        
        # Apply entity tags
        for entity in entities:
            self._apply_entity_tags(tokens, positions, entity, tags)
        
        return TokenizedSpan(
            tokens=tokens,
            tags=tags,
            original_text=text,
            start_positions=[pos[0] for pos in positions],
            end_positions=[pos[1] for pos in positions]
        )
    
    def _simple_tokenize(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Simple whitespace tokenizer with position tracking."""
        tokens = []
        positions = []
        
        # Use regex to find word boundaries and track positions
        for match in re.finditer(r'\S+', text):
            tokens.append(match.group())
            positions.append((match.start(), match.end()))
        
        return tokens, positions
    
    def _apply_entity_tags(self, tokens: List[str], positions: List[Tuple[int, int]], 
                          entity: Entity, tags: List[str]):
        """Apply BIO tags for a single entity."""
        entity_tokens = []
        
        # Find tokens that overlap with entity span
        for i, (start, end) in enumerate(positions):
            if not (end <= entity.start or start >= entity.end):
                entity_tokens.append(i)
        
        if not entity_tokens:
            return
        
        # Apply BIO tagging
        for j, token_idx in enumerate(entity_tokens):
            if j == 0:
                tags[token_idx] = BIOTag.B.with_label(entity.label)
            else:
                tags[token_idx] = BIOTag.I.with_label(entity.label)


class RuNER:
    """
    Really Uncomplicated Named Entity Recognition system.
    
    Simple interface: provide entity mappings, get BIO-tagged results.
    """
    
    def __init__(self, 
                 entity_matcher: EntityMatcher = None,
                 text_normalizer: TextNormalizer = None,
                 bio_tagger: BIOTagger = None):
        """Initialize ruNER with customizable components."""
        self.entity_matcher = entity_matcher or ExactEntityMatcher()
        self.text_normalizer = text_normalizer or BasicTextNormalizer()
        self.bio_tagger = bio_tagger or BIOTagger()
        
        # If matcher doesn't have normalizer, set it
        if hasattr(self.entity_matcher, 'normalizer') and self.entity_matcher.normalizer is None:
            self.entity_matcher.normalizer = self.text_normalizer
    
    def process(self, text: str, entity_map: Dict[str, str]) -> Tuple[List[Entity], TokenizedSpan]:
        """
        Process text with entity mappings.
        
        Args:
            text: Input text to process
            entity_map: Dictionary of {"entity name": "entity type"}
        
        Returns:
            Tuple of (detected entities, BIO-tagged tokens)
        """
        # Find entity matches
        entities = self.entity_matcher.find_matches(text, entity_map)
        
        # Generate BIO tags
        tagged_span = self.bio_tagger.tag_text(text, entities)
        
        return entities, tagged_span
    
    def get_entity_labels(self, entity_map: Dict[str, str]) -> Set[str]:
        """Get all unique entity labels from mapping."""
        return set(entity_map.values())
    
    def get_bio_labels(self, entity_map: Dict[str, str]) -> List[str]:
        """Get all BIO labels for the given entity mapping."""
        labels = ["O"]
        for entity_type in sorted(self.get_entity_labels(entity_map)):
            labels.extend([
                BIOTag.B.with_label(entity_type),
                BIOTag.I.with_label(entity_type)
            ])
        return labels


# Convenience function for simple usage
def create_runer(entity_matcher: EntityMatcher = None) -> RuNER:
    """Create a ruNER instance with sensible defaults."""
    return RuNER(entity_matcher=entity_matcher)