"""
Really Uncomplicated NER (ruNER)

Simple, intuitive Named Entity Recognition with minimal setup required.

Usage:
    from llm_utils.ner import runer
    
    # Define what you want to recognize
    entities = {
        "Bilbo Baggins": "Person",
        "Gandalf": "Person", 
        "The Shire": "Location",
        "Middle-earth": "Location"
    }
    
    text = "Bilbo Baggins lived in The Shire before his adventure."
    
    # Get entities and BIO tags automatically
    found_entities, bio_tagged = runer(text, entities)
    
    # Print results
    for entity in found_entities:
        print(f"Found {entity.label}: '{entity.text}' at {entity.start}-{entity.end}")
    
    for token, tag in zip(bio_tagged.tokens, bio_tagged.tags):
        print(f"{token} -> {tag}")
"""

from .core import (
    RuNER, Entity, BIOTag, TokenizedSpan,
    TextNormalizer, BasicTextNormalizer,
    EntityMatcher, ExactEntityMatcher,
    BIOTagger, create_runer
)

# Create a default instance for convenience
_default_runer = create_runer()

def runer(text: str, entity_map: dict) -> tuple:
    """
    Convenience function for simple ruNER usage.
    
    Args:
        text: Text to process
        entity_map: Dictionary mapping entity names to types
                   Example: {"Bilbo Baggins": "Person", "The Shire": "Location"}
    
    Returns:
        Tuple of (entities, bio_tagged_tokens)
    """
    return _default_runer.process(text, entity_map)

def get_bio_labels(entity_map: dict) -> list:
    """Get all possible BIO labels for an entity mapping."""
    return _default_runer.get_bio_labels(entity_map)

__all__ = [
    'runer', 'get_bio_labels',
    'RuNER', 'Entity', 'BIOTag', 'TokenizedSpan',
    'TextNormalizer', 'BasicTextNormalizer', 
    'EntityMatcher', 'ExactEntityMatcher',
    'BIOTagger', 'create_runer'
]