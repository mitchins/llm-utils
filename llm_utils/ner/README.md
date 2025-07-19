# ruNER - Really Uncomplicated Named Entity Recognition

ruNER is designed to make Named Entity Recognition as simple as possible. No complex configuration files, no specific library versions, no XML - just Python dictionaries and intuitive functions.

## Quick Start

```python
from llm_utils.ner import runer

# Define what you want to recognize
entities = {
    "Bilbo Baggins": "Person",
    "The Shire": "Location", 
    "One Ring": "Artifact"
}

text = "Bilbo Baggins found the One Ring in The Shire."

# Get entities and BIO tags automatically
found_entities, bio_tagged = runer(text, entities)

# Print results
for entity in found_entities:
    print(f"Found {entity.label}: '{entity.text}' at {entity.start}-{entity.end}")
```

## Features

### ✅ **Zero Configuration**
- No XML files, no complex setup
- Just provide a dictionary of entities and their types
- BIO tags are generated automatically

### ✅ **Possessive Handling**
- Automatically handles possessives like "Bilbo's hat" and "Bilbo Baggins's house"
- Matches both "Bilbo" and "Bilbo's" to the same entity

### ✅ **Case Insensitive**
- Matches entities regardless of case
- "JOHN SMITH", "john smith", "John Smith" all match

### ✅ **Overlap Resolution**  
- Automatically handles overlapping entities
- Prefers longer matches when entities overlap

### ✅ **SOLID Architecture**
- Follows SOLID principles for maintainability
- Extensible components for custom behavior
- Well-tested with comprehensive test suite

### ✅ **Embedding Support**
- Built-in support for embeddings for co-referencing
- Optional transformer-based embeddings
- Simple coreference resolution

## Core Components

### Entity Definition
Simply define entities as a dictionary:

```python
entities = {
    "Entity Name": "Entity Type",
    "Albert Einstein": "Person",
    "Princeton University": "Organization", 
    "Theory of Relativity": "Concept"
}
```

### BIO Tag Generation
BIO tags are generated automatically:

```python
from llm_utils.ner import get_bio_labels

bio_labels = get_bio_labels(entities)
# Returns: ['O', 'B-CONCEPT', 'I-CONCEPT', 'B-ORGANIZATION', 'I-ORGANIZATION', 'B-PERSON', 'I-PERSON']
```

### Results
Get both detected entities and BIO-tagged tokens:

```python
entities, tagged = runer(text, entity_map)

# Entities with position information
for entity in entities:
    print(f"{entity.text} -> {entity.label} [{entity.start}:{entity.end}]")

# BIO-tagged tokens
for token, tag in zip(tagged.tokens, tagged.tags):
    print(f"{token} -> {tag}")
```

## Advanced Usage

### Custom Components
For advanced use cases, you can customize the components:

```python
from llm_utils.ner import RuNER, ExactEntityMatcher, BasicTextNormalizer, BIOTagger

# Create custom ruNER instance
runer = RuNER(
    entity_matcher=ExactEntityMatcher(),
    text_normalizer=BasicTextNormalizer(), 
    bio_tagger=BIOTagger()
)

entities, tagged = runer.process(text, entity_map)
```

### Embedding Support
For co-referencing capabilities:

```python
from llm_utils.ner.embeddings import create_embedding_runer
from llm_utils.ner import create_runer

base_runer = create_runer()
embedding_runer = create_embedding_runer(base_runer, use_transformers=True)

entities, tagged, coref_clusters = embedding_runer.process_with_coref(text, entity_map)

# View coreference clusters
for cluster in coref_clusters:
    print(f"Cluster: {cluster.get_mentions()}")
```

## Examples

### News Article Processing
```python
entities = {
    "Joe Biden": "Person",
    "White House": "Location",
    "Congress": "Organization"
}

text = "Joe Biden met with Congress at the White House."
found_entities, bio_tagged = runer(text, entities)
```

### Academic Text Processing  
```python
entities = {
    "BERT": "Model",
    "Stanford University": "Organization",
    "Natural Language Processing": "Field"
}

text = "BERT advanced Natural Language Processing research at Stanford University."
found_entities, bio_tagged = runer(text, entities)
```

### Fantasy Literature
```python
entities = {
    "Frodo Baggins": "Person",
    "Mount Doom": "Location",
    "One Ring": "Artifact"
}

text = "Frodo Baggins carried the One Ring to Mount Doom."
found_entities, bio_tagged = runer(text, entities)
```

## Why ruNER?

### Problems with Existing NER Frameworks
- **Complex Setup**: Require specific XML configurations, model training pipelines
- **Version Dependencies**: Break when library versions change
- **Overbaked**: Too many features for simple use cases
- **Hard to Extend**: Difficult to customize for specific domains

### ruNER Solutions
- **Simple Interface**: Just provide entity mappings as dictionaries
- **No Dependencies**: Works with standard Python libraries
- **Just Enough Features**: Possessive handling, case-insensitive matching, BIO tagging
- **Easy Extension**: SOLID architecture allows easy customization
- **Well Tested**: Comprehensive test suite ensures reliability

## Performance Notes

ruNER is optimized for simplicity and correctness, not speed. For high-performance applications with large entity lists, consider:

1. Using more sophisticated matching algorithms
2. Pre-indexing entities for faster lookup
3. Implementing custom `EntityMatcher` strategies

## Future Enhancements

- **Fuzzy Matching**: Handle minor spelling variations
- **ML-based Entity Recognition**: Train custom models from entity mappings
- **Advanced Coreference**: More sophisticated coreference resolution
- **Performance Optimizations**: Faster matching for large entity sets

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/ner/ -v
```

Or see the demo:

```bash
python examples/runer_demo.py
```

## Philosophy

ruNER follows the principle of "Really Uncomplicated" - it should be so simple and intuitive that it doesn't need explanation. If you find yourself reading complex documentation to use ruNER, we've failed our goal.

The system is designed to be:
- **Intuitive**: Uses familiar Python patterns
- **Predictable**: Consistent behavior across use cases  
- **Maintainable**: Clean, well-tested code following SOLID principles
- **Extensible**: Easy to customize without breaking existing functionality