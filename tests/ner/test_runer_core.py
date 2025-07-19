"""
Tests for ruNER core functionality.
"""

import pytest
from llm_utils.ner.core import (
    Entity, BIOTag, TokenizedSpan,
    BasicTextNormalizer, ExactEntityMatcher, 
    BIOTagger, RuNER, create_runer
)


class TestBasicTextNormalizer:
    
    def test_simple_normalization(self):
        normalizer = BasicTextNormalizer()
        assert normalizer.normalize("Hello World") == "hello world"
        assert normalizer.normalize("  EXTRA   SPACES  ") == "extra spaces"
    
    def test_possessive_handling(self):
        normalizer = BasicTextNormalizer()
        
        # Standard possessive
        assert normalizer.normalize("Bilbo's hat") == "bilbo hat"
        assert normalizer.normalize("John's car") == "john car"
        
        # Plural possessive
        assert normalizer.normalize("Students' books") == "students books"
        assert normalizer.normalize("Dogs' toys") == "dogs toys"
        
        # End of string possessive
        assert normalizer.normalize("That is Bilbo's") == "that is bilbo"
        assert normalizer.normalize("The students'") == "the students"
    
    def test_complex_possessives(self):
        normalizer = BasicTextNormalizer()
        text = "Bilbo Baggins's adventure started when Gandalf's plan began."
        expected = "bilbo baggins adventure started when gandalf plan began."
        assert normalizer.normalize(text) == expected


class TestExactEntityMatcher:
    
    def test_simple_matching(self):
        matcher = ExactEntityMatcher()
        entity_map = {"Bilbo Baggins": "Person", "The Shire": "Location"}
        text = "Bilbo Baggins lived in The Shire."
        
        entities = matcher.find_matches(text, entity_map)
        
        assert len(entities) == 2
        assert entities[0].text == "Bilbo Baggins"
        assert entities[0].label == "Person"
        assert entities[0].start == 0
        assert entities[0].end == 13
        
        assert entities[1].text == "The Shire"
        assert entities[1].label == "Location"
    
    def test_possessive_matching(self):
        matcher = ExactEntityMatcher()
        entity_map = {"Bilbo Baggins": "Person"}
        text = "This is Bilbo Baggins's house."
        
        entities = matcher.find_matches(text, entity_map)
        
        assert len(entities) == 1
        assert entities[0].text == "Bilbo Baggins"
        assert entities[0].label == "Person"
    
    def test_case_insensitive_matching(self):
        matcher = ExactEntityMatcher()
        entity_map = {"Bilbo Baggins": "Person"}
        text = "BILBO BAGGINS was a hobbit."
        
        entities = matcher.find_matches(text, entity_map)
        
        assert len(entities) == 1
        assert entities[0].text == "BILBO BAGGINS"
        assert entities[0].label == "Person"
    
    def test_overlapping_entities(self):
        matcher = ExactEntityMatcher()
        entity_map = {
            "John Smith": "Person",
            "John": "Person",
            "Smith Corporation": "Organization"
        }
        text = "John Smith works at Smith Corporation."
        
        entities = matcher.find_matches(text, entity_map)
        
        # Should keep longest non-overlapping entities
        assert len(entities) == 2
        entity_texts = [e.text for e in entities]
        assert "John Smith" in entity_texts
        assert "Smith Corporation" in entity_texts
        assert "John" not in entity_texts  # Overlapped by "John Smith"
    
    def test_multiple_occurrences(self):
        matcher = ExactEntityMatcher()
        entity_map = {"John": "Person"}
        text = "John met John at John's house."
        
        entities = matcher.find_matches(text, entity_map)
        
        # Should find all occurrences
        assert len(entities) == 3
        assert all(e.text == "John" and e.label == "Person" for e in entities)
        assert entities[0].start == 0
        assert entities[1].start == 9
        assert entities[2].start == 17


class TestBIOTagger:
    
    def test_simple_tagging(self):
        tagger = BIOTagger()
        entities = [
            Entity("John", "Person", 0, 4),
            Entity("Microsoft", "Organization", 14, 23)
        ]
        text = "John works at Microsoft."
        
        tagged = tagger.tag_text(text, entities)
        
        expected_tokens = ["John", "works", "at", "Microsoft."]
        expected_tags = ["B-PERSON", "O", "O", "B-ORGANIZATION"]
        
        assert tagged.tokens == expected_tokens
        assert tagged.tags == expected_tags
        assert tagged.original_text == text
    
    def test_multi_token_entity(self):
        tagger = BIOTagger()
        entities = [Entity("John Smith", "Person", 0, 10)]
        text = "John Smith works here."
        
        tagged = tagger.tag_text(text, entities)
        
        expected_tokens = ["John", "Smith", "works", "here."]
        expected_tags = ["B-PERSON", "I-PERSON", "O", "O"]
        
        assert tagged.tokens == expected_tokens
        assert tagged.tags == expected_tags
    
    def test_adjacent_entities(self):
        tagger = BIOTagger()
        entities = [
            Entity("John", "Person", 0, 4),
            Entity("Smith", "Person", 5, 10)
        ]
        text = "John Smith"
        
        tagged = tagger.tag_text(text, entities)
        
        expected_tokens = ["John", "Smith"]
        expected_tags = ["B-PERSON", "B-PERSON"]  # Adjacent entities get B- tags
        
        assert tagged.tokens == expected_tokens
        assert tagged.tags == expected_tags
    
    def test_empty_entities(self):
        tagger = BIOTagger()
        text = "No entities here."
        
        tagged = tagger.tag_text(text, [])
        
        expected_tokens = ["No", "entities", "here."]
        expected_tags = ["O", "O", "O"]
        
        assert tagged.tokens == expected_tokens
        assert tagged.tags == expected_tags


class TestRuNER:
    
    def test_end_to_end_processing(self):
        runer = RuNER()
        entity_map = {
            "Bilbo Baggins": "Person",
            "The Shire": "Location",
            "Gandalf": "Person"
        }
        text = "Bilbo Baggins lived in The Shire with Gandalf."
        
        entities, tagged = runer.process(text, entity_map)
        
        # Check entities
        assert len(entities) == 3
        entity_labels = {e.text: e.label for e in entities}
        assert entity_labels["Bilbo Baggins"] == "Person"
        assert entity_labels["The Shire"] == "Location" 
        assert entity_labels["Gandalf"] == "Person"
        
        # Check BIO tags
        assert "B-PERSON" in tagged.tags
        assert "I-PERSON" in tagged.tags
        assert "B-LOCATION" in tagged.tags
        assert "O" in tagged.tags
    
    def test_possessive_end_to_end(self):
        runer = RuNER()
        entity_map = {"Bilbo Baggins": "Person"}
        text = "This is Bilbo Baggins's house."
        
        entities, tagged = runer.process(text, entity_map)
        
        assert len(entities) == 1
        assert entities[0].text == "Bilbo Baggins"
        assert entities[0].label == "Person"
        
        # Should have B- and I- tags for the person
        assert "B-PERSON" in tagged.tags
        assert "I-PERSON" in tagged.tags
    
    def test_get_bio_labels(self):
        runer = RuNER()
        entity_map = {
            "John": "Person",
            "Microsoft": "Organization",
            "Seattle": "Location"
        }
        
        bio_labels = runer.get_bio_labels(entity_map)
        
        expected = [
            "O",
            "B-LOCATION", "I-LOCATION",
            "B-ORGANIZATION", "I-ORGANIZATION", 
            "B-PERSON", "I-PERSON"
        ]
        assert sorted(bio_labels) == sorted(expected)
    
    def test_empty_entity_map(self):
        runer = RuNER()
        text = "No entities to find."
        
        entities, tagged = runer.process(text, {})
        
        assert len(entities) == 0
        assert all(tag == "O" for tag in tagged.tags)
    
    def test_no_matches(self):
        runer = RuNER()
        entity_map = {"Nonexistent Person": "Person"}
        text = "This text has no matching entities."
        
        entities, tagged = runer.process(text, entity_map)
        
        assert len(entities) == 0
        assert all(tag == "O" for tag in tagged.tags)


class TestConvenienceFunctions:
    
    def test_create_runer(self):
        runer = create_runer()
        assert isinstance(runer, RuNER)
        assert runer.entity_matcher is not None
        assert runer.text_normalizer is not None
        assert runer.bio_tagger is not None


class TestBIOTag:
    
    def test_bio_tag_with_label(self):
        assert BIOTag.B.with_label("Person") == "B-PERSON"
        assert BIOTag.I.with_label("Location") == "I-LOCATION"
        assert BIOTag.O.with_label("anything") == "O"
    
    def test_bio_tag_case_handling(self):
        assert BIOTag.B.with_label("person") == "B-PERSON"
        assert BIOTag.I.with_label("LOCATION") == "I-LOCATION"


class TestEntity:
    
    def test_entity_creation(self):
        entity = Entity("John", "Person", 0, 4)
        assert entity.text == "John"
        assert entity.label == "Person"
        assert entity.start == 0
        assert entity.end == 4
        assert entity.confidence == 1.0
        assert entity.normalized_text == "John"  # Auto-set by post_init
    
    def test_entity_with_normalized_text(self):
        entity = Entity("John", "Person", 0, 4, normalized_text="john")
        assert entity.normalized_text == "john"