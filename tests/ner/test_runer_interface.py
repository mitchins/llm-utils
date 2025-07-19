"""
Tests for ruNER public interface and convenience functions.
"""

import pytest
from llm_utils.ner import runer, get_bio_labels, Entity


class TestPublicInterface:
    
    def test_runer_convenience_function(self):
        """Test the main runer() convenience function."""
        entity_map = {
            "Bilbo Baggins": "Person",
            "The Shire": "Location"
        }
        text = "Bilbo Baggins lived in The Shire."
        
        entities, tagged = runer(text, entity_map)
        
        # Should find both entities
        assert len(entities) == 2
        assert isinstance(entities[0], Entity)
        
        # Check entity details
        entity_dict = {e.text: e.label for e in entities}
        assert entity_dict["Bilbo Baggins"] == "Person"
        assert entity_dict["The Shire"] == "Location"
        
        # Check BIO tagging
        assert len(tagged.tokens) > 0
        assert "B-PERSON" in tagged.tags
        assert "B-LOCATION" in tagged.tags
    
    def test_get_bio_labels_function(self):
        """Test the get_bio_labels convenience function."""
        entity_map = {
            "John": "Person",
            "Microsoft": "Organization"
        }
        
        labels = get_bio_labels(entity_map)
        
        expected = ["O", "B-ORGANIZATION", "I-ORGANIZATION", "B-PERSON", "I-PERSON"]
        assert sorted(labels) == sorted(expected)
    
    def test_complex_example(self):
        """Test a more complex real-world example."""
        entity_map = {
            "Frodo Baggins": "Person",
            "Samwise Gamgee": "Person", 
            "Mount Doom": "Location",
            "One Ring": "Artifact",
            "Mordor": "Location",
            "Sauron": "Person"
        }
        
        text = """Frodo Baggins and Samwise Gamgee traveled to Mount Doom in Mordor 
                  to destroy the One Ring and defeat Sauron."""
        
        entities, tagged = runer(text, entity_map)
        
        # Should find most entities (exact matching)
        assert len(entities) >= 4  # At least several should match
        
        # Check specific entities
        found_entities = {e.text: e.label for e in entities}
        assert "Frodo Baggins" in found_entities
        assert found_entities["Frodo Baggins"] == "Person"
        
        # Check BIO structure
        person_tags = [tag for tag in tagged.tags if "PERSON" in tag]
        assert len(person_tags) > 0
        assert any(tag.startswith("B-") for tag in person_tags)
    
    def test_possessive_examples(self):
        """Test possessive handling in real examples."""
        entity_map = {
            "Bilbo Baggins": "Person",
            "Bilbo": "Person",  # Add shortened form
            "Bag End": "Location"
        }
        
        # Test various possessive forms
        test_cases = [
            "Bilbo Baggins's house",
            "Bilbo's adventure", 
            "Bag End's green door"
        ]
        
        for text in test_cases:
            entities, tagged = runer(text, entity_map)
            # Should find at least one entity in each case
            assert len(entities) >= 1, f"No entities found in: {text}"
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        
        # Empty entity map
        entities, tagged = runer("Some text here", {})
        assert len(entities) == 0
        assert all(tag == "O" for tag in tagged.tags)
        
        # Empty text
        entities, tagged = runer("", {"John": "Person"})
        assert len(entities) == 0
        assert len(tagged.tokens) == 0
        
        # No matches
        entities, tagged = runer("Nothing matches here", {"Bilbo": "Person"})
        assert len(entities) == 0
        assert all(tag == "O" for tag in tagged.tags)
    
    def test_case_insensitive_matching(self):
        """Test that matching works regardless of case."""
        entity_map = {"john smith": "Person"}
        
        test_cases = [
            "John Smith",
            "JOHN SMITH", 
            "john smith",
            "John SMITH"
        ]
        
        for text in test_cases:
            entities, tagged = runer(text, entity_map)
            assert len(entities) == 1
            assert entities[0].label == "Person"
    
    def test_overlapping_entity_handling(self):
        """Test how overlapping entities are handled."""
        entity_map = {
            "New York": "Location",
            "New York City": "Location",
            "York": "Location"
        }
        
        text = "I visited New York City last summer."
        entities, tagged = runer(text, entity_map)
        
        # Should prefer longer matches and avoid overlaps
        entity_texts = [e.text for e in entities]
        assert "New York City" in entity_texts
        # Shorter overlapping entities should be excluded
        assert "New York" not in entity_texts or "York" not in entity_texts
    
    def test_multiple_entity_types(self):
        """Test processing with multiple entity types."""
        entity_map = {
            "Albert Einstein": "Person",
            "Princeton University": "Organization", 
            "Princeton": "Location",
            "Theory of Relativity": "Concept",
            "Nobel Prize": "Award"
        }
        
        text = """Albert Einstein worked at Princeton University in Princeton 
                  and developed the Theory of Relativity, winning the Nobel Prize."""
        
        entities, tagged = runer(text, entity_map)
        
        # Check we get multiple entity types
        found_labels = {e.label for e in entities}
        assert len(found_labels) >= 3  # Should find multiple types
        
        # Check BIO labels include all types
        bio_labels = get_bio_labels(entity_map)
        expected_types = {"PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "AWARD"}
        found_bio_types = set()
        for label in bio_labels:
            if "-" in label:
                found_bio_types.add(label.split("-")[1])
        
        assert expected_types.issubset(found_bio_types)


class TestRealWorldExamples:
    """Test ruNER with realistic use cases."""
    
    def test_news_article_example(self):
        """Test with a news-like text."""
        entity_map = {
            "Joe Biden": "Person",
            "United States": "Country",
            "White House": "Location",
            "Congress": "Organization"
        }
        
        text = "Joe Biden addressed Congress from the White House about United States policy."
        
        entities, tagged = runer(text, entity_map)
        
        assert len(entities) >= 2  # Should find several entities
        person_entities = [e for e in entities if e.label == "Person"]
        assert len(person_entities) >= 1
    
    def test_academic_paper_example(self):
        """Test with academic/technical text."""
        entity_map = {
            "BERT": "Model",
            "Stanford University": "Organization",
            "Natural Language Processing": "Field",
            "Transformer": "Architecture"
        }
        
        text = "BERT, developed using Transformer architecture, advanced Natural Language Processing research at Stanford University."
        
        entities, tagged = runer(text, entity_map)
        
        # Should identify technical entities
        found_labels = {e.label for e in entities}
        assert "Model" in found_labels or "Architecture" in found_labels
    
    def test_fantasy_literature_example(self):
        """Test with fantasy/fictional content."""
        entity_map = {
            "Aragorn": "Person",
            "Legolas": "Person",
            "Gimli": "Person",
            "Fellowship of the Ring": "Group",
            "Minas Tirith": "Location",
            "Middle-earth": "World"
        }
        
        text = "Aragorn, Legolas, and Gimli were members of the Fellowship of the Ring in Middle-earth."
        
        entities, tagged = runer(text, entity_map)
        
        # Should find multiple characters and locations
        person_entities = [e for e in entities if e.label == "Person"]
        assert len(person_entities) >= 2