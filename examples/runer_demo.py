#!/usr/bin/env python3
"""
ruNER Demo - Really Uncomplicated Named Entity Recognition

This demo shows how simple it is to use ruNER for entity recognition.
No complex configuration, no XML, no specific library versions required.
"""

from llm_utils.ner import runer, get_bio_labels


def basic_demo():
    """Basic ruNER usage demonstration."""
    print("🚀 ruNER Basic Demo")
    print("=" * 50)
    
    # Define what you want to recognize - it's that simple!
    entities = {
        "Bilbo Baggins": "Person",
        "Gandalf": "Person",
        "The Shire": "Location", 
        "Bag End": "Location",
        "One Ring": "Artifact",
        "Sauron": "Person"
    }
    
    # Some test text
    text = """Bilbo Baggins lived in Bag End in The Shire. 
              When Gandalf arrived, Bilbo's adventure began. 
              The One Ring was Sauron's creation."""
    
    print(f"📝 Text: {text.strip()}")
    print(f"\n🎯 Entity Map: {entities}")
    
    # Process with ruNER
    found_entities, bio_tagged = runer(text, entities)
    
    print(f"\n✨ Found {len(found_entities)} entities:")
    for entity in found_entities:
        print(f"  • '{entity.text}' → {entity.label} [{entity.start}:{entity.end}]")
    
    print(f"\n🏷️  BIO Tags:")
    for token, tag in zip(bio_tagged.tokens, bio_tagged.tags):
        print(f"  {token:15} → {tag}")
    
    print(f"\n📋 All possible BIO labels for this entity map:")
    bio_labels = get_bio_labels(entities)
    print(f"  {bio_labels}")


def possessive_demo():
    """Demo showing possessive handling."""
    print("\n\n👑 Possessive Handling Demo")
    print("=" * 50)
    
    entities = {
        "Bilbo Baggins": "Person",
        "Frodo Baggins": "Person"
    }
    
    test_cases = [
        "Bilbo Baggins's house is cozy.",
        "This is Frodo's ring.",
        "The Bagginses' home is in the Shire.",
        "Bilbo and Frodo's adventure started here."
    ]
    
    for text in test_cases:
        print(f"\n📝 Text: {text}")
        found_entities, _ = runer(text, entities)
        
        if found_entities:
            for entity in found_entities:
                print(f"  ✅ Found: '{entity.text}' → {entity.label}")
        else:
            print("  ❌ No entities found")


def case_insensitive_demo():
    """Demo showing case insensitive matching."""
    print("\n\n🔤 Case Insensitive Demo")
    print("=" * 50)
    
    entities = {"john smith": "Person"}
    
    test_cases = [
        "John Smith works here.",
        "JOHN SMITH is the manager.", 
        "john smith likes coffee.",
        "Meet John SMITH tomorrow."
    ]
    
    for text in test_cases:
        print(f"\n📝 Text: {text}")
        found_entities, _ = runer(text, entities)
        
        for entity in found_entities:
            print(f"  ✅ Found: '{entity.text}' → {entity.label}")


def overlapping_entities_demo():
    """Demo showing how overlapping entities are handled."""
    print("\n\n🔗 Overlapping Entities Demo")
    print("=" * 50)
    
    entities = {
        "John": "Person",
        "John Smith": "Person", 
        "Smith Corporation": "Organization"
    }
    
    text = "John Smith founded Smith Corporation."
    
    print(f"📝 Text: {text}")
    print(f"🎯 Entity Map: {entities}")
    
    found_entities, bio_tagged = runer(text, entities)
    
    print(f"\n✨ Found entities (longest matches preferred):")
    for entity in found_entities:
        print(f"  • '{entity.text}' → {entity.label}")
    
    print(f"\n🏷️  BIO Tags:")
    for token, tag in zip(bio_tagged.tokens, bio_tagged.tags):
        print(f"  {token:20} → {tag}")


def real_world_demo():
    """Demo with a more realistic example."""
    print("\n\n🌍 Real World Demo")
    print("=" * 50)
    
    # Example: Processing a news article
    entities = {
        "Joe Biden": "Person",
        "Kamala Harris": "Person", 
        "White House": "Location",
        "United States": "Country",
        "Congress": "Organization",
        "Supreme Court": "Organization"
    }
    
    text = """Joe Biden and Kamala Harris met at the White House yesterday.
              They discussed United States policy with Congress representatives.
              The Supreme Court's recent decision was also reviewed."""
    
    print(f"📰 News Text: {text}")
    print(f"🎯 Political Entities: {list(entities.keys())}")
    
    found_entities, bio_tagged = runer(text, entities)
    
    print(f"\n✨ Recognized Entities:")
    by_type = {}
    for entity in found_entities:
        if entity.label not in by_type:
            by_type[entity.label] = []
        by_type[entity.label].append(entity.text)
    
    for entity_type, texts in by_type.items():
        print(f"  {entity_type}: {texts}")


def main():
    """Run all demos."""
    print("🎭 ruNER Demonstration")
    print("Really Uncomplicated Named Entity Recognition")
    print("=" * 60)
    
    basic_demo()
    possessive_demo() 
    case_insensitive_demo()
    overlapping_entities_demo()
    real_world_demo()
    
    print("\n\n🎉 Demo Complete!")
    print("ruNER makes NER really uncomplicated - no complex setup needed!")


if __name__ == "__main__":
    main()