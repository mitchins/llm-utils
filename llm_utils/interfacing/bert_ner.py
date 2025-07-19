# RUNER - Really Uncomplicated Name Entity Recognizer

from enum import Enum
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Optional, Dict, Union
import numpy as np

class EntityType(Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    MISCELLANEOUS = "MISC"

class BERTSpanNER(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", entity_types: Optional[List[EntityType]] = None):
        super().__init__()
        
        if entity_types is None:
            entity_types = list(EntityType)
        
        self.entity_types = entity_types
        self.num_labels = len(entity_types) * 2 + 1  # B-TAG, I-TAG for each type + O
        
        # Create label mapping
        self.id2label = {0: "O"}
        self.label2id = {"O": 0}
        
        idx = 1
        for entity_type in entity_types:
            self.id2label[idx] = f"B-{entity_type.value}"
            self.id2label[idx + 1] = f"I-{entity_type.value}"
            self.label2id[f"B-{entity_type.value}"] = idx
            self.label2id[f"I-{entity_type.value}"] = idx + 1
            idx += 2
        
        # BERT backbone
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        
        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {"loss": loss, "logits": logits}

class SpanExtractor:
    def __init__(self, model: BERTSpanNER, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def extract_spans(self, text: str, return_confidence: bool = False) -> List[Dict[str, Union[str, int, float, None]]]:
        """Extract named entity spans from text"""
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop("offset_mapping")[0]
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs["logits"], dim=-1)[0]
            
        if return_confidence:
            probs = torch.softmax(outputs["logits"], dim=-1)[0]
            confidences = torch.max(probs, dim=-1)[0]
        
        # Convert predictions to spans
        spans = []
        current_span = None
        
        for i, (pred_id, offset) in enumerate(zip(predictions, offset_mapping)):
            label = self.model.id2label[int(pred_id.item())]
            start_char, end_char = offset.tolist()
            
            # Skip special tokens (offset is [0,0])
            if start_char == 0 and end_char == 0:
                continue
                
            if label.startswith("B-"):
                # Start new span
                if current_span:
                    spans.append(current_span)
                
                entity_type = label[2:]  # Remove "B-" prefix
                current_span = {
                    "text": text[start_char:end_char],
                    "start": start_char,
                    "end": end_char,
                    "label": entity_type,
                    "confidence": confidences[i].item() if return_confidence else None
                }
                
            elif label.startswith("I-") and current_span:
                # Continue current span
                entity_type = label[2:]  # Remove "I-" prefix
                if current_span["label"] == entity_type:
                    current_span["text"] = text[current_span["start"]:end_char]
                    current_span["end"] = end_char
                    if return_confidence:
                        # Average confidence across tokens in span
                        current_span["confidence"] = (current_span["confidence"] + confidences[i].item()) / 2
                else:
                    # Mismatched I- tag, end current span and start new one
                    spans.append(current_span)
                    current_span = {
                        "text": text[start_char:end_char],
                        "start": start_char,
                        "end": end_char,
                        "label": entity_type,
                        "confidence": confidences[i].item() if return_confidence else None
                    }
            else:
                # O tag - end current span if exists
                if current_span:
                    spans.append(current_span)
                    current_span = None
        
        # Don't forget the last span
        if current_span:
            spans.append(current_span)
            
        return spans

# Usage example
def create_ner_pipeline(entity_types: Optional[List[EntityType]] = None):
    """Create a ready-to-use NER pipeline"""
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BERTSpanNER(entity_types=entity_types)
    extractor = SpanExtractor(model, tokenizer)
    
    return extractor

# Example usage:
if __name__ == "__main__":
    # Create pipeline with custom entity types
    custom_entities = [EntityType.PERSON, EntityType.ORGANIZATION]
    ner = create_ner_pipeline(custom_entities)
    
    # Note: This is untrained - you'd need to train it first
    sample_text = "John Smith works at Microsoft in Seattle."
    spans = ner.extract_spans(sample_text, return_confidence=True)
    
    for span in spans:
        print(f"{span.get('text', '')!r} -> {span.get('label', '')} ({span.get('start', 0)}-{span.get('end', 0)})")