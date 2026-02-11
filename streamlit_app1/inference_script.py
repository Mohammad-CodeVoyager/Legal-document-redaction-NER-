"""
Inference module for legal document redaction using BERT-NER

This module provides the redaction functionality for the Streamlit app. It loads a Hugging Face NER model and exposes functions to redact sensitive entities from text.
"""
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import os

# Model configuration
HF_MODEL_NAME = "sayeedlearnstech/LDR-NER"
model_path = "ldr_ner_model"

# Entity types that can be redacted
REDACTED_ENTITIES = {
    "PER": "Person Names",
    "ORG": "Organizations",
    "LOC": "Locations",
    "CODE": "Code",
    "DATETIME": "Date/Time",
    "DEM": "Demographic Info",
    "MISC": "Miscellaneous",
    "QUANTITY": "Quantities"
}

def download_model():
    """
    Download the model and tokenizer from Hugging Face if not available locally.
    Returns:
        True if download and save successful, False otherwise.
    """
    print(f"Model not found locally. Downloading from Hugging Face...")
    print(f"Model: {HF_MODEL_NAME}")
    print("This may take a few minutes...")
    try:
        os.makedirs(model_path, exist_ok=True)
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        print("Downloading model...")
        model = AutoModelForTokenClassification.from_pretrained(HF_MODEL_NAME)
        print(f"Saving to {model_path}...")
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        print(f"✅ Model successfully downloaded and saved!")
        return True
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

# Initialize the NER pipeline
my_ner = None
try:
    # Try to load from local folder first
    if not os.path.exists(model_path):
        print(f"Local model not found at '{model_path}'")
        if download_model():
            print("Attempting to load downloaded model...")
        else:
            raise Exception("Failed to download model")
    
    my_ner = pipeline(
        "token-classification",
        model=model_path,
        tokenizer=model_path,
        aggregation_strategy="simple"
    )
    print(f"✅ Model loaded successfully from '{model_path}'")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    my_ner = None


def redact_text(text, selected_entities=None):
    """
    Redact sensitive information from text using the NER model.
    Redacts: CODE, DATETIME, DEM, LOC, MISC, ORG, PER, QUANTITY, and O (other tags)

    Args:
        text (str): Input text to redact
        selected_entities (list, optional): List of entity types to redact (e.g., ["PER", "LOC", "ORG"]).
            If None, redacts all entity types.

    Returns:
        str: Redacted text with sensitive info replaced by black lines (█)
    """
    if my_ner is None:
        raise Exception("NER model not loaded. Please check the model path.")
    # Default to all entities if none specified
    if selected_entities is None:
        selected_entities = list(REDACTED_ENTITIES.keys())
    entities = my_ner(text)
    entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    for item in entities:
        if item['entity_group'] in selected_entities:
            start = item['start']
            end = item['end']
            redacted_text = text[start:end]
            # Replace with black line (█) maintaining text length
            text = text[:start] + "█" * len(redacted_text) + text[end:]
    return text


def get_ner_pipeline():
    """
    Get the NER pipeline for advanced usage.
    Returns:
        transformers.Pipeline: The loaded NER pipeline, or None if not loaded.
    """
    return my_ner
