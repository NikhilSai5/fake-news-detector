import os
import json
import uuid
import re
from datetime import datetime

import spacy
from sentence_transformers import CrossEncoder

import dateparser


# ---------------------------------------------
# Load Models
# ---------------------------------------------
nlp = spacy.load("en_core_web_sm")
nli_model = CrossEncoder("cross-encoder/nli-roberta-base")


# ========================================================
# 1. Sentence Splitting
# ========================================================
def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


# ========================================================
# 2. Filter Claim-like Sentences
# ========================================================
def is_claim_like(sentence):
    if sentence.endswith("?"):
        return False

    if len(sentence.split()) < 4:
        return False

    subjective_markers = [
        "i think", "in my opinion", "i believe",
        "experts believe", "some people think",
        "people believe"
    ]

    if any(x in sentence.lower() for x in subjective_markers):
        return False

    return True


# ========================================================
# 3. NLI-based Factuality Check
# ========================================================
def nli_is_claim(sentence):
    hypothesis = "This sentence makes a factual claim."
    pred = nli_model.predict([(sentence, hypothesis)])

    # pred shape: (1, 3) → [contradiction, neutral, entailment]
    label = int(pred.argmax())

    return label == 2  # entailment = factual claim


# ========================================================
# 4. Normalize Dates
# ========================================================
def normalize_dates(text, reference_date):
    dt = dateparser.parse(text, settings={"RELATIVE_BASE": reference_date})
    if dt:
        return [dt.strftime("%Y-%m-%d")]
    return []


# ========================================================
# 5. Normalize Numbers
# ========================================================
import re

def normalize_numbers(text: str):
    """
    Lightweight numeric extractor that works on ANY Python version.
    Extracts integers, floats, percentages, and basic units.
    """

    numbers = []

    # Find numbers like 8, 8.3, 1000, 1.5%, 12%, 2.4kg etc.
    pattern = r'(\d+(?:\.\d+)?)(\s?(?:%|percent|kg|km|m|billion|million|thousand)?)'
    
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    for val, unit in matches:
        try:
            numbers.append({
                "value": float(val),
                "unit": unit.strip() if unit else ""
            })
        except:
            pass

    return numbers


# ========================================================
# 6. Extract Entities (NO Linking)
# ========================================================
def extract_entities(text):
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        ents.append({
            "text": ent.text,
            "label": ent.label_
        })
    return ents


# ========================================================
# 7. Build Claim JSON Object
# ========================================================
def build_claim_obj(sentence, reference_date):
    return {
        "claim_text": sentence,
        "normalized_text": sentence,
        "numbers": normalize_numbers(sentence),
        "dates": normalize_dates(sentence, reference_date),
        "entities": extract_entities(sentence)
    }


# ========================================================
# 8. Save to JSON in extracted_claims/
# ========================================================
def save_claims(claims, folder="extracted_claims"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"claims_{uuid.uuid4().hex}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(claims, f, indent=2)

    return path


# ========================================================
# 9. FULL PIPELINE: Extract + Normalize + Save
# ========================================================
# def extract_and_normalize_claims(text, publish_date=None):
#     if not publish_date:
#         publish_date = datetime.now()

#     final_claims = []

#     sentences = split_sentences(text)

#     for sent in sentences:

#         # Pass basic filters
#         if not is_claim_like(sent):
#             continue

#         # NLI determines if it's factual
#         if not nli_is_claim(sent):
#             continue

#         # Use ORIGINAL full sentence as claim
#         claim_obj = build_claim_obj(sent, publish_date)
#         final_claims.append(claim_obj)

#     # Save JSON
#     path = save_claims(final_claims)

#     return final_claims, path


def extract_and_normalize_claims(text, publish_date=None):
    if not publish_date:
        publish_date = datetime.now()

    final_claims = []
    sentences = split_sentences(text)

    for sent in sentences:

        # HARD LIMIT
        if len(final_claims) >= 20:
            break

        # Pass basic filters
        if not is_claim_like(sent):
            continue

        # NLI determines if it's factual
        if not nli_is_claim(sent):
            continue

        claim_obj = build_claim_obj(sent, publish_date)
        final_claims.append(claim_obj)

    # Save JSON
    path = save_claims(final_claims)
    return final_claims, path
