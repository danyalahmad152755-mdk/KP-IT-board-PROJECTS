import re
import string

def clean_text(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize(text):
    return text.split()

CROP_KEYWORDS = {
    'tomato': ['tomato', 'tomatoes'],
    'potato': ['potato', 'potatoes']
}

DISEASE_KEYWORDS = {
    'Early_blight': ['early blight', 'erly blight', 'earlyblight'],
    'Late_blight': ['late blight', 'lateblight'],
    'Bacterial_spot': ['bacterial spot', 'bacterial'],
    'healthy': ['healthy']
}

SYMPTOM_KEYWORDS = [
    'yellow leaves', 'black spots', 'brown spots',
    'lesions', 'wilting', 'yellowing', 'curling', 'mold', 'rot', 'holes',
    'spots'
]

INTENT_KEYWORDS = {
    'treatment_management': [
        'treat', 'treatment', 'cure', 'manage', 'management', 'control',
        'get rid of', 'fix', 'what should i do', 'what to do', 'how do i deal',
        'after detecting', 'what should i do if', 'what should i do after'
    ],
    'prevention': ['prevent', 'prevention', 'avoid', 'stop from', 'protect', 'next season', 'next time'],
    'symptoms': ['symptom', 'symptoms', 'signs', 'look like', 'identify'],
    'watering': ['water', 'watering', 'irrigate', 'irrigation'],
    'fertilizer': ['fertilizer', 'fertilize', 'nutrient', 'feed the plant'],
    'general_advice': ['advice', 'help', 'suggest', 'recommend']
}

def detect_crop(text):
    for crop, keywords in CROP_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return crop
    return None

def detect_disease(text):
    for disease, keywords in DISEASE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return disease
    return None

def detect_symptoms(text):
    found = []
    matched_text_spans = []
    for phrase in SYMPTOM_KEYWORDS:
        if phrase in text:
            if any(phrase in longer for longer in matched_text_spans):
                continue
            found.append(phrase)
            matched_text_spans.append(phrase)
    return found

def detect_intents(text, disease_detected):
    found_intents = []
    specific_intents = {k: v for k, v in INTENT_KEYWORDS.items() if k != 'general_advice'}
    for intent, keywords in specific_intents.items():
        for kw in keywords:
            if kw in text:
                found_intents.append(intent)
                break

    if found_intents:
        return list(dict.fromkeys(found_intents))

    general_matched = any(kw in text for kw in INTENT_KEYWORDS['general_advice'])

    if disease_detected is not None:
        return ['treatment_management']

    if general_matched:
        return ['general_advice']

    return ['general_advice']

def process_farmer_question(raw_question):
    cleaned = clean_text(raw_question)
    tokens = tokenize(cleaned)

    crop = detect_crop(cleaned)
    disease = detect_disease(cleaned)
    symptoms = detect_symptoms(cleaned)
    intents = detect_intents(cleaned, disease)

    return {
        'raw_question': raw_question,
        'cleaned_question': cleaned,
        'tokens': tokens,
        'crop': crop,
        'disease': disease,
        'symptoms': symptoms,
        'intents': intents
    }

# --- ADDED ALIAS TO FIX APP.PY IMPORT ERROR ---
process_question = process_farmer_question