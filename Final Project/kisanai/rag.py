from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from kisanai.knowledge_base import KNOWLEDGE_BASE

def kb_entry_to_text(entry):
    return (
        f"{entry['title']}. {entry['description']} "
        f"Symptoms: {entry['symptoms']} "
        f"Causes: {entry['causes']} "
        f"Prevention: {entry['prevention']} "
        f"Management: {entry['management']}"
    )

kb_texts = [kb_entry_to_text(entry) for entry in KNOWLEDGE_BASE]
vectorizer = TfidfVectorizer(stop_words='english')
kb_vectors = vectorizer.fit_transform(kb_texts)

INTENT_TO_CATEGORIES = {
    'watering': ['watering'],
    'fertilizer': ['fertilizer'],
    'symptoms': ['disease'],
    'prevention': ['disease', 'crop_care'],
    'treatment_management': ['disease'],
    'general_advice': ['crop_care', 'healthy']
}

def retrieve_knowledge_v2(nlp_result, top_k=2, verbose_scores=True):
    crop = nlp_result['crop']
    disease = nlp_result['disease']
    intents = nlp_result['intents']
    question_text = nlp_result['cleaned_question']

    primary_intent = intents[0]
    allowed_categories = set(INTENT_TO_CATEGORIES.get(primary_intent, ['crop_care']))

    if disease is not None:
        allowed_categories.add('disease')

    candidates = [e for e in KNOWLEDGE_BASE if e['category'] in allowed_categories]
    if not candidates:
        candidates = KNOWLEDGE_BASE

    question_vector = vectorizer.transform([question_text])
    scored_candidates = []

    for entry in candidates:
        score = 0.0

        if crop is not None:
            if entry['crop'] == crop:
                score += 5
            elif entry['crop'] == 'general':
                score += 2
            else:
                score -= 3
        else:
            if entry['crop'] == 'general':
                score += 1

        if disease is not None and entry['category'] == 'disease':
            if entry['disease'] == disease:
                score += 8
            else:
                score -= 5

        if entry['category'] in INTENT_TO_CATEGORIES.get(primary_intent, []):
            score += 3

        entry_text = kb_entry_to_text(entry)
        entry_vector = vectorizer.transform([entry_text])
        similarity = cosine_similarity(question_vector, entry_vector)[0][0]
        score += similarity * 2

        scored_candidates.append((entry, score))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    if verbose_scores:
        return scored_candidates[:top_k]
    else:
        return [entry for entry, score in scored_candidates[:top_k]]

# --- ADDED BRIDGE FUNCTION TO FIX APP.PY IMPORT ERROR ---
from kisanai.nlp import process_question

def retrieve_context(query, crop, disease):
    """
    Acts as a bridge between App.py and retrieve_knowledge_v2.
    It takes the image model's confirmed crop/disease and uses it 
    to boost the RAG search accuracy.
    """
    # 1. Process the raw text to find the farmer's intent
    nlp_result = process_question(query)
    
    # 2. Override text-guesses with the highly accurate image classifications
    nlp_result['crop'] = crop
    nlp_result['disease'] = disease
    
    # 3. Fetch context using your original v2 algorithm (returning pure text entries)
    return retrieve_knowledge_v2(nlp_result, top_k=2, verbose_scores=False)