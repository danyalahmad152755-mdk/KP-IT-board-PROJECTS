from kisanai.nlp import process_farmer_question
from kisanai.rag import retrieve_knowledge_v2
from kisanai.gemini_client import generate_gemini_response

def kisan_ai_full_pipeline(farmer_question, disease_prediction=None, confidence=None, top_k=2):
    nlp_result = process_farmer_question(farmer_question)
    ranked_results = retrieve_knowledge_v2(nlp_result, top_k=top_k, verbose_scores=True)

    final_answer = generate_gemini_response(
        farmer_question=farmer_question,
        nlp_result=nlp_result,
        retrieved_entries=ranked_results,
        disease_prediction=disease_prediction,
        confidence=confidence
    )

    return {
        'nlp_result': nlp_result,
        'retrieved_entries': ranked_results,
        'final_answer': final_answer
    }