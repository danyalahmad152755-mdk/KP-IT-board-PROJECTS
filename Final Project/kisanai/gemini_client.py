import time
import streamlit as st
from google import genai


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# BUILD GEMINI PROMPT
# ============================================================

def build_gemini_prompt(
    farmer_question,
    nlp_result,
    retrieved_entries,
    disease_prediction=None,
    confidence=None
):

    context_blocks = []

    for entry in retrieved_entries:

        block = (
            f"- Title: {entry['title']}\n"
            f"  Category: {entry['category']}\n"
            f"  Description: {entry['description']}\n"
            f"  Symptoms: {entry['symptoms']}\n"
            f"  Causes: {entry['causes']}\n"
            f"  Prevention: {entry['prevention']}\n"
            f"  Management: {entry['management']}\n"
            f"  Source: {entry['source']['organization']} - "
            f"\"{entry['source']['title']}\" "
            f"({entry['source']['url']})"
        )

        context_blocks.append(block)

    if context_blocks:

        rag_context_text = "\n\n".join(context_blocks)

    else:

        rag_context_text = (
            "No relevant knowledge base entries were retrieved."
        )


    # ========================================================
    # IMAGE PREDICTION
    # ========================================================

    if disease_prediction is not None and confidence is not None:

        prediction_text = (
            f"Predicted class: {disease_prediction}\n"
            f"Confidence: {confidence:.2f}%"
        )

    else:

        prediction_text = (
            "No image was provided for this question."
        )


    # ========================================================
    # GEMINI PROMPT
    # ========================================================

    prompt = f"""
You are KisanAI, an agricultural advisory assistant.

Your job is to help farmers understand crop diseases and
provide practical agricultural advice.

IMPORTANT RULES:

1. Use the Retrieved Knowledge Base Context as the primary
   factual source for disease-specific advice.

2. Do NOT invent pesticide names, doses, chemical
   concentrations, or application frequencies.

3. If the knowledge base does not provide a specific
   treatment or pesticide recommendation, tell the farmer
   to follow locally approved agricultural labels or consult
   a local agricultural extension expert.

4. Do NOT invent agricultural facts that are not supported
   by the retrieved knowledge.

5. If the retrieved knowledge is insufficient to safely
   answer the question, clearly say so.

6. Clearly distinguish between:
   - Image model prediction
   - Farmer's own description
   - Knowledge-base information

7. The image prediction is a MODEL ESTIMATE and NOT a
   laboratory-confirmed diagnosis.

8. If confidence is below 80%, clearly mention that the
   prediction is uncertain.

9. If the model predicts "healthy" but the farmer believes
   the plant has a disease, do NOT claim that the plant has
   that disease.

10. Keep the answer simple, practical, and farmer-friendly.

11. Preserve the provided sources and URLs.

12. Do not create fake sources.

------------------------------------------------------------
FARMER'S QUESTION
------------------------------------------------------------

{farmer_question}


------------------------------------------------------------
DETECTED INFORMATION
------------------------------------------------------------

Crop:
{nlp_result.get('crop')}

Disease mentioned by farmer:
{nlp_result.get('disease')}

Symptoms:
{nlp_result.get('symptoms')}

Intent:
{nlp_result.get('intents')}


------------------------------------------------------------
IMAGE MODEL PREDICTION
------------------------------------------------------------

{prediction_text}


------------------------------------------------------------
RETRIEVED KNOWLEDGE BASE
------------------------------------------------------------

{rag_context_text}


------------------------------------------------------------
RESPONSE FORMAT
------------------------------------------------------------

Use exactly these sections:

**Assessment**

Briefly explain what the image model predicted and/or
what the farmer described.

If an image prediction exists, clearly state that it is
a model estimate.

**What you should do**

Give practical next steps based on the retrieved
knowledge.

**Prevention / management**

Give prevention and management recommendations supported
by the retrieved knowledge.

**Important note**

Mention uncertainty, limitations, or when the farmer
should consult an agricultural expert.

**Sources**

List the sources from the retrieved knowledge base.

Do not invent additional sources.
"""

    return prompt


# ============================================================
# GENERATE GEMINI RESPONSE
# ============================================================

def generate_gemini_response(
    farmer_question,
    nlp_result,
    retrieved_entries,
    disease_prediction=None,
    confidence=None,
    model_name="gemini-3.6-flash",
    max_retries=2,
    retry_delay=5
):

    # --------------------------------------------------------
    # Clean RAG results
    # --------------------------------------------------------

    clean_entries = []

    for item in retrieved_entries:

        if isinstance(item, tuple):

            clean_entries.append(item[0])

        else:

            clean_entries.append(item)


    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = build_gemini_prompt(
        farmer_question=farmer_question,
        nlp_result=nlp_result,
        retrieved_entries=clean_entries,
        disease_prediction=disease_prediction,
        confidence=confidence
    )


    # --------------------------------------------------------
    # Gemini Interactions API
    # --------------------------------------------------------

    for attempt in range(max_retries):

        try:

            interaction = client.interactions.create(
                model=model_name,
                input=prompt
            )

            if hasattr(interaction, "output_text"):

                return interaction.output_text

            return str(interaction)


        except Exception as e:

            if attempt < max_retries - 1:

                time.sleep(retry_delay)

                continue


            return (
                f"Gemini API Error\n\n"
                f"Error Type: {type(e).__name__}\n\n"
                f"Error Details: {str(e)}"
            )


    return "Gemini API request failed."


# ============================================================
# BRIDGE FUNCTION
# ============================================================

def get_agri_advice(
    query,
    diagnosis=None,
    nlp_info=None,
    retrieved_context=None,
    api_key=None
):

    if nlp_info is None:

        nlp_info = {}


    if retrieved_context is None:

        retrieved_context = []


    disease_pred = None
    conf = None


    if diagnosis:

        disease_pred = (
            f"{diagnosis.get('crop')} - "
            f"{diagnosis.get('disease')}"
        )

        raw_conf = diagnosis.get(
            'confidence',
            0.0
        )

        if raw_conf <= 1.0:

            conf = raw_conf * 100

        else:

            conf = raw_conf


    return generate_gemini_response(

        farmer_question=query,

        nlp_result=nlp_info,

        retrieved_entries=retrieved_context,

        disease_prediction=disease_pred,

        confidence=conf
    )