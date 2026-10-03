"""
App.py
KisanAI Two-Stage Crop Advisory Web Application
"""

import streamlit as st
from PIL import Image, ImageOps
import os

from kisanai.validator import LeafValidator
from kisanai.classifier import DiseaseClassifier
from kisanai.nlp import process_farmer_question as process_question
from kisanai.rag import retrieve_context
from kisanai.gemini_client import get_agri_advice

st.set_page_config(
    page_title="KisanAI - Smart Crop Advisory",
    page_icon="🌱",
    layout="wide"
)

# Cache model weights in memory
@st.cache_resource
def load_all_models():
    validator = LeafValidator()
    classifier = DiseaseClassifier()
    return validator, classifier

try:
    validator, classifier = load_all_models()
except Exception as e:
    st.error(f"Error loading model weights: {str(e)}")
    st.stop()

st.title("🌱 KisanAI: Crop Disease Detection & Advisory")
st.markdown("Upload a potato or tomato leaf image to diagnose diseases and ask questions.")

col1, col2 = st.columns([1, 1], gap="medium")

# Track active diagnosis across the session
if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None

with col1:
    st.subheader("📷 Step 1: Leaf Image Diagnosis")
    uploaded_file = st.file_uploader(
        "Upload a clear leaf photo (Tomato or Potato)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with st.spinner("Validating image..."):
                is_valid_leaf, leaf_conf = validator.validate(image)

            if not is_valid_leaf:
                st.session_state.diagnosis_result = None
                st.error("❌ **Image Rejected**")
                st.warning(
                    f"This image does not appear to be a valid crop leaf (Confidence: {leaf_conf*100:.1f}%). "
                    "Please upload a clear, focused photograph of a tomato or potato leaf."
                )
            else:
                st.success(f"✅ **Valid Leaf Confirmed** ({leaf_conf*100:.1f}%)")
                
                with st.spinner("Diagnosing crop disease..."):
                    # Preprocess image to center-fit and crop out background distraction
                    target_size = (224, 224)
                    processed_image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
                    
                    crop, disease, conf = classifier.predict(processed_image)

                st.session_state.diagnosis_result = {
                    "crop": crop,
                    "disease": disease,
                    "confidence": conf
                }

                st.markdown("### Diagnosis Result")
                m1, m2 = st.columns(2)
                m1.metric(label="Target Crop", value=crop)
                m2.metric(label="Condition", value=disease)
                st.metric(label="Prediction Confidence", value=f"{conf * 100:.2f}%")

                # --- UPDATED THRESHOLD TO 80% ---
                if conf < 0.80:
                    st.info("ℹ️ The confidence score is below 80%. Consider retaking the photo in direct, even lighting.")

        except Exception as err:
            st.error(f"Could not process this file: {str(err)}")

with col2:
    st.subheader("💬 Step 2: Farmer Advisory Chat")
    
    # Display context banner if diagnosis exists
    if st.session_state.diagnosis_result:
        d = st.session_state.diagnosis_result
        st.info(f"🌿 Active diagnosis: **{d['crop']} - {d['disease']}** ({d['confidence']*100:.1f}%)")
    
    farmer_query = st.text_area(
        "Ask a question regarding symptoms, organic/chemical treatment, or prevention:",
        placeholder="e.g., What are the best organic treatments for late blight on my crop?",
        height=130
    )
    
    get_advice = st.button("Get Agricultural Advice", type="primary")

    if get_advice:
        if not farmer_query.strip():
            st.warning("Please type a question before asking for advice.")
        else:
            with st.spinner("Consulting agricultural knowledge base..."):
                nlp_info = process_question(farmer_query)
                
                # Prioritize confirmed image diagnosis over text-extracted guesses
                active_crop = st.session_state.diagnosis_result["crop"] if st.session_state.diagnosis_result else nlp_info.get("crop")
                active_disease = st.session_state.diagnosis_result["disease"] if st.session_state.diagnosis_result else nlp_info.get("disease")
                
                context_chunks = retrieve_context(
                    query=farmer_query,
                    crop=active_crop,
                    disease=active_disease
                )
                
                api_key = st.secrets.get("GEMINI_API_KEY")
                if not api_key:
                    st.error("API configuration error: GEMINI_API_KEY missing in `.streamlit/secrets.toml`.")
                else:
                    response_text = get_agri_advice(
                        query=farmer_query,
                        diagnosis=st.session_state.diagnosis_result,
                        nlp_info=nlp_info,
                        retrieved_context=context_chunks,
                        api_key=api_key
                    )
                    st.markdown("### 🚜 Advisory Recommendations")
                    st.write(response_text)