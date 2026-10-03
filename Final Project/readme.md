# KisanAI — AI-Powered Crop Disease Detection & Agricultural Assistant

KisanAI is an AI-powered agricultural assistant designed to help farmers identify **tomato and potato leaf diseases** and obtain relevant agricultural care and management information.

The project combines **Computer Vision, Transfer Learning, MobileNetV2, ResNet50, NLP, Retrieval-Augmented Generation (RAG), and Generative AI** into a single application.

---

## Project Overview

KisanAI consists of two major AI components:

### 1. Image-Based Crop Disease Detection

KisanAI uses a **two-stage computer vision pipeline**.

First, a **MobileNetV2** model validates whether the uploaded image contains a plant leaf.

If the image is classified as a valid leaf, it is passed to the **ResNet50 disease classification model**.

```text
                    Uploaded Image
                          │
                          ▼
                    MobileNetV2
                   Leaf Validator
                          │
              ┌───────────┴───────────┐
              │                       │
          Non-Leaf                   Leaf
              │                       │
              ▼                       ▼
        Reject Image              ResNet50
                                      │
                                      ▼
                              Disease Prediction
```

This additional validation stage helps prevent unrelated images, screenshots, or other non-leaf images from being incorrectly classified as crop diseases.

### 2. Agricultural Question Answering

KisanAI also allows users to ask agricultural questions.

The system processes the question using NLP, retrieves relevant information from an agricultural knowledge base, and uses Gemini to generate a context-aware response.

```text
Farmer Question
       │
       ▼
NLP Processing
       │
       ├── Intent
       ├── Crop
       └── Disease
       │
       ▼
Intent-Aware RAG Retrieval
       │
       ▼
Agricultural Knowledge Base
       │
       ▼
Relevant Context
       │
       ▼
Gemini
       │
       ▼
Agricultural Response
```

---

# Key Features

* Tomato and potato disease detection
* Leaf/non-leaf image validation
* MobileNetV2-based leaf validation
* ResNet50-based disease classification
* Transfer learning
* NLP-based intent extraction
* Crop and disease extraction
* Intent-aware RAG retrieval
* Agricultural knowledge base
* Source organization, title, and URL metadata
* Gemini-powered agricultural responses
* Streamlit-based user interface
* Image preprocessing and validation
* Confidence-based leaf validation

---

# Computer Vision Pipeline

KisanAI uses two separate deep learning models with different responsibilities.

## Stage 1 — MobileNetV2 Leaf Validation

The first model is **MobileNetV2**.

Its purpose is not to identify the disease.

Instead, it answers:

> **"Is this image a valid plant leaf image?"**

This is important because a disease classification model can produce a prediction even when an unrelated image is provided.

For example:

```text
Ubuntu Screenshot
       │
       ▼
MobileNetV2
       │
       ▼
Not a Leaf
       │
       ▼
Image Rejected
```

For a valid leaf:

```text
Tomato Leaf
       │
       ▼
MobileNetV2
       │
       ▼
Leaf Detected
       │
       ▼
ResNet50
       │
       ▼
Tomato Disease Prediction
```

---

## MobileNetV2 Leaf Validator

The current implementation uses a confidence threshold of **0.80**.

A prediction is considered a valid leaf when:

```text
Leaf Probability >= 0.80
```

### Implementation

```python
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


class LeafValidator:
    def __init__(
        self,
        model_path="model/kisanai_leaf_validator.keras",
        threshold=0.80
    ):
        self.threshold = threshold
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = (224, 224)

    def preprocess_image(
        self,
        image: Image.Image
    ) -> np.ndarray:

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Center-crop to minimize background distraction
        image = ImageOps.fit(
            image,
            self.img_size,
            Image.Resampling.LANCZOS
        )

        img_array = tf.keras.utils.img_to_array(image)

        return np.expand_dims(img_array, axis=0)

    def validate(self, image: Image.Image):

        processed_input = self.preprocess_image(image)

        raw_pred = self.model.predict(
            processed_input,
            verbose=0
        )

        # Calculate leaf probability based on binary output shape
        leaf_prob = (
            float(1.0 - raw_pred[0][0])
            if raw_pred.shape[-1] == 1
            else float(raw_pred[0][0])
        )

        is_valid_leaf = leaf_prob >= self.threshold

        return is_valid_leaf, leaf_prob
```

### Leaf Validation Logic

The validator performs the following steps:

```text
Input Image
     │
     ▼
Convert to RGB
     │
     ▼
Resize / Center Crop
     │
     ▼
224 × 224
     │
     ▼
MobileNetV2
     │
     ▼
Leaf Probability
     │
     ▼
Probability >= 0.80?
     │
 ┌───┴────┐
 │        │
Yes       No
 │        │
 ▼        ▼
Leaf    Non-Leaf
 │        │
 ▼        ▼
ResNet50 Reject
```

---

# Stage 2 — ResNet50 Disease Classification

After an image passes the leaf validation stage, it is passed to the **ResNet50** disease classification model.

The ResNet50 model uses **transfer learning** with pretrained ImageNet weights.

### Model Configuration

```text
Architecture: ResNet50
Input Size: 224 × 224 × 3
Number of Classes: 7
Training Method: Transfer Learning
```

The model classifies the leaf into one of seven supported classes.

---

# Supported Disease Classes

## Tomato

* Tomato — Healthy
* Tomato — Early Blight
* Tomato — Late Blight
* Tomato — Bacterial Spot

## Potato

* Potato — Healthy
* Potato — Early Blight
* Potato — Late Blight

---

# Dataset

The disease classification model was trained using a selected subset of the **PlantVillage dataset** containing tomato and potato leaf images.

## Dataset Statistics

| Class                   |    Images |
| ----------------------- | --------: |
| Tomato — Healthy        |     1,591 |
| Tomato — Early Blight   |     1,000 |
| Tomato — Late Blight    |     1,909 |
| Tomato — Bacterial Spot |     2,127 |
| Potato — Healthy        |       152 |
| Potato — Early Blight   |     1,000 |
| Potato — Late Blight    |     1,000 |
| **Total**               | **8,779** |

## Dataset Split

| Dataset    |    Images |
| ---------- | --------: |
| Training   |     6,143 |
| Validation |     1,315 |
| Testing    |     1,321 |
| **Total**  | **8,779** |

---

# ResNet50 Model Performance

The current ResNet50 disease classification model achieved approximately:

```text
Test Accuracy: 95.76%
```

This result represents performance on the project's test set.

> **Important:** Test-set accuracy does not guarantee the same performance on real-world field images. Real agricultural images can contain different lighting conditions, backgrounds, leaf orientations, disease stages, and image quality.

---

# Why Two Models?

Using two models separates two different computer vision tasks.

### MobileNetV2

**Task:**

```text
Is this a leaf?
```

### ResNet50

**Task:**

```text
If it is a leaf, what disease/class does it belong to?
```

Therefore:

```text
                 Image
                   │
                   ▼
             MobileNetV2
             Leaf Validation
                   │
          ┌────────┴────────┐
          │                 │
       Non-Leaf             Leaf
          │                 │
          ▼                 ▼
       Reject            ResNet50
                             │
                             ▼
                     Disease Prediction
```

This architecture creates a validation layer before disease classification.

---

# NLP and RAG System

The second major component of KisanAI is an agricultural question-answering system.

Users can ask questions such as:

```text
How can I manage tomato early blight?
```

The system extracts relevant information from the question.

### NLP Processing

The NLP component identifies:

* Intent
* Crop
* Disease

For example:

```text
Question:
How can I manage tomato early blight?

Intent:
Management

Crop:
Tomato

Disease:
Early Blight
```

This structured information is then used by the retrieval system.

---

# Retrieval-Augmented Generation

KisanAI uses **Retrieval-Augmented Generation (RAG)** to provide the language model with relevant agricultural information.

The process is:

```text
User Question
      │
      ▼
NLP
      │
      ▼
Intent / Crop / Disease
      │
      ▼
Intent-Aware Retrieval
      │
      ▼
Agricultural Knowledge Base
      │
      ▼
Relevant Documents
      │
      ▼
Gemini
      │
      ▼
Final Response
```

The agricultural knowledge base includes source metadata such as:

* Source organization
* Document title
* Source URL

This allows the retrieved information to maintain its source context.

---

# Gemini Integration

KisanAI uses **Gemini 3.6 Flash through the Interactions API** to generate responses based on the retrieved agricultural context.

The language model is not used as the disease classifier.

Instead:

```text
ResNet50
   │
   └── Disease Prediction

RAG + Gemini
   │
   └── Agricultural Question Answering
```

This separation keeps the computer vision and agricultural question-answering components as distinct parts of the system.

---

# Complete KisanAI Architecture

```text
                         KISANAI
                            │
              ┌─────────────┴─────────────┐
              │                           │
         IMAGE INPUT                FARMER QUESTION
              │                           │
              ▼                           ▼
        MobileNetV2                     NLP
       Leaf Validation                   │
              │                   ┌───────┼───────┐
        ┌─────┴─────┐             │       │       │
        │           │          Intent    Crop   Disease
     Non-Leaf      Leaf             │       │       │
        │           │               └───────┼───────┘
        ▼           ▼                       │
     Reject      ResNet50                   ▼
                    │                 RAG Retrieval
                    │                       │
                    ▼                       ▼
             Disease Prediction      Knowledge Base
                                            │
                                            ▼
                                         Gemini
                                            │
                                            ▼
                                   Agricultural Advice
```

---

# Technologies Used

## Machine Learning

* Python
* TensorFlow
* Keras
* MobileNetV2
* ResNet50
* Transfer Learning
* NumPy
* Pandas
* Scikit-learn

## Natural Language Processing

* NLP
* Intent extraction
* Crop extraction
* Disease extraction
* Retrieval-Augmented Generation

## Generative AI

* Gemini
* Gemini Interactions API

## Application

* Streamlit

## Development Environment

* Google Colab
* Google Drive

---

# Project Structure

A possible project structure is:

```text
KisanAI/
│
├── model/
│   ├── kisanai_leaf_validator.keras
│   └── kisanai_resnet50_head.h5
│
├── app/
│   └── App.py
│
├── rag/
│   ├── retrieval.py
│   └── knowledge_base.py
│
├── data/
│   └── knowledge_base/
│
├── notebooks/
│   ├── leaf_validator_training.ipynb
│   └── disease_model_training.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

> The exact folder structure may differ depending on the final version of the project.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/KisanAI.git
cd KisanAI
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

KisanAI requires a Gemini API key for the generative AI component.

Create a `.env` file:

```text
GEMINI_API_KEY=your_api_key_here
```

Never upload your API key to GitHub.

Add the following to `.gitignore`:

```text
.env
```

---

# Running the Application

Run the Streamlit application:

```bash
streamlit run App.py
```

The application will open in your browser.

---

# Example Image Workflow

```text
1. User uploads an image
          ↓
2. MobileNetV2 validates the image
          ↓
3. If it is not a leaf → reject
          ↓
4. If it is a leaf → send to ResNet50
          ↓
5. ResNet50 predicts the crop/disease class
          ↓
6. Prediction is displayed
```

---

# Example Question Workflow

```text
1. Farmer enters a question
          ↓
2. NLP processes the question
          ↓
3. Intent, crop and disease are identified
          ↓
4. Relevant information is retrieved
          ↓
5. Retrieved context is passed to Gemini
          ↓
6. Gemini generates the agricultural response
```

---

# Limitations

KisanAI is currently an **educational and research-oriented prototype**.

Current limitations include:

* Disease detection is limited to tomato and potato.
* Only seven disease/healthy classes are supported.
* The disease classifier was trained primarily on controlled leaf-image data.
* Real-world field images may differ significantly from the training dataset.
* Poor lighting and complex backgrounds may affect predictions.
* Disease symptoms can vary according to disease stage and environmental conditions.
* MobileNetV2 leaf validation is dependent on the quality and diversity of its training data.
* The RAG knowledge base is limited to the agricultural sources incorporated into the project.
* Agricultural recommendations should be independently verified before important farming decisions are made.

---

# Future Improvements

Possible future improvements include:

* Expand disease detection to additional crops.
* Add more real-world field images.
* Improve leaf validation using a larger and more diverse dataset.
* Improve robustness against complex backgrounds.
* Add additional disease classes.
* Add multilingual support.
* Add Urdu-language interaction.
* Improve RAG retrieval accuracy.
* Expand the agricultural knowledge base.
* Add disease severity estimation.
* Provide confidence scores and explanations.
* Deploy the application publicly.
* Collect real-world user feedback.
* Evaluate the models on field-collected agricultural images.

---

# Learning Objectives

This project demonstrates practical implementation of:

* Data preprocessing
* Image classification
* Convolutional Neural Networks
* Transfer Learning
* MobileNetV2
* ResNet50
* Model evaluation
* Binary classification
* Multiclass classification
* NLP
* Intent extraction
* Information retrieval
* Retrieval-Augmented Generation
* Large Language Models
* Generative AI
* API integration
* Streamlit application development

---

# Project Status

**Status: In Development**

The core components of KisanAI have been implemented:

* MobileNetV2 leaf validation
* ResNet50 disease classification
* NLP processing
* Intent-aware RAG retrieval
* Agricultural knowledge base
* Gemini-based response generation
* Streamlit application

Further development is focused on improving robustness, usability, and real-world agricultural applicability.

---

# Disclaimer

KisanAI is an educational and research-oriented project.

The disease predictions and agricultural information provided by the system should not be considered a substitute for professional agricultural diagnosis or consultation.

Users should independently verify important agricultural decisions with qualified agricultural professionals or reliable agricultural sources.

---

# Author

**Danyal Ahmad**

Machine Learning / AI Student

---

# Acknowledgements

* PlantVillage dataset
* TensorFlow and Keras
* MobileNetV2
* ResNet50
* ImageNet pretrained models
* Google Gemini
* Agricultural information sources used in the knowledge base
