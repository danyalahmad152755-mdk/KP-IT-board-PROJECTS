# KisanAI: Intelligent Crop Disease Detection & Advisory System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

## Overview
This repository contains the core deep learning computer vision pipeline and backend integrations for **KisanAI**, a comprehensive agricultural advisory system submitted as the Final Project for the KP IT Board. 

The system bridges the gap between raw visual data and actionable farming advice. It utilizes a dual-model computer vision approach to diagnose Tomato and Potato leaf diseases, which seamlessly feeds into an advanced Natural Language Processing (NLP) pipeline to provide farmers with customized, natural language solutions.

## System Architecture & Tech Stack
KisanAI is designed as an end-to-end platform, built on a modern, multi-layered tech stack:
*   **Computer Vision Engine:** Convolutional Neural Networks built with TensorFlow and Keras to classify crop diseases from leaf images.
*   **NLP & Advisory Engine:** A Retrieval-Augmented Generation (RAG) context retrieval pipeline utilizing TF-IDF cosine similarity, powered by the Google Gemini API to generate accurate agricultural advice.
*   **User Interface:** An interactive, accessible web interface deployed via Streamlit.
*   **Database Management:** PostgreSQL for robust relational data handling, record insertion, and user management.

## Vision Model Architectures
To evaluate the trade-offs between computational efficiency and diagnostic accuracy, two distinct architectures were fine-tuned for the classification task:

1.  **MobileNetV2:** A lightweight, highly efficient architecture optimized for mobile and edge-device deployment, ensuring rapid inference without a significant drop in accuracy.
2.  **ResNet50:** A deeper, more complex residual network utilized to capture intricate feature representations and establish a high-accuracy baseline.

Both models process input images resized to `(224, 224, 3)` and output probabilities across 7 target classes.

## Dataset Details
The models were trained on a curated subset of the PlantVillage dataset. The data was strictly filtered to include only the necessary 7 classes relevant to the KisanAI scope, yielding a total of **8,779 images**.

To ensure robust model evaluation and prevent data leakage, the dataset was deterministically shuffled (Seed: 42) and split into Train (70%), Validation (15%), and Test (15%) sets.

| Class Name | Total Images | Train (70%) | Val (15%) | Test (15%) |
| :--- | :--- | :--- | :--- | :--- |
| Potato: Early Blight | 1,000 | 700 | 150 | 150 |
| Potato: Late Blight | 1,000 | 700 | 150 | 150 |
| Potato: Healthy | 152 | 106 | 22 | 24 |
| Tomato: Bacterial Spot | 2,127 | 1,488 | 319 | 320 |
| Tomato: Early Blight | 1,000 | 700 | 150 | 150 |
| Tomato: Late Blight | 1,909 | 1,336 | 286 | 287 |
| Tomato: Healthy | 1,591 | 1,113 | 238 | 240 |
| **TOTAL** | **8,779** | **6,143** | **1,315** | **1,321** |

## Setup & Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/danyalahmad152755-mdk/KP-IT-board-PROJECTS.git](https://github.com/danyalahmad152755-mdk/KP-IT-board-PROJECTS.git)[cite: 2]
cd "KP-IT-board-PROJECTS/Final Project"[cite: 2]
