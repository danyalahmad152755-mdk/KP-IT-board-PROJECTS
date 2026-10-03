import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

print("=" * 60)
print("KISANAI DIAGNOSTIC CHECK")
print("=" * 60)

# Detect script directory (c:/Users/ABC/OneDrive/Desktop/kisan AI)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

val_path = os.path.join(MODEL_DIR, "kisanai_leaf_validator.keras")
dis_path = os.path.join(MODEL_DIR, "kisanai_resnet50_head.h5")

print(f"Project Directory : {SCRIPT_DIR}")
print(f"Model Directory   : {MODEL_DIR}")
print("-" * 60)

# Check directory contents
if os.path.exists(MODEL_DIR):
    print(f"Files inside '{MODEL_DIR}':")
    for f in os.listdir(MODEL_DIR):
        print(f"  - {f}")
else:
    print(f"[CRITICAL] Folder does not exist: {MODEL_DIR}")

print("-" * 60)
print(f"Checking Validator Path: {'FOUND' if os.path.exists(val_path) else 'MISSING!'}")
print(f"Checking Disease Path  : {'FOUND' if os.path.exists(dis_path) else 'MISSING!'}")
print("-" * 60)

# 1. TEST VALIDATOR LOGIC
if os.path.exists(val_path):
    print("\n[INFO] Testing Validator Model...")
    val_model = tf.keras.models.load_model(val_path)
    
    # Test random noise (should be non-leaf)
    dummy_non_leaf = np.random.uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)
    raw_val_pred = float(val_model.predict(dummy_non_leaf, verbose=0)[0][0])
    
    # Check orientation: class 0 vs class 1
    print(f"Raw sigmoid output on random noise: {raw_val_pred:.4f}")
    leaf_confidence = 1.0 - raw_val_pred
    print(f"Computed Leaf Confidence          : {leaf_confidence*100:.2f}%")
    print(f"Decision                          : {'LEAF (Accepted)' if leaf_confidence >= 0.70 else 'NON-LEAF (Rejected)'}")
else:
    print("\n[WARNING] Skipping validator test because file was not found.")

# 2. TEST DISEASE MODEL
if os.path.exists(dis_path):
    print("\n[INFO] Testing Disease Model Preprocessing...")
    dis_model = tf.keras.models.load_model(dis_path)
    
    test_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
    test_img[:, :, :, 1] = 200.0  # Green synthetic patch
    
    pred_a = dis_model.predict(preprocess_input(test_img.copy()), verbose=0)[0]
    pred_b = dis_model.predict(test_img.copy() / 255.0, verbose=0)[0]
    pred_c = dis_model.predict(test_img.copy(), verbose=0)[0]

    print("\nPredictions under different input scales:")
    print(f" - Mode A (preprocess_input): Class {np.argmax(pred_a)} ({np.max(pred_a)*100:.2f}%)")
    print(f" - Mode B (rescaled / 255.0): Class {np.argmax(pred_b)} ({np.max(pred_b)*100:.2f}%)")
    print(f" - Mode C (raw 0-255)       : Class {np.argmax(pred_c)} ({np.max(pred_c)*100:.2f}%)")
else:
    print("\n[WARNING] Skipping disease model test because file was not found.")

print("=" * 60)