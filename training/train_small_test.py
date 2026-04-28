"""
==========================================================
  ProjectPothole — Quick Test on Pothole.v1 Dataset
==========================================================
Run this in Google Colab to test ONLY the new smaller dataset.
It only takes a few minutes because the dataset is tiny (48MB).
"""

# !pip install ultralytics --quiet

import os
import yaml
from ultralytics import YOLO
import zipfile

# --- 1. Configuration ---
ZIP_PATH = "/content/Pothole.v1-raw.yolov8.zip"
DATASET_DIR = "/content/pothole_v1_dataset"

# For a "quick guess", we use Nano and 50 epochs.
MODEL_SIZE = "yolov8n.pt"  
EPOCHS = 50                

# --- 2. Unzip Dataset ---
print("=" * 50)
print("1. Extracting small dataset...")
os.makedirs(DATASET_DIR, exist_ok=True)
if os.path.exists(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)
    print("✅ Extracted!")
else:
    raise FileNotFoundError(f"❌ Cannot find {ZIP_PATH}. Did you upload it to Colab?")

# --- 3. Fix Roboflow paths in data.yaml ---
print("2. Patching data.yaml paths...")
yaml_path = os.path.join(DATASET_DIR, "data.yaml")

with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

# Roboflow sometimes uses relative paths. We force absolute paths for Colab.
data['train'] = os.path.join(DATASET_DIR, "train", "images")
data['val'] = os.path.join(DATASET_DIR, "valid", "images")  # Roboflow usually uses 'valid'
if 'test' in data:
    data['test'] = os.path.join(DATASET_DIR, "test", "images")

with open(yaml_path, 'w') as f:
    yaml.dump(data, f, default_flow_style=False)
print("✅ Paths patched!")

# --- 4. Train! ---
print("=" * 50)
print("3. Starting YOLOv8 Quick Test...")
print("=" * 50)

model = YOLO(MODEL_SIZE)

results = model.train(
    data=yaml_path,
    epochs=EPOCHS,
    imgsz=640,
    batch=16,
    project="/content/quick_test",
    name="run_1",
    plots=True
)

# --- 5. Evaluate ---
print("\n" + "=" * 50)
print("4. Final Accuracy on Small Dataset:")
metrics = model.val()
print(f"   mAP50:    {metrics.box.map50:.4f}")
print(f"   Precision: {metrics.box.mp:.4f}")
print(f"   Recall:    {metrics.box.mr:.4f}")
print("=" * 50)
print("If the mAP50 is high, this dataset is pure gold and we should merge it!")
