"""
==========================================================
  ProjectPothole — YOLOv8 Training Resume Script for Colab
==========================================================

HOW TO USE IN GOOGLE COLAB:
  1. Open Google Colab (https://colab.research.google.com)
  2. Change runtime type to GPU (Runtime -> Change runtime type -> GPU)
  3. Mount Google Drive and make sure your dataset zip and last.pt checkpoint 
     are stored in Google Drive.
  4. Copy-paste this entire script into a Colab cell.
  5. Edit the paths in the CONFIGURATION section below to match your Drive paths.
  6. Run the cell to resume training from the checkpoint.
"""

# ============================================================
# STEP 0: Install dependencies (Uncomment if running in Colab)
# ============================================================
# !pip install ultralytics --quiet

import os
import shutil
import zipfile
import yaml
from ultralytics import YOLO

# ============================================================
# STEP 1: Configuration — EDIT THESE PATHS
# ============================================================

# 1. Dataset ZIP file on Google Drive
ZIP_PATH = "/content/drive/MyDrive/RDD2022.zip"

# 2. Path to the last.pt checkpoint on Google Drive
# Make sure to edit this path to match where your pothole_training runs are saved!
CHECKPOINT_PATH = "/content/drive/MyDrive/pothole_training/road_damage_v15/weights/last.pt"

# 3. Where to extract the dataset on Colab's fast local scratch disk
DATASET_DIR = "/content/dataset"

# ============================================================
# STEP 2: Mount Google Drive
# ============================================================
print("=" * 60)
print("STEP 2: Mounting Google Drive...")
print("=" * 60)

try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive mounted successfully!")
except ImportError:
    print("⚠️ 'google.colab' module not found. Assuming local run or non-Colab Jupyter environment.")
    # For local testing, we can simulate or adjust paths
    ZIP_PATH = "./data/RDD2022.zip"
    CHECKPOINT_PATH = "./pothole_training/road_damage_v15/weights/last.pt"
    DATASET_DIR = "./dataset"

# ============================================================
# STEP 3: Extract Dataset (If not already extracted)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Checking and Extracting Dataset...")
print("=" * 60)

os.makedirs(DATASET_DIR, exist_ok=True)

# Check if train/val directories already exist to avoid wasting time re-extracting
is_extracted = False
for root, dirs, files in os.walk(DATASET_DIR):
    if "train" in dirs and "val" in dirs:
        is_extracted = True
        break

if not is_extracted:
    if os.path.exists(ZIP_PATH):
        print(f"📦 Extracting {ZIP_PATH} to local scratch disk {DATASET_DIR}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        print("✅ Extraction complete!")
    else:
        print(f"❌ ZIP file not found at {ZIP_PATH}!")
        print("   Please check the path to your dataset ZIP file on Google Drive.")
        raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")
else:
    print("✅ Dataset already extracted. Skipping extraction step to save time.")

# Find the actual dataset root containing train/val
split_dir = None
for root, dirs, files in os.walk(DATASET_DIR):
    if "train" in dirs and "val" in dirs:
        split_dir = root
        break

if split_dir is None:
    raise FileNotFoundError("Could not find train/val folders in the dataset directory!")

print(f"📂 Dataset root path: {split_dir}")

# ============================================================
# STEP 4: Re-create data.yaml in current directory
# ============================================================
# When YOLO resumes, it reads `data: data.yaml` from the checkpoint's internal configuration.
# To ensure it resolves to our newly extracted local dataset in Colab, we write a local data.yaml.

print("\n" + "=" * 60)
print("STEP 4: Configuring data.yaml for Colab...")
print("=" * 60)

data_config = {
    'path': split_dir,
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': 5,
    'names': [
        'D00_Longitudinal_Crack',
        'D10_Transverse_Crack',
        'D20_Alligator_Crack',
        'D40_Pothole',
        'D44_Other_Damage'
    ]
}

# Save data.yaml in current working directory (e.g. /content/data.yaml)
data_yaml_path = os.path.join(os.getcwd(), "data.yaml")
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print(f"✅ data.yaml created in working directory: {data_yaml_path}")
print(f"   Config path points to: {split_dir}")

# ============================================================
# STEP 5: Resume Training from Checkpoint
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Resuming YOLOv8 Training!")
print("=" * 60)

if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ Checkpoint file not found at {CHECKPOINT_PATH}!")
    print("   Please verify the CHECKPOINT_PATH variable points to your last.pt file on Google Drive.")
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

print(f"🔄 Loading checkpoint: {CHECKPOINT_PATH}")
# Load the model from the checkpoint file
model = YOLO(CHECKPOINT_PATH)

print("🚀 Starting training resumption...")
# Setting resume=True tells YOLO to restore all parameters and settings from the checkpoint
# Note: In YOLOv8, you do not pass other arguments (like data or epochs) to model.train() when resuming,
# because they are loaded automatically from the checkpoint's config.
results = model.train(resume=True)

print("\n" + "=" * 60)
print("🎉 Resume training completed!")
print("=" * 60)
