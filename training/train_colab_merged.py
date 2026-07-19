"""
==========================================================
  ProjectPothole — YOLOv8 MERGED Training Script
==========================================================

This script takes your main RDD2022 dataset and your new high-accuracy
Pothole.v1 dataset, automatically fixes the label conflicts (0 -> 3),
merges them together into a super-dataset, and trains a medium YOLOv8 model
on it to cross the 70% accuracy mark.

HOW TO USE IN GOOGLE COLAB:
  1. Open Google Colab (https://colab.research.google.com)
  2. Change runtime type to GPU (Runtime -> Change runtime type -> GPU)
  3. Upload this script to Colab OR paste it in a cell.
  4. Ensure both your RDD2022.zip and Pothole.v1-raw.yolov8.zip are in your Google Drive.
  5. Run the cell to start training or resume training from last.pt automatically.
"""

import os
import shutil
import zipfile
import yaml
from ultralytics import YOLO

# ============================================================
# 1. Configuration
# ============================================================

# Paths to the zip archives on Google Drive (default location is Drive root)
MAIN_ZIP = "/content/drive/MyDrive/RDD2022.zip"
NEW_ZIP = "/content/drive/MyDrive/Pothole.v1-raw.yolov8.zip"

# Where to extract datasets on Colab's fast local scratch disk
MAIN_DIR = "/content/dataset_main"
NEW_DIR = "/content/dataset_new"

# Using Medium model and more epochs to hit 70%+
MODEL_SIZE = "yolov8m.pt"  
EPOCHS = 100               
BATCH_SIZE = 8             
IMG_SIZE = 640             

# Save project checkpoints directly to Google Drive so they persist on disconnects
DRIVE_PATH = "/content/drive/MyDrive"
PROJECT_NAME = f"{DRIVE_PATH}/pothole_training" if os.path.exists(DRIVE_PATH) else "pothole_training"

# ============================================================
# 2. Mount Google Drive (Automatically if in Colab)
# ============================================================
print("=" * 60)
print("1. Checking Google Drive...")
print("=" * 60)

try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive mounted successfully!")
except ImportError:
    print("⚠️ 'google.colab' module not found. Assuming local run or non-Colab Jupyter environment.")
    # Fallback to local paths for local testing
    MAIN_ZIP = "./data/RDD2022.zip"
    NEW_ZIP = "./data/Pothole.v1-raw.yolov8.zip"
    MAIN_DIR = "./dataset_main"
    NEW_DIR = "./dataset_new"
    PROJECT_NAME = "./pothole_training"

# Re-evaluate project path post-mount
if os.path.exists(DRIVE_PATH):
    PROJECT_NAME = f"{DRIVE_PATH}/pothole_training"

# ============================================================
# 3. Extract & Merge Datasets (With Caching to avoid duplicate work)
# ============================================================
print("\n" + "=" * 60)
print("2. Extracting and Merging Datasets...")
print("=" * 60)

# Check if the merged dataset is already fully extracted and merged locally
rdd_root = None
for root, dirs, files in os.walk(MAIN_DIR):
    if "train" in dirs and "val" in dirs:
        rdd_root = root
        break

dataset_already_merged = False
if rdd_root is not None:
    # Check if we have some files in the train images folder to ensure it's not empty
    train_img_dir = os.path.join(rdd_root, "train", "images")
    if os.path.exists(train_img_dir) and len(os.listdir(train_img_dir)) > 100:
        dataset_already_merged = True

if not dataset_already_merged:
    # --- A. Extract MAIN Dataset (RDD2022) ---
    print("📦 Extracting RDD2022 main dataset...")
    os.makedirs(MAIN_DIR, exist_ok=True)
    if os.path.exists(MAIN_ZIP):
        with zipfile.ZipFile(MAIN_ZIP, 'r') as zip_ref:
            zip_ref.extractall(MAIN_DIR)
        print("✅ RDD2022 main dataset extracted.")
    else:
        raise FileNotFoundError(f"❌ Missing RDD2022 main dataset at {MAIN_ZIP}")

    # Find actual RDD root containing train and val folders
    rdd_root = None
    for root, dirs, files in os.walk(MAIN_DIR):
        if "train" in dirs and "val" in dirs:
            rdd_root = root
            break
            
    if rdd_root is None:
        raise FileNotFoundError("❌ Could not find train/val folders in extracted RDD2022 dataset!")

    # --- B. Extract & Merge NEW Dataset (Pothole.v1) ---
    print("📦 Extracting and merging Pothole.v1 dataset...")
    os.makedirs(NEW_DIR, exist_ok=True)
    if os.path.exists(NEW_ZIP):
        with zipfile.ZipFile(NEW_ZIP, 'r') as zip_ref:
            zip_ref.extractall(NEW_DIR)
        print("✅ Pothole.v1 dataset extracted.")
    else:
        raise FileNotFoundError(f"❌ Missing Pothole.v1 dataset at {NEW_ZIP}")

    fixed_count = 0
    # Map Roboflow splits to RDD splits (valid -> val)
    targets = {'train': 'train', 'valid': 'val'}
    
    for origin_split, target_split in targets.items():
        origin_imgs = os.path.join(NEW_DIR, origin_split, "images")
        origin_lbls = os.path.join(NEW_DIR, origin_split, "labels")
        
        target_imgs = os.path.join(rdd_root, target_split, "images")
        target_lbls = os.path.join(rdd_root, target_split, "labels")
        
        if not os.path.exists(origin_imgs):
            continue
        
        # Ensure target directories exist
        os.makedirs(target_imgs, exist_ok=True)
        os.makedirs(target_lbls, exist_ok=True)
        
        for f in os.listdir(origin_lbls):
            if not f.endswith('.txt'):
                continue
            
            # Read label and map class 0 (Pothole in Roboflow) to 3 (Pothole in RDD)
            with open(os.path.join(origin_lbls, f), 'r') as file:
                lines = file.readlines()
                
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] == '0':
                    parts[0] = '3' # Fix class ID to match RDD Pothole class ID
                    new_lines.append(" ".join(parts) + "\n")
                    
            # Write mapped labels to target RDD folder
            with open(os.path.join(target_lbls, f), 'w') as file:
                file.writelines(new_lines)
                
            # Copy corresponding image to RDD folder
            img_name = f.replace('.txt', '.jpg')
            if os.path.exists(os.path.join(origin_imgs, img_name)):
                shutil.copy(os.path.join(origin_imgs, img_name), os.path.join(target_imgs, img_name))
                
            fixed_count += 1

    print(f"✅ Successfully converted and merged {fixed_count} new high-quality pothole images!")
else:
    print(f"✅ Merged dataset already exists at {rdd_root}. Skipping extraction & merging to save time.")

# ============================================================
# 4. Create Master data.yaml (Both in dataset folder and local folder)
# ============================================================
print("\n" + "=" * 60)
print("3. Setting up data.yaml...")
print("=" * 60)

data_config = {
    'path': rdd_root,
    'train': 'train/images',
    'val': 'val/images',
    'nc': 5,
    'names': [
        'D00_Longitudinal_Crack',
        'D10_Transverse_Crack',
        'D20_Alligator_Crack',
        'D40_Pothole',
        'D44_Other_Damage'
    ]
}

# Write to dataset folder
data_yaml_path = os.path.join(rdd_root, "data.yaml")
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

# Write to current working directory so YOLOv8 resume finds it relative to execution folder
local_data_yaml_path = os.path.join(os.getcwd(), "data.yaml")
with open(local_data_yaml_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print(f"✅ data.yaml configured at: {data_yaml_path}")
print(f"   and copied to local working directory: {local_data_yaml_path}")

# ============================================================
# 5. Train the YOLOv8m Model (Resume or Fresh)
# ============================================================
print("\n" + "=" * 60)
print("4. Starting or Resuming YOLOv8m Training...")
print("=" * 60)

# Check for checkpoints to resume
default_last_pt = os.path.join(PROJECT_NAME, "road_damage_merged", "weights", "last.pt")
uploaded_last_pt = "/content/last.pt"

model_loaded = False
model = None

# Try loading from local /content first (e.g. manually uploaded)
if os.path.exists(uploaded_last_pt):
    print(f"✅ Found uploaded checkpoint at {uploaded_last_pt}. Preparing to resume training...")
    model = YOLO(uploaded_last_pt)
    model_loaded = True
# Try loading from Google Drive path
elif os.path.exists(default_last_pt):
    print(f"✅ Found checkpoint in Google Drive at {default_last_pt}. Preparing to resume training...")
    model = YOLO(default_last_pt)
    model_loaded = True

if model_loaded:
    print("🚀 Resuming training from last.pt checkpoint...")
    # Passing resume=True is sufficient for YOLO to pick up where it left off
    results = model.train(resume=True)
else:
    print("🚀 No previous checkpoint found. Starting a fresh training run!")
    model = YOLO(MODEL_SIZE)
    
    results = model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project=PROJECT_NAME,
        name="road_damage_merged",
        patience=30,          
        save=True,
        plots=True,
        verbose=True,
        device=0,             
        # Hyperparameters for maximum accuracy
        cos_lr=True,          
        mosaic=1.0,           
        mixup=0.1,            
        close_mosaic=10       
    )

# ============================================================
# 6. Final Evaluation
# ============================================================
print("\n" + "=" * 60)
print("5. Evaluating Model on Validation Set...")
print("=" * 60)

metrics = model.val()
print(f"\n📊 Evaluation Results:")
print(f"   mAP50:    {metrics.box.map50 * 100:.2f}%")
print(f"   mAP50-95: {metrics.box.map * 100:.2f}%")
print(f"   Precision: {metrics.box.mp * 100:.2f}%")
print(f"   Recall:    {metrics.box.mr * 100:.2f}%")
print("=" * 60)
print("🎉 Training process finished! The best weights file is ready.")
