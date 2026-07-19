"""
======================================================================
  Pothole Patrol — Google Colab Merged Training Script
======================================================================

Paste this entire script into a single cell in Google Colab.
Make sure your Google Colab runtime is set to GPU:
  -> Runtime -> Change runtime type -> Hardware accelerator -> GPU

Prerequisites:
  1. Upload `RDD2022.zip` and `Pothole.v1-raw.yolov8.zip` to your Google Drive.
  2. By default, they should be in the root of your Google Drive:
     - My Drive/RDD2022.zip
     - My Drive/Pothole.v1-raw.yolov8.zip
"""

import os
import shutil
import zipfile
import yaml
from ultralytics import YOLO

# ============================================================
# 1. Configuration & Paths
# ============================================================
# Paths as they will appear inside Google Colab environment
MAIN_ZIP = "/content/drive/MyDrive/RDD2022.zip"
NEW_ZIP = "/content/drive/MyDrive/Pothole.v1-raw.yolov8.zip"

# Extraction destinations on Google Colab's fast local scratch disk
MAIN_DIR = "/content/dataset_main"
NEW_DIR = "/content/dataset_new"

# Training parameters
MODEL_SIZE = "yolov8m.pt"  # Medium model
EPOCHS = 100               # Target epochs for high accuracy
BATCH_SIZE = 8             # Batch size (optimized for Colab T4 GPU VRAM)
IMG_SIZE = 640             # Image resolution

# Google Drive folder path to save weight checkpoints so they persist
DRIVE_PATH = "/content/drive/MyDrive"
PROJECT_NAME = f"{DRIVE_PATH}/pothole_training" if os.path.exists(DRIVE_PATH) else "pothole_training"

# ============================================================
# 2. Mount Google Drive
# ============================================================
print("=" * 60)
print("Step 1: Mounting Google Drive...")
print("=" * 60)

try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive mounted successfully!")
except ImportError:
    print("⚠️ 'google.colab' module not found. Using local paths...")
    # Clean fallbacks for local run if needed
    MAIN_ZIP = "./data/RDD2022.zip"
    NEW_ZIP = "./data/Pothole.v1-raw.yolov8.zip"
    MAIN_DIR = "./dataset_main"
    NEW_DIR = "./dataset_new"
    PROJECT_NAME = "./pothole_training"

# ============================================================
# 3. Extract & Merge Datasets
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Extracting & Merging Datasets...")
print("=" * 60)

rdd_root = None
for root, dirs, files in os.walk(MAIN_DIR):
    if "train" in dirs and "val" in dirs:
        rdd_root = root
        break

# Avoid duplicate extraction on repeated runs
dataset_already_merged = False
if rdd_root is not None:
    train_img_dir = os.path.join(rdd_root, "train", "images")
    if os.path.exists(train_img_dir) and len(os.listdir(train_img_dir)) > 100:
        dataset_already_merged = True

if not dataset_already_merged:
    # A. Extract RDD2022 (Main Dataset)
    if os.path.exists(MAIN_ZIP):
        print("📦 Extracting main RDD2022 dataset...")
        os.makedirs(MAIN_DIR, exist_ok=True)
        with zipfile.ZipFile(MAIN_ZIP, 'r') as zip_ref:
            zip_ref.extractall(MAIN_DIR)
        print("✅ RDD2022 extracted.")
    else:
        raise FileNotFoundError(f"❌ Missing RDD2022 dataset zip on Google Drive at: {MAIN_ZIP}")

    # Relocate RDD root
    for root, dirs, files in os.walk(MAIN_DIR):
        if "train" in dirs and "val" in dirs:
            rdd_root = root
            break
            
    if rdd_root is None:
        raise FileNotFoundError("❌ Could not find train/val directories inside RDD2022 extraction!")

    # B. Extract and Merge Pothole.v1 (New Dataset)
    if os.path.exists(NEW_ZIP):
        print("📦 Extracting Pothole.v1 dataset...")
        os.makedirs(NEW_DIR, exist_ok=True)
        with zipfile.ZipFile(NEW_ZIP, 'r') as zip_ref:
            zip_ref.extractall(NEW_DIR)
        print("✅ Pothole.v1 extracted.")
    else:
        raise FileNotFoundError(f"❌ Missing Pothole.v1 dataset zip on Google Drive at: {NEW_ZIP}")

    print("⚙️ Merging datasets and fixing label IDs (mapping Roboflow 0 to RDD 3 for Pothole type)...")
    fixed_count = 0
    splits_map = {'train': 'train', 'valid': 'val'}
    
    for src_split, dest_split in splits_map.items():
        src_imgs = os.path.join(NEW_DIR, src_split, "images")
        src_lbls = os.path.join(NEW_DIR, src_split, "labels")
        
        dest_imgs = os.path.join(rdd_root, dest_split, "images")
        dest_lbls = os.path.join(rdd_root, dest_split, "labels")
        
        if not os.path.exists(src_imgs):
            continue
        
        os.makedirs(dest_imgs, exist_ok=True)
        os.makedirs(dest_lbls, exist_ok=True)
        
        for f in os.listdir(src_lbls):
            if not f.endswith('.txt'):
                continue
            
            # Map Class ID 0 (Roboflow Pothole) to 3 (RDD Pothole)
            with open(os.path.join(src_lbls, f), 'r') as file:
                lines = file.readlines()
                
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] == '0':
                    parts[0] = '3'  # Realign with RDD labeling schema
                    new_lines.append(" ".join(parts) + "\n")
                    
            # Save labeled output to master folder
            with open(os.path.join(dest_lbls, f), 'w') as file:
                file.writelines(new_lines)
                
            # Copy corresponding image
            img_filename = f.replace('.txt', '.jpg')
            if os.path.exists(os.path.join(src_imgs, img_filename)):
                shutil.copy(os.path.join(src_imgs, img_filename), os.path.join(dest_imgs, img_filename))
                
            fixed_count += 1

    print(f"✅ Successfully merged {fixed_count} images from Pothole.v1 into RDD2022.")
else:
    print(f"✅ Dataset already merged at {rdd_root}. Skipping extraction step to save time.")

# ============================================================
# 4. Generate Master data.yaml
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Creating YOLO Configuration data.yaml...")
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

data_yaml_path = os.path.join(rdd_root, "data.yaml")
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

# Copy to current working directory so relative paths work nicely
local_data_yaml_path = os.path.join(os.getcwd(), "data.yaml")
with open(local_data_yaml_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print(f"✅ data.yaml generated successfully:")
print(f"   - Merged Dir: {data_yaml_path}")
print(f"   - Local Dir:  {local_data_yaml_path}")

# ============================================================
# 5. Execute Training (Resume Check or Start Fresh)
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Starting YOLOv8m Merged Training...")
print("=" * 60)

# Check locations for existing training checkpoints to resume training
default_last_pt = os.path.join(PROJECT_NAME, "road_damage_merged", "weights", "last.pt")
uploaded_last_pt = "/content/last.pt"

model = None
is_resuming = False

if os.path.exists(uploaded_last_pt):
    print(f"✅ Found checkpoint uploaded directly to Colab: {uploaded_last_pt}")
    model = YOLO(uploaded_last_pt)
    is_resuming = True
elif os.path.exists(default_last_pt):
    print(f"✅ Found checkpoint in Google Drive: {default_last_pt}")
    model = YOLO(default_last_pt)
    is_resuming = True

if is_resuming:
    print("🚀 Resuming training from the last saved milestone...")
    model.train(resume=True)
else:
    print("🚀 Starting a fresh training run using pre-trained YOLOv8m...")
    model = YOLO(MODEL_SIZE)
    model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project=PROJECT_NAME,
        name="road_damage_merged",
        patience=30,      # Stop early if no improvement after 30 epochs
        save=True,
        plots=True,
        verbose=True,
        device=0,         # Run on GPu
        cos_lr=True,      # Cosine learning rate scheduler
        mosaic=1.0,       # Data augmentation
        mixup=0.1,        # Data augmentation
        close_mosaic=10   # Turn off mosaic for final 10 epochs
    )

# ============================================================
# 6. Evaluation Output
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Verifying Validation Metrics...")
print("=" * 60)

metrics = model.val()
print("\n📊 Evaluation Summary:")
print(f"   mAP50:    {metrics.box.map50 * 100:.2f}%")
print(f"   mAP50-95: {metrics.box.map * 100:.2f}%")
print(f"   Precision: {metrics.box.mp * 100:.2f}%")
print(f"   Recall:    {metrics.box.mr * 100:.2f}%")
print("=" * 60)
print("🎉 YOLOv8m training run complete! Best weights saved inside Google Drive /pothole_training/road_damage_merged/weights/best.pt")
