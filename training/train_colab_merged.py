"""
==========================================================
  ProjectPothole — YOLOv8 MERGED Training Script
==========================================================

This script does something magical: It takes your main RDD2022 dataset, 
takes your new high-accuracy Pothole.v1 dataset, automatically fixes the 
label conflicts (0 -> 3), merges them together into a super-dataset, 
and trains a medium YOLOv8 model on it to cross that 70% accuracy mark!

Upload this script alongside BOTH your RDD2022.zip and Pothole.v1-raw.yolov8.zip.
"""

import os
import shutil
import zipfile
import yaml
from ultralytics import YOLO

# ============================================================
# 1. Configuration
# ============================================================
MAIN_ZIP = "/content/RDD2022.zip"
NEW_ZIP = "/content/Pothole.v1-raw.yolov8.zip"
MAIN_DIR = "/content/dataset_main"
NEW_DIR = "/content/dataset_new"

# Using Medium model and more epochs to hit 70%+
MODEL_SIZE = "yolov8m.pt"  
EPOCHS = 100               
BATCH_SIZE = 8             
IMG_SIZE = 640             

drive_path = "/content/drive/MyDrive"
PROJECT_NAME = f"{drive_path}/pothole_training" if os.path.exists(drive_path) else "pothole_training"

# ============================================================
# 2. Extract MAIN Dataset
# ============================================================
print("=" * 60)
print("1. Extracting RDD2022...")
os.makedirs(MAIN_DIR, exist_ok=True)
if os.path.exists(MAIN_ZIP):
    with zipfile.ZipFile(MAIN_ZIP, 'r') as zip_ref:
        zip_ref.extractall(MAIN_DIR)
else:
    raise FileNotFoundError(f"Missing {MAIN_ZIP}")

# Find actual RDD root
rdd_root = None
for root, dirs, files in os.walk(MAIN_DIR):
    if "train" in dirs and "val" in dirs:
        rdd_root = root
        break

# ============================================================
# 3. Extract & Merge NEW Dataset (Pothole.v1)
# ============================================================
print("=" * 60)
print("2. Extracting Pothole.v1 and Fixing Labels...")
os.makedirs(NEW_DIR, exist_ok=True)
if os.path.exists(NEW_ZIP):
    with zipfile.ZipFile(NEW_ZIP, 'r') as zip_ref:
        zip_ref.extractall(NEW_DIR)
else:
    raise FileNotFoundError(f"Missing {NEW_ZIP}")

fixed_count = 0
# We want to move data into rdd_root/train and rdd_root/val
targets = {'train': 'train', 'valid': 'val'} # Roboflow uses 'valid', RDD uses 'val'

for origin_split, target_split in targets.items():
    origin_imgs = os.path.join(NEW_DIR, origin_split, "images")
    origin_lbls = os.path.join(NEW_DIR, origin_split, "labels")
    
    target_imgs = os.path.join(rdd_root, target_split, "images")
    target_lbls = os.path.join(rdd_root, target_split, "labels")
    
    if not os.path.exists(origin_imgs): continue
    
    for f in os.listdir(origin_lbls):
        if not f.endswith('.txt'): continue
        
        # 1. Read label and map 0 -> 3
        with open(os.path.join(origin_lbls, f), 'r') as file:
            lines = file.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == '0':
                parts[0] = '3' # FIX CLASS ID
                new_lines.append(" ".join(parts) + "\n")
                
        # 2. Write fixed label to RDD folder
        with open(os.path.join(target_lbls, f), 'w') as file:
            file.writelines(new_lines)
            
        # 3. Copy image across
        img_name = f.replace('.txt', '.jpg')
        if os.path.exists(os.path.join(origin_imgs, img_name)):
            shutil.copy(os.path.join(origin_imgs, img_name), os.path.join(target_imgs, img_name))
            
        fixed_count += 1

print(f"✅ Successfully converted and merged {fixed_count} new high-quality pothole images!")

# ============================================================
# 4. Create Master data.yaml
# ============================================================
data_yaml_path = os.path.join(rdd_root, "data.yaml")
data_config = {
    'path': rdd_root,
    'train': 'train/images',
    'val': 'val/images',
    'nc': 5,
    'names': ['D00_Longitudinal_Crack', 'D10_Transverse_Crack', 'D20_Alligator_Crack', 'D40_Pothole', 'D44_Other_Damage']
}
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

# ============================================================
# 5. Train the 70%+ Accuracy Model
# ============================================================
print("=" * 60)
print("3. Starting SUPERVISED YOLOv8m TRAINING (Target 70%+)")
print("=" * 60)

# Check if a past checkpoint exists to resume training
default_last_pt = os.path.join(PROJECT_NAME, "road_damage_merged", "weights", "last.pt")
uploaded_last_pt = "/content/last.pt"

if os.path.exists(uploaded_last_pt):
    print(f"✅ Found uploaded checkpoint at {uploaded_last_pt}. Resuming training...")
    model = YOLO(uploaded_last_pt)
    results = model.train(resume=True)
elif os.path.exists(default_last_pt):
    print(f"✅ Found checkpoint in Drive at {default_last_pt}. Resuming training...")
    model = YOLO(default_last_pt)
    results = model.train(resume=True)
else:
    print("🚀 No previous checkpoint found. Starting fresh training!")
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

# Evaluate
print("=" * 60)
print("4. Final Teacher Evaluation:")
metrics = model.val()
print(f"   mAP50:    {metrics.box.map50*100:.2f}%")
print(f"   Precision: {metrics.box.mp*100:.2f}%")
print("=" * 60)
print("🎉 Bring this best.pt to your teacher!")
