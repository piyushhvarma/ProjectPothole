"""
==========================================================
  ProjectPothole — YOLOv8 Training Script for Google Colab
==========================================================

HOW TO USE:
  1. Open Google Colab (https://colab.research.google.com)
  2. Go to Runtime → Change runtime type → GPU (T4 is free)
  3. Upload your RDD2022.zip to Google Drive OR directly to Colab
  4. Copy-paste this entire script into a Colab cell and run it
  5. Download the trained best.pt when done

NOTE: If uploading to Google Drive, mount Drive first:
   from google.colab import drive
   drive.mount('/content/drive')
   Then change ZIP_PATH below to point to your Drive location.
"""

# ============================================================
# STEP 0: Install dependencies
# ============================================================
# !pip install ultralytics --quiet

import os
import yaml
from ultralytics import YOLO

# ============================================================
# STEP 1: Configuration — EDIT THESE PATHS
# ============================================================

# Where is your RDD2022.zip?
# Option A: Uploaded directly to Colab
ZIP_PATH = "/content/RDD2022.zip"

# Option B: On Google Drive (uncomment and edit)
# ZIP_PATH = "/content/drive/MyDrive/RDD2022.zip"

# Where to extract the dataset
DATASET_DIR = "/content/dataset"

# Training parameters
# 1. UPGRADE MODEL: yolov8s.pt (Small) or yolov8m.pt (Medium) dramatically improve accuracy over 'n'.
MODEL_SIZE = "yolov8s.pt"
# 2. INCREASE EPOCHS: 50 is often too low for road damage data. 100-150 is better.
EPOCHS = 100               
BATCH_SIZE = 8             # REDUCED TO 8 TO PREVENT CRASHING (OOM)
IMG_SIZE = 640             # Image size for training

# Automatically save checkpoints to Google Drive if mounted so you don't lose them!
import os
drive_path = "/content/drive/MyDrive"
PROJECT_NAME = f"{drive_path}/pothole_training" if os.path.exists(drive_path) else "pothole_training"

# ============================================================
# STEP 2: Extract Dataset
# ============================================================

print("=" * 60)
print("STEP 2: Extracting dataset...")
print("=" * 60)

os.makedirs(DATASET_DIR, exist_ok=True)

if os.path.exists(ZIP_PATH):
    import zipfile
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)
    print(f"✅ Extracted to {DATASET_DIR}")
else:
    print(f"❌ ZIP not found at {ZIP_PATH}")
    print("   Please upload RDD2022.zip or update ZIP_PATH")
    raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")

# Find the actual dataset root (handles nested folders)
split_dir = None
for root, dirs, files in os.walk(DATASET_DIR):
    if "train" in dirs and "val" in dirs:
        split_dir = root
        break

if split_dir is None:
    raise FileNotFoundError("Could not find train/val folders in extracted dataset!")

print(f"✅ Dataset root found: {split_dir}")

# Count images
train_imgs = len([f for f in os.listdir(os.path.join(split_dir, "train", "images")) if f.endswith(('.jpg', '.png'))])
val_imgs = len([f for f in os.listdir(os.path.join(split_dir, "val", "images")) if f.endswith(('.jpg', '.png'))])
print(f"📊 Training images: {train_imgs}")
print(f"📊 Validation images: {val_imgs}")

# ============================================================
# STEP 3: Create Correct data.yaml
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Creating data.yaml configuration...")
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

data_yaml_path = os.path.join(split_dir, "data.yaml")
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print(f"✅ data.yaml created at: {data_yaml_path}")
print(f"   Classes: {data_config['names']}")

# ============================================================
# STEP 4: Verify Labels
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Verifying label integrity...")
print("=" * 60)

labels_dir = os.path.join(split_dir, "train", "labels")
class_counts = {}
total_files = 0
bad_files = 0

for f in os.listdir(labels_dir):
    if not f.endswith('.txt'):
        continue
    total_files += 1
    filepath = os.path.join(labels_dir, f)
    try:
        with open(filepath) as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id < 5:
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    else:
                        bad_files += 1
    except Exception:
        bad_files += 1

print(f"✅ Checked {total_files} label files ({bad_files} had issues)")
print("📊 Class distribution:")
for cls_id in sorted(class_counts.keys()):
    name = data_config['names'][cls_id]
    count = class_counts[cls_id]
    print(f"   [{cls_id}] {name}: {count} annotations")
print(f"   Total: {sum(class_counts.values())} annotations")

# ============================================================
# STEP 5: Train YOLOv8
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Starting YOLOv8 training!")
print(f"  Model: {MODEL_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Image size: {IMG_SIZE}")
print("=" * 60)

# Load pretrained YOLOv8 model
model = YOLO(MODEL_SIZE)

# Train!
results = model.train(
    data=data_yaml_path,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    project=PROJECT_NAME,
    name="road_damage_v2",  # changed name to avoid mixing with previous run
    patience=25,          # 3. INCREASE PATIENCE: 10 is too low. Sometimes loss plateaus then drops.
    save=True,
    save_period=1,        # SAVE EVERY EPOCH so you don't lose checkpoints if it crashes
    plots=True,           # Generate training plots
    verbose=True,
    device=0,             # Use GPU 0
    # 4. HYPERPARAMETER TUNING FOR ACCURACY:
    cos_lr=True,          # Cosine learning rate scheduling (helps find better minimum)
    mosaic=1.0,           # Mosaic augmentation (combines 4 images, great for small cracks/potholes)
    mixup=0.1,            # Mixup augmentation to generalize better
    close_mosaic=10       # Disable mosaic in the last 10 epochs to stabilize final fine-tuning
)

# ============================================================
# STEP 6: Evaluate the Model
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Evaluating model on validation set...")
print("=" * 60)

metrics = model.val()

print(f"\n📊 Results:")
print(f"   mAP50:    {metrics.box.map50:.4f}")
print(f"   mAP50-95: {metrics.box.map:.4f}")
print(f"   Precision: {metrics.box.mp:.4f}")
print(f"   Recall:    {metrics.box.mr:.4f}")

# ============================================================
# STEP 7: Export & Download
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Your trained model is ready!")
print("=" * 60)

best_pt_path = os.path.join(PROJECT_NAME, "road_damage_v1", "weights", "best.pt")

if os.path.exists(best_pt_path):
    size_mb = os.path.getsize(best_pt_path) / (1024 * 1024)
    print(f"✅ Model saved: {best_pt_path} ({size_mb:.1f} MB)")
    print(f"\n📥 To download, run this in a new Colab cell:")
    print(f"   from google.colab import files")
    print(f"   files.download('{best_pt_path}')")
    print(f"\n   Then replace best.pt in your ProjectPothole folder!")
else:
    print("⚠️ best.pt not found at expected location.")
    print(f"   Check the {PROJECT_NAME}/ folder for your results.")

print("\n" + "=" * 60)
print("🎉 DONE! Copy best.pt to your project and run app.py")
print("=" * 60)
