"""
ProjectPothole — Local Training Script (RTX 4050, 6GB VRAM)
Run this directly: python train_local.py
"""

from ultralytics import YOLO

if __name__ == '__main__':
    # Load pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # Train using your existing data.yaml (already configured!)
    results = model.train(
        data="data.yaml",        # Your existing data.yaml
        epochs=50,               # 50 epochs is a good starting point
        batch=8,                 # Safe for 6GB VRAM (reduce to 4 if OOM)
        imgsz=640,               # Standard YOLO image size
        project="pothole_training",
        name="road_damage_v1",
        patience=10,             # Stop early if no improvement for 10 epochs
        save=True,
        save_period=10,          # Checkpoint every 10 epochs
        plots=True,              # Generate training plots
        verbose=True,
        device=0,                # Your RTX 4050
    )

    # Evaluate
    metrics = model.val()
    print(f"\n📊 Results:")
    print(f"   mAP50:     {metrics.box.map50:.4f}")
    print(f"   mAP50-95:  {metrics.box.map:.4f}")
    print(f"   Precision:  {metrics.box.mp:.4f}")
    print(f"   Recall:     {metrics.box.mr:.4f}")

    # Copy best.pt to project root
    import shutil, os
    best_path = os.path.join("pothole_training", "road_damage_v1", "weights", "best.pt")
    if os.path.exists(best_path):
        shutil.copy(best_path, "best.pt")
        print(f"\n✅ best.pt copied to project root! You're ready to run app.py")
    else:
        print(f"\n⚠️  Check pothole_training/ folder for results")
