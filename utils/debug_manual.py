from ultralytics import YOLO
import cv2
import os

model = YOLO("best.pt")
img_path = "route_photos_vinayak/road_manual_26.85000_75.56800.jpg"

print(f"Running deep inference on: {img_path}")
results = model(img_path, verbose=False)

print("\n--- ALL DETECTIONS FOUND ---")
for box in results[0].boxes:
    conf = box.conf[0].item()
    cls = int(box.cls[0].item())
    print(f"Class: {cls}, Confidence: {conf:.4f}")

# Plot and save NO MATTER WHAT
annotated_img = results[0].plot()
cv2.imwrite("detections_vinayak/det_road_manual_DEBUG.jpg", annotated_img)
print("\nDebug image saved: detections_vinayak/det_road_manual_DEBUG.jpg")
