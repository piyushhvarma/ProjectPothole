import cv2
import os
from ultralytics import YOLO
import folium
import time

# --- CONFIGURATION ---
MODEL_PATH = "models/best.pt"        # Your trained model
VIDEO_PATH = "data/videos/test_video.mp4" # Your test video
CONFIDENCE_THRESHOLD = 0.45   # CHANGED from 0.9: 90% was way too high and likely hiding valid potholes!
SCREENSHOT_DIR = "outputs/pothole_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
# --- CLASS COLORS (BGR for OpenCV) ---
# Each road damage type gets its own color for easy identification
CLASS_COLORS = {
    0: (255, 165, 0),    # D00 Longitudinal Crack — Orange
    1: (0, 255, 255),    # D10 Transverse Crack  — Yellow
    2: (0, 165, 255),    # D20 Alligator Crack   — Dark Orange
    3: (0, 0, 255),      # D40 Pothole           — Red (most dangerous!)
    4: (255, 0, 255),    # D44 Other Damage      — Magenta
}

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Other Damage",
}

# Map marker colors for Folium (uses CSS color names)
MAP_COLORS = {
    0: "orange",
    1: "blue",
    2: "darkred",
    3: "red",
    4: "purple",
}

# --- SIMULATED GPS START POINT (Example: Connaught Place, Delhi) ---
# In a real hardware project, this comes from a GPS sensor.
lat = 28.6304
lon = 77.2177

# Initialize
print("Loading AI Model... (This might take a moment)")
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
detected_locations = []  # Stores [lat, lon, class_id, confidence]

# Get model class names for display
model_classes = model.names
print(f"Model loaded! Classes: {model_classes}")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print("Starting Road Damage Patrol... Press 'Q' to stop.\n")

frame_count = 0
detection_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frame_count += 1

    # 1. Run Detection
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # 2. Check for Road Damage
    detected_damage_in_frame = set()
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            
            if conf > CONFIDENCE_THRESHOLD:
                detection_count += 1

                # Get the class name (from model or our lookup)
                class_name = model_classes.get(cls, CLASS_NAMES.get(cls, f"Class_{cls}"))

                # Keep track of what we detected
                detected_damage_in_frame.add(class_name.replace(' ', '_'))

                # Log the location with class info
                detected_locations.append([lat, lon, cls, conf])
                
                # Visual Warning — color-coded by damage type
                color = CLASS_COLORS.get(cls, (255, 255, 255))
                label = f"{class_name} ({int(conf*100)}%)"
                cv2.putText(annotated_frame, label,
                           (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                print(f"  [{frame_count}] {class_name} ({int(conf*100)}%) at ({lat:.5f}, {lon:.5f})")

    # Save screenshot if any road damage was detected in this frame
    if detected_damage_in_frame:
        damage_tags = "_".join(detected_damage_in_frame)
        screenshot_path = os.path.join(SCREENSHOT_DIR, f"{damage_tags}_frame_{frame_count}.jpg")
        cv2.imwrite(screenshot_path, annotated_frame)

    # 3. Simulate Moving Car (Update GPS)
    lat += 0.00005 
    lon += 0.00005

    # 4. Show the Video (original resolution)
    cv2.imshow("Road Damage Patrol — Live Feed", annotated_frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- GENERATE THE MAP ---
print(f"\n{'='*50}")
print(f"Generating Map with {len(detected_locations)} detections marked...")
print(f"{'='*50}")

if detected_locations:
    # Create map centered at the start location
    m = folium.Map(location=[28.6304, 77.2177], zoom_start=15)

    # Summary stats
    type_counts = {}
    for loc in detected_locations:
        cls_id = loc[2]
        name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
        type_counts[name] = type_counts.get(name, 0) + 1

    # Add a colored marker for every detection
    for loc in detected_locations:
        lat_val, lon_val, cls_id, conf = loc
        class_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
        color = MAP_COLORS.get(cls_id, "gray")
        
        folium.CircleMarker(
            location=[lat_val, lon_val],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"{class_name} ({int(conf*200)}%)",
            tooltip=class_name
        ).add_to(m)

    # Add a legend as HTML
    legend_html = """
    <div style="position:fixed; bottom:50px; left:50px; z-index:1000; 
                background:white; padding:10px; border:2px solid grey; border-radius:5px;">
        <h4>Road Damage Types</h4>
        <p><span style="color:orange;">●</span> Longitudinal Crack</p>
        <p><span style="color:blue;">●</span> Transverse Crack</p>
        <p><span style="color:darkred;">●</span> Alligator Crack</p>
        <p><span style="color:red;">●</span> Pothole</p>
        <p><span style="color:purple;">●</span> Other Damage</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save("outputs/maps/pothole_map.html")
    
    print("\n📊 Detection Summary:")
    for name, count in sorted(type_counts.items()):
        print(f"   {name}: {count}")
    print(f"   Total: {len(detected_locations)} detections across {frame_count} frames")
    print("\n✅ SUCCESS! Open 'outputs/maps/pothole_map.html' to see the report.")
else:
    print("No road damage detected. Good roads! 🎉")