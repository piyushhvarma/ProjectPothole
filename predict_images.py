import os
import cv2
from ultralytics import YOLO
import folium
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# ==========================================================
# CONFIGURATION
# ==========================================================
MODEL_PATH = "models/best.pt"
IMAGE_FOLDER = "data/route_photos"  # SOURCE FOLDER
DETECTIONS_FOLDER = "outputs/detections/detections" # SAVE FOLDER
OUTPUT_MAP = "outputs/maps/route_pothole_map.html" # Shorter name to avoid long URL issues
CONFIDENCE_THRESHOLD = 0.40 # Raised to 40% to stop predicting Google logos as potholes!

# Map color codes (CSS)
MAP_COLORS = {
    0: "orange",  # Longitudinal Crack
    1: "blue",    # Transverse Crack
    2: "darkred", # Alligator Crack
    3: "red",     # Pothole (High Priority!)
    4: "purple",  # Other Damage
}

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Other Damage",
}

# ==========================================================
# HELPER: Extract GPS/Info from Filename or EXIF
# ==========================================================

def get_metadata(filename):
    """Try to extract lat/lon from filenames like photo_001_26.843_75.565.jpg"""
    try:
        parts = filename.replace(".jpg", "").split("_")
        if len(parts) >= 4:
            lat = float(parts[2])
            lon = float(parts[3])
            return lat, lon
    except:
        pass
    return None, None

# ==========================================================
# MAIN PROCESSING
# ==========================================================

if __name__ == "__main__":
    print("Loading AI Model...")
    model = YOLO(MODEL_PATH)
    
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ Folder '{IMAGE_FOLDER}' not found! Run collect_images.py first.")
        exit()

    os.makedirs(DETECTIONS_FOLDER, exist_ok=True)
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"🔍 Found {len(images)} images in '{IMAGE_FOLDER}'")

    m = folium.Map(location=[26.87, 75.67], zoom_start=13)
    detections = []

    for fname in images:
        path = os.path.join(IMAGE_FOLDER, fname)
        lat, lon = get_metadata(fname)
        
        # Run detection
        results = model(path, verbose=False)
        
        # Check if we have any detections above threshold
        has_detection = False
        for box in results[0].boxes:
            if box.conf[0].item() > CONFIDENCE_THRESHOLD:
                has_detection = True
                break
        
        if has_detection or "manual" in fname:
            # Save the annotated image
            annotated_img = results[0].plot()
            cv2.imwrite(os.path.join(DETECTIONS_FOLDER, f"det_{fname}"), annotated_img)
            
            # If manual and no detections, create a custom popup
            if not has_detection and "manual" in fname:
                name = "Manual Damage Report"
                color = "purple"
                popup_html = f"""
                    <div style="width: 320px;">
                        <h4>{name}</h4>
                        <img src="{DETECTIONS_FOLDER}/det_{fname}" width="300" style="border-radius: 5px;">
                        <p>User provided screenshot - High Damage Zone</p>
                        <p>Coords: {lat:.5f}, {lon:.5f}</p>
                    </div>
                """
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=12,
                    color=color,
                    fill=True,
                    fill_color=color,
                    popup=folium.Popup(popup_html, max_width=350),
                    tooltip=name
                ).add_to(m)
                detections.append((lat, lon, name))
                continue
            
            for box in results[0].boxes:
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                if conf > CONFIDENCE_THRESHOLD:
                    name = CLASS_NAMES.get(cls, f"Class_{cls}")
                    print(f"  [FOUND] {name} in {fname} ({int(conf*100)}%)")
                    
                    if lat and lon:
                        color = MAP_COLORS.get(cls, "gray")
                        
                        # Create a nice HTML popup with the image
                        img_filename = f"det_{fname}"
                        popup_html = f"""
                            <div style="width: 320px;">
                                <h4>{name} ({int(conf*100)}%)</h4>
                                <img src="{DETECTIONS_FOLDER}/{img_filename}" width="300" style="border-radius: 5px;">
                                <p>Coords: {lat:.5f}, {lon:.5f}</p>
                            </div>
                        """
                        
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=8,
                            color=color,
                            fill=True,
                            fill_color=color,
                            popup=folium.Popup(popup_html, max_width=350),
                            tooltip=name
                        ).add_to(m)
                        detections.append((lat, lon, name))

    m.save(OUTPUT_MAP)
    print(f"\n🎉 Processed all images. Map saved: {OUTPUT_MAP}")
    print(f"   Total detections marked: {len(detections)}")
    print(f"   Visual results saved to: {DETECTIONS_FOLDER}/")
