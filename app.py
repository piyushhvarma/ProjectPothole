import cv2
from ultralytics import YOLO
import folium
import time

# --- CONFIGURATION ---
MODEL_PATH = "best.pt"       # Your trained brain
VIDEO_PATH = "test_video3.mp4" # Your test video
CONFIDENCE_THRESHOLD = 0.000001    # How sure the AI needs to be (0.5 = 50%)

# --- SIMULATED GPS START POINT (Example: Connaught Place, Delhi) ---
# In a real hardware project, this comes from a GPS sensor.
lat = 28.6304
lon = 77.2177

# Initialize
print("Loading AI Model... (This might take a moment)")
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
detected_locations = []

print("Starting Pothole Patrol... Press 'Q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # End of video

    # 1. Run Detection
    results = model(frame, verbose=False) # verbose=False keeps the terminal clean
    annotated_frame = results[0].plot()

    # 2. Check for Potholes
    # results[0].boxes contains all the detections in this frame
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            
            # If confidence is high enough
            if conf > CONFIDENCE_THRESHOLD:
                # Log the location
                detected_locations.append([lat, lon])
                
                # Visual Warning
                cv2.putText(annotated_frame, f"POTHOLE DETECTED! ({int(conf*100)}%)", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print(f"⚠️ Pothole found at Lat: {lat:.5f}, Lon: {lon:.5f}")

    # 3. Simulate Moving Car (Update GPS)
    # This slightly changes coordinates every frame to simulate driving
    lat += 0.00005 
    lon += 0.00005

    # 4. Show the Video
    # Resize window to fit screen if video is 4K
    display_frame = cv2.resize(annotated_frame, (1020, 600)) 
    cv2.imshow("Pothole Patrol Live Feed", display_frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- GENERATE THE MAP ---
print(f"\nGeneratin Map with {len(detected_locations)} potholes marked...")

if detected_locations:
    # Create map centered at the start location
    m = folium.Map(location=[28.6304, 77.2177], zoom_start=15)

    # Add a red marker for every detection
    for location in detected_locations:
        folium.CircleMarker(
            location=location,
            radius=5,
            color="red",
            fill=True,
            fill_color="red"
        ).add_to(m)

    m.save("pothole_map.html")
    print("✅ SUCCESS! Open 'pothole_map.html' to see the report.")
else:
    print("No potholes detected. Good roads!")