import os
import folium
from ultralytics import YOLO
import cv2
import base64

# ==========================================================
# CONFIGURATION
# ==========================================================
MODEL_PATH = "models/old_best.pt"
OUTPUT_MAP = "outputs/maps/pothole_patrol_master.html"
CONFIDENCE_THRESHOLD = 0.15

# Define our routes — each has a distinct base color
ROUTES = [
    {
        "name": "Highway Route (NH48)",
        "source": "data/route_photos",
        "detections": "outputs/detections/detections",
        "base_color": "#2196F3",   # Blue
    },
    {
        "name": "Vinayak Marg Detour",
        "source": "data/route_photos_vinayak",
        "detections": "outputs/detections/detections_vinayak",
        "base_color": "#9C27B0",   # Purple
    }
]

CLASS_NAMES = {0: "Longitudinal Crack", 1: "Transverse Crack", 2: "Alligator Crack", 3: "Pothole", 4: "Other Damage"}

# Severity coloring for route patches (same for all routes)
SEVERITY_COLORS = {
    "none":   None,       # No overlay
    "low":    "#FFD700",  # Yellow (few detections)
    "medium": "#FF8C00",  # Orange (moderate)
    "high":   "#FF0000",  # Red (severe)
}

SEGMENT_SIZE = 5  # Group every N consecutive photos into one segment

def get_metadata(filename):
    try:
        parts = filename.replace(".jpg", "").replace(".jpeg", "").replace(".png", "").split("_")
        if len(parts) >= 4:
            return float(parts[2]), float(parts[3])
    except:
        pass
    return None, None

def classify_severity(detection_count):
    """Classify a route segment by how many detections it has."""
    if detection_count == 0:
        return "none"
    elif detection_count <= 2:
        return "low"
    elif detection_count <= 5:
        return "medium"
    else:
        return "high"

def image_to_base64(image_path):
    """Convert an image file to a base64 data URI for embedding in HTML."""
    try:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        mime = {"jpg": "image/jpeg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{data}"
    except Exception as e:
        print(f"   [WARN] Could not encode image {image_path}: {e}")
        return ""

def process_routes():
    print("Loading AI Model...")
    model = YOLO(MODEL_PATH)
    
    # Base map centered between MUJ and Railway Station
    m = folium.Map(location=[26.88, 75.67], zoom_start=13, tiles="OpenStreetMap")
    
    total_detections = 0
    
    # We'll collect all route data first, then draw in correct layer order
    all_routes_data = []

    for route in ROUTES:
        source = route["source"]
        dest = route["detections"]
        name = route["name"]
        base_color = route["base_color"]
        
        print(f"\n🚀 Processing Route: {name} ({source})")
        if not os.path.exists(source):
            print(f"   [SKIP] Folder {source} not found.")
            continue
            
        os.makedirs(dest, exist_ok=True)
        images = sorted([f for f in os.listdir(source) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if not images:
            print(f"   [SKIP] No images found in {source}.")
            continue
        
        print(f"   Found {len(images)} images.")
        
        # ============================================
        # PHASE 1: Collect all route points + run model
        # ============================================
        route_points = []      # (lat, lon) for each image in order
        image_detections = []  # list of lists: detections per image
        valid_images = []      # only images with valid coords
        image_b64_cache = {}   # fname -> base64 data URI
        
        for fname in images:
            path = os.path.join(source, fname)
            lat, lon = get_metadata(fname)
            if lat is None or lon is None:
                print(f"   [SKIP] No coords in filename: {fname}")
                continue
            
            route_points.append((lat, lon))
            valid_images.append(fname)
            
            # Run model
            results = model(path, verbose=False)
            
            dets_for_this_image = []
            is_manual = "manual" in fname
            
            if is_manual:
                # Save annotated image
                annot_path = os.path.join(dest, f"det_{fname}")
                cv2.imwrite(annot_path, results[0].plot())
                
                has_detection = any(box.conf[0].item() > CONFIDENCE_THRESHOLD for box in results[0].boxes)
                if not has_detection:
                    # Encode the image for popup
                    image_b64_cache[fname] = image_to_base64(annot_path)
                    
                    total_detections += 1
                    dets_for_this_image.append({"name": "Manual Report", "conf": 1.0, "cls": -1})
                    image_detections.append(dets_for_this_image)
                    continue
            
            has_any = False
            for box in results[0].boxes:
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                if conf > CONFIDENCE_THRESHOLD:
                    if not has_any:
                        # Save annotated image (once per image)
                        annot_path = os.path.join(dest, f"det_{fname}")
                        cv2.imwrite(annot_path, results[0].plot())
                        # Encode for popup
                        image_b64_cache[fname] = image_to_base64(annot_path)
                        has_any = True
                    
                    det_name = CLASS_NAMES.get(cls, f"Class_{cls}")
                    dets_for_this_image.append({"name": det_name, "conf": conf, "cls": cls})
                    total_detections += 1
                    print(f"   [FOUND] {det_name} ({int(conf*100)}%) @ {lat:.5f},{lon:.5f}")
            
            image_detections.append(dets_for_this_image)
        
        if not route_points:
            print(f"   [SKIP] No valid route points for {name}.")
            continue
        
        route_det_count = sum(len(d) for d in image_detections)
        print(f"   ✅ {len(route_points)} valid points, {route_det_count} total detections on this route.")
        
        all_routes_data.append({
            "name": name,
            "base_color": base_color,
            "dest": dest,
            "route_points": route_points,
            "image_detections": image_detections,
            "valid_images": valid_images,
            "image_b64_cache": image_b64_cache,
        })
    
    # ============================================
    # DRAWING PHASE — Layered for visibility
    # ============================================
    # Layer 1: Draw ALL base route lines first (bottom layer)
    for rd in all_routes_data:
        # For Vinayak, draw a thick purple glow so it's always visible
        if "Vinayak" in rd["name"]:
            # Thick purple glow underneath everything
            folium.PolyLine(
                locations=rd["route_points"],
                color=rd["base_color"],
                weight=14,
                opacity=0.35,
                tooltip=f"{rd['name']} — Full Route (purple highlight)",
            ).add_to(m)
        
        # Normal base line
        folium.PolyLine(
            locations=rd["route_points"],
            color=rd["base_color"],
            weight=5,
            opacity=0.8,
            tooltip=f"{rd['name']} — Full Route",
        ).add_to(m)
    
    # Layer 2: Draw severity overlays for Highway ONLY first
    for rd in all_routes_data:
        if "Vinayak" in rd["name"]:
            continue  # Skip Vinayak for now, draw it on top later
        _draw_severity_overlays(m, rd)
    
    # Layer 3: Re-draw Vinayak purple line ON TOP of Highway severity
    for rd in all_routes_data:
        if "Vinayak" not in rd["name"]:
            continue
        # Thick purple glow on top
        folium.PolyLine(
            locations=rd["route_points"],
            color=rd["base_color"],
            weight=12,
            opacity=0.45,
            tooltip=f"{rd['name']} — Full Route",
        ).add_to(m)
        # Solid purple center line
        folium.PolyLine(
            locations=rd["route_points"],
            color=rd["base_color"],
            weight=5,
            opacity=0.9,
            tooltip=f"{rd['name']} — Full Route",
        ).add_to(m)
    
    # Layer 4: Draw Vinayak severity overlays on top
    for rd in all_routes_data:
        if "Vinayak" not in rd["name"]:
            continue
        _draw_severity_overlays(m, rd)
    
    # Layer 5: Place ALL detection markers (top layer)
    for rd in all_routes_data:
        _draw_detection_markers(m, rd)

    # ============================================
    # LEGEND
    # ============================================
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000; 
                background:white; padding:15px 20px; border:2px solid #333; 
                border-radius:8px; font-family:Arial; font-size:13px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
        <h4 style="margin:0 0 10px 0; font-size:15px;">🛣️ Pothole Patrol — Legend</h4>
        <p style="margin:3px 0;"><span style="display:inline-block;width:30px;height:4px;background:#2196F3;vertical-align:middle;border-radius:2px;"></span> Highway Route (NH48)</p>
        <p style="margin:3px 0;"><span style="display:inline-block;width:30px;height:6px;background:#9C27B0;vertical-align:middle;border-radius:2px;"></span> Vinayak Marg Detour</p>
        <hr style="margin:8px 0;">
        <p style="margin:3px 0;"><span style="display:inline-block;width:30px;height:8px;background:#FFD700;vertical-align:middle;border-radius:2px;"></span> Minor Damage (1-2)</p>
        <p style="margin:3px 0;"><span style="display:inline-block;width:30px;height:8px;background:#FF8C00;vertical-align:middle;border-radius:2px;"></span> Moderate Damage (3-5)</p>
        <p style="margin:3px 0;"><span style="display:inline-block;width:30px;height:8px;background:#FF0000;vertical-align:middle;border-radius:2px;"></span> Severe Damage (6+)</p>
        <hr style="margin:8px 0;">
        <p style="margin:3px 0;"><span style="color:orange;">●</span> Crack Detection</p>
        <p style="margin:3px 0;"><span style="color:red;">●</span> Pothole Detection</p>
        <p style="margin:3px 0;"><span style="color:purple;">📌</span> User Report</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ============================================
    # TITLE
    # ============================================
    title_html = """
    <div style="position:fixed; top:15px; left:50%; transform:translateX(-50%); z-index:1000; 
                background:linear-gradient(135deg, #1a1a2e, #16213e); color:white; 
                padding:12px 30px; border-radius:30px; font-family:Arial; 
                font-size:18px; font-weight:bold; text-align:center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
        🚗 Pothole Patrol — MUJ to Jaipur Station
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    m.save(OUTPUT_MAP)
    print(f"\n🎉 MASTER MAP SAVED: {OUTPUT_MAP}")
    print(f"   Total detections across all routes: {total_detections}")


def _draw_severity_overlays(m, rd):
    """Draw severity-colored overlay segments on a route."""
    name = rd["name"]
    route_points = rd["route_points"]
    image_detections = rd["image_detections"]
    num_images = len(route_points)
    
    for seg_start in range(0, num_images, SEGMENT_SIZE):
        seg_end = min(seg_start + SEGMENT_SIZE, num_images)
        
        # Count detections in this segment
        seg_detection_count = 0
        for i in range(seg_start, seg_end):
            seg_detection_count += len(image_detections[i])
        
        severity = classify_severity(seg_detection_count)
        
        if severity == "none":
            continue  # Don't overlay clean segments
        
        color = SEVERITY_COLORS[severity]
        
        # Draw the segment as a thick colored overlay
        seg_points = route_points[seg_start:seg_end + 1] if seg_end < num_images else route_points[seg_start:]
        
        if len(seg_points) >= 2:
            severity_label = {
                "low": "⚠️ Minor Damage",
                "medium": "🟠 Moderate Damage",
                "high": "🔴 Severe Damage"
            }[severity]
            
            tooltip_text = f"{name}: {severity_label} ({seg_detection_count} detections)"
            
            folium.PolyLine(
                locations=seg_points,
                color=color,
                weight=9,
                opacity=0.85,
                tooltip=tooltip_text,
            ).add_to(m)


def _draw_detection_markers(m, rd):
    """Place individual detection markers with embedded images."""
    name = rd["name"]
    route_points = rd["route_points"]
    image_detections = rd["image_detections"]
    valid_images = rd["valid_images"]
    image_b64_cache = rd["image_b64_cache"]
    dest = rd["dest"]
    
    for i, (lat, lon) in enumerate(route_points):
        if not image_detections[i]:
            continue
        
        fname = valid_images[i]
        
        for det in image_detections[i]:
            if det["cls"] == -1:
                # Manual report marker
                img_b64 = image_b64_cache.get(fname, "")
                img_tag = f"<img src='{img_b64}' width='300' style='border-radius:5px;'>" if img_b64 else "<p><i>Image unavailable</i></p>"
                
                popup_html = f"""
                    <div style='width:320px;'>
                        <h4>User Reported Damage (High Damage)</h4>
                        <p><b>Route:</b> {name}</p>
                        {img_tag}
                        <p>Coords: {lat:.5f}, {lon:.5f}</p>
                    </div>
                """
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_html, max_width=350),
                    tooltip=f"{name}: Manual Report",
                    icon=folium.Icon(color='darkpurple', icon='info-sign')
                ).add_to(m)
                continue
            
            det_name = det["name"]
            conf = det["conf"]
            
            # Marker color based on detection type
            if det_name == "Pothole":
                marker_color = "red"
            elif "Crack" in det_name:
                marker_color = "orange"
            else:
                marker_color = "gray"
            
            # Get embedded image
            img_b64 = image_b64_cache.get(fname, "")
            if not img_b64:
                # Try loading it now (might have been saved but not cached)
                annot_path = os.path.join(dest, f"det_{fname}")
                if os.path.exists(annot_path):
                    img_b64 = image_to_base64(annot_path)
            
            img_tag = f"<img src='{img_b64}' width='300' style='border-radius:5px;'>" if img_b64 else "<p><i>Image unavailable</i></p>"
            
            popup_html = f"""
                <div style='width:320px;'>
                    <h4>{det_name} ({int(conf*100)}%)</h4>
                    <p><b>Route:</b> {name}</p>
                    {img_tag}
                    <p>Coords: {lat:.5f}, {lon:.5f}</p>
                </div>
            """
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=350),
                tooltip=f"{det_name} ({int(conf*100)}%)"
            ).add_to(m)


if __name__ == "__main__":
    process_routes()
