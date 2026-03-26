import requests
import os
import time

# ==========================================================
# CONFIGURATION
# ==========================================================
API_KEY = "ENCRYPTED KEY"  # Replace with your actual key
IMG_SIZE = "640x640"
OUTPUT_DIR = "route_photos_vinayak"  # NEW FOLDER FOR THIS TRIP
STOPS = [
    {"name": "Manipal University Jaipur", "lat": 26.8439, "lon": 75.5652},
    {"name": "Jaipur Railway Station", "lat": 26.9181, "lon": 75.7876}
]
NUM_INTERVALS = 20  # Number of images to capture between stops

# ==========================================================
# HELPERS
# ==========================================================

def decode_polyline(polyline_str):
    """Decodes a polyline string into a list of lat/lon dicts."""
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'lat': 0, 'lng': 0}

    while index < len(polyline_str):
        for unit in ['lat', 'lng']:
            shift, result = 0, 0
            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if not byte >= 0x20:
                    break

            if (result & 1):
                changes[unit] = ~(result >> 1)
            else:
                changes[unit] = (result >> 1)

        lat += changes['lat'] / 1e5
        lng += changes['lng'] / 1e5
        coordinates.append({'lat': lat, 'lon': lng})

    return coordinates

def get_route_points(origin, destination, waypoints=None):
    """Fetches points along the road route using Directions API."""
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&key={API_KEY}"
    if waypoints:
        url += f"&waypoints={waypoints}"
    response = requests.get(url).json()
    
    if response['status'] == 'OK':
        polyline = response['routes'][0]['overview_polyline']['points']
        return decode_polyline(polyline)
    else:
        print(f"[ERROR] Directions API failed: {response['status']}")
        if 'error_message' in response:
            print(f"Details: {response['error_message']}")
        return []

def download_streetview(lat, lon, filename):
    url = f"https://maps.googleapis.com/maps/api/streetview?size={IMG_SIZE}&location={lat},{lon}&key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        # Check if the image skip placeholder (8.6KB)
        if len(response.content) < 10000:
            print(f"[SKIP] No imagery for {lat}, {lon}")
            return False
        with open(os.path.join(OUTPUT_DIR, filename), "wb") as f:
            f.write(response.content)
        print(f"[OK] Saved: {filename}")
        return True
    else:
        print(f"[ERROR] Failed to download for {lat}, {lon}: {response.status_code}")
        return False

if __name__ == "__main__":
    import shutil
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"STARTING ROAD TRIP collection: MUJ to Highway via Vinayak Marg...")
    
    # Precise coordinates from the screenshot
    origin = "26.8439,75.5652" # MUJ
    destination = "26.8305,75.5739" # Highway Junction
    waypoints = "26.8584,75.5711" # Vinayak Marg Turn
    
    points = get_route_points(origin, destination, waypoints)
    
    if not points:
        print("❌ Could not find a road route. Make sure 'Directions API' is enabled!")
        exit()

    print(f"Found {len(points)} points along the road. Downloading images...")
    
    count = 0
    # Download every 2nd point to capture ~100 images
    for i in range(0, len(points), 2):
        pt = points[i]
        fname = f"road_{count:03d}_{pt['lat']:.5f}_{pt['lon']:.5f}.jpg"
        if download_streetview(pt['lat'], pt['lon'], fname):
            count += 1
            time.sleep(0.3)
        
    print(f"\nDONE! {count} road photos saved to: {OUTPUT_DIR}")
