# Pothole Patrol — AI-Powered Road Damage Intelligence

[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8m-red.svg)](https://github.com/ultralytics/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Vercel](https://img.shields.io/badge/Deployment-Vercel-black.svg)](https://project-pothole.vercel.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

Pothole Patrol is a computer vision and geospatial intelligence system designed to automate road damage assessment. Using a customized **YOLOv8m** object detection model trained on merged public and custom datasets, the system identifies road distress from video feeds (like dashcams) and static photos, mapping the detections in real time onto interactive geospatial reports.

👉 **Live Landing Page**: [project-pothole.vercel.app](https://project-pothole.vercel.app/)  
👉 **Interactive Map Output**: `outputs/maps/pothole_patrol_master.html` (covers MUJ Campus → Jaipur Railway Station route)

---

## 🌟 Key Features

*   **Deep Learning Detection**: Employs YOLOv8m optimized for edge execution (30+ FPS) to identify and classify 5 distinct road damage classes.
*   **Geospatial Mapping**: Converts visual bounding box coordinates into interactive `folium` leaf maps color-coded by damage priority.
*   **Video Processing (`app.py`)**: Runs inference on continuous video streams, logs detections along a simulated GPS route, captures damage snapshots, and exports an HTML map summary.
*   **Batch Image Mapping (`predict_images.py`)**: Processes directories of geo-tagged images, parses GPS EXIF/filename metadata, and generates an interactive map report with embedded damage images in popups.
*   **Responsive Dashboard**: Sleek, modern responsive landing page built with Syne and DM Sans typography displaying route status, statistics, and project workflows.

---

## 🛣️ Damage Classification System

The model is trained to identify and categorize road anomalies according to international standards (RDD):

| Class ID | Damage Type | Color Code | Description | Severity / Priority |
| :--- | :--- | :---: | :--- | :--- |
| **0** | `Longitudinal Crack` | **Orange** | Cracks parallel to the direction of travel | Low |
| **1** | `Transverse Crack` | **Blue** | Cracks perpendicular to the direction of travel | Low |
| **2** | `Alligator Crack` | **Dark Red** | Interconnected cracks resembling alligator skin | Medium |
| **3** | `Pothole` | **Red** | Bowl-shaped depressions in the pavement | **High** |
| **4** | `Other Damage` | **Purple** | Pothole patches, weathering, or other failures | Medium |

---

## 📁 Repository Structure

```
├── data/                    # Source datasets (videos and route images) [Gitignored]
├── models/                  # Trained weights (YOLOv8 best.pt) [Gitignored]
├── outputs/                 # Output directories
│   ├── detections/          # Annotated image detections
│   ├── maps/                # Generated interactive HTML maps (Folium)
│   └── pothole_screenshots/   # Snapshots captured from video runs
├── utils/                   # Helper utility scripts
├── app.py                   # Real-time video-to-map pipeline
├── predict_images.py        # Batch image geo-mapping script
├── index.html               # Main widescreen landing page
└── README.md                # Project documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the required libraries:
```bash
pip install opencv-python ultralytics folium pillow
```

### 2. Set Up Model & Data
Since models and large media files are ignored by git, place them in the following paths:
*   Save your trained YOLOv8 model weights file to: `models/best.pt`
*   Save your test video file to: `data/videos/test_video.mp4`
*   Place geo-tagged route photos in: `data/route_photos/`

### 3. Run Live Video Pipeline
Process a video stream, capture screenshots of active damage, and export the route map:
```bash
python app.py
```
*   **Interactive Control**: Press `Q` while the OpenCV frame viewer is active to stop processing early and generate the map immediately.
*   **Output**: Opens or saves to `outputs/maps/pothole_map.html`.

### 4. Run Batch Image Mapping
Process a folder of geo-tagged images (incorporating coordinates in filenames like `photo_001_26.843_75.565.jpg`):
```bash
python predict_images.py
```
*   **Output**: Saves interactive maps to `outputs/maps/route_pothole_map.html` with image bounding boxes embedded directly inside map popups!

---

## 🖥️ Landing Page Customizations

The landing page (`index.html`) is structured with premium modern design layouts:
*   **Grid layout**: The header utilizes a responsive 2-column grid displaying content copy and map mockups side-by-side.
*   **Typography**: Implements Google Fonts `Syne` and `DM Sans`.
*   **Icons**: Integrates the Tabler Icons library (`ti-cpu`, `ti-map-2`, `ti-brand-github`) via CDN.

---

## ☁️ Deployment

The frontend dashboard is deployed live via **Vercel**. Since git integration is active, pushing updates to the main branch triggers builds automatically:

```bash
git add .
git commit -m "docs: add comprehensive README"
git push origin main
```
Track deployment builds in real-time in your Vercel Dashboard!

---

Developed by **Piyush Varma** (MUJ AIML)
