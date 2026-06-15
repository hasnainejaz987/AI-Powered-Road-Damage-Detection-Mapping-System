# AI-Powered Road Damage Detection and Mapping System

A modern, high-performance web application designed to automatically detect road anomalies (potholes, structural cracks, surface erosion) from image/video streams and map their exact GPS coordinates for municipal planning.

This project is tailored for road conditions in Pakistan (with simulated telemetry and mapping centered around Islamabad, Rawalpindi, Lahore, and Karachi).

---

## Key Features

1. **Real-Time YOLOv8 Detection**: Inspect road images or stream mock vehicular dashcam video to detect potholes and cracks with bounding boxes and high-precision confidence ratings.
2. **GPS Tagging & Interactive Mapping**: Uses Folium (Leaflet-based mapping) to place coordinates of every detected road damage with visual markers colored dynamically according to damage severity levels.
3. **Severity Classification**:
   - **Low**: Small cracks/deterioration.
   - **Medium**: Average-sized anomalies needing standard maintenance.
   - **High**: Large potholes posing immediate accident hazards.
4. **Analytics Dashboard**: Dynamic charts visualizing road damage trends, hotspot locations, and class breakdowns.
5. **PDF Report Engine**: Export PDF reports detailing active road repair items, coordinates, and priority lists for local authorities (e.g., NHA, CDA, KMC).

---

## Project Structure

- `app.py`: Streamlit-based web dashboard interface.
- `detector.py`: Model loader for YOLOv8 with an interactive, robust mock fallback system.
- `database.py`: Handles SQLite database initialization, storage, and retrieval.
- `reporter.py`: Generates official PDF reports using ReportLab.
- `requirements.txt`: Project dependencies list.

---

## Installation & Running

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```
