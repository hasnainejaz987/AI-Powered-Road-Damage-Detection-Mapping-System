import streamlit as st
import pandas as pd
import numpy as np
import cv2
import folium
from streamlit_folium import st_folium
import os
import time
import random

# Import helper modules
from database import init_db, add_detection, get_all_detections, clear_database
from detector import RoadDamageDetector, DAMAGE_CLASSES
from reporter import generate_pdf_report

# Initialize database
init_db()

# Initialize Detector
@st.cache_resource
def get_detector():
    return RoadDamageDetector()

detector = get_detector()

# Set up Streamlit Page configuration
st.set_page_config(
    page_title="AI Road Damage Detection & Mapping",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
    <style>
    /* Main app styles */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    .main-title {
        font-family: 'Inter', sans-serif;
        color: #0F172A;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
        letter-spacing: -0.025em;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    /* Metric Cards */
    .metric-container {
        display: flex;
        gap: 1.25rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 1.5rem;
        flex: 1;
        border: 1px solid #E2E8F0;
        text-align: left;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        transition: all 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border-color: #CBD5E1;
    }
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748B;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
""", unsafe_allow_html=True)
# Clean Header Section (No Sidebar)
st.markdown("<h1 class='main-title'>InspectAI &mdash; Road Quality Mapping</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Automated AI infrastructure inspection, classification, and GPS logging system.</p>", unsafe_allow_html=True)

# Horizontal Top Tabs for Navigation
tabs = st.tabs([
    "📊 Dashboard Overview", 
    "🚧 Real-Time Detection", 
    "🗺️ Interactive Damage Map", 
    "📈 Analytics & Insights", 
    "📋 Export Reports", 
    "⚙️ Settings"
])

# -----------------
# 1. OVERVIEW PAGE
# -----------------
with tabs[0]:
    # Load stats
    df = get_all_detections()
    total = len(df)
    high = len(df[df['severity'] == 'High']) if total > 0 else 0
    med = len(df[df['severity'] == 'Medium']) if total > 0 else 0
    low = len(df[df['severity'] == 'Low']) if total > 0 else 0
    
    # Render KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total}</div><div class="metric-label">Total Damages</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #EF4444;">{high}</div><div class="metric-label">High Severity</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #F59E0B;">{med}</div><div class="metric-label">Medium Severity</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #3B82F6;">{low}</div><div class="metric-label">Low Severity</div></div>', unsafe_allow_html=True)
        
    st.write("---")
    
    st.subheader("📋 Recent Detections Log")
    if not df.empty:
        # Display nicely styled DataFrame
        st.dataframe(df.head(10)[["timestamp", "damage_type", "severity", "confidence", "road_name", "latitude", "longitude"]], use_container_width=True)
    else:
        st.info("No detections loaded in database.")

# --------------------------
# 2. REAL-TIME DETECTION
# --------------------------
with tabs[1]:
    source = st.radio("Select Input Source:", ["Image Upload", "Video File Upload", "Live Camera Feed"])
    
    # Coordinates mapping simulator for the new detections
    cities = {
        "Islamabad": (33.6844, 73.0479, "Kashmir Highway"),
        "Rawalpindi": (33.5960, 73.0520, "Murree Road"),
        "Lahore": (31.5565, 74.3265, "Mall Road"),
        "Karachi": (24.8716, 67.0594, "Shahrah-e-Faisal")
    }
    
    selected_city = st.selectbox("Assign Detection Coordinates City:", list(cities.keys()))
    city_coords = cities[selected_city]
    
    if source == "Image Upload":
        uploaded_file = st.file_uploader("Choose a road image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Convert uploaded file to cv2 image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            st.write("Running YOLOv8 Inspection...")
            annotated_image, detections = detector.detect(image)
            
            # Show original vs annotated
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="YOLOv8 Annotated", use_container_width=True)
            
            # Detections table and database logging
            if detections:
                st.success(f"Successfully identified {len(detections)} instances of road damage!")
                
                # Show results in a clean table
                st.write("### Detection Breakdown:")
                results_df = pd.DataFrame(detections)[["class_name", "confidence", "severity"]]
                st.table(results_df)
                
                if st.button("Log Detections to Database & Map"):
                    for d in detections:
                        # Introduce small random jitter around selected city coordinates
                        lat = city_coords[0] + np.random.uniform(-0.003, 0.003)
                        lng = city_coords[1] + np.random.uniform(-0.003, 0.003)
                        add_detection(
                            damage_type=d["class_name"],
                            confidence=float(d["confidence"]),
                            severity=d["severity"],
                            latitude=lat,
                            longitude=lng,
                            road_name=f"{city_coords[2]}, {selected_city}"
                        )
                    st.success("Logged successfully! View in 'Interactive Damage Map'.")
            else:
                st.info("No road damage detected in this image.")

    elif source == "Video File Upload":
        uploaded_video = st.file_uploader("Choose a road inspection video...", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            # Initialize session states for video
            if "video_detections" not in st.session_state:
                st.session_state.video_detections = None
            if "video_processed_path" not in st.session_state:
                st.session_state.video_processed_path = None
                
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Original Video Uploaded")
                st.video(uploaded_video)
            
            # Save uploaded video to temp file for OpenCV reading
            import tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.close()
            
            with col2:
                st.write("### AI Analysis Actions")
                analyze_btn = st.button("🚧 Run YOLOv8 Video Inspection")
                
            if analyze_btn:
                cap = cv2.VideoCapture(tfile.name)
                
                # Retrieve video specs
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Set up Output Writer
                out_path = "processed_output.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                frame_placeholder = st.empty()
                
                frame_count = 0
                detected_counts = 0
                all_detections = []
                
                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Apply YOLO/Mock detector
                        annotated, detections = detector.detect(frame)
                        out.write(annotated)
                        
                        # Show current processed frame in dashboard
                        frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        # Track detections
                        if detections:
                            detected_counts += len(detections)
                            for d in detections:
                                all_detections.append(d)
                        
                        frame_count += 1
                        progress = min(float(frame_count) / max(total_frames, 1), 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing Frame {frame_count}/{total_frames} (Found: {detected_counts} issues)...")
                    
                    st.session_state.video_detections = all_detections
                    st.session_state.video_processed_path = out_path
                    st.success("YOLOv8 Video Inspection Completed!")
                    
                finally:
                    cap.release()
                    out.release()
                    try:
                        os.unlink(tfile.name)
                    except Exception:
                        pass
            
            # Render download/logging options if processing is completed
            if st.session_state.video_processed_path is not None:
                st.write("---")
                col_actions1, col_actions2 = st.columns(2)
                
                with col_actions1:
                    if st.session_state.video_detections:
                        st.write(f"### Detections Found: {len(st.session_state.video_detections)}")
                        if st.button("Save Video Detections to Database & Map"):
                            # Sample unique logged damages to prevent flooding database
                            sampled_detections = random.sample(st.session_state.video_detections, min(10, len(st.session_state.video_detections)))
                            for d in sampled_detections:
                                lat = city_coords[0] + np.random.uniform(-0.003, 0.003)
                                lng = city_coords[1] + np.random.uniform(-0.003, 0.003)
                                add_detection(
                                    damage_type=d["class_name"],
                                    confidence=float(d["confidence"]),
                                    severity=d["severity"],
                                    latitude=lat,
                                    longitude=lng,
                                    road_name=f"{city_coords[2]} (Video Link), {selected_city}"
                                )
                            st.success("Successfully logged annotated locations to database!")
                
                with col_actions2:
                    if os.path.exists(st.session_state.video_processed_path):
                        st.write("### Download Results")
                        with open(st.session_state.video_processed_path, "rb") as file:
                            st.download_button(
                                label="📥 Download Annotated Video File",
                                data=file,
                                file_name="Annotated_Road_Damage.mp4",
                                mime="video/mp4"
                            )


    elif source == "Live Camera Feed":
        camera_mode = st.radio("Select Camera Source:", ["Real Webcam (Local USB Camera)", "Simulated Dashcam Feed"])
        
        if camera_mode == "Real Webcam (Local USB Camera)":
            st.write("Reading stream from local camera index 0 (webcam / dashcam)...")
        else:
            st.write("Simulating live high-resolution camera feed from vehicular dashcam/drone...")
        
        # Stream controls
        run = st.button("Start Live Camera Stream")
        
        frame_placeholder = st.empty()
        log_placeholder = st.empty()
        
        if run:
            if camera_mode == "Real Webcam (Local USB Camera)":
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Error: Could not access the local USB camera. Please verify it is connected and not in use by another application.")
                else:
                    try:
                        count = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Failed to capture frame from webcam.")
                                break
                            
                            annotated, detections = detector.detect(frame)
                            frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                            
                            # Log detections periodically
                            if count % 20 == 0 and detections:
                                d = random.choice(detections)
                                lat = city_coords[0] + np.random.uniform(-0.004, 0.004)
                                lng = city_coords[1] + np.random.uniform(-0.004, 0.004)
                                add_detection(
                                    damage_type=d["class_name"],
                                    confidence=float(d["confidence"]),
                                    severity=d["severity"],
                                    latitude=lat,
                                    longitude=lng,
                                    road_name=f"{city_coords[2]} (Real Cam), {selected_city}"
                                )
                                log_placeholder.info(f"💾 Automatically logged **{d['class_name']} ({d['severity']})** at coords: {lat:.5f}, {lng:.5f}")
                            
                            count += 1
                            time.sleep(0.05)
                    finally:
                        cap.release()
            else:
                # Generate a mock highway frame
                base_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
                # Draw simple road perspective lanes
                cv2.line(base_frame, (320, 180), (50, 480), (255, 255, 255), 5)
                cv2.line(base_frame, (320, 180), (590, 480), (255, 255, 255), 5)
                cv2.line(base_frame, (320, 180), (320, 480), (255, 255, 255), 2) # Center dashed lane
                
                count = 0
                while True:
                    frame = base_frame.copy()
                    # Introduce moving noise/elements
                    cv2.putText(frame, f"FPS: {random.randint(28, 30)} | GPS: Locked", (15, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 185, 129), 2)
                    cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (15, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Draw mock pothole/cracks shifting downward to simulate driving
                    offset = (count * 15) % 300
                    if offset < 280:
                        px = 200 + int(offset * 0.4)
                        py = 200 + offset
                        cv2.ellipse(frame, (px, py), (15 + int(offset*0.1), 8 + int(offset*0.05)), 0, 0, 360, (40, 40, 40), -1)
                    
                    annotated, detections = detector.detect(frame)
                    frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    if count % 20 == 0 and detections:
                        d = random.choice(detections)
                        lat = city_coords[0] + np.random.uniform(-0.004, 0.004)
                        lng = city_coords[1] + np.random.uniform(-0.004, 0.004)
                        add_detection(
                            damage_type=d["class_name"],
                            confidence=float(d["confidence"]),
                            severity=d["severity"],
                            latitude=lat,
                            longitude=lng,
                            road_name=f"{city_coords[2]} (Live Cam), {selected_city}"
                        )
                        log_placeholder.info(f"💾 Automatically logged **{d['class_name']} ({d['severity']})** at coords: {lat:.5f}, {lng:.5f}")
                    
                    count += 1
                    time.sleep(0.1)


# -----------------------------
# 3. INTERACTIVE DAMAGE MAP
# -----------------------------
with tabs[2]:
    df = get_all_detections()
    
    if df.empty:
        st.info("No data available to display on map. Log some detections first.")
    else:
        # Filtering Controls
        col1, col2 = st.columns(2)
        with col1:
            selected_severities = st.multiselect("Filter Severity:", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        with col2:
            selected_types = st.multiselect("Filter Damage Type:", DAMAGE_CLASSES, default=DAMAGE_CLASSES)
            
        filtered_df = df[df['severity'].isin(selected_severities) & df['damage_type'].isin(selected_types)]
        
        # Center map on mean of filtered detections, or default to Islamabad
        if not filtered_df.empty:
            center_lat = filtered_df['latitude'].mean()
            center_lng = filtered_df['longitude'].mean()
        else:
            center_lat, center_lng = 33.6844, 73.0479
            
        m = folium.Map(location=[center_lat, center_lng], zoom_start=11, tiles="CartoDB positron")
        
        # Color mapping for severity
        severity_colors = {
            "High": "red",
            "Medium": "orange",
            "Low": "blue"
        }
        
        for idx, row in filtered_df.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"<strong>{row['damage_type']}</strong><br>Severity: {row['severity']}<br>Confidence: {int(row['confidence']*100)}%<br>Location: {row['road_name']}",
                icon=folium.Icon(color=severity_colors.get(row['severity'], 'green'), icon='exclamation-sign')
            ).add_to(m)
            
        st_folium(m, width="100%", height=600)
        st.write(f"Showing {len(filtered_df)} filtered items on the map.")

# -----------------------------
# 4. ANALYTICS & INSIGHTS
# -----------------------------
with tabs[3]:
    df = get_all_detections()
    
    if df.empty:
        st.info("No data logged in database to generate charts.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Damage Type Distribution")
            type_counts = df['damage_type'].value_counts()
            st.bar_chart(type_counts, color="#10B981")
            
        with col2:
            st.write("### Severity Level Breakdown")
            sev_counts = df['severity'].value_counts()
            st.bar_chart(sev_counts, color="#F59E0B")
            
        st.write("### Detection Hotspots (Most Damaged Roads)")
        road_counts = df['road_name'].value_counts().head(10)
        st.bar_chart(road_counts, color="#EF4444")

# -----------------------------
# 5. EXPORT REPORTS
# -----------------------------
with tabs[4]:
    df = get_all_detections()
    
    if df.empty:
        st.info("No data available to export.")
    else:
        st.write(f"Generate PDF report of current logs ({len(df)} total items).")
        
        # Date Filter option
        st.write("### Filter Options")
        city_filter = st.multiselect("Select Cities to include:", ["Islamabad", "Rawalpindi", "Lahore", "Karachi"], default=["Islamabad", "Rawalpindi", "Lahore", "Karachi"])
        
        # Filter dataframe by city substring
        regex_pattern = '|'.join(city_filter)
        filtered_df = df[df['road_name'].str.contains(regex_pattern, case=False, na=False)]
        
        if st.button("Build PDF Report"):
            pdf_path = generate_pdf_report(filtered_df, "road_damage_inspection_report.pdf")
            
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_file,
                    file_name="Road_Damage_Report.pdf",
                    mime="application/pdf"
                )
            st.success("PDF Report generated successfully! Click the button above to download.")

# -----------------------------
# 6. SETTINGS & DB MGMT
# -----------------------------
with tabs[5]:
    st.write("### Model Options")
    model_opt = st.selectbox("Select Active YOLO Weights:", ["Mock Simulator Model (Default)", "yolov8n.pt", "best.pt (Custom Trained)"])
    
    st.write("---")
    st.write("### Database Maintenance")
    if st.button("🧹 Clear All Database Logs", help="Permanently delete all logs from database."):
        clear_database()
        st.success("All database entries cleared successfully!")
