import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random

DB_NAME = "pothole_detection.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create detections table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        damage_type TEXT,
        confidence REAL,
        severity TEXT,
        latitude REAL,
        longitude REAL,
        road_name TEXT,
        image_path TEXT
    )
    """)
    conn.commit()
    
    # Check if we need to seed mock data
    cursor.execute("SELECT COUNT(*) FROM detections")
    count = cursor.fetchone()[0]
    
    if count == 0:
        # Seed realistic Pakistan road data
        damage_types = [
            "Pothole", 
            "Longitudinal Crack", 
            "Transverse Crack", 
            "Alligator Crack", 
            "Road Surface Erosion", 
            "Water-Damaged Section"
        ]
        severities = ["Low", "Medium", "High"]
        
        # Roads and locations around Islamabad/Rawalpindi/Lahore/Karachi
        locations = [
            # Islamabad
            {"city": "Islamabad", "road": "Jinnah Avenue, Blue Area", "lat": 33.7104, "lng": 73.0560},
            {"city": "Islamabad", "road": "Kashmir Highway / Srinagar Highway", "lat": 33.6844, "lng": 73.0479},
            {"city": "Islamabad", "road": "IJP Road", "lat": 33.6385, "lng": 73.0784},
            {"city": "Islamabad", "road": "Park Road, Chattha Bakhtawar", "lat": 33.6657, "lng": 73.1378},
            {"city": "Islamabad", "road": "Islamabad Expressway", "lat": 33.6001, "lng": 73.1554},
            # Rawalpindi
            {"city": "Rawalpindi", "road": "Murree Road", "lat": 33.6051, "lng": 73.0640},
            {"city": "Rawalpindi", "road": "Mall Road, Saddar", "lat": 33.5960, "lng": 73.0520},
            {"city": "Rawalpindi", "road": "Peshawar Road", "lat": 33.5900, "lng": 73.0100},
            # Lahore
            {"city": "Lahore", "road": "Mall Road (Shahrah-e-Quaid-e-Azam)", "lat": 31.5565, "lng": 74.3265},
            {"city": "Lahore", "road": "Ferozepur Road", "lat": 31.4800, "lng": 74.3400},
            {"city": "Lahore", "road": "Canal Bank Road, Gulberg", "lat": 31.5200, "lng": 74.3500},
            {"city": "Lahore", "road": "Jail Road", "lat": 31.5398, "lng": 74.3382},
            # Karachi
            {"city": "Karachi", "road": "Shahrah-e-Faisal", "lat": 24.8716, "lng": 67.0594},
            {"city": "Karachi", "road": "Korangi Road", "lat": 24.8250, "lng": 67.0900},
            {"city": "Karachi", "road": "Sunset Boulevard, Clifton", "lat": 24.8160, "lng": 67.0450},
            {"city": "Karachi", "road": "University Road", "lat": 24.9180, "lng": 67.1000}
        ]
        
        # Populate about 40 mock points spread over the last 30 days
        now = datetime.now()
        for i in range(45):
            loc = random.choice(locations)
            d_type = random.choice(damage_types)
            sev = random.choice(severities)
            conf = round(random.uniform(0.65, 0.98), 2)
            
            # Slightly jitter coordinates for variety
            lat = loc["lat"] + random.uniform(-0.005, 0.005)
            lng = loc["lng"] + random.uniform(-0.005, 0.005)
            
            # Timestamp spread
            days_ago = random.randint(0, 30)
            hours_ago = random.randint(0, 23)
            timestamp = (now - timedelta(days=days_ago, hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute("""
            INSERT INTO detections (timestamp, damage_type, confidence, severity, latitude, longitude, road_name, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, d_type, conf, sev, lat, lng, f"{loc['road']}, {loc['city']}", ""))
            
        conn.commit()
    conn.close()

def add_detection(damage_type, confidence, severity, latitude, longitude, road_name, image_path=""):
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO detections (timestamp, damage_type, confidence, severity, latitude, longitude, road_name, image_path)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, damage_type, confidence, severity, latitude, longitude, road_name, image_path))
    conn.commit()
    conn.close()

def get_all_detections():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM detections ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def clear_database():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections")
    conn.commit()
    conn.close()
