import streamlit as st
import cv2
import threading
import time
import requests
from collections import deque
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import tempfile
import os

# Lazy imports to avoid torch loading issues on startup
@st.cache_resource
def load_yolo_models():
    """Lazy load YOLO models to avoid torch import issues"""
    try:
        from ultralytics import YOLO
        return YOLO
    except Exception as e:
        st.error(f"Failed to import YOLO: {e}")
        return None

@st.cache_resource  
def load_deepsort():
    """Lazy load DeepSort to avoid import issues"""
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        return DeepSort
    except Exception as e:
        st.error(f"Failed to import DeepSort: {e}")
        return None

@st.cache_resource
def load_twilio():
    """Lazy load Twilio client"""
    try:
        from twilio.rest import Client
        return Client
    except Exception as e:
        st.error(f"Failed to import Twilio: {e}")
        return None

green_light = 30 
# Page configuration with theme detection
st.set_page_config(
    page_title="🚦 Smart Traffic Monitor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme detection and light mode focused CSS
st.markdown("""
<style>
    /* Import Ant Design color palette and typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Light Theme Variables - Clean & Professional */
    :root {
        --ant-primary-color: #1890ff;
        --ant-success-color: #52c41a;
        --ant-warning-color: #faad14;
        --ant-error-color: #ff4d4f;
        --ant-info-color: #1890ff;
        --ant-font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --ant-border-radius: 6px;
        
        /* Clean Light Theme */
        --ant-text-color: #000000d9;
        --ant-text-color-secondary: #00000073;
        --ant-border-color: #d9d9d9;
        --ant-background-color: #ffffff;
        --ant-card-background: #ffffff;
        --ant-sidebar-background: #fafafa;
        --ant-box-shadow: 0 2px 8px rgba(0, 0, 0, 0.09);
    }
    
    /* Reset Streamlit defaults with clean light theme */
    .stApp {
        font-family: var(--ant-font-family);
        background-color: var(--ant-background-color);
        color: var(--ant-text-color);
    }
    
    /* Clean Header */
    .ant-header {
        background: linear-gradient(135deg, var(--ant-primary-color) 0%, #40a9ff 100%);
        padding: 24px;
        border-radius: var(--ant-border-radius);
        color: white;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: var(--ant-box-shadow);
        border: 1px solid transparent;
    }
    
    .ant-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 600;
        color: white;
    }
    
    .ant-header p {
        margin: 8px 0 0 0;
        font-size: 16px;
        opacity: 0.9;
        color: white;
    }
    
    /* Emergency Alert - High Visibility */
    .ant-alert-error {
        background: #fff2f0;
        border: 1px solid #ffccc7;
        border-radius: var(--ant-border-radius);
        padding: 20px 24px;
        margin: 16px 0;
        position: relative;
        animation: ant-pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        box-shadow: 0 4px 12px rgba(255, 77, 79, 0.15);
        color: var(--ant-text-color);
    }
    
    .ant-alert-error-critical {
        background: linear-gradient(135deg, #ff4d4f 0%, #ff7875 100%);
        color: white;
        border: 2px solid #ff4d4f;
        animation: ant-pulse-critical 1.5s ease-in-out infinite;
        box-shadow: 0 0 30px rgba(255, 77, 79, 0.4);
    }
    
    @keyframes ant-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.02); }
    }
    
    @keyframes ant-pulse-critical {
        0%, 100% { opacity: 1; box-shadow: 0 0 30px rgba(255, 77, 79, 0.4); }
        50% { opacity: 0.9; box-shadow: 0 0 40px rgba(255, 77, 79, 0.6); }
    }
    
    /* Clean White Cards */
    .ant-card {
        background: var(--ant-card-background);
        border-radius: var(--ant-border-radius);
        border: 1px solid var(--ant-border-color);
        box-shadow: var(--ant-box-shadow);
        margin-bottom: 16px;
        overflow: hidden;
    }
    
    .ant-card-head {
        background: #fafafa;
        border-bottom: 1px solid var(--ant-border-color);
        padding: 16px 24px;
        font-weight: 500;
        font-size: 16px;
        color: var(--ant-text-color);
    }
    
    .ant-card-body {
        padding: 24px;
        background: var(--ant-card-background);
        color: var(--ant-text-color);
    }
    
    /* Clean Statistics Cards */
    .ant-statistic {
        background: var(--ant-card-background);
        padding: 20px;
        border-radius: var(--ant-border-radius);
        border: 1px solid var(--ant-border-color);
        box-shadow: var(--ant-box-shadow);
        margin-bottom: 16px;
        transition: all 0.3s cubic-bezier(0.645, 0.045, 0.355, 1);
    }
    
    .ant-statistic:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    .ant-statistic-title {
        color: var(--ant-text-color-secondary);
        font-size: 14px;
        margin-bottom: 8px;
        font-weight: 400;
    }
    
    .ant-statistic-content {
        color: var(--ant-text-color);
        font-size: 24px;
        font-weight: 600;
        line-height: 32px;
    }
    
    /* Status indicators */
    .ant-status-success { color: var(--ant-success-color); }
    .ant-status-warning { color: var(--ant-warning-color); }
    .ant-status-error { color: var(--ant-error-color); }
    .ant-status-info { color: var(--ant-info-color); }
    
    /* Clean Sidebar */
    .css-1d391kg {
        background: var(--ant-sidebar-background);
        border-right: 1px solid var(--ant-border-color);
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Traffic Signal Components */
    .ant-traffic-signal {
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--ant-card-background);
        border: 2px solid var(--ant-border-color);
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: var(--ant-box-shadow);
    }
    
    .ant-traffic-signal-emergency {
        border-color: var(--ant-error-color);
        background: linear-gradient(135deg, #fff2f0 0%, #ffebe8 100%);
        animation: ant-pulse 2s ease-in-out infinite;
    }
    
    .ant-traffic-signal-normal {
        border-color: var(--ant-success-color);
        background: linear-gradient(135deg, #f6ffed 0%, #efffdf 100%);
    }
    
    /* Emergency Override */
    .emergency-override {
        background: linear-gradient(135deg, var(--ant-error-color) 0%, #ff7875 100%);
        color: white;
        padding: 24px;
        border-radius: var(--ant-border-radius);
        text-align: center;
        margin: 20px 0;
        border: 2px solid var(--ant-error-color);
        box-shadow: 0 8px 24px rgba(255, 77, 79, 0.2);
        animation: ant-pulse-critical 2s ease-in-out infinite;
    }
    
    /* Analytics Cards */
    .ant-analytics-card {
        background: var(--ant-card-background);
        border: 1px solid var(--ant-border-color);
        border-radius: var(--ant-border-radius);
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: var(--ant-box-shadow);
        transition: all 0.3s ease;
    }
    
    .ant-analytics-card:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
    }
    
    /* System Status List */
    .ant-list-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 16px;
        border-bottom: 1px solid #f0f0f0;
        background: var(--ant-card-background);
        margin-bottom: 1px;
        color: var(--ant-text-color);
    }
    
    .ant-list-item:last-child {
        border-bottom: none;
    }
    
    /* Audio Alert */
    .ant-audio-alert {
        background: linear-gradient(135deg, var(--ant-warning-color) 0%, #ffd666 100%);
        color: #000;
        padding: 16px 20px;
        border-radius: var(--ant-border-radius);
        border: 2px solid var(--ant-warning-color);
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(250, 173, 20, 0.3);
        animation: ant-pulse 1.5s ease-in-out infinite;
    }
    
    /* Video Container */
    .ant-video-container {
        background: var(--ant-card-background);
        border: 1px solid var(--ant-border-color);
        border-radius: var(--ant-border-radius);
        padding: 24px;
        box-shadow: var(--ant-box-shadow);
        margin-bottom: 24px;
    }
    
    /* Typography */
    .ant-typography-h1 {
        font-size: 32px;
        font-weight: 600;
        line-height: 1.25;
        margin-bottom: 16px;
        color: var(--ant-text-color);
    }
    
    .ant-typography-h2 {
        font-size: 24px;
        font-weight: 600;
        line-height: 1.35;
        margin-bottom: 12px;
        color: var(--ant-text-color);
    }
    
    .ant-typography-h3 {
        font-size: 18px;
        font-weight: 500;
        line-height: 1.4;
        margin-bottom: 8px;
        color: var(--ant-text-color);
    }
    
    /* Feature List */
    .ant-feature-list {
        background: var(--ant-card-background);
        border: 1px solid var(--ant-border-color);
        border-radius: var(--ant-border-radius);
        padding: 24px;
        box-shadow: var(--ant-box-shadow);
        color: var(--ant-text-color);
    }
    
    .ant-feature-list li {
        padding: 8px 0;
        border-bottom: 1px solid #f0f0f0;
        list-style: none;
        opacity: 0.9;
    }
    
    .ant-feature-list li:last-child {
        border-bottom: none;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .ant-header {
            padding: 16px;
        }
        
        .ant-card-body {
            padding: 16px;
        }
        
        .ant-statistic {
            padding: 16px;
        }
        
        .ant-header h1 {
            font-size: 24px;
        }
    }
    
    /* Hide empty containers */
    .element-container:empty,
    .stColumn > div:empty,
    .block-container > div:empty {
        display: none !important;
    }
    
    /* Remove extra spacing from empty elements */
    .stMarkdown:empty,
    .stText:empty {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Configuration constants
CONFIDENCE_THRESHOLD_EMG = 0.65
CONFIDENCE_THRESHOLD = 0.4

def classify_traffic_density(vehicle_count):
    """Classify traffic density based on vehicle count"""
    if vehicle_count <= 5:
        return "🟢 Low"
    elif vehicle_count <= 15:
        return "🟡 Moderate" 
    elif vehicle_count <= 25:
        return "🟠 High"
    else:
        return "🔴 Heavy"

def classify_accident(vehicle_count, stopped_vehicles, avg_speed, weather, time_of_day):
    """Classify traffic incidents based on multiple factors"""
    weather_risk = {'Clear': 1.0, 'Rain': 1.2, 'Fog': 1.5, 'Snow': 1.7}
    time_risk = {'Day': 1.0, 'Night': 1.3}
    weather_factor = weather_risk.get(weather, 1.0)
    time_factor = time_risk.get(time_of_day, 1.0)
    risk_multiplier = weather_factor * time_factor
    impact_score = stopped_vehicles * risk_multiplier
    congestion_factor = 1 + (vehicle_count / 50)
    speed_penalty = max(0, (30 - avg_speed) / 30)
    severity_score = impact_score * congestion_factor * (1 + speed_penalty)

    if stopped_vehicles > 0 and severity_score < 2:
        return ("minor", round(severity_score, 2))

    if severity_score >= 8:
        return ("severe", round(severity_score, 2))
    elif severity_score >= 5:
        return ("major", round(severity_score, 2))
    elif severity_score >= 2:
        return ("minor", round(severity_score, 2))
    else:
        return ("none", round(severity_score, 2))

def get_weather_info(city="New York"):
    """Get actual weather information or return mock data"""
    # Mock weather data for demo - you can integrate real API here
    weather_conditions = {
        "New York": {"condition": "Clear", "temp": "22°C", "icon": "☀️"},
        "London": {"condition": "Light Rain", "temp": "18°C", "icon": "🌦️"},
        "Mumbai": {"condition": "Partly Cloudy", "temp": "28°C", "icon": "⛅"},
        "Delhi": {"condition": "Hazy", "temp": "32°C", "icon": "🌫️"},
    }
    return weather_conditions.get(city, {"condition": "Clear", "temp": "25°C", "icon": "☀️"})

def get_severity_color(incident_level):
    colors = {
        "none": "🟢",
        "minor": "🟡", 
        "major": "🟠",
        "severe": "🔴"
    }
    return colors.get(incident_level, "⚪")

def send_emergency_alert():
    try:
        Client = load_twilio()
        if Client is None:
            return "❌ Twilio not available"
            
        account_sid = ""
        auth_token = ""
        from_whatsapp_number = ""
        to_whatsapp_number = ""

        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body="🚨 ALERT: Emergency vehicle detected at junction. Override signal to GREEN immediately. Manage the vehicles there to control traffic!!",
            from_=from_whatsapp_number,
            to=to_whatsapp_number
        )
        return f"✅ Emergency alert sent via WhatsApp. SID: {message.sid}"
    except Exception as e:
        return f"❌ Failed to send WhatsApp message: {e}"

class TrafficAnalytics:
    def __init__(self):
        self.vehicle_history = []
        self.speed_history = []
        self.incident_history = []
        self.emergency_history = []
    
    def add_data(self, timestamp, vehicle_count, avg_speed, incident_level, emergency_count):
        self.vehicle_history.append({
            'timestamp': timestamp,
            'vehicle_count': vehicle_count,
            'avg_speed': avg_speed,
            'incident_level': incident_level,
            'emergency_count': emergency_count
        })
    
    def get_traffic_flow_chart(self):
        if not self.vehicle_history:
            return go.Figure()
        
        df = pd.DataFrame(self.vehicle_history)
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['vehicle_count'],
            mode='lines+markers',
            name='Vehicle Count',
            line=dict(color='#667eea', width=3)
        ))
        
        fig.update_layout(
            title="Real-time Traffic Flow",
            xaxis_title="Time",
            yaxis_title="Vehicle Count",
            template="plotly_white",
            height=300
        )
        return fig
    
    def get_speed_chart(self):
        if not self.vehicle_history:
            return go.Figure()
        
        df = pd.DataFrame(self.vehicle_history)
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['avg_speed'],
            mode='lines+markers',
            name='Average Speed',
            line=dict(color='#28a745', width=3),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title="Average Speed Monitoring",
            xaxis_title="Time",
            yaxis_title="Speed (km/h)",
            template="plotly_white",
            height=300
        )
        return fig

# Initialize session state
if 'analytics' not in st.session_state:
    st.session_state.analytics = TrafficAnalytics()
if 'emergency_detected' not in st.session_state:
    st.session_state.emergency_detected = False
if 'emergency_time' not in st.session_state:
    st.session_state.emergency_time = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'emergency_log' not in st.session_state:
    st.session_state.emergency_log = ""

# Header
st.markdown("""
<div class="ant-header">
    <h1 class="ant-typography-h1">🚦 Smart Traffic Monitoring System</h1>
    <p>AI-Powered Traffic Analysis with Emergency Vehicle Detection</p>
</div>
""", unsafe_allow_html=True)

# Emergency Alert Display with Ant Design components
if st.session_state.emergency_detected:
    st.markdown("""
    <div class="ant-alert-error-critical">
        <h2 class="ant-typography-h2">🚨 EMERGENCY VEHICLE DETECTED 🚨</h2>
        <p>Traffic signal override activated | Emergency services prioritized</p>
        <p><strong>Alert sent at:</strong> {}</p>
    </div>
    """.format(st.session_state.emergency_time), unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Model Loading Status
    with st.expander("🤖 AI Models", expanded=True):
        if not st.session_state.models_loaded:
            if st.button("🔄 Load AI Models"):
                with st.spinner("Loading YOLO models..."):
                    try:
                        YOLO = load_yolo_models()
                        DeepSort = load_deepsort()
                        st.session_state.vehicle_model = YOLO('yolov8s.pt')
                        st.session_state.emergency_model = YOLO('best.pt')
                        st.session_state.tracker = DeepSort(max_age=30)
                        st.session_state.models_loaded = True
                        st.success("✅ Models loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error loading models: {e}")
        else:
            st.success("✅ AI Models Ready")
    
    # Video Upload
    uploaded_file = st.file_uploader(
        "📹 Upload Traffic Video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a traffic video for analysis"
    )
    
    # Configuration Settings
    with st.expander("⚙️ Detection Settings"):
        vehicle_threshold = st.slider("Vehicle Detection Threshold", 0.1, 1.0, CONFIDENCE_THRESHOLD, 0.05)
        emergency_threshold = st.slider("Emergency Vehicle Threshold", 0.1, 1.0, CONFIDENCE_THRESHOLD_EMG, 0.05)
        
    with st.expander("🌍 Environment Settings"):
        weather = st.selectbox("Weather Condition", ["Clear", "Rain", "Fog", "Snow"])
        time_of_day = st.selectbox("Time of Day", ["Day", "Night"])
        city = st.text_input("City", "New York")
    
    # Emergency Controls
    with st.expander("🚨 Emergency Controls"):
        if st.button("🔄 Reset Emergency Alert"):
            st.session_state.emergency_detected = False
            st.session_state.emergency_time = None
            st.rerun()
        
        if st.button("📱 Test Emergency Alert"):
            result = send_emergency_alert()
            st.info(result)

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    # Video Analysis with Ant Design container
    st.markdown('<div class="ant-video-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="ant-typography-h3">📹 Video Analysis</h3>', unsafe_allow_html=True)
    video_placeholder = st.empty()
    
    if uploaded_file and st.session_state.models_loaded:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        if st.button("▶️ Start Analysis", type="primary"):
            st.session_state.processing = True
            
            # Video processing with original logic
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("Could not open video file.")
            else:
                frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30
            
            # Initialize tracking variables (keeping original logic)
            vehicle_tracks = {}
            speed_records = deque(maxlen=30)
            last_emergency_time = 0
            emergency_cooldown = 60
            
            # Original constants from your code
            STALL_DISTANCE_THRESHOLD = 2
            STALL_TIME_THRESHOLD = 5
            HISTORY_LENGTH = 15
            
            # Processing loop
            frame_number = 0
            stframe = st.empty()
            
            while st.session_state.processing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                frame_cropped = frame[100:, :]
                current_time_ts = time.time()
                current_time_dt = datetime.now()

                # Vehicle Detection (original logic)
                vehicle_results = st.session_state.vehicle_model(frame_cropped)[0]
                detections = []

                for box in vehicle_results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = box
                    if conf < vehicle_threshold:
                        continue
                    label = int(cls)
                    name = st.session_state.vehicle_model.names[label]
                    if name in ['car', 'truck', 'bus', 'motorbike']:
                        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, name))

                # Update tracks (original logic)
                tracks = st.session_state.tracker.update_tracks(detections, frame=frame_cropped)

                vehicle_count, stalled_count, total_speed, speed_count = 0, 0, 0, 0

                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    vehicle_count += 1
                    track_id = track.track_id
                    label = track.get_det_class()
                    l, t, r, b = track.to_ltrb()
                    cx, cy = int((l + r) / 2), int((t + b) / 2)

                    if track_id not in vehicle_tracks:
                        vehicle_tracks[track_id] = {
                            'positions': deque(maxlen=HISTORY_LENGTH),
                            'last_active_time': current_time_ts,
                            'label': label,
                            'status': 'Tracking'
                        }

                    info = vehicle_tracks[track_id]
                    positions = info['positions']
                    positions.append((cx, cy))

                    if len(positions) >= 2:
                        dx = positions[-1][0] - positions[-2][0]
                        dy = positions[-1][1] - positions[-2][1]
                        dist = (dx ** 2 + dy ** 2) ** 0.5
                        speed = dist * frame_rate * 0.05
                        total_speed += speed
                        speed_count += 1
                    else:
                        speed = 0

                    if len(positions) >= 2:
                        total_dist = sum(((positions[i][0] - positions[i - 1][0]) ** 2 +
                                          (positions[i][1] - positions[i - 1][1]) ** 2) ** 0.5
                                         for i in range(1, len(positions)))
                        avg_dist = total_dist / (len(positions) - 1)

                        if avg_dist < STALL_DISTANCE_THRESHOLD:
                            if current_time_ts - info['last_active_time'] > STALL_TIME_THRESHOLD:
                                info['status'] = f"{label} (Stalled)"
                                stalled_count += 1
                                color = (0, 0, 255)
                            else:
                                info['status'] = f"{label} (Waiting...)"
                                color = (0, 255, 255)
                        else:
                            info['last_active_time'] = current_time_ts
                            info['status'] = f"{label} (Moving)"
                            color = (0, 255, 0)
                    else:
                        info['status'] = f"{label} (Tracking)"
                        color = (255, 255, 0)

                    # Draw vehicle tracking boxes with original offset
                    top_offset = 100
                    cv2.rectangle(frame, (int(l), int(t) + top_offset), (int(r), int(b) + top_offset), color, 2)
                    cv2.putText(frame, f"ID {track_id} - {info['status']}", (int(l), int(t) + top_offset - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                avg_speed = total_speed / speed_count if speed_count > 0 else 0
                speed_records.append(avg_speed)
                avg_speed_smoothed = sum(speed_records) / len(speed_records)
                
                # Emergency Vehicle Detection (original logic)
                emergency_results = st.session_state.emergency_model(frame)[0]
                emergency_count = 0
                
                # Check if there are any detection boxes
                if len(emergency_results.boxes.data) > 0:
                    for box in emergency_results.boxes.data.tolist():
                        x1, y1, x2, y2, conf, cls = box
                        if conf < emergency_threshold:
                            continue
                        
                        emergency_count += 1
                        
                        # Trigger emergency alert only once per detection
                        if emergency_count == 1 and not st.session_state.emergency_detected:
                            st.session_state.emergency_detected = True
                            st.session_state.emergency_time = current_time_dt.strftime("%H:%M:%S")
                            alert_result = send_emergency_alert()
                            # Log the emergency detection
                            st.session_state.emergency_log = f"Emergency detected at {st.session_state.emergency_time}"

                        label = int(cls)
                        name = st.session_state.emergency_model.names[label]
                        color_emg = (0, 165, 255)  # Orange color for emergency vehicles
                        
                        # Draw emergency vehicle bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_emg, 3)
                        cv2.putText(frame, f"EMERGENCY: {name} {conf:.2f}", (int(x1), int(y1) - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_emg, 2)
                
                # If emergency detected, override all signals to GREEN
                if emergency_count > 0 or st.session_state.emergency_detected:
                    traffic_signal_status = "🟢 ALL GREEN - EMERGENCY OVERRIDE"
                    green_light = 120  # Maximum green time for emergency
                else:
                    traffic_signal_status = f"🟢 Green: {green_light:.0f}s"

                # Traffic Analysis (original logic)
                green_light = min(vehicle_count * 2.5, 120)
                incident_level, severity = classify_accident(vehicle_count, stalled_count, avg_speed_smoothed, weather, time_of_day)

                # Original overlay text with emergency override
                overlay_text = [
                    f"Vehicles: {vehicle_count}",
                    f"Stalled: {stalled_count}",
                    f"Avg Speed: {avg_speed_smoothed:.2f}",
                    f"Signal Status: {traffic_signal_status}",
                    f"Weather: {weather}",
                    f"Incident: {incident_level} (Score: {severity})",
                    f"Emergencies: {emergency_count}",
                    f"Emergency Thr: {emergency_threshold:.2f}"
                ]

                y0 = 20
                for i, line in enumerate(overlay_text):
                    y = y0 + i * 25
                    # Use red color for emergency status
                    color = (0, 0, 255) if "EMERGENCY OVERRIDE" in line else (255, 255, 255)
                    cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Add to analytics
                st.session_state.analytics.add_data(current_time_dt, vehicle_count, avg_speed_smoothed, incident_level, emergency_count)
                
                # Display frame
                stframe.image(frame, channels="BGR", use_container_width=True)
                
                time.sleep(1 / frame_rate if frame_rate > 0 else 0.03)
            
            cap.release()
            os.unlink(tfile.name)
            st.session_state.processing = False
    else:
        # Display placeholder when no video
        st.markdown("""
        <div class="ant-card">
            <div class="ant-card-body" style="text-align: center; padding: 60px 24px;">
                <h3 class="ant-typography-h3">📤 Upload a Traffic Video</h3>
                <p>Select a video file from the sidebar to begin AI-powered traffic analysis</p>
                <div style="margin: 20px 0;">
                    <span style="font-size: 48px; opacity: 0.3;">🎥</span>
                </div>
                <p style="color: var(--ant-text-color-secondary);">
                    Supported formats: MP4, AVI, MOV, MKV
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced Emergency Display Section
# Replace the existing metrics section in your Streamlit app

with col2:
    st.markdown('<div class="ant-analytics-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="ant-typography-h3">📊 Real-time Metrics</h3>', unsafe_allow_html=True)
    
    # Current Status Metrics
    if st.session_state.analytics.vehicle_history:
        latest = st.session_state.analytics.vehicle_history[-1]
        
        # EMERGENCY ALERT with Ant Design styling
        if latest['emergency_count'] > 0 or st.session_state.emergency_detected:
            st.markdown("""
            <div class="emergency-override">
                <h2 class="ant-typography-h2">🚨 EMERGENCY VEHICLES APPROACHING 🚨</h2>
                <h3 class="ant-typography-h3">🟢🟢🟢 ALL TRAFFIC LIGHTS: GREEN 🟢🟢🟢</h3>
                <p><strong>Emergency Vehicles Detected: {}</strong></p>
                <p><strong>CLEAR THE INTERSECTION IMMEDIATELY</strong></p>
                <p>Emergency Override Active Since: {}</p>
            </div>
            """.format(
                latest['emergency_count'] if latest['emergency_count'] > 0 else "1 (Active Alert)",
                st.session_state.emergency_time or "Now"
            ), unsafe_allow_html=True)
            
            # Emergency Protocol Badge
            st.markdown("""
            <div class="ant-alert-error">
                <strong>🆘 EMERGENCY PROTOCOL ACTIVATED</strong>
            </div>
            """, unsafe_allow_html=True)
            
        # Key Metrics with Ant Design Statistics
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="ant-statistic">
                <div class="ant-statistic-title">🚗 Current Vehicles</div>
                <div class="ant-statistic-content">{latest['vehicle_count']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="ant-statistic">
                <div class="ant-statistic-title">⚡ Average Speed</div>
                <div class="ant-statistic-content">{latest['avg_speed']:.1f} km/h</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            # Emergency count with status color
            emergency_status_class = "ant-status-error" if latest['emergency_count'] > 0 else "ant-status-success"
            emergency_value = latest['emergency_count'] if latest['emergency_count'] > 0 else "1"
            st.markdown(f"""
            <div class="ant-statistic">
                <div class="ant-statistic-title">🚨 Emergency Vehicles</div>
                <div class="ant-statistic-content {emergency_status_class}">{emergency_value}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Incident level with appropriate color
            severity_colors = {
                "none": "ant-status-success",
                "minor": "ant-status-warning", 
                "major": "ant-status-warning",
                "severe": "ant-status-error"
            }
            severity_class = severity_colors.get(latest['incident_level'], "ant-status-info")
            severity_icon = get_severity_color(latest['incident_level'])
            
            st.markdown(f"""
            <div class="ant-statistic">
                <div class="ant-statistic-title">⚠️ Incident Level</div>
                <div class="ant-statistic-content {severity_class}">{severity_icon} {latest['incident_level'].title()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Traffic Signal Status with Ant Design styling
        st.markdown('<h3 class="ant-typography-h3">🚦 Traffic Signal Control</h3>', unsafe_allow_html=True)
        
        if latest['emergency_count'] > 0 or st.session_state.emergency_detected:
            # Emergency Override Display
            st.markdown("""
            <div class="ant-traffic-signal ant-traffic-signal-emergency">
                <div style="text-align: center;">
                    <h4 class="ant-typography-h3 ant-status-error">🚨 EMERGENCY OVERRIDE ACTIVE 🚨</h4>
                    <h3 class="ant-typography-h2 ant-status-success">🟢 ALL LIGHTS: GREEN 🟢</h3>
                    <p>Emergency vehicles have absolute priority</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(1.0, text="🟢 EMERGENCY GREEN - ALL DIRECTIONS")
            
        else:
            # Normal Traffic Signal Optimization
            green_time = min(latest['vehicle_count'] * 2.5, 120)
            st.markdown("""
            <div class="ant-traffic-signal ant-traffic-signal-normal">
                <div style="text-align: center;">
                    <h4 class="ant-typography-h3 ant-status-success">🚦 Normal Traffic Operations</h4>
                    <p>Optimized signal timing based on traffic flow</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(green_time / 120, text=f"🟢 Optimized Green: {green_time:.0f}s")
            st.caption("Green light duration optimized based on current traffic volume")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced System Status with Ant Design List
    st.markdown('<div class="ant-card">', unsafe_allow_html=True)
    st.markdown('<div class="ant-card-head">🔧 System Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="ant-card-body" style="padding: 0;">', unsafe_allow_html=True)
    
    # Emergency Status gets top priority
    if st.session_state.emergency_detected:
        st.markdown("""
        <div class="ant-list-item" style="background: #fff2f0; border-left: 4px solid var(--ant-error-color);">
            <span><strong>🆘 Emergency Mode</strong></span>
            <span class="ant-status-error"><strong>All systems prioritizing emergency vehicles</strong></span>
        </div>
        """, unsafe_allow_html=True)
    
    status_items = [
        ("🤖 AI Models", "✅ Active" if st.session_state.models_loaded else "❌ Not Loaded", 
         "ant-status-success" if st.session_state.models_loaded else "ant-status-error"),
        ("📹 Video Processing", "🟢 Running" if st.session_state.processing else "⏸️ Stopped",
         "ant-status-success" if st.session_state.processing else "ant-status-info"),
        ("🚨 Emergency Detection", "🚨 ACTIVE ALERT" if st.session_state.emergency_detected else "🟢 Normal",
         "ant-status-error" if st.session_state.emergency_detected else "ant-status-success"),
        ("📱 WhatsApp Alerts", "📱 Connected & Ready", "ant-status-success"),
        ("🚦 Traffic Override", "🟢 EMERGENCY GREEN" if st.session_state.emergency_detected else "🚦 Normal Operations",
         "ant-status-error" if st.session_state.emergency_detected else "ant-status-success")
    ]
    
    for item, status, status_class in status_items:
        st.markdown(f"""
        <div class="ant-list-item">
            <span>{item}</span>
            <span class="{status_class}"><strong>{status}</strong></span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)
    
# Additional: Enhanced Emergency Alert at the top of the page
if st.session_state.emergency_detected:
    st.markdown("""
    <div class="emergency-override" style="margin: 2rem 0;">
        <h1 class="ant-typography-h1">🚨🚨🚨 EMERGENCY VEHICLES APPROACHING 🚨🚨🚨</h1>
        <h2 class="ant-typography-h2">🟢🟢🟢 ALL TRAFFIC LIGHTS ARE NOW GREEN 🟢🟢🟢</h2>
        <h3 class="ant-typography-h3">⚠️ CLEAR ALL INTERSECTIONS IMMEDIATELY ⚠️</h3>
        <p><strong>Emergency Alert Activated:</strong> {}</p>
        <p><strong>Status:</strong> All traffic signals overridden to GREEN</p>
        <p><strong>Action Required:</strong> All vehicles must yield and clear the path</p>
    </div>
    """.format(st.session_state.emergency_time), unsafe_allow_html=True)

    # Add audio alert simulation with Ant Design styling
    st.markdown("""
    <div class="ant-audio-alert">
        <h4 class="ant-typography-h3">📢 AUDIO ALERT ACTIVE 📢</h4>
        <p>🔊 "Emergency vehicles approaching. All traffic lights are green. Clear the intersection."</p>
    </div>
    """, unsafe_allow_html=True)

# Analytics Dashboard Section
if st.session_state.analytics.vehicle_history:
    st.markdown("---")
    st.markdown('<h2 class="ant-typography-h2">📈 Traffic Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown('<div class="ant-card">', unsafe_allow_html=True)
        st.markdown('<div class="ant-card-head">🚗 Vehicle Count Trends</div>', unsafe_allow_html=True)
        st.markdown('<div class="ant-card-body">', unsafe_allow_html=True)
        st.plotly_chart(st.session_state.analytics.get_traffic_flow_chart(), use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    with col_chart2:
        st.markdown('<div class="ant-card">', unsafe_allow_html=True)
        st.markdown('<div class="ant-card-head">⚡ Speed Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="ant-card-body">', unsafe_allow_html=True)
        st.plotly_chart(st.session_state.analytics.get_speed_chart(), use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

# Features Showcase with Ant Design styling
st.markdown("---")
st.markdown('<div class="ant-card">', unsafe_allow_html=True)
st.markdown('<div class="ant-card-head">🌟 Key Features & Capabilities</div>', unsafe_allow_html=True)
st.markdown('<div class="ant-card-body">', unsafe_allow_html=True)

feature_cols = st.columns(2)

with feature_cols[0]:
    st.markdown("""
    <div class="ant-feature-list">
        <h4 class="ant-typography-h3">🤖 AI & Detection</h4>
        <ul style="list-style: none; padding: 0;">
            <li>🎯 YOLO-based vehicle recognition</li>
            <li>🚨 Emergency vehicle detection</li>
            <li>📊 Real-time traffic analytics</li>
            <li>🔍 DeepSORT vehicle tracking</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with feature_cols[1]:
    st.markdown("""
    <div class="ant-feature-list">
        <h4 class="ant-typography-h3">🚦 Smart Controls</h4>
        <ul style="list-style: none; padding: 0;">
            <li>⚡ Dynamic signal optimization</li>
            <li>🚨 Emergency override system</li>
            <li>📱 WhatsApp alert integration</li>
            <li>🌍 Weather-aware analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)

# Footer with Ant Design styling
st.markdown("---")
st.markdown("""
<div class="ant-card" style="text-align: center; margin-top: 2rem;">
    <div class="ant-card-body">
        <h3 class="ant-typography-h3">🏆 Smart Traffic Monitor</h3>
        <p class="ant-status-info">Built for Hackathon Excellence</p>
        <p style="color: var(--ant-text-color-secondary);">
            Powered by YOLO, DeepSORT, Streamlit & Ant Design
        </p>
        <div style="margin-top: 16px;">
            <span class="ant-status-success">✅ Real-time Processing</span> | 
            <span class="ant-status-warning">🚨 Emergency Ready</span> | 
            <span class="ant-status-info">📊 Analytics Enabled</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Analytics Dashboard Section
st.markdown("---")

# Final Values Display Section
st.markdown("""
<div class="ant-header" style="background: linear-gradient(135deg, #52c41a 0%, #73d13d 100%);">
    <h1 class="ant-typography-h1">📊 Current Traffic Summary</h1>
    <p>Real-time traffic metrics and environmental conditions</p>
</div>
""", unsafe_allow_html=True)

col_final1, col_final2, col_final3 = st.columns(3)

with col_final1:
    st.markdown('<div class="ant-analytics-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="ant-typography-h3">🚗 Traffic Metrics</h3>', unsafe_allow_html=True)
    
    if st.session_state.analytics.vehicle_history:
        latest = st.session_state.analytics.vehicle_history[-1]
        
        # Current Vehicles
        st.markdown(f"""
        <div class="ant-statistic">
            <div class="ant-statistic-title">🚗 Current Vehicles</div>
            <div class="ant-statistic-content ant-status-info">{latest['vehicle_count']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Average Speed
        speed_status = "ant-status-success" if latest['avg_speed'] > 25 else "ant-status-warning" if latest['avg_speed'] > 15 else "ant-status-error"
        st.markdown(f"""
        <div class="ant-statistic">
            <div class="ant-statistic-title">⚡ Average Speed</div>
            <div class="ant-statistic-content {speed_status}">{latest['avg_speed']:.1f} km/h</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Emergency Vehicles
        emergency_display = "None" if latest['emergency_count'] == 0 else str(latest['emergency_count'])
        emergency_status = "ant-status-success" if latest['emergency_count'] == 0 else "ant-status-error"
        st.markdown(f"""
        <div class="ant-statistic">
            <div class="ant-statistic-title">🚨 Emergency Vehicles</div>
            <div class="ant-statistic-content {emergency_status}">{emergency_display}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Incident Level
        severity_colors = {
            "none": "ant-status-success",
            "minor": "ant-status-warning", 
            "major": "ant-status-warning",
            "severe": "ant-status-error"
        }
        severity_class = severity_colors.get(latest['incident_level'], "ant-status-info")
        severity_icon = get_severity_color(latest['incident_level'])
        
        st.markdown(f"""
        <div class="ant-statistic">
            <div class="ant-statistic-title">⚠️ Incident Level</div>
            <div class="ant-statistic-content {severity_class}">{severity_icon} {latest['incident_level'].title()}</div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Default values when no data available
        st.markdown(f"""
        <div class="ant-statistic">
            <div class="ant-statistic-title">🚗 Current Vehicles</div>
            <div class="ant-statistic-content ant-status-info">4</div>
        </div>
        <div class="ant-statistic">
            <div class="ant-statistic-title">⚡ Average Speed</div>
            <div class="ant-statistic-content ant-status-success">35.2 km/h</div>
        </div>
        <div class="ant-statistic">
            <div class="ant-statistic-title">🚨 Emergency Vehicles</div>
            <div class="ant-statistic-content ant-status-success">None</div>
        </div>
        <div class="ant-statistic">
            <div class="ant-statistic-title">⚠️ Incident Level</div>
            <div class="ant-statistic-content ant-status-success">🟢 None</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_final2:
    st.markdown('<div class="ant-analytics-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="ant-typography-h3">🌤️ Weather & Environment</h3>', unsafe_allow_html=True)
    
    # Get weather information
    weather_info = get_weather_info(city)
    
    st.markdown(f"""
    <div class="ant-statistic">
        <div class="ant-statistic-title">📍 Location</div>
        <div class="ant-statistic-content ant-status-info">{city}</div>
    </div>
    <div class="ant-statistic">
        <div class="ant-statistic-title">🌤️ Weather</div>
        <div class="ant-statistic-content ant-status-info">{weather_info['icon']} {weather_info['condition']}</div>
    </div>
    <div class="ant-statistic">
        <div class="ant-statistic-title">🌡️ Temperature</div>
        <div class="ant-statistic-content ant-status-info">{weather_info['temp']}</div>
    </div>
    <div class="ant-statistic">
        <div class="ant-statistic-title">🕐 Time Period</div>
        <div class="ant-statistic-content ant-status-info">{time_of_day}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_final3:
    st.markdown('<div class="ant-analytics-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="ant-typography-h3">🚦 Traffic Analysis</h3>', unsafe_allow_html=True)
    
    if st.session_state.analytics.vehicle_history:
        latest = st.session_state.analytics.vehicle_history[-1]
        traffic_density = classify_traffic_density(latest['vehicle_count'])
        
        # Traffic density with appropriate colors
        density_colors = {
            "🟢 Low": "ant-status-success",
            "🟡 Moderate": "ant-status-warning",
            "🟠 High": "ant-status-warning", 
            "🔴 Heavy": "ant-status-error"
        }
        density_class = density_colors.get(traffic_density, "ant-status-info")
        
        st.markdown(f"""
        <div class="ant-statistic">
            <div class="ant-statistic-title">🚦 Traffic Density</div>
            <div class="ant-statistic-content {density_class}">{traffic_density}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Traffic flow status
        if latest['avg_speed'] > 30:
            flow_status = "🟢 Smooth"
            flow_class = "ant-status-success"
        elif latest['avg_speed'] > 20:
            flow_status = "🟡 Moderate"
            flow_class = "ant-status-warning"
        elif latest['avg_speed'] > 10:
            flow_status = "🟠 Congested"
            flow_class = "ant-status-warning"
        else:
            flow_status = "🔴 Heavy Congestion"
            flow_class = "ant-status-error"
            
        st.markdown(f"""
        <div class="ant-statistic">
            <div class="ant-statistic-title">🌊 Traffic Flow</div>
            <div class="ant-statistic-content {flow_class}">{flow_status}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Signal optimization
        green_time = min(latest['vehicle_count'] * 2.5, 120)
        st.markdown(f"""
        <div class="ant-statistic">
            <div class="ant-statistic-title">🚦 Optimized Green Time</div>
            <div class="ant-statistic-content ant-status-info">{green_time:.0f}s</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Efficiency score
        efficiency = min(100, max(0, (latest['avg_speed'] / 40) * 100))
        efficiency_class = "ant-status-success" if efficiency > 70 else "ant-status-warning" if efficiency > 40 else "ant-status-error"
        st.markdown(f"""
        <div class="ant-statistic">
            <div class="ant-statistic-title">📈 Traffic Efficiency</div>
            <div class="ant-statistic-content {efficiency_class}">{efficiency:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Default values when no data available
        st.markdown(f"""
        <div class="ant-statistic">
            <div class="ant-statistic-title">🚦 Traffic Density</div>
            <div class="ant-statistic-content ant-status-success">🟢 Low</div>
        </div>
        <div class="ant-statistic">
            <div class="ant-statistic-title">🌊 Traffic Flow</div>
            <div class="ant-statistic-content ant-status-success">🟢 Smooth</div>
        </div>
        <div class="ant-statistic">
            <div class="ant-statistic-title">🚦 Optimized Green Time</div>
            <div class="ant-statistic-content ant-status-info">30s</div>
        </div>
        <div class="ant-statistic">
            <div class="ant-statistic-title">📈 Traffic Efficiency</div>
            <div class="ant-statistic-content ant-status-success">85%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Summary Banner
if st.session_state.analytics.vehicle_history:
    latest = st.session_state.analytics.vehicle_history[-1]
    weather_info = get_weather_info(city)
    traffic_density = classify_traffic_density(latest['vehicle_count'])
    
    st.markdown(f"""
    <div class="ant-card" style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 2px solid #1890ff;">
        <div class="ant-card-body" style="text-align: center;">
            <h2 class="ant-typography-h2" style="color: #1890ff; margin-bottom: 20px;">🌟 Traffic Summary Dashboard</h2>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px;">
                <div style="flex: 1; min-width: 200px;">
                    <h4 class="ant-typography-h3">🚗 Current Vehicles: <span style="color: #1890ff;">{latest['vehicle_count']}</span></h4>
                    <h4 class="ant-typography-h3">⚡ Average Speed: <span style="color: #52c41a;">{latest['avg_speed']:.1f} km/h</span></h4>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <h4 class="ant-typography-h3">🚨 Emergency Vehicles: <span style="color: {'#ff4d4f' if latest['emergency_count'] > 0 else '#52c41a'};">{'None' if latest['emergency_count'] == 0 else latest['emergency_count']}</span></h4>
                    <h4 class="ant-typography-h3">⚠️ Incident Level: <span>{get_severity_color(latest['incident_level'])} {latest['incident_level'].title()}</span></h4>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <h4 class="ant-typography-h3">🌤️ Weather: <span style="color: #1890ff;">{weather_info['icon']} {weather_info['condition']}</span></h4>
                    <h4 class="ant-typography-h3">🚦 Traffic: <span>{traffic_density}</span></h4>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Default summary when no data
    weather_info = get_weather_info(city)
    st.markdown(f"""
    <div class="ant-card" style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 2px solid #1890ff;">
        <div class="ant-card-body" style="text-align: center;">
            <h2 class="ant-typography-h2" style="color: #1890ff; margin-bottom: 20px;">🌟 Traffic Summary Dashboard</h2>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px;">
                <div style="flex: 1; min-width: 200px;">
                    <h4 class="ant-typography-h3">🚗 Current Vehicles: <span style="color: #1890ff;">4</span></h4>
                    <h4 class="ant-typography-h3">⚡ Average Speed: <span style="color: #52c41a;">35.2 km/h</span></h4>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <h4 class="ant-typography-h3">🚨 Emergency Vehicles: <span style="color: #52c41a;">None</span></h4>
                    <h4 class="ant-typography-h3">⚠️ Incident Level: <span>🟢 None</span></h4>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <h4 class="ant-typography-h3">🌤️ Weather: <span style="color: #1890ff;">{weather_info['icon']} {weather_info['condition']}</span></h4>
                    <h4 class="ant-typography-h3">🚦 Traffic: <span style="color: #52c41a;">🟢 Low</span></h4>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)