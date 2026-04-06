"""
Tracking.py — Unified professional UI with enhanced Face + Anomaly detection.
Now uses YOLOv11 and face_recognition for 128-d embeddings.
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import os
import base64
import yaml
from utils import recognize, submitNew, get_info_from_id, deleteOne

# ── Paths ───────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_BASE, "data", "config.yaml")

if not os.path.exists(_CONFIG_PATH):
    # Fallback to root for dev
    _CONFIG_PATH = os.path.join(_BASE, "config.yaml")

with open(_CONFIG_PATH, "r") as _f:
    _cfg = yaml.safe_load(_f)

# Use paths from config or defaults
YOLO_MODEL_PATH  = os.path.join(_BASE, _cfg["YOLO"].get("MODEL_PATH", "models/yolo11n.pt"))
ASSETS_DIR       = os.path.join(_BASE, "assets")
SAMPLE_IMAGE_DIR = os.path.join(ASSETS_DIR, "Sample_images")

st.set_page_config(page_title="Black Eyes", layout="wide", page_icon="👁️",
                   initial_sidebar_state="expanded")

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #030712 !important;
    color: #e2e8f0;
    font-family: 'Exo 2', sans-serif;
}
[data-testid="stSidebar"] {
    background: #050b18 !important;
    border-right: 1px solid #0f2a4a !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Scanlines Effect */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,170,0.012) 2px, rgba(0,255,170,0.012) 4px);
    pointer-events: none;
    z-index: 9999;
}

/* Sidebar nav */
.nav-logo {
    padding: 24px 20px 16px;
    border-bottom: 1px solid #0f2a4a;
    margin-bottom: 8px;
}
.nav-logo-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: 4px;
    text-transform: uppercase;
}
.nav-logo-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: #00ffaa;
    letter-spacing: 3px;
    margin-top: 2px;
}
.nav-section {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: #1e3a5f;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 12px 20px 4px;
}

/* Radio as nav items */
[data-testid="stSidebar"] .stRadio > label { display: none !important; }
[data-testid="stSidebar"] .stRadio > div {
    flex-direction: column !important;
    gap: 2px !important;
    padding: 0 8px;
}
[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 16px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #475569 !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: #0a1628 !important;
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: #0a1628 !important;
    color: #00ffaa !important;
    border-left: 2px solid #00ffaa !important;
}
[data-testid="stSidebar"] .stRadio > div > label > div { display: none !important; }
[data-testid="stSidebar"] .stRadio > div > label > div:last-child {
    display: block !important;
    margin-left: 0 !important;
}

/* Sidebar sliders / selects */
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 10px !important;
    color: #334155 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: #0a1628 !important;
    border-color: #1e3a5f !important;
    color: #94a3b8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
}

/* Page header */
.page-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    padding: 28px 0 20px;
    border-bottom: 1px solid #0f2a4a;
    margin-bottom: 24px;
}
.page-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: 3px;
    text-transform: uppercase;
    line-height: 1;
}
.live-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #00ffaa12;
    border: 1px solid #00ffaa33;
    border-radius: 20px;
    padding: 6px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: #00ffaa;
    letter-spacing: 2px;
}
.live-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00ffaa;
    box-shadow: 0 0 6px #00ffaa;
    animation: blink 1.5s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* Status cards */
.s-card {
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 10px;
    border: 1px solid #0f2a4a;
    background: #070f1f;
}
.s-card.known { border-color: #00ffaa33; background: #00ffaa06; }
.s-card.unknown { border-color: #ff444433; background: #ff444406; }
.s-card.threat { border-color: #ff6b0033; background: #ff6b0006; animation: threatPulse 1.2s infinite; }
.s-card.clear { border-color: #00ffaa33; background: #00ffaa06; }
@keyframes threatPulse { 0%,100%{border-color:#ff6b0033} 50%{border-color:#ff6b00aa; box-shadow:0 0 16px #ff6b0022} }

.s-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: #334155;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.s-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 1px;
    line-height: 1.1;
}
.s-value.known { color: #00ffaa; }
.s-value.unknown { color: #ff4444; }
.s-value.threat { color: #ff6b00; }
.s-value.clear { color: #00ffaa; }
</style>
""", unsafe_allow_html=True)

# ── Load AI Models ────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo(path):
    if not os.path.exists(path):
        # Fallback to v11 nano if custom model is missing
        return YOLO("yolo11n.pt")
    m = YOLO(path)
    # Ensure custom names if needed (standard COCO includes knife at index 43)
    return m

yolo_model = load_yolo(YOLO_MODEL_PATH)

def draw_yolo(frame, results):
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = box.conf[0].item()
            cls   = int(box.cls[0].item())
            label = f"{yolo_model.model.names.get(cls, str(cls))}: {conf:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 107, 0), 2)
            cv2.putText(frame, label, (x1+2, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 107, 0), 1)
    return frame

# ── Session state ─────────────────────────────────────────────────────────────
if "cam_running" not in st.session_state:
    st.session_state.cam_running = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div class="nav-logo">
            <div class="nav-logo-text">👁 Black Eyes</div>
            <div class="nav-logo-sub">● SYSTEM ONLINE</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">// navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["TRACKING", "DATABASE", "MANAGEMENT"], label_visibility="collapsed")

    st.markdown('<div class="nav-section">// settings</div>', unsafe_allow_html=True)
    TOLERANCE = st.slider("Match Tolerance", 0.1, 1.0, _cfg["DETECTION"].get("FACE_TOLERANCE", 0.4), 0.05)
    input_src = st.selectbox("Input Source", ["Live Camera", "Upload Image"])

    st.markdown("---")
    st.markdown(f"""
        <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#1e3a5f;line-height:2;padding:0 8px">
        BUILD v3.0.0 (PROFESSIONAL)<br>ENGINE: YOLOv11 + DLib<br>STATUS: { "ACTIVE" if st.session_state.cam_running else "IDLE" }
        </div>
    """, unsafe_allow_html=True)

# ── Routing ───────────────────────────────────────────────────────────────────
if page == "TRACKING":
    st.markdown("""
        <div class="page-header">
            <div>
                <div class="page-title">Live Surveillance</div>
                <div id="status-line" style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#334155;letter-spacing:3px;margin-top:6px">
                // dual mode: biometric identity + anomaly detection
                </div>
            </div>
            <div class="live-badge"><div class="live-dot"></div>LIVE</div>
        </div>
    """, unsafe_allow_html=True)

    if input_src == "Upload Image":
        file = st.file_uploader("Scan Static Frame", type=["jpg","jpeg","png"])
        if file:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run Detections
            out_img, name, _ = recognize(rgb, TOLERANCE)
            yolo_results = yolo_model(img, conf=0.5, verbose=False)
            out_img_bgr  = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            final_frame  = draw_yolo(out_img_bgr, yolo_results)
            
            detections = [yolo_model.model.names.get(int(b.cls[0])) for r in yolo_results for b in r.boxes]
            
            col1, col2 = st.columns([3,2])
            with col1:
                st.image(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            with col2:
                # Identity card
                _cls = "known" if name != "Unknown" else "unknown"
                st.markdown(f"""
                    <div class="s-card {_cls}">
                        <div class="s-label">// biometric identity</div>
                        <div class="s-value {_cls}">{name}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Anomaly card
                if detections:
                    st.markdown(f"""
                        <div class="s-card threat">
                            <div class="s-label">// anomalies detected</div>
                            { "".join([f'<div class="s-value threat" style="font-size:18px">⚠ {d.upper()}</div>' for d in set(detections)]) }
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown('<div class="s-card clear"><div class="s-label">// anomaly scan</div><div class="s-value clear">✓ CLEAR</div></div>', unsafe_allow_html=True)

    else:  # Live Camera
        col_feed, col_panel = st.columns([3, 2])
        with col_panel:
            st.markdown('<div class="s-label">// control center</div>', unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            if b1.button("▶ START SYSTEM"): st.session_state.cam_running = True
            if b2.button("■ STOP SYSTEM"): st.session_state.cam_running = False
            
            identity_ph = st.empty()
            anomaly_ph  = st.empty()

        with col_feed:
            frame_ph = st.empty()

        if st.session_state.cam_running:
            cap = cv2.VideoCapture(0)
            while st.session_state.cam_running:
                ret, frame = cap.read()
                if not ret: break

                # Recognition (RGB)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out_img, name, pid = recognize(rgb, TOLERANCE)

                # Anomaly (BGR)
                results = yolo_model(frame, verbose=False, conf=0.5)
                out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                final   = draw_yolo(out_bgr, results)
                final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

                frame_ph.image(final_rgb, channels="RGB", use_column_width=True)

                # UI Updates
                _cls = "known" if name != "Unknown" else "unknown"
                identity_ph.markdown(f'<div class="s-card {_cls}"><div class="s-label">// identity</div><div class="s-value {_cls}">{name}</div></div>', unsafe_allow_html=True)
                
                detections = list(set([yolo_model.model.names.get(int(b.cls[0])) for r in results for b in r.boxes]))
                if detections:
                    threats_html = "".join([f'<div class="s-value threat" style="font-size:18px">⚠ {d.upper()}</div>' for d in detections])
                    anomaly_ph.markdown(f'<div class="s-card threat"><div class="s-label">// anomalies</div>{threats_html}</div>', unsafe_allow_html=True)
                else:
                    anomaly_ph.markdown('<div class="s-card clear"><div class="s-label">// anomalies</div><div class="s-value clear">✓ CLEAR</div></div>', unsafe_allow_html=True)

            cap.release()

elif page == "DATABASE":
    import pickle
    st.markdown('<div class="page-header"><div class="page-title">Identity Registry</div></div>', unsafe_allow_html=True)
    
    db = get_database()
    if not db:
        st.info("Registry is empty. Register subjects in Management pannel.")
    else:
        for idx, person in db.items():
            c1, c2 = st.columns([1,4])
            with c1:
                if "image" in person: st.image(person["image"], width=100)
            with c2:
                st.markdown(f"""
                    <div style="padding:10px">
                        <div style="color:#00ffaa;font-weight:700;font-size:20px">{person['name']}</div>
                        <div style="color:#334155;font-size:12px">UUID: {person['id']}</div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("---")

elif page == "MANAGEMENT":
    st.markdown('<div class="page-header"><div class="page-title">Registry Management</div></div>', unsafe_allow_html=True)
    
    action = st.tabs(["Add Personnel", "Remove Personnel"])
    
    with action[0]:
        name = st.text_input("Name")
        pid  = st.text_input("Employee ID")
        img_file = st.file_uploader("Upload Profile Image", type=["jpg","png"])
        if st.button("REGISTER"):
            if name and pid and img_file:
                img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
                # Note: We pass RGB to submitNew
                ret = submitNew(name, pid, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if ret is True: st.success("Subject Authorized.")
                elif ret == -1: st.error("No face detected.")
                else: st.error("ID collision.")
    
    with action[1]:
        del_id = st.text_input("ID to Remove")
        if st.button("DELETE"):
            if deleteOne(del_id): st.success("Removed.")
            else: st.error("ID not found.")
