"""
Tracking.py — Unified professional UI, face + anomaly in one feed
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import os
import base64
import yaml
from utils import recognize

_BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_BASE, "config.yaml"), "r") as _f:
    _cfg = yaml.safe_load(_f)

YOLO_MODEL_PATH  = os.path.join(_BASE, _cfg["YOLO"]["MODEL_PATH"])
SAMPLE_IMAGE_DIR = os.path.join(_BASE, "Sample_images")

st.set_page_config(page_title="Black Eyes", layout="wide", page_icon="👁️",
                   initial_sidebar_state="expanded")

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

/* Scanlines */
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

/* Buttons */
.stButton > button {
    background: #0a1628 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 6px !important;
    color: #64748b !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    padding: 10px 16px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    border-color: #00ffaa66 !important;
    color: #00ffaa !important;
    background: #0d1f3c !important;
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
.page-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: #334155;
    letter-spacing: 3px;
    margin-top: 6px;
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

/* Feed panel */
.feed-wrap {
    background: #070f1f;
    border: 1px solid #0f2a4a;
    border-radius: 10px;
    overflow: hidden;
}
.feed-bar {
    background: #050b18;
    border-bottom: 1px solid #0f2a4a;
    padding: 9px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: #334155;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.rec-dot { color: #ff4444; animation: blink 1s infinite; }

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
.s-meta {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: #334155;
    margin-top: 4px;
}

/* Metric row */
.m-row { display: flex; gap: 8px; margin: 10px 0; }
.m-tile {
    flex:1; background:#070f1f; border:1px solid #0f2a4a;
    border-radius:8px; padding:12px 14px;
}
.m-val { font-family:'Share Tech Mono',monospace; font-size:20px; color:#f1f5f9; }
.m-lbl { font-family:'Share Tech Mono',monospace; font-size:9px; color:#1e3a5f; letter-spacing:2px; text-transform:uppercase; margin-top:3px; }

/* Image */
[data-testid="stImage"] img { border-radius: 0 !important; display: block; }
[data-testid="stImage"] { line-height: 0; }

/* File uploader */
[data-testid="stFileUploader"] section {
    background: #070f1f !important;
    border: 1px dashed #1e3a5f !important;
    border-radius: 8px !important;
}

/* Divider */
hr { border-color: #0f2a4a !important; }
</style>
""", unsafe_allow_html=True)

# ── Load YOLO ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo(path):
    if not os.path.exists(path):
        return YOLO("yolov8s.pt")
    m = YOLO(path)
    m.model.names = {0: 'fire', 1: 'other', 2: 'smoke', 3: 'gun', 4: 'knife'}
    return m

yolo_model = load_yolo(YOLO_MODEL_PATH)

def draw_yolo(frame, results):
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = box.conf[0].item()
            cls   = int(box.cls[0].item())
            label = f"{yolo_model.model.names.get(cls, str(cls))}: {conf:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 170), 2)
            cv2.rectangle(frame, (x1, y1-20), (x1 + len(label)*8, y1), (0, 255, 170), -1)
            cv2.putText(frame, label, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
    return frame

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {"cam_running": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div class="nav-logo">
            <div class="nav-logo-text">👁 Black Eyes</div>
            <div class="nav-logo-sub">● SYSTEM ONLINE</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">// navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["TRACKING", "DATABASE", "UPDATING"], label_visibility="collapsed")

    st.markdown('<div class="nav-section">// settings</div>', unsafe_allow_html=True)
    TOLERANCE = st.slider("Face Tolerance", 0.1, 1.0, 0.5, 0.05)
    input_src = st.selectbox("Input Source", ["Live Camera", "Upload Image / Sample"])

    st.markdown("---")
    st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#1e3a5f;line-height:2;padding:0 8px">
        BUILD v2.1.0<br>ENGINE: OPENCV + YOLOV8<br>MODE: LOCAL TEST
        </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TRACKING PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "TRACKING":
    st.markdown("""
        <div class="page-header">
            <div>
                <div class="page-title">Live Surveillance</div>
                <div class="page-sub">// face recognition + anomaly detection — unified feed</div>
            </div>
            <div class="live-badge"><div class="live-dot"></div>LIVE</div>
        </div>
    """, unsafe_allow_html=True)

    if input_src == "Upload Image / Sample":
        tab1, tab2 = st.tabs(["📷  Face Recognition", "🔍  Anomaly Detection"])

        with tab1:
            file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
            if file:
                img_rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                out_img, name, pid = recognize(img_rgb, TOLERANCE)
                c1, c2 = st.columns([3,2])
                with c1:
                    st.markdown('<div class="feed-wrap"><div class="feed-bar"><span>▶ FRAME ANALYSIS</span><span>STATIC INPUT</span></div>', unsafe_allow_html=True)
                    st.image(out_img, channels="RGB", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with c2:
                    _cls = "known" if name != "Unknown" else "unknown"
                    st.markdown(f"""
                        <div class="s-card {_cls}">
                            <div class="s-label">// identity result</div>
                            <div class="s-value {_cls}">{name}</div>
                            <div class="s-meta">ID: {pid}</div>
                        </div>
                        <div class="s-card {"clear" if name != "Unknown" else "unknown"}">
                            <div class="s-label">// access status</div>
                            <div class="s-value {"clear" if name != "Unknown" else "unknown"}">
                                {"✓ AUTHORISED" if name != "Unknown" else "⚠ DENIED"}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        with tab2:
            choice = st.selectbox("Select sample", ["knife", "fire", "smoke"])
            img_path = os.path.join(SAMPLE_IMAGE_DIR, f"{choice}.jpg")
            if os.path.exists(img_path):
                frame = cv2.imread(img_path)
                results = yolo_model(frame, conf=0.5)
                frame = draw_yolo(frame, results)
                detections = list(set([yolo_model.model.names.get(int(b.cls[0]), "?") for r in results for b in r.boxes]))
                c1, c2 = st.columns([3,2])
                with c1:
                    st.markdown('<div class="feed-wrap"><div class="feed-bar"><span>▶ ANOMALY SCAN</span><span>STATIC INPUT</span></div>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with c2:
                    if detections:
                        for d in detections:
                            emoji = {"knife":"🔪","fire":"🔥","smoke":"💨"}.get(d,"⚠")
                            st.markdown(f"""
                                <div class="s-card threat">
                                    <div class="s-label">// threat detected</div>
                                    <div class="s-value threat">{emoji} {d.upper()}</div>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="s-card clear"><div class="s-label">// scan result</div><div class="s-value clear">✓ ALL CLEAR</div></div>', unsafe_allow_html=True)

    else:  # Live Camera
        col_feed, col_panel = st.columns([3, 2])

        with col_panel:
            st.markdown('<div class="s-label" style="padding:4px 0 8px">// camera controls</div>', unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            start_btn = b1.button("▶  START")
            stop_btn  = b2.button("■  STOP")

            if start_btn:
                st.session_state.cam_running = True
            if stop_btn:
                st.session_state.cam_running = False

            st.markdown("---")
            identity_ph = st.empty()
            anomaly_ph  = st.empty()
            st.markdown(f"""
                <div class="m-row">
                    <div class="m-tile"><div class="m-val">{TOLERANCE:.2f}</div><div class="m-lbl">Tolerance</div></div>
                    <div class="m-tile"><div class="m-val">DUAL</div><div class="m-lbl">Mode</div></div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("""
                <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#1e3a5f;margin-top:8px">
                FACE RECOG + ANOMALY RUNNING IN PARALLEL
                </div>
            """, unsafe_allow_html=True)

        with col_feed:
            st.markdown('<div class="feed-wrap"><div class="feed-bar"><span>▶ LIVE FEED — CAM 01</span><span class="rec-dot">● REC</span></div>', unsafe_allow_html=True)
            frame_ph = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.cam_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Camera not accessible.")
            else:
                while st.session_state.cam_running:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Run both: face recognition + YOLO anomaly on same frame
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    out_img, name, pid = recognize(rgb, TOLERANCE)

                    yolo_results = yolo_model(frame, verbose=False, conf=0.5)
                    out_img_bgr  = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                    out_img_bgr  = draw_yolo(out_img_bgr, yolo_results)
                    final_rgb    = cv2.cvtColor(out_img_bgr, cv2.COLOR_BGR2RGB)
                    detections   = list(set([yolo_model.model.names.get(int(b.cls[0]),"?") for r in yolo_results for b in r.boxes]))

                    frame_ph.image(final_rgb, channels="RGB", use_container_width=True)

                    # Identity panel
                    _cls = "known" if name != "Unknown" else "unknown"
                    identity_ph.markdown(f"""
                        <div class="s-card {_cls}">
                            <div class="s-label">// identity</div>
                            <div class="s-value {_cls}">{name}</div>
                            <div class="s-meta">ID: {pid} &nbsp;|&nbsp; {"✓ AUTHORISED" if name != "Unknown" else "⚠ UNIDENTIFIED"}</div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Anomaly panel
                    if detections:
                        emojis = {"knife":"🔪","fire":"🔥","smoke":"💨"}
                        threat_html = "".join([f'<div class="s-value threat" style="font-size:18px">{emojis.get(d,"⚠")} {d.upper()}</div>' for d in detections])
                        anomaly_ph.markdown(f'<div class="s-card threat"><div class="s-label">// anomaly</div>{threat_html}</div>', unsafe_allow_html=True)
                    else:
                        anomaly_ph.markdown('<div class="s-card clear"><div class="s-label">// anomaly</div><div class="s-value clear" style="font-size:18px">✓ CLEAR</div></div>', unsafe_allow_html=True)

                cap.release()

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "DATABASE":
    import pickle

    st.markdown("""
        <div class="page-header">
            <div>
                <div class="page-title">Identity Database</div>
                <div class="page-sub">// registered subjects — authorised personnel</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    PKL_PATH = os.path.join(_BASE, _cfg["PATH"]["PKL_PATH"])
    try:
        with open(PKL_PATH, "rb") as f:
            database = pickle.load(f)
    except:
        database = {}

    if not database:
        st.markdown('<div class="s-card"><div class="s-label">// status</div><div class="s-value unknown">NO RECORDS FOUND</div><div class="s-meta">Add subjects via the Updating page</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="m-row">
                <div class="m-tile"><div class="m-val">{len(database)}</div><div class="m-lbl">Total Subjects</div></div>
                <div class="m-tile"><div class="m-val">ACTIVE</div><div class="m-lbl">DB Status</div></div>
                <div class="m-tile"><div class="m-val">OpenCV</div><div class="m-lbl">Engine</div></div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        for idx, person in database.items():
            c1, c2, c3 = st.columns([1, 2, 6])
            with c1:
                img = person.get("image")
                if isinstance(img, np.ndarray) and img.size > 0:
                    st.image(img, width=80)
                else:
                    st.markdown('<div style="width:80px;height:80px;background:#0a1628;border:1px solid #1e3a5f;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:24px">👤</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                    <div style="padding:8px 0">
                        <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#334155;letter-spacing:2px">// ID</div>
                        <div style="font-family:'Rajdhani',sans-serif;font-size:20px;font-weight:700;color:#00ffaa">{person.get("id","—")}</div>
                    </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                    <div style="padding:8px 0">
                        <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#334155;letter-spacing:2px">// NAME</div>
                        <div style="font-family:'Rajdhani',sans-serif;font-size:22px;font-weight:700;color:#f1f5f9">{person.get("name","Unknown")}</div>
                        <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#475569">INDEX: {idx} &nbsp;|&nbsp; STATUS: AUTHORISED</div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('<hr style="border-color:#0a1628;margin:4px 0">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# UPDATING PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "UPDATING":
    import sys
    sys.path.insert(0, _BASE)
    from utils import submitNew, get_info_from_id, deleteOne

    st.markdown("""
        <div class="page-header">
            <div>
                <div class="page-title">Manage Database</div>
                <div class="page-sub">// add, remove, or update authorised subjects</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    action = st.radio("", ["Add Subject", "Delete Subject", "Update Subject"],
                      horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    def handle_submit(name, pid, image, old_idx=None):
        if not name.strip() or not pid.strip():
            st.error("Name and ID are required.")
            return
        ret = submitNew(name.strip(), pid.strip(), image, old_idx=old_idx)
        if ret is True:
            st.success(f"✅ {name} added/updated successfully.")
        elif ret == 0:
            st.error("ID already exists. Use Update to modify.")
        elif ret == -1:
            st.error("No face detected in image. Use a clear frontal photo.")

    if action == "Add Subject":
        c1, c2 = st.columns(2)
        name = c1.text_input("Full Name", placeholder="e.g. John Doe")
        pid  = c2.text_input("Subject ID", placeholder="e.g. 5")
        src  = st.radio("Image source", ["Upload", "Webcam"], horizontal=True)
        if src == "Upload":
            f = st.file_uploader("Photo", type=["jpg","png","jpeg"])
            if f:
                img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=200)
                if st.button("▶  REGISTER SUBJECT"):
                    handle_submit(name, pid, img)
        else:
            snap = st.camera_input("Capture")
            if snap:
                img = cv2.imdecode(np.frombuffer(snap.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                if st.button("▶  REGISTER SUBJECT"):
                    handle_submit(name, pid, img)

    elif action == "Delete Subject":
        pid = st.text_input("Subject ID to remove", placeholder="Enter ID")
        if st.button("LOOK UP"):
            name, image, _ = get_info_from_id(pid)
            if name:
                st.session_state["del_target"] = (name, image, pid)
            else:
                st.error("ID not found.")
        if "del_target" in st.session_state:
            n, img, p = st.session_state["del_target"]
            st.markdown(f'<div class="s-card unknown"><div class="s-label">// subject to remove</div><div class="s-value unknown">{n}</div><div class="s-meta">ID: {p}</div></div>', unsafe_allow_html=True)
            if isinstance(img, np.ndarray) and img.size > 0:
                st.image(img, width=150)
            if st.button("■  CONFIRM DELETE"):
                deleteOne(p)
                st.success(f"✅ {n} removed.")
                del st.session_state["del_target"]

    elif action == "Update Subject":
        pid = st.text_input("Subject ID to update", placeholder="Enter ID")
        if st.button("LOOK UP"):
            old_name, old_img, old_idx = get_info_from_id(pid)
            if old_name:
                st.session_state["upd_target"] = (old_name, old_img, old_idx, pid)
            else:
                st.error("ID not found.")
        if "upd_target" in st.session_state:
            old_name, old_img, old_idx, old_pid = st.session_state["upd_target"]
            c1, c2 = st.columns([2,1])
            new_name = c1.text_input("New Name", value=old_name)
            new_id   = c1.text_input("New ID",   value=old_pid)
            new_file = c1.file_uploader("New Photo (optional)", type=["jpg","png","jpeg"])
            if isinstance(old_img, np.ndarray) and old_img.size > 0:
                c2.image(old_img, caption="Current photo", width=150)
            if st.button("▶  SAVE CHANGES"):
                if new_file:
                    new_img = cv2.imdecode(np.frombuffer(new_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                else:
                    new_img = cv2.cvtColor(old_img, cv2.COLOR_RGB2BGR) if isinstance(old_img, np.ndarray) else old_img
                handle_submit(new_name, new_id, new_img, old_idx=old_idx)
                del st.session_state["upd_target"]