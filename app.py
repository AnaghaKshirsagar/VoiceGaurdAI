import streamlit as st
import os
import hashlib
import traceback
import numpy as np
import time
from datetime import datetime

import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from audio_analysis import analyze_audio


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="VoiceGuard AI | Voice Integrity Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "file_path" not in st.session_state:
    st.session_state.file_path = None
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "page" not in st.session_state:
    st.session_state.page = "New Scan"
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# THEME ENGINE
# ============================================================
def apply_theme(theme: str):
    if theme == "Light":
        css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
        .stApp { background: #f5f8f6; color: #1a2e24; }
        #MainMenu, header, footer, [data-testid="stToolbar"] { visibility: hidden !important; }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #eaf1ed 0%, #dde8e2 100%) !important;
            border-right: 1px solid rgba(0,120,60,0.15) !important;
        }
        section[data-testid="stSidebar"] .stRadio label {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 13px !important;
            color: #3d5c4a !important;
            padding: 9px 14px !important;
            border-radius: 8px !important;
        }
        section[data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(0,160,80,0.1) !important;
            color: #007a3d !important;
        }

        .sidebar-brand { display:flex; align-items:center; gap:12px; padding:6px 4px 22px 4px; border-bottom:1px solid rgba(0,120,60,0.12); margin-bottom:18px; }
        .sidebar-logo { width:42px; height:42px; background:linear-gradient(135deg,#00c853,#00a0e0); border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:20px; color:#fff; font-weight:800; box-shadow:0 4px 14px rgba(0,200,83,0.3); }
        .sidebar-title { font-family:'Orbitron',sans-serif; font-weight:700; font-size:16px; color:#007a3d; letter-spacing:1px; }
        .sidebar-sub { font-size:10px; color:#5a7a6a; letter-spacing:1.5px; margin-top:2px; }

        .status-badge { display:inline-flex; align-items:center; gap:6px; background:rgba(0,160,80,0.1); border:1px solid rgba(0,160,80,0.25); border-radius:20px; padding:5px 12px; font-family:'JetBrains Mono',monospace; font-size:11px; color:#007a3d; }
        .status-dot { width:7px; height:7px; background:#00c853; border-radius:50%; box-shadow:0 0 8px #00c853; animation:pulse 1.8s infinite; }
        @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }

        .main-hero h1 { font-family:'Orbitron',sans-serif; font-size:30px; font-weight:700; color:#1a2e24; margin:0 0 8px 0; }
        .main-hero h1 span { color:#007a3d; }
        .main-hero p { color:#4a6a5a; font-size:14px; max-width:620px; line-height:1.6; }

        .upload-card { background:#ffffff; border:1px dashed rgba(0,140,60,0.3); border-radius:16px; padding:38px 28px; text-align:center; margin-bottom:22px; box-shadow:0 4px 18px rgba(0,0,0,0.04); }
        .upload-icon { width:62px; height:62px; margin:0 auto 16px auto; background:linear-gradient(135deg,#00c853,#00a0e0); border-radius:16px; display:flex; align-items:center; justify-content:center; font-size:26px; color:#fff; box-shadow:0 6px 18px rgba(0,200,83,0.3); }
        .upload-title { font-family:'Orbitron',sans-serif; font-size:19px; font-weight:600; color:#1a2e24; margin-bottom:6px; }
        .upload-sub { color:#5a7a6a; font-size:13px; margin-bottom:18px; }
        .format-pills { display:flex; justify-content:center; gap:8px; flex-wrap:wrap; margin-bottom:20px; }
        .pill { background:rgba(0,160,80,0.08); border:1px solid rgba(0,160,80,0.2); border-radius:20px; padding:4px 12px; font-family:'JetBrains Mono',monospace; font-size:11px; color:#007a3d; }

        .score-panel { background:#ffffff; border:1px solid rgba(0,140,60,0.2); border-radius:16px; padding:26px; margin-bottom:20px; box-shadow:0 4px 18px rgba(0,0,0,0.04); }
        .score-label { font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:2px; color:#5a7a6a; margin-bottom:10px; }
        .score-value { font-family:'Orbitron',sans-serif; font-size:40px; font-weight:800; line-height:1; margin-bottom:6px; }
        .score-high { color:#d32f2f; } .score-medium { color:#f9a825; } .score-low { color:#007a3d; }
        .score-desc { color:#5a7a6a; font-size:13px; }

        .metric-card { background:#ffffff; border:1px solid rgba(0,140,60,0.12); border-radius:12px; padding:18px; position:relative; box-shadow:0 2px 10px rgba(0,0,0,0.03); }
        .metric-card::before { content:""; position:absolute; top:0; left:0; width:3px; height:100%; background:#00c853; border-radius:12px 0 0 12px; }
        .metric-label { font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:1.5px; color:#5a7a6a; margin-bottom:10px; }
        .metric-value { font-family:'Orbitron',sans-serif; font-size:23px; font-weight:700; }
        .green { color:#007a3d; } .cyan { color:#0088cc; } .yellow { color:#f9a825; } .red { color:#d32f2f; }

        .section-card { background:#ffffff; border:1px solid rgba(0,140,60,0.1); border-radius:14px; padding:20px; margin-bottom:16px; box-shadow:0 2px 10px rgba(0,0,0,0.03); }
        .section-title { font-family:'Orbitron',sans-serif; font-size:13px; letter-spacing:1.5px; color:#007a3d; margin-bottom:14px; display:flex; align-items:center; gap:8px; }
        .section-title::before { content:""; width:4px; height:14px; background:#00c853; border-radius:2px; }

        .hash-box { background:#f0f5f2; border:1px solid rgba(0,140,60,0.15); border-radius:8px; padding:14px 16px; font-family:'JetBrains Mono',monospace; font-size:12px; color:#3d5c4a; word-break:break-all; }

        .about-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:18px; }
        .about-card { background:#ffffff; border:1px solid rgba(0,140,60,0.12); border-radius:14px; padding:22px; box-shadow:0 2px 10px rgba(0,0,0,0.03); }
        .about-card h3 { font-family:'Orbitron',sans-serif; font-size:15px; color:#007a3d; margin:0 0 10px 0; }
        .about-card p { color:#4a6a5a; font-size:13px; line-height:1.6; margin:0; }

        .history-item { background:#ffffff; border:1px solid rgba(0,140,60,0.12); border-radius:12px; padding:16px 18px; margin-bottom:10px; transition:all 0.2s; }
        .history-item:hover { border-color:#00c853; box-shadow:0 4px 14px rgba(0,200,83,0.1); }

        .stButton > button {
            width:100%; height:46px; border-radius:10px !important;
            background:linear-gradient(90deg,rgba(0,200,83,0.15),rgba(0,160,220,0.1)) !important;
            border:1px solid rgba(0,160,80,0.4) !important; color:#007a3d !important;
            font-family:'Orbitron',sans-serif !important; font-weight:600 !important; font-size:13px !important; letter-spacing:1px !important;
        }
        .stButton > button:hover { background:linear-gradient(90deg,rgba(0,200,83,0.25),rgba(0,160,220,0.18)) !important; box-shadow:0 0 18px rgba(0,200,83,0.2) !important; }

        [data-testid="stFileUploaderDropzone"] { background:rgba(0,160,80,0.04) !important; border:1px dashed rgba(0,160,80,0.3) !important; border-radius:12px !important; }
        .stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid rgba(0,140,60,0.12); }
        .stTabs [data-baseweb="tab"] { background:transparent !important; color:#5a7a6a !important; border-radius:8px 8px 0 0 !important; font-family:'JetBrains Mono',monospace !important; font-size:12px !important; padding:9px 16px !important; }
        .stTabs [aria-selected="true"] { background:rgba(0,160,80,0.1) !important; color:#007a3d !important; border-bottom:2px solid #00c853 !important; }
        .stProgress > div > div > div { background:linear-gradient(90deg,#00c853,#00a0e0) !important; }
        .stTextArea textarea { background:#f8fbf9 !important; color:#1a2e24 !important; border:1px solid rgba(0,140,60,0.2) !important; border-radius:8px !important; font-family:'JetBrains Mono',monospace !important; }
        </style>
        """
    else:
        css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
        .stApp { background: #030505; color: #e0f7ef; }
        #MainMenu, header, footer, [data-testid="stToolbar"] { visibility: hidden !important; }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #050a08 0%, #020403 100%) !important;
            border-right: 1px solid rgba(0,255,100,0.12) !important;
        }
        section[data-testid="stSidebar"] .stRadio label {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 13px !important;
            color: #8ba89a !important;
            padding: 9px 14px !important;
            border-radius: 8px !important;
        }
        section[data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(0,255,100,0.08) !important;
            color: #00ff64 !important;
        }

        .sidebar-brand { display:flex; align-items:center; gap:12px; padding:6px 4px 22px 4px; border-bottom:1px solid rgba(0,255,100,0.1); margin-bottom:18px; }
        .sidebar-logo { width:42px; height:42px; background:linear-gradient(135deg,#00ff64,#00ccff); border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:20px; color:#000; font-weight:800; box-shadow:0 0 18px rgba(0,255,100,0.4); }
        .sidebar-title { font-family:'Orbitron',sans-serif; font-weight:700; font-size:16px; color:#00ff64; letter-spacing:1px; }
        .sidebar-sub { font-size:10px; color:#5a7a6a; letter-spacing:1.5px; margin-top:2px; }

        .status-badge { display:inline-flex; align-items:center; gap:6px; background:rgba(0,255,100,0.1); border:1px solid rgba(0,255,100,0.3); border-radius:20px; padding:5px 12px; font-family:'JetBrains Mono',monospace; font-size:11px; color:#00ff64; }
        .status-dot { width:7px; height:7px; background:#00ff64; border-radius:50%; box-shadow:0 0 8px #00ff64; animation:pulse 1.8s infinite; }
        @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }

        .main-hero h1 { font-family:'Orbitron',sans-serif; font-size:30px; font-weight:700; color:#f0fff8; margin:0 0 8px 0; }
        .main-hero h1 span { color:#00ff64; text-shadow:0 0 18px rgba(0,255,100,0.4); }
        .main-hero p { color:#7a9b8a; font-size:14px; max-width:620px; line-height:1.6; }

        .upload-card { background:linear-gradient(145deg,rgba(8,18,14,0.95),rgba(4,10,8,0.95)); border:1px dashed rgba(0,255,100,0.25); border-radius:16px; padding:38px 28px; text-align:center; margin-bottom:22px; }
        .upload-icon { width:62px; height:62px; margin:0 auto 16px auto; background:linear-gradient(135deg,#00ff64,#00ccff); border-radius:16px; display:flex; align-items:center; justify-content:center; font-size:26px; color:#000; box-shadow:0 0 26px rgba(0,255,100,0.35); }
        .upload-title { font-family:'Orbitron',sans-serif; font-size:19px; font-weight:600; color:#e8fff0; margin-bottom:6px; }
        .upload-sub { color:#6a8a7a; font-size:13px; margin-bottom:18px; }
        .format-pills { display:flex; justify-content:center; gap:8px; flex-wrap:wrap; margin-bottom:20px; }
        .pill { background:rgba(0,255,100,0.08); border:1px solid rgba(0,255,100,0.2); border-radius:20px; padding:4px 12px; font-family:'JetBrains Mono',monospace; font-size:11px; color:#00ff64; }

        .score-panel { background:linear-gradient(145deg,rgba(6,16,12,0.98),rgba(3,10,8,0.98)); border:1px solid rgba(0,255,100,0.18); border-radius:16px; padding:26px; margin-bottom:20px; }
        .score-label { font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:2px; color:#5a7a6a; margin-bottom:10px; }
        .score-value { font-family:'Orbitron',sans-serif; font-size:40px; font-weight:800; line-height:1; margin-bottom:6px; }
        .score-high { color:#ff3b5c; text-shadow:0 0 18px rgba(255,59,92,0.4); }
        .score-medium { color:#ffc857; text-shadow:0 0 18px rgba(255,200,87,0.4); }
        .score-low { color:#00ff64; text-shadow:0 0 18px rgba(0,255,100,0.4); }
        .score-desc { color:#7a9b8a; font-size:13px; }

        .metric-card { background:rgba(6,14,11,0.9); border:1px solid rgba(0,255,100,0.12); border-radius:12px; padding:18px; position:relative; }
        .metric-card::before { content:""; position:absolute; top:0; left:0; width:3px; height:100%; background:#00ff64; border-radius:12px 0 0 12px; }
        .metric-label { font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:1.5px; color:#5a7a6a; margin-bottom:10px; }
        .metric-value { font-family:'Orbitron',sans-serif; font-size:23px; font-weight:700; }
        .green { color:#00ff64; } .cyan { color:#00d4ff; } .yellow { color:#ffc857; } .red { color:#ff3b5c; }

        .section-card { background:rgba(6,14,11,0.85); border:1px solid rgba(0,255,100,0.1); border-radius:14px; padding:20px; margin-bottom:16px; }
        .section-title { font-family:'Orbitron',sans-serif; font-size:13px; letter-spacing:1.5px; color:#00ff64; margin-bottom:14px; display:flex; align-items:center; gap:8px; }
        .section-title::before { content:""; width:4px; height:14px; background:#00ff64; border-radius:2px; }

        .hash-box { background:#020403; border:1px solid rgba(0,255,100,0.15); border-radius:8px; padding:14px 16px; font-family:'JetBrains Mono',monospace; font-size:12px; color:#6b9e88; word-break:break-all; }

        .about-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:18px; }
        .about-card { background:rgba(6,14,11,0.9); border:1px solid rgba(0,255,100,0.12); border-radius:14px; padding:22px; }
        .about-card h3 { font-family:'Orbitron',sans-serif; font-size:15px; color:#00ff64; margin:0 0 10px 0; }
        .about-card p { color:#8ba89a; font-size:13px; line-height:1.6; margin:0; }

        .history-item { background:rgba(6,14,11,0.9); border:1px solid rgba(0,255,100,0.12); border-radius:12px; padding:16px 18px; margin-bottom:10px; transition:all 0.2s; }
        .history-item:hover { border-color:#00ff64; box-shadow:0 0 16px rgba(0,255,100,0.15); }

        .stButton > button {
            width:100%; height:46px; border-radius:10px !important;
            background:linear-gradient(90deg,rgba(0,255,100,0.15),rgba(0,200,255,0.1)) !important;
            border:1px solid rgba(0,255,100,0.4) !important; color:#00ff64 !important;
            font-family:'Orbitron',sans-serif !important; font-weight:600 !important; font-size:13px !important; letter-spacing:1px !important;
        }
        .stButton > button:hover { background:linear-gradient(90deg,rgba(0,255,100,0.25),rgba(0,200,255,0.18)) !important; box-shadow:0 0 22px rgba(0,255,100,0.25) !important; color:#fff !important; }

        [data-testid="stFileUploaderDropzone"] { background:rgba(0,255,100,0.03) !important; border:1px dashed rgba(0,255,100,0.3) !important; border-radius:12px !important; }
        .stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid rgba(0,255,100,0.1); }
        .stTabs [data-baseweb="tab"] { background:transparent !important; color:#6a8a7a !important; border-radius:8px 8px 0 0 !important; font-family:'JetBrains Mono',monospace !important; font-size:12px !important; padding:9px 16px !important; }
        .stTabs [aria-selected="true"] { background:rgba(0,255,100,0.1) !important; color:#00ff64 !important; border-bottom:2px solid #00ff64 !important; }
        .stProgress > div > div > div { background:linear-gradient(90deg,#00ff64,#00d4ff) !important; }
        .stTextArea textarea { background:#020403 !important; color:#c8e6d4 !important; border:1px solid rgba(0,255,100,0.15) !important; border-radius:8px !important; font-family:'JetBrains Mono',monospace !important; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

apply_theme(st.session_state.theme)


# ============================================================
# HELPERS
# ============================================================
def normalize_score(value):
    try:
        value = float(value)
        if value <= 1:
            value *= 100
        return max(0, min(100, value))
    except:
        return 0

def get_value(data, keys, default=None):
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default

def detector_data(result):
    detector = result.get("detector", {}) or result.get("deepfake_analysis", {})
    fake = get_value(detector, ["fake_score", "fake_probability", "synthetic_score"], result.get("fake_score", 0))
    real = get_value(detector, ["real_score", "real_probability", "authentic_score"], result.get("real_score", 0))
    raw  = get_value(detector, ["raw_results", "raw_output", "results"])
    return normalize_score(fake), normalize_score(real), raw

def fraud_data(result):
    fraud = result.get("fraud_analysis", {})
    return (
        str(get_value(fraud, ["risk_level", "risk", "level"], "UNKNOWN")).upper(),
        get_value(fraud, ["risk_points", "points", "score"], 0),
        get_value(fraud, ["signals", "fraud_signals"], [])
    )

def render_waveform(path):
    try:
        audio, sr = librosa.load(path, sr=None, mono=True)
        bg = "#030505" if st.session_state.theme == "Dark" else "#f5f8f6"
        color = "#00ff64" if st.session_state.theme == "Dark" else "#007a3d"
        fig, ax = plt.subplots(figsize=(11, 2.3), facecolor=bg)
        ax.set_facecolor(bg)
        librosa.display.waveshow(audio, sr=sr, ax=ax, color=color, alpha=0.9)
        ax.set_xlabel("Time (s)", color="#5a7a6a", fontsize=8)
        ax.set_ylabel("Amp", color="#5a7a6a", fontsize=8)
        ax.tick_params(colors="#5a7a6a", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#1a2e24")
        ax.grid(True, alpha=0.1, color=color)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.warning("Waveform unavailable")
        with st.expander("Error"):
            st.code(str(e))

def render_spectrogram(path):
    try:
        audio, sr = librosa.load(path, sr=None, mono=True)
        if len(audio) > sr * 60:
            audio = audio[:int(sr * 60)]
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        db = librosa.power_to_db(mel, ref=np.max)
        bg = "#030505" if st.session_state.theme == "Dark" else "#f5f8f6"
        title_color = "#00ff64" if st.session_state.theme == "Dark" else "#007a3d"
        fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=bg)
        ax.set_facecolor(bg)
        img = librosa.display.specshow(db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma", fmax=8000)
        ax.set_title("VOICE FREQUENCY SIGNATURE", fontsize=10, color=title_color, fontfamily="monospace", pad=8)
        ax.tick_params(colors="#5a7a6a", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#1a2e24")
        cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
        cbar.ax.yaxis.set_tick_params(color="#5a7a6a", labelsize=7)
        cbar.outline.set_edgecolor("#1a2e24")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#5a7a6a")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.warning("Spectrogram unavailable")
        with st.expander("Error details"):
            st.code(str(e))

def add_to_history(result, file_name, file_hash, path):
    fake, real, _ = detector_data(result)
    risk, points, signals = fraud_data(result)
    entry = {
        "id": len(st.session_state.history) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": file_name,
        "file_hash": file_hash,
        "file_path": path,
        "fake_score": fake,
        "real_score": real,
        "risk_level": risk,
        "risk_points": points,
        "signals_count": len(signals) if signals else 0,
        "duration": result.get("duration"),
        "transcript": result.get("transcript", ""),
        "result": result
    }
    st.session_state.history.insert(0, entry)
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[:50]


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">◈</div>
        <div>
            <div class="sidebar-title">VoiceGuard</div>
            <div class="sidebar-sub">VOICE INTEGRITY AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**WORKSPACE**")
    page = st.radio(
        "Navigation",
        ["New Scan", "Overview", "History", "About Us", "Settings"],
        label_visibility="collapsed",
        key="nav_radio"
    )
    st.session_state.page = page

    st.markdown("---")
    st.markdown("**SYSTEM**")
    st.markdown("""
    <div class="status-badge">
        <span class="status-dot"></span>
        System Online · v2.4.1
    </div>
    """, unsafe_allow_html=True)
    st.caption("AI Voice Integrity Platform")


# ============================================================
# NEW SCAN
# ============================================================
if st.session_state.page == "New Scan":
    st.markdown("""
    <div class="main-hero">
        <h1>Voice <span>Authenticity</span> Detection Suite</h1>
        <p>Deepfake-aware audio forensics with explainable AI. Upload a recording for synthetic voice detection, fraud-language analysis, transcription and digital evidence hashing.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-card">
        <div class="upload-icon">↑</div>
        <div class="upload-title">Drag & Drop Audio</div>
        <div class="upload-sub">Or browse to upload — analysis begins instantly</div>
        <div class="format-pills">
            <span class="pill">MP3</span><span class="pill">WAV</span><span class="pill">M4A</span>
            <span class="pill">FLAC</span><span class="pill">OGG</span><span class="pill">AAC</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload audio", type=["wav","mp3","m4a","flac","ogg","aac"], label_visibility="collapsed")

    if uploaded:
        os.makedirs("uploads", exist_ok=True)
        data = uploaded.getvalue()
        file_hash = hashlib.sha256(data).hexdigest()
        extension = os.path.splitext(uploaded.name)[1].lower()
        path = os.path.join("uploads", file_hash + extension)

        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(data)

        st.session_state.file_path = path
        st.session_state.file_hash = file_hash
        st.session_state.file_name = uploaded.name

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_btn = st.button("▶  ANALYZE VOICE", use_container_width=True)

        if analyze_btn:
            with st.status("Running forensic pipeline...", expanded=True) as status:
                st.write("INITIALIZING FORENSIC ENGINE...")
                time.sleep(0.2)
                st.write("ANALYZING VOICE SIGNATURE...")
                time.sleep(0.2)
                st.write("TRANSCRIBING SPEECH...")
                time.sleep(0.2)
                st.write("SCANNING FRAUD INDICATORS...")
                time.sleep(0.2)
                st.write("GENERATING THREAT ASSESSMENT...")
                try:
                    result = analyze_audio(path)
                    st.session_state.analysis_result = result
                    add_to_history(result, uploaded.name, file_hash, path)
                    status.update(label="Analysis complete", state="complete")
                except Exception:
                    status.update(label="Analysis failed", state="error")
                    st.error("Forensic engine error")
                    with st.expander("Technical details"):
                        st.code(traceback.format_exc())

    result = st.session_state.analysis_result
    if result:
        fake, real, raw = detector_data(result)
        risk, points, signals = fraud_data(result)
        transcript = result.get("transcript", "")
        duration = result.get("duration", None)
        sample_rate = result.get("sample_rate", None)

        if fake >= 75 or risk == "HIGH":
            threat_level, threat_class = "HIGH RISK", "score-high"
            threat_score = max(fake, points * 10) if points else fake
        elif fake >= 50 or risk == "MEDIUM":
            threat_level, threat_class = "MEDIUM RISK", "score-medium"
            threat_score = max(fake, points * 8) if points else fake
        else:
            threat_level, threat_class = "LOW RISK", "score-low"
            threat_score = max(fake, points * 5) if points else fake

        st.markdown(f"""
        <div class="score-panel">
            <div class="score-label">VOICE THREAT ASSESSMENT</div>
            <div class="score-value {threat_class}">{threat_level}</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:14px; color:#8ba89a; margin-bottom:8px;">
                Threat Score: {threat_score:.0f} · Combined authenticity + fraud signals
            </div>
            <div class="score-desc">Assessment derived from synthetic voice probability and fraud-language pattern matching.</div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">SYNTHETIC SIGNAL</div><div class="metric-value red">{fake:.1f}%</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">AUTHENTICITY</div><div class="metric-value green">{real:.1f}%</div></div>', unsafe_allow_html=True)
        with m3:
            risk_c = "red" if risk == "HIGH" else "yellow" if risk == "MEDIUM" else "green"
            st.markdown(f'<div class="metric-card"><div class="metric-label">FRAUD RISK</div><div class="metric-value {risk_c}">{risk}</div></div>', unsafe_allow_html=True)
        with m4:
            dur = f"{float(duration):.1f}s" if duration else "N/A"
            st.markdown(f'<div class="metric-card"><div class="metric-label">DURATION</div><div class="metric-value cyan">{dur}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Voice Forensics", "Transcript", "Fraud Intelligence", "Evidence"])

        with tab1:
            st.markdown('<div class="section-title">AUTHENTICITY ANALYSIS</div>', unsafe_allow_html=True)
            st.progress(min(fake / 100, 1.0))
            st.caption(f"Synthetic voice signal detected: {fake:.1f}%")
            st.markdown('<div class="section-title">FRAUD SUMMARY</div>', unsafe_allow_html=True)
            st.write(f"**Risk Level:** {risk}  |  **Points:** {points}  |  **Signals:** {len(signals) if signals else 0}")

        with tab2:
            st.markdown('<div class="section-title">VOICE WAVEFORM</div>', unsafe_allow_html=True)
            render_waveform(st.session_state.file_path)
            st.markdown('<div class="section-title">VOICE FREQUENCY SIGNATURE</div>', unsafe_allow_html=True)
            render_spectrogram(st.session_state.file_path)
            c1, c2 = st.columns(2)
            with c1: st.metric("Sample Rate", f"{sample_rate} Hz" if sample_rate else "N/A")
            with c2: st.metric("Duration", f"{float(duration):.2f} s" if duration else "N/A")

        with tab3:
            st.markdown('<div class="section-title">TRANSCRIBED SPEECH</div>', unsafe_allow_html=True)
            if transcript:
                st.text_area("Speech content", transcript, height=280, label_visibility="collapsed")
                st.caption("Generated with AI speech recognition")
            else:
                st.info("No speech detected in the recording.")

        with tab4:
            st.markdown('<div class="section-title">FRAUD INTELLIGENCE</div>', unsafe_allow_html=True)
            st.write(f"**Fraud Risk:** {risk}  |  **Risk Points:** {points}")
            if signals:
                for sig in signals:
                    if isinstance(sig, dict):
                        cat = sig.get("category", "Unknown").replace("_", " ").title()
                        pts = sig.get("points", 0)
                        matches = sig.get("matches", [])
                        st.markdown(f'<div class="section-card"><strong>{cat}</strong><br><span style="font-size:12px;color:#7a9b8a;">Risk points: {pts}</span></div>', unsafe_allow_html=True)
                        if matches: st.write("Matched:", matches)
            else:
                st.success("No significant fraud-language indicators detected.")

        with tab5:
            st.markdown('<div class="section-title">DIGITAL EVIDENCE INTEGRITY</div>', unsafe_allow_html=True)
            st.write("**File Name**"); st.code(st.session_state.file_name)
            st.write("**SHA-256 Hash**")
            st.markdown(f'<div class="hash-box">{st.session_state.file_hash}</div>', unsafe_allow_html=True)
            if raw:
                with st.expander("Advanced Detector Information"):
                    try: st.json(raw)
                    except: st.write(raw)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↻  START NEW ANALYSIS"):
            st.session_state.analysis_result = None
            st.session_state.file_path = None
            st.session_state.file_hash = None
            st.session_state.file_name = None
            st.rerun()


# ============================================================
# OVERVIEW
# ============================================================
elif st.session_state.page == "Overview":
    st.markdown("""
    <div class="main-hero">
        <h1>Forensic <span>Overview</span></h1>
        <p>Platform status and summary of recent activity.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-label">PLATFORM STATUS</div><div class="metric-value green">ONLINE</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">TOTAL SCANS</div><div class="metric-value cyan">{len(st.session_state.history)}</div></div>', unsafe_allow_html=True)
    with c3:
        high_count = sum(1 for h in st.session_state.history if h["risk_level"] == "HIGH")
        st.markdown(f'<div class="metric-card"><div class="metric-label">HIGH RISK</div><div class="metric-value red">{high_count}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">THEME</div><div class="metric-value yellow">{st.session_state.theme.upper()}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div class="section-title">RECENT SCANS</div>', unsafe_allow_html=True)
        for entry in st.session_state.history[:6]:
            risk_color = "red" if entry["risk_level"] == "HIGH" else "yellow" if entry["risk_level"] == "MEDIUM" else "green"
            st.markdown(f"""
            <div class="history-item">
                <strong>{entry['file_name']}</strong> &nbsp;·&nbsp; {entry['timestamp']}<br>
                <span style="font-size:12px; color:#7a9b8a;">
                    Risk: <span class="{risk_color}">{entry['risk_level']}</span> &nbsp;|&nbsp;
                    Synthetic: {entry['fake_score']:.1f}% &nbsp;|&nbsp;
                    Signals: {entry['signals_count']}
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No scans yet. Go to **New Scan** to analyze your first audio file.")


# ============================================================
# HISTORY
# ============================================================
elif st.session_state.page == "History":
    st.markdown("""
    <div class="main-hero">
        <h1>Scan <span>History</span></h1>
        <p>All previous forensic analyses performed in this session.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No scan history available yet. Perform an analysis in **New Scan** to build history.")
    else:
        col_a, col_b = st.columns([3, 1])
        with col_b:
            if st.button("Clear All History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        st.write(f"**{len(st.session_state.history)}** scan(s) recorded")

        for entry in st.session_state.history:
            with st.expander(f"{entry['file_name']}  ·  {entry['timestamp']}  ·  {entry['risk_level']}"):
                st.write(f"**SHA-256:** `{entry['file_hash']}`")
                st.write(f"**Synthetic Signal:** {entry['fake_score']:.1f}%")
                st.write(f"**Authenticity Signal:** {entry['real_score']:.1f}%")
                st.write(f"**Fraud Risk:** {entry['risk_level']} ({entry['risk_points']} points)")
                st.write(f"**Signals Detected:** {entry['signals_count']}")
                if entry.get("duration"):
                    st.write(f"**Duration:** {float(entry['duration']):.2f} s")
                if entry.get("transcript"):
                    st.text_area("Transcript", entry["transcript"], height=120, key=f"hist_t_{entry['id']}")

                if st.button("Load this scan", key=f"load_{entry['id']}"):
                    st.session_state.analysis_result = entry["result"]
                    st.session_state.file_path = entry["file_path"]
                    st.session_state.file_hash = entry["file_hash"]
                    st.session_state.file_name = entry["file_name"]
                    st.session_state.page = "New Scan"
                    st.rerun()


# ============================================================
# ABOUT US
# ============================================================
elif st.session_state.page == "About Us":
    st.markdown("""
    <div class="main-hero">
        <h1>About <span>VoiceGuard AI</span></h1>
        <p>Professional voice integrity and audio forensics platform.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-grid">
        <div class="about-card">
            <h3>◈ Mission</h3>
            <p>VoiceGuard AI detects synthetic and deepfake voices as well as scam language patterns in audio recordings. We combine AI audio classification, speech transcription and rule-based fraud intelligence into a single forensic workspace.</p>
        </div>
        <div class="about-card">
            <h3>◈ What We Detect</h3>
            <p>• AI-generated / deepfake speech<br>
            • Financial & credential request language<br>
            • Urgency and impersonation tactics<br>
            • Digital evidence integrity via SHA-256 hashing</p>
        </div>
        <div class="about-card">
            <h3>◈ Technology</h3>
            <p>Powered by Hugging Face Transformers, OpenAI Whisper, librosa signal analysis and a custom fraud-pattern engine. Designed for transparent, explainable voice threat assessment.</p>
        </div>
        <div class="about-card">
            <h3>◈ Vision</h3>
            <p>Voice is becoming a critical attack surface. VoiceGuard AI provides security teams, financial institutions and individuals with a fast, reliable tool to verify whether a voice can be trusted.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.success("VoiceGuard AI — Detect the voice. Expose the threat.")


# ============================================================
# SETTINGS
# ============================================================
elif st.session_state.page == "Settings":
    st.markdown("""
    <div class="main-hero">
        <h1>System <span>Settings</span></h1>
        <p>Customize appearance and manage platform preferences.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">APPEARANCE</div>', unsafe_allow_html=True)

    theme_choice = st.radio(
        "Theme Mode",
        ["Dark", "Light"],
        index=0 if st.session_state.theme == "Dark" else 1,
        horizontal=True
    )

    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">DATA MANAGEMENT</div>', unsafe_allow_html=True)
    st.write(f"Current scan history contains **{len(st.session_state.history)}** entries.")

    if st.button("Clear Scan History"):
        st.session_state.history = []
        st.success("History cleared.")
        time.sleep(0.6)
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">PLATFORM INFORMATION</div>', unsafe_allow_html=True)
    st.write("""
    **VoiceGuard AI** combines:
    - Deepfake / synthetic voice detection  
    - Speech-to-text transcription  
    - Fraud & social-engineering language analysis  
    - Cryptographic evidence hashing (SHA-256)  
    - Waveform and spectrogram forensics  
    """)
    st.caption("All analysis runs locally in this session. No audio is stored permanently beyond the current browser session.")