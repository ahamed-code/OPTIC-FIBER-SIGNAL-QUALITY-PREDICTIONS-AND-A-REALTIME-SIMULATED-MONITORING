import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import shap
import os
import time
import random

# ======================================================
# PATH HANDLING
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

# ======================================================
# LOAD MODEL & ARTIFACTS
# ======================================================
try:
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))
    shap_background = joblib.load(os.path.join(MODEL_DIR, "shap_background.pkl"))
except Exception:
    st.error("❌ Model files missing. Train the model first.")
    st.stop()

# ======================================================
# STREAMLIT CONFIG
# ======================================================
st.set_page_config(
    page_title="AI Optical Fiber Monitoring",
    layout="wide"
)

st.title("📡 AI-Based Optical Signal Quality Prediction with Explainable AI & Real-Time Monitoring")
st.markdown(
    "Real-time **optical fiber / AirFiber signal monitoring** with **Explainable AI (XAI)** for NOC operations."
)

# ======================================================
# REAL-TIME AIRFIBER SIMULATION
# ======================================================
def read_airfiber_realtime():
    return {
        "power": random.uniform(-22, -9),
        "noise": random.uniform(0.01, 0.08),
        "ber": 10 ** random.uniform(-8, -4),
        "q_factor": random.uniform(6, 18),
        "wavelength": 1550,
        "distance": random.uniform(5, 40)
    }

# ======================================================
# SIDEBAR INPUTS
# ======================================================
st.sidebar.markdown("### 🔁 Prediction Mode")
mode = st.sidebar.radio(
    "Select input source:",
    ["Manual Input", "Real-Time Example Simulation"]
)

if mode == "Manual Input":
    power = st.sidebar.slider("Power (dBm)", -25.0, -5.0, -12.0)
    noise = st.sidebar.slider("Noise (dB)", 0.001, 0.1, 0.02)
    ber = st.sidebar.slider("Bit Error Rate (BER)", 1e-10, 1e-3, 1e-6, format="%.10f")
    q_factor = st.sidebar.slider("Q-Factor", 3.0, 20.0, 14.0)
    wavelength = st.sidebar.selectbox("Wavelength (nm)", [1540, 1550, 1552, 1555])
    distance = st.sidebar.slider("Distance (km)", 1.0, 120.0, 25.0)

    input_dict = {
        "power": power,
        "noise": noise,
        "ber": ber,
        "q_factor": q_factor,
        "wavelength": wavelength,
        "distance": distance
    }

else:
    # Realtime simulation
    input_dict = read_airfiber_realtime()
    st.sidebar.success("📡 Live AirFiber connected")
    for k, v in input_dict.items():
        st.sidebar.write(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# ======================================================
# PREPARE INPUT
# ======================================================
X = np.array([[input_dict[col] for col in columns]])
X_scaled = scaler.transform(X)

# ======================================================
# MODEL PREDICTION
# ======================================================
probs = model.predict_proba(X_scaled)[0]
pred = int(np.argmax(probs))
confidence = probs[pred]

status_map = {0: "DEGRADED", 1: "HEALTHY", 2: "CRITICAL"}
color_map = {0: "orange", 1: "green", 2: "red"}

status = status_map.get(pred, "UNKNOWN")
color = color_map.get(pred, "gray")

st.markdown(f"## Signal Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
st.write(f"### Prediction Confidence: **{confidence * 100:.2f}%**")

# ======================================================
# LIVE SIGNAL WAVEFORM
# ======================================================
st.subheader("📈 Live Optical Signal Waveform")

t = np.linspace(0, 1, 500)
frequency = 4 + (20 - input_dict["q_factor"]) / 3
signal = np.sin(2 * np.pi * frequency * t)
signal *= np.exp(-input_dict["distance"] / 80)
signal += np.random.normal(0, input_dict["noise"], len(t))

fig = go.Figure()
fig.add_trace(go.Scatter(y=signal, mode="lines", name="Optical Signal"))
fig.update_layout(
    height=350,
    title="Real-Time Optical Signal",
    xaxis_title="Time",
    yaxis_title="Amplitude"
)
st.plotly_chart(fig, use_container_width=True)

# ======================================================
# RULE-BASED FAULT DIAGNOSIS
# ======================================================
st.subheader("📌 Fault Cause Analysis")
if input_dict["ber"] > 1e-5:
    st.error("🚨 High BER — Noise or interference detected")
elif input_dict["q_factor"] < 8:
    st.warning("⚠️ Low Q-Factor — Dispersion or attenuation issue")
elif input_dict["power"] < -18:
    st.warning("⚠️ Low Power — Fiber bending or connector loss")
else:
    st.success("✅ Signal parameters within acceptable range")

# ======================================================
# EXPLAINABLE AI (SHAP)
# ======================================================
st.subheader("🧠 Explainable AI — Feature Impact")
explainer = shap.Explainer(model.predict_proba, shap_background, feature_names=columns)
shap_values = explainer(X_scaled)
shap_class = shap_values.values[0][:, pred]

shap_df = pd.DataFrame({"Feature": columns, "Impact": np.abs(shap_class)}).sort_values(by="Impact", ascending=False)
st.bar_chart(shap_df.set_index("Feature"))

top_feature = shap_df.iloc[0]["Feature"]
reason_map = {
    "power": "Low optical power",
    "noise": "High noise level",
    "ber": "Elevated bit error rate",
    "q_factor": "Poor signal quality",
    "distance": "High attenuation due to distance",
    "wavelength": "Wavelength mismatch"
}
st.info(f"🔍 **Primary decision factor:** {reason_map.get(top_feature, top_feature)}")

# ======================================================
# NOC ALERT SIMULATION
# ======================================================
if pred in [0, 2]:
    st.error("🚨 NOC ALERT GENERATED")
    st.write(f"📍 Root cause likely related to **{top_feature.upper()}**")

# ======================================================
# AUTO REFRESH EVERY 3 SECONDS FOR REAL-TIME MODE
# ======================================================

if mode == "Real-Time Example Simulation":
    time.sleep(3)
    st.rerun()  # works in latest Streamlit
# ======================================================
# FOOTER
st.markdown("---")
st.caption(
    "Final Year Project | AI-based Optical Fiber & AirFiber Monitoring with Explainable AI | Done by Nityasri"
)
