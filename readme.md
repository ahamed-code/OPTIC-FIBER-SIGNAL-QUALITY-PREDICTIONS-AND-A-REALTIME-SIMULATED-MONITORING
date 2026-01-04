# 📡 AI-Based Optical Fiber Signal Monitoring with Explainable AI

An **AI-driven Optical Fiber & AirFiber signal quality monitoring system** designed for **telecom Network Operations Centers (NOC)**.  
The system predicts **signal health**, explains **why a fault occurs using Explainable AI (XAI)**, and generates **proactive alerts** before customer complaints arise.

---

## 🚀 Project Highlights

- 🔍 **Predicts Optical Signal Status**  
  - HEALTHY  
  - DEGRADED  
  - CRITICAL  

- 🧠 **Explainable AI (SHAP)**  
  - Explains which parameters caused the prediction  
  - Builds trust for telecom engineers  

- 📈 **Real-Time Signal Visualization**  
  - Simulated live AirFiber / Optical waveform  
  - Noise, attenuation & dispersion effects  

- 🚨 **NOC Alert System**  
  - Automatically generates alerts for degraded/critical links  
  - Suggests operational actions  

- ☁️ **Cloud Deployable**  
  - Built using Streamlit  
  - Easily deployable on Streamlit Cloud  

---

## 🏗️ System Architecture

Customer
↓
Optical Fiber / AirFiber
↓
Network Monitoring Sensors (Simulated)
↓
NOC (Network Operations Center)
↓
AI Prediction Engine
↓
Explainable AI (SHAP)
↓
Alerts & Preventive Actions

yaml
Copy code

---

## 📊 Input Parameters

| Feature | Description |
|------|------------|
| Power (dBm) | Optical signal strength |
| Noise | Signal noise level |
| BER | Bit Error Rate |
| Q-Factor | Signal quality metric |
| Wavelength (nm) | Transmission wavelength |
| Distance (km) | Fiber length |

---

## 🧪 Prediction Output

| Status | Meaning | NOC Action |
|------|--------|-----------|
| HEALTHY | Signal stable | No action required |
| DEGRADED | Signal weakening | Monitor & preventive maintenance |
| CRITICAL | Severe fault / fiber cut | Immediate field dispatch |

---

## 🧠 Explainable AI (XAI)

This project uses **SHAP (SHapley Additive exPlanations)** to:
- Identify **top contributing features**
- Explain predictions in **human-readable language**
- Improve decision transparency for NOC engineers

Example explanation:
> *“Low optical power is the primary reason for signal degradation, indicating possible fiber bending or connector loss.”*

---

## 🖥️ Technology Stack

- **Frontend:** Streamlit  
- **ML Model:** XGBoost  
- **Explainable AI:** SHAP  
- **Visualization:** Plotly  
- **Backend:** Python  
- **Deployment:** Streamlit Cloud  

---

## 📁 Project Structure

optical/
│
├── src/
│ └── app.py # Streamlit application
│
├── model/
│ ├── xgb_model.pkl # Trained ML model
│ ├── scaler.pkl # Feature scaler
│ ├── columns.pkl # Feature names
│ └── shap_background.pkl # SHAP background data
│
├── requirements.txt
└── README.md

yaml
Copy code

---

## ▶️ How to Run Locally

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
2️⃣ Run the App
bash
Copy code
streamlit run src/app.py
☁️ Deployment (Streamlit Cloud)
Push project to GitHub

Visit 👉 https://share.streamlit.io

Select repository

Set main file:

bash
Copy code
src/app.py
Click Deploy

🎯 Real-World Relevance
Prevents customer complaints

Enables predictive maintenance

Reduces downtime and operational cost

Aligns with telecom NOC workflows

⚠️ Note:
This project uses simulated real-time data.
In real deployments, data can be sourced via SNMP, OTDR, NetFlow, or OSS systems.

🎓 Academic Significance
Suitable for Final Year Engineering Project

Covers:

Machine Learning

Explainable AI

Telecom Networks

Cloud Deployment

👨‍💻 Author
BASHEER AHAMED A
Final Year Project – AI & Telecom Networks

📜 License
This project is for academic and educational purposes.#

