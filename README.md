# AI-Powered Predictive Maintenance System

A comprehensive machine learning and web-based system designed to monitor industrial rotating equipment, predict failures, estimate Remaining Useful Life (RUL), explain model decisions using XAI, and provide actionable insights through an interactive Digital Twin dashboard and an LLM maintenance assistant.

---

## 👥 Team Members
* Salah Mohy Mohamed
* Youssef Mohamed Abdelrahman
* Jana Mohamed Fadl
* Zeyad Yasser Mahrous
* Mariam Khaled Gamal
* Tasneem Ahmed Mohamed

---

## 🚀 Key Features
* **Time-Series Preprocessing & FFT:** Automated sensor data validation, noise filtering, and Fast Fourier Transform for vibration signals.
* **Anomaly Detection & RUL Prediction:** Deep learning models (LSTM, GRU, 1D-CNN) to detect failures and forecast Remaining Useful Life with confidence intervals.
* **Explainable AI (XAI):** Integrated SHAP and LIME visualizations explaining the top contributing sensor factors for critical alerts.
* **Interactive Digital Twin Dashboard:** A role-based Streamlit interface featuring live monitoring, alert management, and a What-If simulation sandbox.
* **LLM Maintenance Assistant:** LangChain-powered conversational assistant grounded in real-time machine logs (RAG) with a local model fallback.
* **Automated Diagnostic Reporting:** Instant PDF report generation containing health summaries, XAI insights, and AI-generated narratives.
* **MLOps & Tracking:** Complete experiment tracking, model versioning, and drift monitoring using MLflow.

---

## 🛠️ Tech Stack
* **Backend / API:** FastAPI, Python
* **Frontend / UI:** Streamlit, Plotly
* **AI / ML / XAI:** Scikit-learn, TensorFlow / PyTorch, SHAP, LIME, LangChain
* **MLOps & Tracking:** MLflow

---

## ⚙️ Project Architecture & Workflow
1. Sensor data stream ingestion & validation.
2. Feature engineering & frequency-domain transformation (FFT).
3. FastAPI inference service running anomaly detection & RUL models.
4. Streamlit dashboard rendering real-time health, charts, XAI plots, and LLM chat.

---

## 📥 Installation & Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME
