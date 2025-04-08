# --- Your existing imports remain unchanged ---
import streamlit as st
from streamlit_option_menu import option_menu
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import joblib
import os
from io import StringIO
import requests
from codebase.dashboard_graphs import MaternalHealthDashboard
import cohere

# --- File Paths ---
model_path = r"C:\Users\Madhvi\OneDrive\Desktop\pragati_hackathon_pregnancy_risk_prediction\model\maternal_risk_model.pkl"
scaler_path = r"C:\Users\Madhvi\OneDrive\Desktop\pragati_hackathon_pregnancy_risk_prediction\model\scaler.pkl"
label_path = r"C:\Users\Madhvi\OneDrive\Desktop\pragati_hackathon_pregnancy_risk_prediction\model\label_encoder.pkl"

fetal_model = joblib.load(open("model\\fetal_health_model.pkl", 'rb'))
scaler_f = joblib.load(open("model\\fetal_scaler.pkl", 'rb'))

# --- Load Maternal Model Assets ---
maternal_model = joblib.load(open(model_path, 'rb'))
scaler = joblib.load(open(scaler_path, 'rb'))
label_encoder = joblib.load(open(label_path, 'rb'))

# --- Sidebar Menu ---
with st.sidebar:
    st.title("💙 MedPredict")
    selected = option_menu(
        'MedPredict',
        ['About us', 'Pregnancy Risk Prediction', '📄Fetal Health Prediction', 'Dashboard'],
        icons=['chat-square-text', 'heart-pulse',' ', 'bar-chart-line'],
        default_index=0
    )

# --- About Us Page ---
if selected == 'About us':
    st.markdown("<h1 style='text-align: center; color: #6C63FF;'> Welcome to <em>MedPredict</em></h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size:18px;'>
        <p>Empowering maternal and fetal healthcare through AI-driven predictive solutions.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3><img src="https://tse4.mm.bing.net/th?id=OIP.2PivYzvysBIObm7mT4NwbwHaHa&pid=Api&P=0&h=180" width="30" style="vertical-align: middle; margin-right: 10px;">Pregnancy Risk Prediction</h3>', unsafe_allow_html=True)
        st.markdown("""
        <ul>
            <li>AI-powered maternal risk analysis.</li>
            <li>Track key vitals: BP, glucose, temp, heart rate.</li>
        </ul>
        """, unsafe_allow_html=True)
        st.image("graphics/pregnancy_risk_image.jpg", caption="Empowering Safe Pregnancies", use_container_width=True)

    with col2:
        st.markdown('<h3><img src="https://cdn-icons-png.flaticon.com/512/6322/6322022.png" width="30" style="vertical-align: middle; margin-right: 10px;">Fetal health Predictor</h3>', unsafe_allow_html=True)
        st.markdown("""
        <ul>
            <li>Interpretation of CTG data.</li>
            <li>Classification: Normal / Suspect / Pathological.</li>
        </ul>
        """, unsafe_allow_html=True)
        st.image("graphics/fetal_health_image.jpg", caption="AI-Assisted Healthy Beginnings", use_container_width=True)

    # ChatBot only here
    st.markdown("---")
    st.markdown('<h3><img src="https://tse1.mm.bing.net/th?id=OIP.87ZRkORvjBkzOuSeBZ0mrAHaHa&pid=Api&P=0&h=180" width="30" style="vertical-align: middle; margin-right: 10px;">Pregnancy AI Chat Bot </h3>', unsafe_allow_html=True)
    with st.expander("💬 Ask about maternal health"):
        user_query = st.text_input("📝 Your Question:", placeholder="e.g., What is normal BP during pregnancy?")
        if st.button("Ask"):
            if user_query.strip():
                with st.spinner("Thinking..."):
                    try:
                        co = cohere.Client("P439t9JWBvJhMi6RjwaPPaPI8NTj1zdQgM2yTg32")
                        response = co.chat(
                            message=user_query,
                            preamble="You are a caring, knowledgeable assistant for maternal and fetal health.",
                            temperature=0.5
                        )
                        st.success(response.text)
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            else:
                st.warning("Please enter a question.")

# --- Pregnancy Risk Prediction Page ---
if selected == 'Pregnancy Risk Prediction':
    st.title(' Pregnancy Risk Prediction')
    st.markdown("Predict potential pregnancy risks with simple input parameters.")

    with st.container():
        st.markdown("####  Enter the following health parameters:")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input('Age', key="age")
        with col2:
            systolicBP = st.text_input('Systolic BP (mmHg)', key="systolicBP")
        with col3:
            diastolicBP = st.text_input('Diastolic BP (mmHg)', key="diastolicBP")

        with col1:
            BS = st.text_input('Blood Glucose (mmol/L)')
        with col2:
            bodyTemp = st.text_input('Body Temperature (°F)')
        with col3:
            heartRate = st.text_input('Heart Rate (bpm)')

    def predict_risk(input_data):
        input_array = np.array(input_data, dtype=float).reshape(1, -1)
        input_array = scaler.transform(input_array)
        prediction = maternal_model.predict(input_array)
        return label_encoder.inverse_transform(prediction)[0]

    if st.button('🔍 Predict Pregnancy Risk'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                predicted_risk = predict_risk([age, systolicBP, diastolicBP, BS, bodyTemp, heartRate])
                if predicted_risk == "low risk":
                    st.success("🟢 Low Risk")
                elif predicted_risk == "mid risk":
                    st.warning("🟠 Medium Risk")
                else:
                    st.error("🔴 High Risk")
            except Exception as e:
                st.error(f"Prediction error: {e}")

    if st.button("🔁 Clear"):
        st.experimental_rerun()

# --- Fetal Health Prediction Page ---
if selected == '📄Fetal Health Prediction':
    st.title(' Fetal Health Prediction')
    st.markdown("Enter CTG report values to assess fetal well-being.")

    # Add CTG reference image
    st.image("ctg_report.png", caption="📄 Sample CTG Report", width=400)

    features = [
        'Baseline Value', 'Accelerations', 'Fetal Movement', 'Uterine Contractions',
        'Light Decelerations', 'Severe Decelerations', 'Prolonged Decelerations',
        'Abnormal Short Term Variability', 'Mean Value Of Short Term Variability',
        'Percentage Of Time With ALTV', 'Mean Value Long Term Variability',
        'Histogram Width', 'Histogram Min', 'Histogram Max', 'Histogram Number Of Peaks',
        'Histogram Number Of Zeroes', 'Histogram Mode', 'Histogram Mean',
        'Histogram Median', 'Histogram Variance', 'Histogram Tendency'
    ]

    user_inputs = []
    st.markdown("####  Enter the CTG Parameters:")
    for i in range(0, len(features), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(features):
                user_inputs.append(cols[j].text_input(features[i + j]))

    if st.button('🔍 Predict Fetal Health'):
        try:
            input_array = np.array(user_inputs, dtype=float).reshape(1, -1)
            input_array = scaler_f.transform(input_array)
            predicted = fetal_model.predict(input_array)[0]

            if predicted == 0:
                st.success("🟢 Normal")
            elif predicted == 1:
                st.warning("🟠 Suspect")
            else:
                st.error("🔴 Pathological")
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("🔁 Clear"):
        st.experimental_rerun()

# --- Dashboard Page ---
if selected == "Dashboard":
    st.title("📊 Maternal Health Dashboard")
    st.markdown("Explore visual insights across regions.")

    api_key = "579b464db66ec23bdd00000139b0d95a6ee4441c5f37eeae13f3a0b2"
    api_endpoint = f"https://api.data.gov.in/resource/6d6a373a-4529-43e0-9cff-f39aa8aa5957?api-key={api_key}&format=csv"

    dashboard = MaternalHealthDashboard(api_endpoint)

    st.markdown("### Regional Bubble Chart")
    dashboard.create_bubble_chart()
    with st.expander("Show Bubble Data"):
        st.markdown(f"<div style='white-space: pre-wrap;'>{dashboard.get_bubble_chart_data()}</div>", unsafe_allow_html=True)

    st.markdown("### Pregnancy Risk Pie Chart")
    dashboard.create_pie_chart()
    with st.expander("Show Pie Data"):
        st.markdown(f"<div style='white-space: pre-wrap;'>{dashboard.get_pie_graph_data()}</div>", unsafe_allow_html=True)
