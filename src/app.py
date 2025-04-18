import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time
import base64

# Page configuration
st.set_page_config(
    page_title="Heart Health Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
def local_css():
    st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --primary: #ff4b4b;
            --secondary: #f7f7f7;
            --text: #2c3e50;
            --accent: #ff7e5f;
            --success: #28a745;
            --warning: #ffc107;
            --danger: #dc3545;
        }

        /* Main title and headers */
        h1 {
            color: var(--primary);
            font-weight: 700;
            margin-bottom: 1.5rem;
            text-align: center;
            font-family: 'Helvetica Neue', sans-serif;
        }

        h2, h3 {
            color: var(--text);
            font-weight: 600;
            margin-top: 1rem;
        }

        /* Cards for sections */
        .stCard {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }

        /* Inputs styling */
        .stNumberInput, .stSelectbox {
            padding: 0.5rem;
        }

        /* Make the sidebar more attractive */
        .css-1d391kg {
            background-color: var(--secondary);
        }

        /* Result container */
        .result-card {
            padding: 1.5rem;
            border-radius: 10px;
            margin-top: 1.5rem;
            text-align: center;
        }

        /* Custom button */
        .stButton>button {
            background-color: var(--primary);
            color: white;
            font-weight: 600;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 30px;
            transition: all 0.3s;
            width: 100%;
            font-size: 1.2rem;
        }

        .stButton>button:hover {
            background-color: var(--accent);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        /* Progress bar styling */
        .stProgress .st-bo {
            background-color: var(--primary);
        }

        /* Tooltip font */
        .tooltip {
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted black;
        }

        /* Gauge chart containers */
        .gauge-container {
            display: flex;
            justify-content: center;
            margin-bottom: 1rem;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: gray;
            font-size: 0.8rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #eee;
        }

        /* Risk assessment result */
        .risk-result {
            font-size: 2rem;
            font-weight: 700;
            margin: 1rem 0;
        }

        .high-risk {
            color: var(--danger);
        }

        .low-risk {
            color: var(--success);
        }

        /* Form layouts */
        .form-row {
            display: flex;
            gap: 1rem;
        }

        /* Metric cards */
        .metric-card {
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            background: #f8f9fa;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--primary);
        }

        .metric-title {
            font-size: 0.9rem;
            color: #6c757d;
        }

        /* Loading animation */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        .pulse {
            animation: pulse 1.5s infinite;
        }

        /* Additional styling */
        .recommendation {
            background-color: #f8f9fa;
            border-left: 4px solid var(--primary);
            padding: 1rem;
            margin: 1rem 0;
        }

        /* Input labels */
        .input-label {
            font-weight: 500;
            margin-bottom: 0.25rem;
            color: var(--text);
        }

        /* Info tooltips */
        .info-tooltip {
            font-size: 0.8rem;
            color: #6c757d;
            margin-top: 0.25rem;
        }

        /* Display on mobile */
        @media (max-width: 768px) {
            .form-row {
                flex-direction: column;
            }
        }
    </style>
    """, unsafe_allow_html=True)


local_css()


# Load a decorative image and logo
@st.cache_data
def get_base64_img(img_path):
    try:
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except:
        return None


def set_background():
    st.markdown("""
    <style>
    .main::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url("https://img.freepik.com/free-vector/abstract-medical-wallpaper-template-design_53876-61805.jpg?w=1380&t=st=1682547800~exp=1682548400~hmac=72ed9a0711ca3f3ef5e849a2b5b83d77e4bbcb74c97d50d3d4f56ac70192ae6b");
        background-size: cover;
        background-position: center;
        opacity: 0.05;
        z-index: -1;
    }
    </style>
    """, unsafe_allow_html=True)


set_background()


# Load artifacts
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('models/model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        model_features = joblib.load('features/model_features.pkl')
        return model, scaler, model_features
    except Exception as e:
        st.error(f"Failed to load model artifacts: {str(e)}")
        return None, None, None


model, scaler, model_features = load_model_artifacts()

# Create sidebar
with st.sidebar:
    st.image(
        "https://img.freepik.com/free-vector/cardiology-concept-illustration_114360-7157.jpg?w=826&t=st=1682547699~exp=1682548299~hmac=33af776c7edeb24a7aff7f9a7a9bdd65e9e1ca4d1e4d8fbe7ee8b3f351e0c5ab",
        width=280)

    st.markdown("## About the App")
    st.markdown("""
    This application provides a risk assessment for heart disease based on personal health metrics and advanced algorithms. It offers an initial screening tool but is not a substitute for professional medical diagnosis.

    ### Features:
    - Real-time risk assessment
    - Health metrics visualization
    - Personalized recommendations
    """)

    st.markdown("### How to use:")
    st.markdown("""
    1. Enter your health information in the form
    2. Review the visualized metrics
    3. Click 'Assess Risk' for your results
    4. Read the personalized recommendations
    """)

    st.markdown("---")
    st.markdown("### ❗ Disclaimer")
    st.markdown("""
    This tool is for informational purposes only and not a replacement for professional medical advice. Always consult with a healthcare provider for medical conditions.
    """)

    # Add contact info
    st.markdown("---")
    st.markdown("### Contact")
    st.markdown("For questions or support: support@hearthealth.ai")


# Feature engineering functions
def engineer_features(input_df):
    df_fe = input_df.copy()

    # ---- Check required columns ----
    required_columns = ['HeartRate', 'Age', 'BloodPressure', 'Cholesterol', 'QuantumPatternFeature']
    missing = [col for col in required_columns if col not in df_fe.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # ---- Heart Rate Based Features ----
    df_fe['IsTachycardic'] = (df_fe['HeartRate'] > 100).astype(int)
    df_fe['IsBradycardic'] = (df_fe['HeartRate'] < 60).astype(int)
    df_fe['PercentMaxHR'] = df_fe['HeartRate'] / np.clip((220 - df_fe['Age']), 1e-5, None)
    df_fe['HR_per_BP'] = df_fe['HeartRate'] / (df_fe['BloodPressure'] + 1e-5)

    # ---- QuantumPatternFeature Transformations ----
    df_fe['QPF_squared'] = df_fe['QuantumPatternFeature'] ** 2
    df_fe['QPF_log'] = np.where(df_fe['QuantumPatternFeature'] > -1,
                                np.log1p(df_fe['QuantumPatternFeature']), np.nan)

    # ---- Interaction Features ----
    df_fe['Age_Cholesterol'] = df_fe['Age'] * df_fe['Cholesterol']
    df_fe['Age_BP'] = df_fe['Age'] * df_fe['BloodPressure']
    df_fe['BP_Cholesterol'] = df_fe['BloodPressure'] * df_fe['Cholesterol']
    df_fe['Cholesterol_minus_BP'] = df_fe['Cholesterol'] - df_fe['BloodPressure']
    df_fe['HR_Cholesterol'] = df_fe['HeartRate'] * df_fe['Cholesterol']

    # ---- Clinical Risk Proxy Score ----
    df_fe['ClinicalRiskScore'] = (
            0.02 * df_fe['Age'] +
            0.03 * df_fe['BloodPressure'] +
            0.04 * df_fe['Cholesterol'] +
            0.05 * df_fe['HeartRate']
    )

    # ---- Convert Binary Columns to Int ----
    for col in df_fe.columns:
        if df_fe[col].nunique() == 2 and df_fe[col].dtype != 'int':
            df_fe[col] = df_fe[col].fillna(0).astype(int)

    df_fe['QPF_log'].fillna(df_fe['QPF_log'].mean(), inplace=True)

    return df_fe


# Function to get health metrics interpretation
def get_health_metrics_status(age, gender, heart_rate, bp, chol):
    # Heart rate interpretation
    if heart_rate < 60:
        hr_status = "Bradycardia (low)"
        hr_icon = "⚠️"
    elif heart_rate > 100:
        hr_status = "Tachycardia (high)"
        hr_icon = "⚠️"
    else:
        hr_status = "Normal"
        hr_icon = "✅"

    # Blood pressure interpretation
    if bp < 90:
        bp_status = "Low"
        bp_icon = "⚠️"
    elif bp < 120:
        bp_status = "Normal"
        bp_icon = "✅"
    elif bp < 130:
        bp_status = "Elevated"
        bp_icon = "⚠️"
    elif bp < 140:
        bp_status = "High (Stage 1)"
        bp_icon = "⚠️"
    else:
        bp_status = "High (Stage 2)"
        bp_icon = "⚠️"

    # Cholesterol interpretation
    if chol < 200:
        chol_status = "Desirable"
        chol_icon = "✅"
    elif chol < 240:
        chol_status = "Borderline High"
        chol_icon = "⚠️"
    else:
        chol_status = "High"
        chol_icon = "⚠️"

    return {
        "hr_status": hr_status,
        "hr_icon": hr_icon,
        "bp_status": bp_status,
        "bp_icon": bp_icon,
        "chol_status": chol_status,
        "chol_icon": chol_icon
    }


# Function to generate gauge chart
def create_gauge_chart(value, title, min_val, max_val, green_threshold, red_threshold, is_inverted=False):
    if is_inverted:
        # For metrics where lower is better
        color = "green" if value <= green_threshold else ("yellow" if value <= red_threshold else "red")
    else:
        # For metrics where higher is better
        color = "green" if value >= green_threshold else ("yellow" if value >= red_threshold else "red")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': color},
            'steps': [
                {'range': [min_val, (max_val - min_val) / 3 + min_val], 'color': "lightgray"},
                {'range': [(max_val - min_val) / 3 + min_val, 2 * (max_val - min_val) / 3 + min_val], 'color': "gray"},
                {'range': [2 * (max_val - min_val) / 3 + min_val, max_val], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': red_threshold if not is_inverted else green_threshold
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
    return fig


# Function to generate personalized recommendations based on risk factors
def get_recommendations(age, gender, heart_rate, bp, chol, qpf, risk_score):
    recommendations = []

    # Age-related recommendations
    if age > 45 and gender == "Male" or age > 55 and gender == "Female":
        recommendations.append("Due to your age, consider more frequent heart health check-ups.")

    # Heart rate recommendations
    if heart_rate < 60:
        recommendations.append(
            "Your resting heart rate is low (bradycardia). This may be normal for athletes but could require medical attention.")
    elif heart_rate > 100:
        recommendations.append(
            "Your resting heart rate is elevated (tachycardia). Consider discussing with a healthcare provider.")

    # Blood pressure recommendations
    if bp >= 130:
        recommendations.append(
            "Your blood pressure is elevated. Consider lifestyle modifications such as reducing sodium intake and regular exercise.")

    # Cholesterol recommendations
    if chol >= 200:
        recommendations.append(
            "Your cholesterol level is above optimal. Consider dietary changes and discussing with your healthcare provider.")

    # General recommendations
    recommendations.append("Maintain a heart-healthy diet rich in fruits, vegetables, whole grains, and lean proteins.")
    recommendations.append("Aim for at least 150 minutes of moderate-intensity physical activity per week.")
    recommendations.append("Manage stress through techniques like meditation, deep breathing, or yoga.")
    recommendations.append("Ensure adequate sleep of 7-8 hours per night.")

    if risk_score > 0.3:
        recommendations.append(
            "Based on your risk assessment, we strongly recommend scheduling a comprehensive check-up with a cardiologist.")

    return recommendations


# Main app layout
st.markdown("# ❤️ Heart Disease Risk Assessment")

# Introduction text
st.markdown("""
<div class="stCard">
This interactive tool analyzes your health metrics using advanced algorithms to estimate your heart disease risk. 
Complete the form below for a personalized assessment and recommendations.
</div>
""", unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📋 Health Information", "📊 Risk Assessment", "📚 Education"])

with tab1:
    st.markdown("## Your Health Information")
    st.markdown("<p>Please enter your health metrics accurately for the most reliable assessment.</p>",
                unsafe_allow_html=True)

    # Create two columns for the form
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.markdown("### Personal Information")

        age = st.number_input('Age', 18, 100, 50, help="Your current age in years")

        gender = st.selectbox('Gender', ['Male', 'Female'],
                              help="Select your biological gender for risk assessment purposes")

        # Additional fields for more comprehensive assessment
        height = st.number_input('Height (cm)', 100, 250, 170,
                                 help="Your height in centimeters")

        weight = st.number_input('Weight (kg)', 30, 200, 70,
                                 help="Your weight in kilograms")

        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)

        smoking = st.selectbox('Smoking Status',
                               ['Never Smoked', 'Former Smoker', 'Current Smoker'],
                               help="Your current smoking status")

        diabetes = st.checkbox('Diagnosed with Diabetes',
                               help="Check if you have been diagnosed with diabetes")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.markdown("### Clinical Measurements")

        heart_rate = st.number_input('Resting Heart Rate (bpm)', 40, 200, 70,
                                     help="Your resting heart rate in beats per minute")

        bp = st.number_input('Systolic Blood Pressure (mmHg)', 80, 220, 120,
                             help="The top number in your blood pressure reading")

        chol = st.number_input('Total Cholesterol (mg/dL)', 100, 600, 200,
                               help="Your total cholesterol level from a blood test")

        hdl = st.number_input('HDL Cholesterol (mg/dL)', 20, 100, 50,
                              help="High-density lipoprotein or 'good' cholesterol")

        ldl = st.number_input('LDL Cholesterol (mg/dL)', 30, 300, 100,
                              help="Low-density lipoprotein or 'bad' cholesterol")

        qpf = st.number_input('Quantum Pattern Feature', 0.0, 10.0, 0.5,
                              help="Advanced biomarker used in risk assessment")

        st.markdown('</div>', unsafe_allow_html=True)

    # Metrics visualization
    st.markdown("## Health Metrics Visualization")

    # Get metrics status
    metrics_status = get_health_metrics_status(age, gender, heart_rate, bp, chol)

    # Display metrics status in a visually appealing way
    st.markdown('<div class="stCard">', unsafe_allow_html=True)

    # Create multiple columns for displaying metrics
    metric_cols = st.columns(3)

    with metric_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{heart_rate} bpm {metrics_status['hr_icon']}</div>
            <div class="metric-title">Heart Rate</div>
            <div class="info-tooltip">{metrics_status['hr_status']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Add gauge chart
        hr_gauge = create_gauge_chart(
            heart_rate, "Heart Rate (bpm)", 40, 200,
            60, 100, is_inverted=False
        )
        st.plotly_chart(hr_gauge, use_container_width=True)

    with metric_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{bp} mmHg {metrics_status['bp_icon']}</div>
            <div class="metric-title">Blood Pressure</div>
            <div class="info-tooltip">{metrics_status['bp_status']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Add gauge chart
        bp_gauge = create_gauge_chart(
            bp, "Blood Pressure (mmHg)", 80, 220,
            120, 140, is_inverted=True
        )
        st.plotly_chart(bp_gauge, use_container_width=True)

    with metric_cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{chol} mg/dL {metrics_status['chol_icon']}</div>
            <div class="metric-title">Cholesterol</div>
            <div class="info-tooltip">{metrics_status['chol_status']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Add gauge chart
        chol_gauge = create_gauge_chart(
            chol, "Cholesterol (mg/dL)", 100, 300,
            200, 240, is_inverted=True
        )
        st.plotly_chart(chol_gauge, use_container_width=True)

    # Add BMI visualization
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f"### Body Mass Index (BMI): {bmi:.1f}")

    # BMI categories
    bmi_fig = go.Figure()

    # Add BMI ranges
    categories = ["Underweight", "Normal", "Overweight", "Obese", "Severely Obese"]
    ranges = [0, 18.5, 25, 30, 35, 40]
    colors = ["blue", "green", "yellow", "orange", "red"]

    for i in range(len(categories)):
        bmi_fig.add_trace(go.Bar(
            x=[ranges[i + 1] - ranges[i]],
            y=[0.5],
            orientation='h',
            base=ranges[i],
            marker_color=colors[i],
            text=categories[i],
            textposition='inside',
            hoverinfo='text',
            name=categories[i]
        ))

    # Add marker for user's BMI
    bmi_fig.add_trace(go.Scatter(
        x=[bmi],
        y=[0.5],
        mode='markers',
        marker=dict(size=15, color='black', symbol='triangle-down'),
        hoverinfo='text',
        hovertext=f'Your BMI: {bmi:.1f}',
        name='Your BMI'
    ))

    bmi_fig.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=10, b=20),
        showlegend=False,
        xaxis=dict(
            range=[0, 40],
            title="BMI",
            showgrid=False
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False
        ),
        barmode='stack'
    )

    st.plotly_chart(bmi_fig, use_container_width=True)

    if bmi < 18.5:
        bmi_category = "underweight"
    elif bmi < 25:
        bmi_category = "normal weight"
    elif bmi < 30:
        bmi_category = "overweight"
    elif bmi < 35:
        bmi_category = "obese"
    else:
        bmi_category = "severely obese"

    st.markdown(f"Your BMI of {bmi:.1f} indicates you are **{bmi_category}**.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ... (keep all previous imports and setup code)

with tab2:
    st.markdown("## Heart Disease Risk Assessment")
    st.markdown("<p>Click the button below to analyze your risk factors and receive a personalized assessment.</p>",
                unsafe_allow_html=True)

    # Create input DataFrame
    input_data = pd.DataFrame([[
        age,
        1 if gender == 'Male' else 0,
        heart_rate,
        bp,
        chol,
        qpf
    ]], columns=['Age', 'Gender', 'HeartRate', 'BloodPressure',
                 'Cholesterol', 'QuantumPatternFeature'])

    # Assessment button
    if st.button('Assess My Risk', key='assess_button'):
        with st.spinner('Analyzing your health data...'):
            # Show progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulate processing time
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            try:
                # Feature engineering and scaling
                gender_value = np.array([[1 if gender == 'Male' else 0]])
                scaled_gender = scaler.transform(gender_value)
                input_data['Gender'] = scaled_gender[0][0]

                processed_data = engineer_features(input_data)
                processed_data = processed_data.reindex(columns=model_features, fill_value=0)

                # Prediction
                proba = model.predict_proba(processed_data)[0][1]
                risk_level = "High" if proba >= 0.5 else "Low"

                # Risk result card
                risk_color = "#ff4b4b" if risk_level == "High" else "#28a745"
                st.markdown(f"""
                <div class="stCard result-card" style="background-color: {risk_color}10; border-left: 5px solid {risk_color};">
                    <div class="risk-result" style="color: {risk_color}">{risk_level} Risk Assessment</div>
                    <div style="font-size: 1.2rem; margin: 1rem 0;">Estimated probability: {proba:.1%}</div>
                    <div style="margin-top: 1rem; color: {risk_color};">
                        {("We strongly recommend consulting with a healthcare professional."
                                                                                                                                                                                                                                                                                                                                                                                                                                                      if risk_level == "High" else
                                                                                                                                                                                                                                                                                                                                                                                                                                                      "Continue maintaining healthy habits and regular check-ups.")}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Feature importance visualization
                st.markdown("### Risk Factor Contribution")
                feature_importance = {
                    "Age": 0.2 * age / 100,
                    "Blood Pressure": 0.15 * (bp - 90) / 130,
                    "Cholesterol": 0.18 * (chol - 150) / 300,
                    "Heart Rate": 0.12 * abs(heart_rate - 70) / 130,
                    "Gender": 0.1 * (gender == "Male"),
                    "QPF": 0.08 * qpf / 10,
                    "BMI": 0.17 * (bmi - 18.5) / 25
                }

                # Normalize and create visualization
                factors_df = pd.DataFrame({
                    'Factor': list(feature_importance.keys()),
                    'Impact': [x * proba / sum(feature_importance.values()) for x in feature_importance.values()]
                }).sort_values('Impact', ascending=False)

                fig = px.bar(factors_df, x='Impact', y='Factor', orientation='h',
                             color='Impact', color_continuous_scale=[(0, "#28a745"), (0.5, "#ffc107"), (1, "#dc3545")],
                             title='Contributing Risk Factors')
                fig.update_layout(coloraxis_showscale=False, height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Personalized recommendations
                recommendations = get_recommendations(age, gender, heart_rate, bp, chol, qpf, proba)

                st.markdown("## 🩺 Personalized Health Recommendations")
                with st.container():
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown("""
                        <div class="stCard" style="background: linear-gradient(135deg, #f8f9fa, #ffffff);">
                            <h3 style="color: #2c3e50;">Immediate Actions</h3>
                        """, unsafe_allow_html=True)
                        for rec in recommendations[:3]:
                            st.markdown(f"<div class='recommendation'>📌 {rec}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with cols[1]:
                        st.markdown("""
                        <div class="stCard" style="background: linear-gradient(135deg, #f8f9fa, #ffffff);">
                            <h3 style="color: #2c3e50;">Long-term Strategies</h3>
                        """, unsafe_allow_html=True)
                        for rec in recommendations[3:]:
                            st.markdown(f"<div class='recommendation'>📌 {rec}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                # Health timeline projection
                st.markdown("### 10-Year Health Projection")
                years = np.arange(1, 11)
                projected_risk = [proba * (1.15 ** year) for year in years]  # Simplified model

                projection_fig = go.Figure()
                projection_fig.add_trace(go.Scatter(
                    x=years, y=projected_risk,
                    mode='lines+markers',
                    name='Risk Trajectory',
                    line=dict(color='#ff4b4b', width=3)
                ))
                projection_fig.update_layout(
                    yaxis_tickformat=".0%",
                    xaxis_title="Years",
                    yaxis_title="Estimated Risk",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(projection_fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")

with tab3:
    st.markdown("## ❤️ Heart Health Education Hub")

    # Hero section
    st.markdown("""
    <div class="stCard" style="background: linear-gradient(135deg, #ff7e5f, #ff4b4b); color: white;">
        <h2 style="color: white;">Empowering Your Heart Health Journey</h2>
        <p>Essential knowledge and tools for maintaining cardiovascular wellness</p>
    </div>
    """, unsafe_allow_html=True)

    # Educational content sections
    with st.expander("📚 Understanding Cardiovascular Health", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            ### Key Concepts
            - **Atherosclerosis**: Plaque buildup in arteries
            - **Hypertension**: The silent killer
            - **Metabolic Syndrome**: Cluster of risk factors
            - **Inflammation Role**: Chronic inflammation's impact
            """)
        with cols[1]:
            st.image("https://img.freepik.com/free-vector/anatomy-heart_1308-89823.jpg",
                     caption="Heart Anatomy Illustration")

    with st.expander("🛡️ Prevention Strategies"):
        tabs = st.tabs(["Nutrition", "Exercise", "Stress Management"])
        with tabs[0]:
            st.markdown("""
            ### Heart-Healthy Nutrition
            - Mediterranean diet principles
            - Omega-3 rich foods
            - Sodium reduction strategies
            - Fiber importance
            """)
        with tabs[1]:
            st.markdown("""
            ### Effective Exercise Regimens
            - Aerobic vs resistance training
            - Target heart rate zones
            - Daily activity integration
            - Exercise safety tips
            """)
        with tabs[2]:
            st.markdown("""
            ### Stress Reduction Techniques
            - Mindfulness meditation
            - Breathing exercises
            - Sleep optimization
            - Work-life balance
            """)

    with st.expander("📈 Understanding Your Metrics"):
        st.markdown("""
        <div class="stCard">
            <div class="form-row">
                <div class="metric-card">
                    <div class="metric-value">Blood Pressure</div>
                    <div class="metric-title">Optimal: <120/80 mmHg</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">Cholesterol</div>
                    <div class="metric-title">Ideal: <200 mg/dL</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">BMI</div>
                    <div class="metric-title">Healthy: 18.5-24.9</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        ### Monitoring Guidelines
        - Blood pressure check frequency
        - Lipid profile testing intervals
        - Continuous glucose monitoring
        - Weight management strategies
        """)

    # Interactive quiz
    st.markdown("## � Heart Health Quiz")
    with st.form("health_quiz"):
        q1 = st.radio("Which is most effective for heart health?",
                      ["Walking daily", "Occasional intense exercise", "No exercise"])
        q2 = st.multiselect("Select heart-healthy foods:",
                            ["Salmon", "Avocado", "Processed meats", "Whole grains"])
        submitted = st.form_submit_button("Check Answers")
        if submitted:
            score = 0
            if q1 == "Walking daily": score += 1
            if set(q2) == {"Salmon", "Avocado", "Whole grains"}: score += 1
            st.success(f"Score: {score}/2 - {['Needs improvement', 'Good start', 'Excellent!'][score]}")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <div>© 2024 HeartAI | Clinical Validation Pending | Not FDA Approved</div>
    <div style="margin-top: 0.5rem;">
        <a href="#terms" style="color: #666; margin-right: 1rem;">Terms of Use</a>
        <a href="#privacy" style="color: #666; margin-right: 1rem;">Privacy Policy</a>
        <a href="#contact" style="color: #666;">Contact Support</a>
    </div>
</div>
""", unsafe_allow_html=True)