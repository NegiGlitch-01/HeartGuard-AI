
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="HeartGuard AI | Negi",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    model = joblib.load("SVM_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_models()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at top left, #172554 0%, transparent 40%),
        linear-gradient(135deg, #080d1b 0%, #101827 55%, #0b1220 100%);
    color: #f8fafc;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #172554, #111827 65%, #3f1728);
    border: 1px solid #293650;
    padding: 32px;
    border-radius: 22px;
    margin-bottom: 26px;
    box-shadow: 0 15px 45px #00000030;
}

.hero-tag {
    display: inline-block;
    color: #67e8f9;
    background: #164e633b;
    border: 1px solid #155e75;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.hero h1 {
    font-size: clamp(27px, 4vw, 43px);
    font-weight: 800;
    color: #ffffff;
    margin: 16px 0 8px;
}

.hero p {
    color: #bdc9dc;
    font-size: 15px;
    line-height: 1.7;
    margin-bottom: 0;
}

.section-card {
    background: #111c2d;
    border: 1px solid #29374b;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}

.section-title {
    color: #67e8f9;
    font-size: 19px;
    font-weight: 700;
    margin-bottom: 6px;
}

.section-subtitle {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 20px;
}

div[data-testid="stForm"] {
    background: #111c2d;
    border: 1px solid #29374b;
    border-radius: 18px;
    padding: 24px;
}

label, .stMarkdown p {
    color: #e2e8f0;
}

div[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #67e8f9;
}

.stButton > button,
.stFormSubmitButton > button {
    width: 100%;
    min-height: 50px;
    border: 0;
    border-radius: 12px;
    background: linear-gradient(90deg, #06b6d4, #2563eb);
    color: white;
    font-size: 16px;
    font-weight: 700;
    transition: 0.2s ease;
}

.stFormSubmitButton > button:hover {
    background: linear-gradient(90deg, #0891b2, #1d4ed8);
    border: 0;
    transform: translateY(-2px);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #0b1423;
    border-color: #34445b;
    border-radius: 9px;
}

div[data-testid="stAlert"] {
    border-radius: 14px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    border-top: 1px solid #29374b;
    padding-top: 20px;
    margin-top: 35px;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO HEADER ----------------
st.markdown("""
<div class="hero">
    <span class="hero-tag">✦ MACHINE LEARNING PROJECT</span>
    <h1>❤️ HeartGuard AI</h1>
    <p>
        Intelligent Heart Disease Risk Prediction<br>
        Enter the health parameters below to generate
        an ML model prediction.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- INFO CARDS ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="section-card">
        <div style="font-size:25px">🧠</div>
        <b>AI-Powered</b>
        <p style="color:#94a3b8;font-size:12px">
        SVM machine learning model
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="section-card">
        <div style="font-size:25px">📋</div>
        <b>11 Parameters</b>
        <p style="color:#94a3b8;font-size:12px">
        Health data-based prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="section-card">
        <div style="font-size:25px">⚡</div>
        <b>Instant Result</b>
        <p style="color:#94a3b8;font-size:12px">
        Prediction from your trained model
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- INPUT FORM ----------------
st.markdown("""
<div class="section-title">🩺 Patient Health Assessment</div>
<div class="section-subtitle">
Complete the fields using the available health information.
</div>
""", unsafe_allow_html=True)

with st.form("heart_prediction_form"):

    st.markdown("#### 👤 Basic Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 31)

    with col2:
        sex = st.selectbox("Sex", ["M", "F"])

    st.markdown("---")
    st.markdown("#### 🫀 Heart & Clinical Parameters")

    col1, col2 = st.columns(2)

    with col1:
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "TA", "ASY"],
            help="ATA: Atypical Angina | NAP: Non-Anginal Pain | "
                 "TA: Typical Angina | ASY: Asymptomatic"
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80, max_value=200, value=120
        )

        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=100, max_value=600, value=200
        )

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            [0, 1],
            format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
        )

    with col2:
        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"]
        )

        max_hr = st.slider(
            "Maximum Heart Rate",
            60, 220, 150
        )

        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["N", "Y"],
            format_func=lambda x: "Yes" if x == "Y" else "No"
        )

        oldpeak = st.slider(
            "Oldpeak (ST Depression)",
            0.0, 6.0, 1.0, step=0.1
        )

    st.markdown("---")
    st.markdown("#### 📈 ECG Stress Test Indicator")

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

    submitted = st.form_submit_button(
        "🚀  ANALYZE HEART RISK",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------
if submitted:
    try:
        raw_input = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            "Sex_" + sex: 1,
            "ChestPainType_" + chest_pain: 1,
            "RestingECG_" + resting_ecg: 1,
            "ExerciseAngina_" + exercise_angina: 1,
            "ST_Slope_" + st_slope: 1
        }

        input_df = pd.DataFrame([raw_input])

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        st.markdown("---")
        st.markdown(
            '<div class="section-title">📊 Prediction Report</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:
            st.error(
                "⚠️ **Higher-risk classification**\n\n"
                "The model classified this input as positive for "
                "the target condition. Please consult a qualified "
                "healthcare professional for proper assessment."
            )
        else:
            st.success(
                "✅ **Lower-risk classification**\n\n"
                "The model classified this input as negative for "
                "the target condition. This does not rule out "
                "heart disease."
            )

        with st.expander("🔍 View submitted model inputs"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that the model, scaler and columns files are "
            "compatible with the same training pipeline."
        )

# ---------------- DISCLAIMER & FOOTER ----------------
st.warning(
    "Medical disclaimer: This project is for educational and "
    "demonstration purposes only. It is not a diagnosis or a "
    "replacement for professional medical advice."
)

st.markdown("""
<div class="footer">
    Built with ❤️ by <b>Karan Negi👽🤷‍♂️</b><br>
    HeartGuard AI · Machine Learning · Streamlit
</div>
""", unsafe_allow_html=True)
