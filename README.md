# ❤️ HeartGuard AI — Heart Disease Risk Prediction

An interactive Machine Learning web application built with Python and Streamlit to predict heart disease risk based on user-provided health parameters.

## 🚀 Features

- Interactive and user-friendly interface
- Heart disease risk prediction using an SVM model
- Health parameter input form
- Data preprocessing using a saved scaler
- Instant model predictions
- Prediction input summary
- Responsive dashboard design

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Support Vector Machine (SVM)

## 📋 Input Parameters

The application uses the following parameters:

- Age and Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- ST Slope

## 📂 Project Structure

```text
HeartGuard-AI/
├── app.py
├── requirements.txt
├── SVM_heart.pkl
├── scaler.pkl
├── columns.pkl
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/NegiGlitch-01/HeartGuard-AI.git
cd HeartGuard-AI
```

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

## 🧠 Machine Learning

The application uses a trained Support Vector Machine (SVM) model. A saved scaler transforms the input features, and a saved column list ensures that the model receives features in the expected order.

## 👨‍💻 Author

**Karan Negi**

GitHub: [NegiGlitch-01](https://github.com/NegiGlitch-01)

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes only. It is not a medical diagnostic tool and must not replace professional medical advice.
