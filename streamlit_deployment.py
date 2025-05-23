import streamlit as st
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

@st.cache_data
def load_data():
    df = pd.read_csv("parkinsons_updrs.data")
    return df

df = load_data()

df['sex'] = df['sex'].astype(int)  
X = df[["age", "sex", "Jitter(%)", "Shimmer", "NHR", "HNR", "RPDE", "DFA", "PPE"]]
y = df['motor_UPDRS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_model():
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_model()

def extract_voice_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    features = {
        "Jitter(%)": librosa.feature.rms(y=y)[0].mean(),
        "Shimmer": librosa.feature.zero_crossing_rate(y)[0].mean(),
        "NHR": librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean(),
        "HNR": librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean(),
        "RPDE": librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean(),
        "DFA": librosa.feature.mfcc(y=y, sr=sr)[0].mean(),
        "PPE": librosa.feature.chroma_stft(y=y, sr=sr)[0].mean()
    }
    return features

# Severity classification
def classify_severity(score):
    if score <= 30:
        return "Mild", "🟢"
    elif score <= 70:
        return "Moderate", "🟡"
    else:
        return "Severe", "🔴"

# Streamlit UI
st.title("Parkinson's Disease Severity Predictor")
st.markdown("Predict Motor UPDRS score and severity using voice features.")

tab1, tab2 = st.tabs(["Upload Audio", "Manual Input"])

with tab1:
    st.header("Voice Recording Analysis")
    age = st.number_input("Enter Age", min_value=10, max_value=100, step=1)
    sex = st.radio("Select Sex", options=["Male", "Female"])
    sex_val = 1 if sex == "Male" else 0

    audio_file = st.file_uploader("Upload a voice recording (WAV format)", type=["wav"])
    if audio_file:
        st.audio(audio_file)
        if st.button("Analyze Voice"):
            features = extract_voice_features(audio_file)
            features["age"] = age
            features["sex"] = sex_val
            features_df = pd.DataFrame([features])[X_train.columns]
            prediction = model.predict(features_df)[0]
            severity, emoji = classify_severity(prediction)
            
            st.subheader("Results")
            st.metric("Predicted Motor UPDRS Score", f"{prediction:.2f}")
            st.metric("Severity", f"{severity} {emoji}")
            st.caption(f"UPDRS Range: 0 (Normal) - 132 (Severe)")

with tab2:
    st.header("Manual Feature Input")
    age = st.number_input("Enter Age", min_value=10, max_value=100, step=1, key="age2")
    sex = st.radio("Select Sex", options=["Male", "Female"], key="sex2")
    sex_val = 1 if sex == "Male" else 0

    jitter = st.slider("Jitter(%)", 0.0, 1.0, 0.5)
    shimmer = st.slider("Shimmer", 0.0, 1.0, 0.5)
    nhr = st.slider("NHR", 0.0, 1.0, 0.5)
    hnr = st.slider("HNR", 0.0, 1.0, 0.5)
    rpde = st.slider("RPDE", 0.0, 1.0, 0.5)
    dfa = st.slider("DFA", 0.0, 1.0, 0.5)
    ppe = st.slider("PPE", 0.0, 1.0, 0.5)
    
    if st.button("Predict"):
        features = {
            "age": age,
            "sex": sex_val,
            "Jitter(%)": jitter,
            "Shimmer": shimmer,
            "NHR": nhr,
            "HNR": hnr,
            "RPDE": rpde,
            "DFA": dfa,
            "PPE": ppe
        }
        features_df = pd.DataFrame([features])[X_train.columns]
        prediction = model.predict(features_df)[0]
        severity, emoji = classify_severity(prediction)
        
        st.subheader("Results")
        st.metric("Predicted Motor UPDRS Score", f"{prediction:.2f}")
        st.metric("Severity", f"{severity} {emoji}")

