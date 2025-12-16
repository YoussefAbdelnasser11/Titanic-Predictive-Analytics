import streamlit as st
import joblib
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

model = joblib.load('../models/best_model.pkl')

st.title("🚢 Titanic Survival Predictor")

img = Image.open('assets/ship.png')
st.image(img, use_column_width=True)

st.sidebar.header("Passenger Information")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.slider("Age", 1, 80, 25)
fare = st.sidebar.slider("Fare", 0.0, 500.0, 50.0)
sibsp = st.sidebar.slider("Siblings/Spouses", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children", 0, 6, 0)

sex = 1 if sex == "Male" else 0

input_data = np.array([[pclass, sex, age, fare, sibsp, parch]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"🟢 Passenger Survived (Probability: {prob:.2f})")
    else:
        st.error(f"🔴 Passenger Did Not Survive (Probability: {prob:.2f})")
